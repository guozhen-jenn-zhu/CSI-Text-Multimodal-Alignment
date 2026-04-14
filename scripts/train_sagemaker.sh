#!/usr/bin/env bash
#
# End-to-end training pipeline for SageMaker Studio g5 instances.
# Tested on: g5.8xlarge (1x A10G), g5.24xlarge (4x), g5.48xlarge (8x).
#
# Stages:
#   0. Create symlinks for multitask shared H5 files
#   1. Install dependencies
#   2. Check pair registry exists (error if missing)
#   3. Generate precomputed text embeddings (if missing)
#   4. Launch multi-GPU training with torchrun
#
# Usage:
#   # Train HAR task (default)
#   bash scripts/train_sagemaker.sh
#
#   # Train a specific task
#   TASK=fall_detection bash scripts/train_sagemaker.sh
#
#   # Train all CSI-Bench tasks sequentially
#   TASK=all bash scripts/train_sagemaker.sh
#
#   # Customize method and encoder
#   TASK=har METHOD=siglip ENCODER=patchtst bash scripts/train_sagemaker.sh

set -euo pipefail

# ------------------------------------------------------------------
# Configuration (override via environment variables)
# ------------------------------------------------------------------
DATASET="${DATASET:-csibench}"
TASK="${TASK:-har}"
METHOD="${METHOD:-clip_infonce}"
ENCODER="${ENCODER:-resnet18_1d}"
TEXT_ENCODER="${TEXT_ENCODER:-clip_vit_b32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Task definitions: task_name -> data_root_suffix
declare -A CSIBENCH_STANDALONE_TASKS=(
    [fall_detection]="csibench/fall_detection"
    [motion_source]="csibench/motion_source"
)
declare -A CSIBENCH_MULTITASK_TASKS=(
    [har]="csibench/multitask"
    [proximity]="csibench/multitask"
    [user_id]="csibench/multitask"
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

detect_gpus() {
    local count
    count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null) || true
    if [[ -z "$count" ]] || [[ "$count" -le 0 ]]; then
        count=$(nvidia-smi -L 2>/dev/null | wc -l) || true
    fi
    if [[ -z "$count" ]] || [[ "$count" -le 0 ]]; then
        count=1
    fi
    echo "$count"
}

resolve_task_root() {
    # Returns the task_root directory containing metadata/ and splits/
    # For multitask: the per-task subfolder (H5 files accessed via symlink)
    # For standalone: the task folder directly
    local task_name="$1"
    if [[ -v CSIBENCH_STANDALONE_TASKS[$task_name] ]]; then
        echo "${DATA_ROOT}/${CSIBENCH_STANDALONE_TASKS[$task_name]}"
    elif [[ -v CSIBENCH_MULTITASK_TASKS[$task_name] ]]; then
        echo "${DATA_ROOT}/${CSIBENCH_MULTITASK_TASKS[$task_name]}/${task_name}"
    else
        echo "${DATA_ROOT}/${DATASET}/${task_name}"
    fi
}

resolve_registry_dir() {
    # Registry lives alongside the task metadata
    resolve_task_root "$1"
}

# ------------------------------------------------------------------
# Stage 0: Create symlinks for multitask shared H5 directory
# ------------------------------------------------------------------
setup_symlinks() {
    local mt_root="${DATA_ROOT}/csibench/multitask"
    local shared_h5="${mt_root}/sub_Human"

    if [[ ! -d "$shared_h5" ]]; then
        log "WARNING: Shared H5 directory not found at ${shared_h5}"
        return
    fi

    for task_dir in har proximity user_id; do
        local link_target="${mt_root}/${task_dir}/sub_Human"
        if [[ -L "$link_target" ]]; then
            continue
        fi
        if [[ -d "${mt_root}/${task_dir}" ]] && [[ ! -e "$link_target" ]]; then
            ln -s ../sub_Human "$link_target"
            log "Created symlink: ${link_target} -> ../sub_Human"
        fi
    done
}

# ------------------------------------------------------------------
# Stage 1: Install dependencies
# ------------------------------------------------------------------
install_deps() {
    log "Stage 1: Installing dependencies"
    cd "$REPO_ROOT"
    pip install -q -r requirements.txt 2>&1 | tail -5
    log "Dependencies installed"
}

# ------------------------------------------------------------------
# Stage 2: Check pair registry exists
# ------------------------------------------------------------------
check_registry() {
    local task_name="$1"
    local reg_dir
    reg_dir=$(resolve_registry_dir "$task_name")
    local reg_path="${reg_dir}/pair_registry.json"

    if [[ -f "$reg_path" ]]; then
        log "Pair registry found: ${reg_path}"
        return
    fi

    log "ERROR: Pair registry not found at: ${reg_path}"
    log "Please add pair_registry.json to the dataset before training."
    log "You can generate it with:"
    log "  cd ${REPO_ROOT}"
    log "  PYTHONPATH=. python scripts/generate_all_texts.py \\"
    log "      --datasets ${DATASET} --tasks ${task_name} \\"
    log "      --data_root ${DATA_ROOT}/csibench --output_dir ${reg_dir}"
    exit 1
}

# ------------------------------------------------------------------
# Stage 3: Generate text embeddings (if missing)
# ------------------------------------------------------------------
generate_embeddings() {
    local task_name="$1"
    local reg_dir
    reg_dir=$(resolve_registry_dir "$task_name")
    local reg_path="${reg_dir}/pair_registry.json"
    local emb_path="${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5"

    if [[ -f "$emb_path" ]]; then
        log "Text embeddings already exist: ${emb_path}"
        return
    fi

    if [[ ! -f "$reg_path" ]]; then
        log "ERROR: Cannot generate embeddings — pair registry not found: ${reg_path}"
        exit 1
    fi

    log "Generating text embeddings for ${DATASET}/${task_name} with ${TEXT_ENCODER}"
    cd "$REPO_ROOT"
    python -m scripts.generate_text_embeddings \
        --registry_path "$reg_path" \
        --output_path "$emb_path" \
        --encoder "$TEXT_ENCODER" \
        --batch_size 256

    log "Text embeddings saved: ${emb_path}"
}

# ------------------------------------------------------------------
# Stage 4: Launch training
# ------------------------------------------------------------------
train_task() {
    local task_name="$1"
    local num_gpus
    num_gpus=$(detect_gpus)

    local task_root
    task_root=$(resolve_task_root "$task_name")
    local reg_dir
    reg_dir=$(resolve_registry_dir "$task_name")
    local reg_path="${reg_dir}/pair_registry.json"
    local emb_path="${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5"

    log "Stage 4: Training ${DATASET}/${task_name} | method=${METHOD} encoder=${ENCODER}"
    log "  GPUs: ${num_gpus} | batch_size: ${BATCH_SIZE} (per GPU) | effective: $((BATCH_SIZE * num_gpus))"
    log "  task_root: ${task_root}"
    log "  registry:  ${reg_path}"
    log "  embeddings: ${emb_path}"

    cd "$REPO_ROOT"

    local overrides=(
        "dataset=${DATASET}"
        "task=${task_name}"
        "method=${METHOD}"
        "csi_encoder.name=${ENCODER}"
        "text_encoder.name=${TEXT_ENCODER}"
        "task_root=${task_root}"
        "registry_path=${reg_path}"
        "text_embeddings_path=${emb_path}"
        "training.batch_size=${BATCH_SIZE}"
        "training.max_epochs=${MAX_EPOCHS}"
        "training.learning_rate=${LR}"
        "training.num_workers=${NUM_WORKERS}"
        "logging.checkpoint_dir=${MODEL_ROOT}/checkpoints"
    )

    if [[ "$num_gpus" -gt 1 ]]; then
        python -m torch.distributed.run \
            --nproc_per_node "$num_gpus" \
            --master_port 29500 \
            -m training.train \
            --config "$CONFIG" \
            --override "${overrides[@]}"
    else
        python -m training.train \
            --config "$CONFIG" \
            --override "${overrides[@]}"
    fi

    log "Training complete for ${DATASET}/${task_name}"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    log "=== CSI-Text Multimodal Alignment Training Pipeline ==="
    log "GPU diagnostics:"
    nvidia-smi -L 2>/dev/null || log "  nvidia-smi not available"
    python3 -c "import torch; print(f'  PyTorch CUDA available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}')" 2>/dev/null || true
    log "Detected GPUs: $(detect_gpus)"

    install_deps
    setup_symlinks

    local tasks=()
    if [[ "$TASK" == "all" ]]; then
        tasks=(har proximity user_id fall_detection motion_source)
    else
        tasks=("$TASK")
    fi

    for t in "${tasks[@]}"; do
        log "=========================================="
        log "Processing task: ${t}"
        log "=========================================="
        check_registry "$t"
        generate_embeddings "$t"
        train_task "$t"
    done

    log "=== All tasks complete ==="
}

main "$@"
