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
#   bash scripts/pipelines/sagemaker_scripts/train_sagemaker.sh
#
#   # Train a specific task
#   TASK=fall_detection bash scripts/pipelines/sagemaker_scripts/train_sagemaker.sh
#
#   # Train all CSI-Bench tasks sequentially
#   TASK=all bash scripts/pipelines/sagemaker_scripts/train_sagemaker.sh
#
#   # Train specific CSI-Bench tasks (comma-separated)
#   TASK=har,fall_detection,proximity bash scripts/pipelines/sagemaker_scripts/train_sagemaker.sh
#
#   # Train SignFI
#   DATASET=signfi TASK=sign_recognition bash scripts/pipelines/sagemaker_scripts/train_sagemaker.sh
#
#   # Train XRF55
#   DATASET=xrf55 TASK=action_recognition bash scripts/pipelines/sagemaker_scripts/train_sagemaker.sh
#
#   # Customize method and encoder
#   TASK=har METHOD=siglip ENCODER=patchtst bash scripts/pipelines/sagemaker_scripts/train_sagemaker.sh

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
MAX_EPOCHS="${MAX_EPOCHS:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_configs/sagemaker_g5.yaml"

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
declare -A SIGNFI_TASKS=(
    [sign_recognition]="signfi/sign_recognition"
)
declare -A XRF55_TASKS=(
    [action_recognition]="xrf55/action_recognition"
)

ALL_CSIBENCH_TASKS=(har proximity user_id fall_detection motion_source)
ALL_SIGNFI_TASKS=(sign_recognition)
ALL_XRF55_TASKS=(action_recognition)

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
    # Returns the task_root directory containing metadata/ and splits/.
    # Task names are unique across datasets, so we check all maps.
    local task_name="$1"
    if [[ -n "${CSIBENCH_STANDALONE_TASKS[$task_name]+x}" ]]; then
        echo "${DATA_ROOT}/${CSIBENCH_STANDALONE_TASKS[$task_name]}"
    elif [[ -n "${CSIBENCH_MULTITASK_TASKS[$task_name]+x}" ]]; then
        echo "${DATA_ROOT}/${CSIBENCH_MULTITASK_TASKS[$task_name]}/${task_name}"
    elif [[ -n "${SIGNFI_TASKS[$task_name]+x}" ]]; then
        echo "${DATA_ROOT}/${SIGNFI_TASKS[$task_name]}"
    elif [[ -n "${XRF55_TASKS[$task_name]+x}" ]]; then
        echo "${DATA_ROOT}/${XRF55_TASKS[$task_name]}"
    else
        echo "${DATA_ROOT}/${DATASET}/${task_name}"
    fi
}

resolve_registry_dir() {
    # Registry lives alongside the task metadata
    resolve_task_root "$1"
}

resolve_dataset_for_task() {
    # Maps a task name to its parent dataset identifier.
    local task_name="$1"
    if [[ -n "${CSIBENCH_STANDALONE_TASKS[$task_name]+x}" ]] \
    || [[ -n "${CSIBENCH_MULTITASK_TASKS[$task_name]+x}" ]]; then
        echo "csibench"
    elif [[ -n "${SIGNFI_TASKS[$task_name]+x}" ]]; then
        echo "signfi"
    elif [[ -n "${XRF55_TASKS[$task_name]+x}" ]]; then
        echo "xrf55"
    else
        echo "$DATASET"
    fi
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
# Stage 5: Run evaluation
# ------------------------------------------------------------------
evaluate_task() {
    local task_name="$1"

    local task_root
    task_root=$(resolve_task_root "$task_name")
    local reg_dir
    reg_dir=$(resolve_registry_dir "$task_name")
    local reg_path="${reg_dir}/pair_registry.json"
    local ckpt_path="${MODEL_ROOT}/checkpoints/${DATASET}-${task_name}-${METHOD}/best.pt"
    local results_dir="${MODEL_ROOT}/results/${DATASET}/${task_name}/${METHOD}_${ENCODER}"

    if [[ ! -f "$ckpt_path" ]]; then
        log "ERROR: Checkpoint not found: ${ckpt_path}"
        return 1
    fi

    log "Stage 5: Evaluating ${DATASET}/${task_name}"
    log "  checkpoint: ${ckpt_path}"
    log "  results:    ${results_dir}"

    cd "$REPO_ROOT"
    local registry_parent
    registry_parent=$(dirname "$reg_dir")

    python -m scripts.run_evaluate \
        --checkpoint "$ckpt_path" \
        --config "$CONFIG" \
        --override \
            "task_root=${task_root}" \
            "registry_path=${reg_path}" \
            "text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5" \
        --eval_protocols zero_shot cross_task visualization \
        --registry_dir "$registry_parent" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "Evaluation results saved to: ${results_dir}"
    if [[ -f "${results_dir}/metrics.json" ]]; then
        log "Metrics:"
        python3 -c "
import json
with open('${results_dir}/metrics.json') as f:
    m = json.load(f)
zs = m.get('zero_shot', {})
print(f'  Zero-shot accuracy:    {zs.get(\"accuracy\", \"N/A\")}')
print(f'  Zero-shot weighted F1: {zs.get(\"weighted_f1\", \"N/A\")}')
far = zs.get('false_alarm_rate', {})
if far:
    print(f'  False alarm rate:      {far.get(\"macro\", \"N/A\")}')
lp = m.get('visualization', {}).get('linear_probe', {})
if lp:
    print(f'  Linear probe accuracy: {lp.get(\"accuracy\", \"N/A\")}')
ct = m.get('cross_task', {})
if ct:
    print(f'  Cross-task transfer:')
    for pair, res in ct.items():
        if isinstance(res, dict) and 'accuracy' in res:
            print(f'    {pair}: accuracy={res[\"accuracy\"]:.4f}  f1={res.get(\"weighted_f1\", 0):.4f}')
"
    fi
    if [[ -f "${results_dir}/summary.md" ]]; then
        log "Summary:"
        cat "${results_dir}/summary.md"
    fi
}

# ------------------------------------------------------------------
# Multitask: training
# ------------------------------------------------------------------
train_multitask() {
    # $1 = config path, $2... = task names
    local mt_config="$1"
    shift
    local mt_tasks=("$@")
    local num_gpus
    num_gpus=$(detect_gpus)

    log "Multitask training: tasks=${mt_tasks[*]} | method=${METHOD} encoder=${ENCODER}"
    log "  GPUs: ${num_gpus} | batch_size: ${BATCH_SIZE} (per GPU)"

    cd "$REPO_ROOT"

    local overrides=(
        "method=${METHOD}"
        "csi_encoder.name=${ENCODER}"
        "text_encoder.name=${TEXT_ENCODER}"
        "training.batch_size=${BATCH_SIZE}"
        "training.max_epochs=${MAX_EPOCHS}"
        "training.learning_rate=${LR}"
        "training.num_workers=${NUM_WORKERS}"
        "logging.checkpoint_dir=${MODEL_ROOT}/checkpoints"
    )

    # Add per-task registry_path and text_embeddings_path overrides
    for i in "${!mt_tasks[@]}"; do
        local t="${mt_tasks[$i]}"
        local reg_dir
        reg_dir=$(resolve_registry_dir "$t")
        overrides+=(
            "multitask.tasks.${i}.registry_path=${reg_dir}/pair_registry.json"
            "multitask.tasks.${i}.text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5"
        )
    done

    if [[ "$num_gpus" -gt 1 ]]; then
        python -m torch.distributed.run \
            --nproc_per_node "$num_gpus" \
            --master_port 29500 \
            -m training.multitask_train \
            --config "$mt_config" \
            --override "${overrides[@]}"
    else
        python -m training.multitask_train \
            --config "$mt_config" \
            --override "${overrides[@]}"
    fi

    log "Multitask training complete"
}

# ------------------------------------------------------------------
# Multitask: per-task evaluation
# ------------------------------------------------------------------
evaluate_multitask() {
    # $1 = multitask config path, $2... = task names
    local mt_config="$1"
    shift
    local mt_tasks=("$@")

    local mt_dataset
    mt_dataset=$(python3 -c "import yaml; print(yaml.safe_load(open('${mt_config}'))['dataset'])")
    local ckpt_path="${MODEL_ROOT}/checkpoints/${mt_dataset}-multitask-${METHOD}/best.pt"
    local base_results_dir="${MODEL_ROOT}/results/multitask_${mt_dataset}/${METHOD}_${ENCODER}"

    if [[ ! -f "$ckpt_path" ]]; then
        log "ERROR: Multitask checkpoint not found: ${ckpt_path}"
        return 1
    fi

    for t in "${mt_tasks[@]}"; do
        local task_ds
        task_ds=$(resolve_dataset_for_task "$t")
        local task_root
        task_root=$(resolve_task_root "$t")
        local reg_dir
        reg_dir=$(resolve_registry_dir "$t")
        local reg_path="${reg_dir}/pair_registry.json"
        local results_dir="${base_results_dir}/${t}"
        local registry_parent
        registry_parent=$(dirname "$reg_dir")

        log "Evaluating multitask model on task: ${t} (dataset=${task_ds})"

        cd "$REPO_ROOT"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt_path" \
            --config "$mt_config" \
            --override \
                "dataset=${task_ds}" \
                "task=${t}" \
                "task_root=${task_root}" \
                "registry_path=${reg_path}" \
                "text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5" \
            --eval_protocols zero_shot cross_task visualization \
            --registry_dir "$registry_parent" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        log "Results for ${t} saved to: ${results_dir}"
        if [[ -f "${results_dir}/metrics.json" ]]; then
            python3 -c "
import json
with open('${results_dir}/metrics.json') as f:
    m = json.load(f)
zs = m.get('zero_shot', {})
print(f'  [${t}] Zero-shot accuracy:    {zs.get(\"accuracy\", \"N/A\")}')
print(f'  [${t}] Zero-shot weighted F1: {zs.get(\"weighted_f1\", \"N/A\")}')
ct = m.get('cross_task', {})
if ct:
    for pair, res in ct.items():
        if isinstance(res, dict) and 'accuracy' in res:
            print(f'  [${t}] Cross-task {pair}: accuracy={res[\"accuracy\"]:.4f}  f1={res.get(\"weighted_f1\", 0):.4f}')
lp = m.get('visualization', {}).get('linear_probe', {})
if lp:
    print(f'  [${t}] Linear probe accuracy: {lp.get(\"accuracy\", \"N/A\")}')
"
        fi
    done

    # Generate combined summary across all tasks
    log "Generating combined multitask summary..."
    cd "$REPO_ROOT"
    python3 -c "
import json, os
from pathlib import Path

base = Path('${base_results_dir}')
all_results = {}
for task_dir in sorted(base.iterdir()):
    mf = task_dir / 'metrics.json'
    if task_dir.is_dir() and mf.is_file():
        with open(mf) as f:
            all_results[task_dir.name] = json.load(f)

if all_results:
    with open(base / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    lines = ['# Multitask Evaluation Summary', '',
             '| Task | Accuracy | Weighted F1 |',
             '|------|----------|-------------|']
    for t, res in sorted(all_results.items()):
        zs = res.get('zero_shot', res)
        acc = zs.get('accuracy', 'N/A')
        f1 = zs.get('weighted_f1', 'N/A')
        if isinstance(acc, float):
            lines.append(f'| {t} | {acc:.4f} | {f1:.4f} |')
        else:
            lines.append(f'| {t} | {acc} | {f1} |')
    lines.append('')

    has_ct = False
    for t, res in sorted(all_results.items()):
        ct = res.get('cross_task', {})
        valid = {k: v for k, v in ct.items() if isinstance(v, dict) and 'accuracy' in v}
        if valid:
            if not has_ct:
                lines.extend(['## Cross-Task Transfer', ''])
                has_ct = True
            lines.extend([f'### Source: {t}', '',
                          '| Direction | Accuracy | Weighted F1 |',
                          '|-----------|----------|-------------|'])
            for pair, r in sorted(valid.items()):
                lines.append(f'| {pair} | {r[\"accuracy\"]:.4f} | {r.get(\"weighted_f1\", 0):.4f} |')
            lines.append('')

    with open(base / 'summary.md', 'w') as f:
        f.write('\n'.join(lines))

    print('Combined summary saved to: ${base_results_dir}/')
    print()
    print('\n'.join(lines))
"
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

    # ---- Multitask modes ----
    if [[ "$TASK" == "multitask_csibench" ]]; then
        local mt_tasks=("${ALL_CSIBENCH_TASKS[@]}")
        local mt_config="${REPO_ROOT}/configs/experiment/sagemaker_configs/sagemaker_multitask_csibench.yaml"
        for t in "${mt_tasks[@]}"; do
            check_registry "$t"
            generate_embeddings "$t"
        done
        train_multitask "$mt_config" "${mt_tasks[@]}"
        evaluate_multitask "$mt_config" "${mt_tasks[@]}"
        log "=== Multitask (CSI-Bench) complete ==="
        return
    fi

    if [[ "$TASK" == "multitask_all" ]]; then
        local mt_tasks=("${ALL_CSIBENCH_TASKS[@]}" "${ALL_SIGNFI_TASKS[@]}" "${ALL_XRF55_TASKS[@]}")
        local mt_config="${REPO_ROOT}/configs/experiment/sagemaker_configs/sagemaker_multitask_all.yaml"
        for t in "${mt_tasks[@]}"; do
            check_registry "$t"
            generate_embeddings "$t"
        done
        train_multitask "$mt_config" "${mt_tasks[@]}"
        evaluate_multitask "$mt_config" "${mt_tasks[@]}"
        log "=== Multitask (all datasets) complete ==="
        return
    fi

    # ---- Single-task modes ----
    local tasks=()
    if [[ "$TASK" == "all" ]]; then
        if [[ "$DATASET" == "csibench" ]]; then
            tasks=("${ALL_CSIBENCH_TASKS[@]}")
        elif [[ "$DATASET" == "signfi" ]]; then
            tasks=("${ALL_SIGNFI_TASKS[@]}")
        elif [[ "$DATASET" == "xrf55" ]]; then
            tasks=("${ALL_XRF55_TASKS[@]}")
        else
            log "ERROR: Unknown dataset '${DATASET}' for TASK=all"
            exit 1
        fi
    elif [[ "$TASK" == *","* ]]; then
        IFS=',' read -ra tasks <<< "$TASK"
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
        evaluate_task "$t"
    done

    log "=== All tasks complete ==="
}

main "$@"
