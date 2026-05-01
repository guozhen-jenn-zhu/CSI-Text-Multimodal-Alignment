#!/usr/bin/env bash
#
# End-to-end training pipeline for local single-machine runs.
# Tested on: Linux/macOS bash, Windows Git Bash / WSL.
#
# Stages:
#   1. Auto-resolve REPO_ROOT, DATA_ROOT, MODEL_ROOT (env-overridable)
#   2. Check pair registry exists for the (dataset, task)
#   3. Generate precomputed text embeddings if missing (optional)
#   4. Launch single- or multi-GPU training via scripts/launch_ddp.py
#
# All experiment-wide overrides (data_root, checkpoint_dir, s3.enabled=false)
# are injected on the command line so you can keep using
# configs/experiment/default.yaml as the base config.
#
# Usage (from any directory inside the repo):
#   bash scripts/pipelines/train_local.sh
#
#   # Pick a task / method / encoder
#   DATASET=csibench TASK=har METHOD=clip_infonce ENCODER=resnet18_1d \
#       bash scripts/pipelines/train_local.sh
#
#   # Force CPU (useful for smoke-testing the pipeline)
#   NUM_GPUS=0 bash scripts/pipelines/train_local.sh
#
#   # Train all CSI-Bench standalone tasks sequentially
#   TASK=all bash scripts/pipelines/train_local.sh
#
#   # Override hyper-parameters and paths
#   BATCH_SIZE=32 MAX_EPOCHS=50 LR=5e-4 \
#   DATA_ROOT=/d/Multimodal_CSI_Text/Dataset MODEL_ROOT=/d/Multimodal_CSI_Text/Model \
#       bash scripts/pipelines/train_local.sh
#
#   # Skip text-embedding generation (e.g. you precomputed them)
#   SKIP_TEXT_EMBEDDINGS=1 bash scripts/pipelines/train_local.sh

set -euo pipefail

# ------------------------------------------------------------------
# Paths (auto-detect; override via env)
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR is scripts/pipelines/, so REPO_ROOT is two levels up.
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/runs}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/experiment/default.yaml}"

# ------------------------------------------------------------------
# Experiment knobs (env-overridable)
# ------------------------------------------------------------------
DATASET="${DATASET:-csibench}"
TASK="${TASK:-har}"
METHOD="${METHOD:-clip_infonce}"
ENCODER="${ENCODER:-resnet18_1d}"
TEXT_ENCODER="${TEXT_ENCODER:-clip_vit_b32}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-3}"
NUM_GPUS="${NUM_GPUS:--1}"          # -1 = auto-detect, 0 = CPU, >=1 = explicit
SKIP_TEXT_EMBEDDINGS="${SKIP_TEXT_EMBEDDINGS:-0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# Task definitions (mirrors train_sagemaker.sh layout under DATA_ROOT)
declare -A CSIBENCH_STANDALONE_TASKS=(
    [fall_detection]="csibench/fall_detection"
    [motion_source]="csibench/motion_source"
)
declare -A CSIBENCH_MULTITASK_TASKS=(
    [har]="csibench/multitask/har"
    [proximity]="csibench/multitask/proximity"
    [user_id]="csibench/multitask/user_id"
)
declare -A XRF55_TASKS=(
    [action_recognition]="xrf55/action_recognition"
)
declare -A SIGNFI_TASKS=(
    [sign_recognition]="signfi/sign_recognition"
)

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() { echo -e "\n\033[1;34m[$(date '+%Y-%m-%d %H:%M:%S')] $*\033[0m"; }
log_warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
log_error() { echo -e "\033[1;31m[ERROR] $*\033[0m"; }

# Resolve a Python interpreter. Prefer $PYTHON_BIN, else `python`, else `python3`.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        log_error "No Python interpreter found. Install Python 3.10+ or set PYTHON_BIN."
        exit 1
    fi
fi
log "Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

resolve_task_root() {
    local dataset="$1" task="$2"
    case "$dataset" in
        csibench)
            if [[ -n "${CSIBENCH_STANDALONE_TASKS[$task]:-}" ]]; then
                echo "${DATA_ROOT}/${CSIBENCH_STANDALONE_TASKS[$task]}"
            elif [[ -n "${CSIBENCH_MULTITASK_TASKS[$task]:-}" ]]; then
                echo "${DATA_ROOT}/${CSIBENCH_MULTITASK_TASKS[$task]}"
            else
                log_error "Unknown CSI-Bench task: ${task}"
                return 1
            fi
            ;;
        xrf55)
            echo "${DATA_ROOT}/${XRF55_TASKS[$task]:-xrf55/${task}}"
            ;;
        signfi)
            echo "${DATA_ROOT}/${SIGNFI_TASKS[$task]:-signfi/${task}}"
            ;;
        *)
            log_error "Unknown dataset: ${dataset}"
            return 1
            ;;
    esac
}

detect_num_gpus() {
    if [[ "$NUM_GPUS" -ge 0 ]] 2>/dev/null; then
        echo "$NUM_GPUS"
        return
    fi
    "$PYTHON_BIN" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
}

ensure_text_embeddings() {
    # Mirrors scripts/pipelines/train_sagemaker.sh:generate_embeddings -- writes
    # an encoder-suffixed file so swapping TEXT_ENCODER (e.g. for the Priority
    # 5B ablation) does not silently reuse a CLIP-generated text_embeddings.h5
    # whose dimensionality would mismatch a sentence-transformer projection.
    local task_root="$1"
    if [[ "$SKIP_TEXT_EMBEDDINGS" == "1" ]]; then
        return 0
    fi
    local registry="${task_root}/pair_registry.json"
    local emb="${task_root}/text_embeddings_${TEXT_ENCODER}.h5"
    if [[ ! -f "$registry" ]]; then
        log_warn "No pair_registry.json at ${registry} -- skipping text-embedding generation."
        log_warn "Run scripts/generate_all_texts.py first."
        return 0
    fi
    if [[ -f "$emb" ]]; then
        return 0
    fi
    log "Generating text embeddings -> ${emb}"
    # Invoke as a module from the repo root so 'from data.pair_registry'
    # inside generate_text_embeddings.py resolves regardless of how
    # PYTHONPATH ends up being interpreted (Git Bash on Windows can mangle
    # MSYS-style /d/... paths). Mirrors train_sagemaker.sh.
    ( cd "$REPO_ROOT" && "$PYTHON_BIN" -m scripts.generate_text_embeddings \
        --registry_path "$registry" \
        --output_path "$emb" \
        --encoder "$TEXT_ENCODER" )
}

run_training() {
    local dataset="$1" task="$2" method="$3" encoder="$4"
    local task_root
    task_root=$(resolve_task_root "$dataset" "$task")

    # NOTE: `logging.checkpoint_dir` must be the *parent* directory.
    # The trainer (training/train.py / training/multitask_train.py) appends
    # f"{dataset}-{task}-{method}" itself, so passing the full per-run dir
    # here would yield a doubled path like .../csibench-har-clip_infonce/
    # csibench-har-clip_infonce/best.pt. This matches the convention used in
    # scripts/pipelines/train_sagemaker.sh.
    local ckpt_dir="${MODEL_ROOT}/checkpoints"
    mkdir -p "$ckpt_dir" "${MODEL_ROOT}/logs"

    ensure_text_embeddings "$task_root"

    local n_gpus
    n_gpus=$(detect_num_gpus)
    log "Training: dataset=${dataset} task=${task} method=${method} encoder=${encoder} (GPUs=${n_gpus})"

    # Match scripts/pipelines/train_sagemaker.sh: pass task_root, registry_path
    # and the encoder-suffixed text_embeddings_path explicitly so the trainer
    # never has to fall back to dataset_config + auto-discovery.  This keeps
    # encoder ablations honest (sentence_transformer cannot accidentally pick
    # up a CLIP-generated text_embeddings.h5).
    local overrides=(
        "dataset=${dataset}"
        "task=${task}"
        "method=${method}"
        "csi_encoder.name=${encoder}"
        "text_encoder.name=${TEXT_ENCODER}"
        "task_root=${task_root}"
        "registry_path=${task_root}/pair_registry.json"
        "text_embeddings_path=${task_root}/text_embeddings_${TEXT_ENCODER}.h5"
        "training.batch_size=${BATCH_SIZE}"
        "training.max_epochs=${MAX_EPOCHS}"
        "training.num_workers=${NUM_WORKERS}"
        "training.learning_rate=${LR}"
        "dataset_config.data_root=${DATA_ROOT}/${dataset}"
        "logging.checkpoint_dir=${ckpt_dir}"
        "s3.enabled=false"
    )

    if [[ -n "$EXTRA_OVERRIDES" ]]; then
        # shellcheck disable=SC2206
        local extra_arr=($EXTRA_OVERRIDES)
        overrides+=("${extra_arr[@]}")
    fi

    cd "$REPO_ROOT"
    if [[ "$n_gpus" -ge 2 ]]; then
        # Multi-GPU: launch DDP. USE_LIBUV=0 sidesteps the missing-libuv error
        # in PyTorch's Windows wheels (torch >= 2.4).
        USE_LIBUV=0 "$PYTHON_BIN" "${REPO_ROOT}/scripts/launch_ddp.py" \
            --config "$CONFIG" \
            --num_gpus "$n_gpus" \
            --override "${overrides[@]}"
    elif [[ "$n_gpus" -eq 1 ]]; then
        # Single GPU: skip DDP entirely (no torchrun, no rendezvous, no libuv).
        "$PYTHON_BIN" -m training.train \
            --config "$CONFIG" \
            --override "${overrides[@]}"
    else
        log_warn "No CUDA GPUs detected -- training on CPU (will be slow)."
        "$PYTHON_BIN" -m training.train \
            --config "$CONFIG" \
            --override "${overrides[@]}"
    fi
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    log "REPO_ROOT:  ${REPO_ROOT}"
    log "DATA_ROOT:  ${DATA_ROOT}"
    log "MODEL_ROOT: ${MODEL_ROOT}"
    log "CONFIG:     ${CONFIG}"

    local task_list=()
    if [[ "$TASK" == "all" ]]; then
        case "$DATASET" in
            csibench) task_list=(har fall_detection motion_source proximity user_id) ;;
            xrf55)    task_list=(action_recognition) ;;
            signfi)   task_list=(sign_recognition) ;;
            *)        log_error "Unknown DATASET=${DATASET}"; exit 1 ;;
        esac
    else
        IFS=',' read -ra task_list <<< "$TASK"
    fi

    for t in "${task_list[@]}"; do
        run_training "$DATASET" "$t" "$METHOD" "$ENCODER"
    done
}

main "$@"
