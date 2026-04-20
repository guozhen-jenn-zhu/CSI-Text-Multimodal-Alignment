#!/usr/bin/env bash
#
# Train supervised ResNet-18 baselines for XRF55 and SignFi.
#
# Usage (SageMaker):
#   bash scripts/train_supervised_baselines.sh
#
# Usage (custom paths):
#   DATA_ROOT=/path/to/data RESULTS_DIR=/path/to/output bash scripts/train_supervised_baselines.sh
#
# To train only one dataset:
#   DATASET=xrf55 bash scripts/train_supervised_baselines.sh
#   DATASET=signfi bash scripts/train_supervised_baselines.sh

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/paper_results/supervised_baselines}"
DATASET="${DATASET:-all}"    # "all", "xrf55", or "signfi"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

# ------------------------------------------------------------------
# XRF55 Action Recognition (55 classes)
# ------------------------------------------------------------------
train_xrf55() {
    local task_root="${DATA_ROOT}/xrf55/action_recognition"
    local output_dir="${RESULTS_DIR}/xrf55_action_recognition"

    if [[ ! -d "${task_root}/metadata" ]]; then
        log "ERROR: XRF55 task_root not found: ${task_root}"
        log "  Expected structure: ${task_root}/metadata/sample_metadata.csv"
        return 1
    fi

    log "=== Training Supervised Baseline: XRF55 Action Recognition (55 classes) ==="
    log "  task_root:  ${task_root}"
    log "  output_dir: ${output_dir}"

    python -m scripts.train_supervised \
        --task_root "$task_root" \
        --num_classes 55 \
        --h5_key auto \
        --output_dir "$output_dir" \
        --input_channels 1 \
        --target_subcarriers 64 \
        --target_time_steps 500 \
        --embed_dim 256 \
        --dropout 0.3 \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --max_epochs "$MAX_EPOCHS" \
        --patience 20 \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED"

    log "XRF55 supervised baseline complete. Results:"
    if [[ -f "${output_dir}/supervised_results.json" ]]; then
        python3 -c "
import json
with open('${output_dir}/supervised_results.json') as f:
    r = json.load(f)
print(f'  Accuracy:    {r[\"test_accuracy_pct\"]:.2f}%')
print(f'  Weighted F1: {r[\"test_weighted_f1\"]:.4f}')
print(f'  Best Epoch:  {r[\"best_epoch\"]}')
"
    fi
}

# ------------------------------------------------------------------
# SignFi Sign Recognition (276 classes)
# ------------------------------------------------------------------
train_signfi() {
    local task_root="${DATA_ROOT}/signfi/sign_recognition"
    local output_dir="${RESULTS_DIR}/signfi_sign_recognition"

    if [[ ! -d "${task_root}/metadata" ]]; then
        log "ERROR: SignFi task_root not found: ${task_root}"
        log "  Expected structure: ${task_root}/metadata/sample_metadata.csv"
        return 1
    fi

    log "=== Training Supervised Baseline: SignFi Sign Recognition (276 classes) ==="
    log "  task_root:  ${task_root}"
    log "  output_dir: ${output_dir}"

    python -m scripts.train_supervised \
        --task_root "$task_root" \
        --num_classes 276 \
        --h5_key amplitude \
        --output_dir "$output_dir" \
        --input_channels 1 \
        --target_subcarriers 64 \
        --target_time_steps 500 \
        --embed_dim 256 \
        --dropout 0.3 \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --max_epochs "$MAX_EPOCHS" \
        --patience 20 \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED"

    log "SignFi supervised baseline complete. Results:"
    if [[ -f "${output_dir}/supervised_results.json" ]]; then
        python3 -c "
import json
with open('${output_dir}/supervised_results.json') as f:
    r = json.load(f)
print(f'  Accuracy:    {r[\"test_accuracy_pct\"]:.2f}%')
print(f'  Weighted F1: {r[\"test_weighted_f1\"]:.4f}')
print(f'  Best Epoch:  {r[\"best_epoch\"]}')
"
    fi
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    log "=== Supervised Baseline Training ==="
    log "  REPO_ROOT:   ${REPO_ROOT}"
    log "  DATA_ROOT:   ${DATA_ROOT}"
    log "  RESULTS_DIR: ${RESULTS_DIR}"
    log "  DATASET:     ${DATASET}"

    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi -L 2>/dev/null || true
    fi
    python3 -c "import torch; print(f'  PyTorch CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" 2>/dev/null || true

    mkdir -p "$RESULTS_DIR"

    case "$DATASET" in
        xrf55)
            train_xrf55
            ;;
        signfi)
            train_signfi
            ;;
        all)
            train_xrf55
            train_signfi
            ;;
        *)
            log "ERROR: Unknown DATASET '${DATASET}'. Use 'all', 'xrf55', or 'signfi'."
            exit 1
            ;;
    esac

    log "=== All supervised baselines complete ==="
    log "Results directory: ${RESULTS_DIR}"
    ls -la "${RESULTS_DIR}"/*/supervised_results.json 2>/dev/null || true
}

main "$@"
