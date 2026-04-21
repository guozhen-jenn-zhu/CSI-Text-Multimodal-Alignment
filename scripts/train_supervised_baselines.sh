#!/usr/bin/env bash
#
# Train supervised ResNet-18 baselines for all tasks.
#
# Usage (SageMaker):
#   bash scripts/train_supervised_baselines.sh
#
# Usage (custom paths):
#   DATA_ROOT=/path/to/data RESULTS_DIR=/path/to/output bash scripts/train_supervised_baselines.sh
#
# To train only specific dataset(s):
#   DATASET=xrf55 bash scripts/train_supervised_baselines.sh
#   DATASET=signfi bash scripts/train_supervised_baselines.sh
#   DATASET=csibench bash scripts/train_supervised_baselines.sh
#   DATASET=csibench_standalone bash scripts/train_supervised_baselines.sh  # fall_detection + motion_source only

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/paper_results/supervised_baselines}"
DATASET="${DATASET:-all}"    # "all", "xrf55", "signfi", or "csibench"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

print_results() {
    local output_dir="$1"
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

run_supervised() {
    local task_root="$1"
    local num_classes="$2"
    local h5_key="$3"
    local output_dir="$4"
    local label="$5"

    if [[ ! -d "${task_root}/metadata" ]]; then
        log "ERROR: task_root not found: ${task_root}"
        log "  Expected structure: ${task_root}/metadata/sample_metadata.csv"
        return 1
    fi

    log "=== Training Supervised Baseline: ${label} (${num_classes} classes) ==="
    log "  task_root:  ${task_root}"
    log "  output_dir: ${output_dir}"

    python -m scripts.train_supervised \
        --task_root "$task_root" \
        --num_classes "$num_classes" \
        --h5_key "$h5_key" \
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

    log "${label} supervised baseline complete. Results:"
    print_results "$output_dir"
}

# ------------------------------------------------------------------
# Per-dataset functions
# ------------------------------------------------------------------
train_xrf55() {
    run_supervised \
        "${DATA_ROOT}/xrf55/action_recognition" \
        55 auto \
        "${RESULTS_DIR}/xrf55_action_recognition" \
        "XRF55 Action Recognition"
}

train_signfi() {
    run_supervised \
        "${DATA_ROOT}/signfi/sign_recognition" \
        276 amplitude \
        "${RESULTS_DIR}/signfi_sign_recognition" \
        "SignFi Sign Recognition"
}

train_csibench_har() {
    run_supervised \
        "${DATA_ROOT}/csibench/multitask/har" \
        5 CSI_amps \
        "${RESULTS_DIR}/csibench_har" \
        "CSI-Bench HAR"
}

train_csibench_proximity() {
    run_supervised \
        "${DATA_ROOT}/csibench/multitask/proximity" \
        4 CSI_amps \
        "${RESULTS_DIR}/csibench_proximity" \
        "CSI-Bench Proximity"
}

train_csibench_user_id() {
    run_supervised \
        "${DATA_ROOT}/csibench/multitask/user_id" \
        7 CSI_amps \
        "${RESULTS_DIR}/csibench_user_id" \
        "CSI-Bench User ID"
}

train_csibench_fall_detection() {
    run_supervised \
        "${DATA_ROOT}/csibench/fall_detection" \
        2 CSI_amps \
        "${RESULTS_DIR}/csibench_fall_detection" \
        "CSI-Bench Fall Detection"
}

train_csibench_motion_source() {
    run_supervised \
        "${DATA_ROOT}/csibench/motion_source" \
        4 CSI_amps \
        "${RESULTS_DIR}/csibench_motion_source" \
        "CSI-Bench Motion Source"
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
        csibench)
            train_csibench_har
            train_csibench_proximity
            train_csibench_user_id
            train_csibench_fall_detection
            train_csibench_motion_source
            ;;
        csibench_standalone)
            train_csibench_fall_detection
            train_csibench_motion_source
            ;;
        all)
            train_xrf55
            train_signfi
            train_csibench_har
            train_csibench_proximity
            train_csibench_user_id
            train_csibench_fall_detection
            train_csibench_motion_source
            ;;
        *)
            log "ERROR: Unknown DATASET '${DATASET}'. Use 'all', 'xrf55', 'signfi', 'csibench', or 'csibench_standalone'."
            exit 1
            ;;
    esac

    log "=== All supervised baselines complete ==="
    log "Results directory: ${RESULTS_DIR}"
    ls -la "${RESULTS_DIR}"/*/supervised_results.json 2>/dev/null || true
}

main "$@"
