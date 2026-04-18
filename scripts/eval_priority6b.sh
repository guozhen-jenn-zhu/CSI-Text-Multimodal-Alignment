#!/usr/bin/env bash
#
# Evaluate Priority 6B: Fine-tuned AM-FM encoder (encoder_lr_scale=0.1)
#
# The training completed but evaluation was never run.  This script runs
# the full evaluation suite on the saved 6B checkpoint and writes results
# to the standard output directory so they sit alongside the resnet18_1d
# multitask results.
#
# Usage (on SageMaker):
#   bash scripts/eval_priority6b.sh
#
# The checkpoint is expected at:
#   ${MODEL_ROOT}/checkpoints/all-multitask-clip_infonce/best.pt
# which corresponds to 6B best_ep156_vl3.6111.pt.

set -euo pipefail

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
CONFIG_AMFM="${REPO_ROOT}/configs/experiment/amfm_alignment.yaml"

CKPT="${MODEL_ROOT}/checkpoints/all-multitask-clip_infonce/best.pt"
RESULTS_DIR="${MODEL_ROOT}/results/multitask_all/clip_infonce_amfm"

ALL_TASKS="har proximity user_id fall_detection motion_source sign_recognition action_recognition"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

# ------------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------------
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    echo "Try syncing from S3 first:"
    echo "  aws s3 cp s3://rnd-sagemaker/Multimodal_CSI_Text/Model/checkpoints/all-multitask-clip_infonce/best.pt ${CKPT}"
    exit 1
fi

log "=== Priority 6B Evaluation: Fine-tuned AM-FM Encoder ==="
log "Checkpoint : ${CKPT}"
log "Config     : ${CONFIG_AMFM}"
log "Output     : ${RESULTS_DIR}"
log "Tasks      : ${ALL_TASKS}"

# ------------------------------------------------------------------
# Run full multitask evaluation
# ------------------------------------------------------------------
# Override csi_encoder.freeze=false to match the 6B training config.
# The encoder_lr_scale override is not needed at eval time but keeps
# the saved config accurate.
cd "$REPO_ROOT"

log "Phase 1/2: Zero-shot + cross-task + visualization"
python -m scripts.run_evaluate \
    --checkpoint "$CKPT" \
    --config "$CONFIG_AMFM" \
    --override \
        "csi_encoder.freeze=false" \
        "csi_encoder.encoder_lr_scale=0.1" \
    --tasks $ALL_TASKS \
    --eval_protocols zero_shot cross_task visualization \
    --registry_dir "${DATA_ROOT}/csibench" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RESULTS_DIR" \
    --no_timestamp

log "Phase 2/2: Ambient QA (CSI-Bench tasks only)"
python -m scripts.run_evaluate \
    --checkpoint "$CKPT" \
    --config "$CONFIG_AMFM" \
    --override \
        "csi_encoder.freeze=false" \
        "csi_encoder.encoder_lr_scale=0.1" \
    --tasks har proximity user_id \
    --eval_protocols ambient_qa \
    --registry_dir "${DATA_ROOT}/csibench" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RESULTS_DIR" \
    --no_timestamp

# ------------------------------------------------------------------
# Sync results to S3
# ------------------------------------------------------------------
log "Syncing results to S3..."
aws s3 sync "$RESULTS_DIR" \
    "s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results/multitask_all/clip_infonce_amfm/" \
    --quiet

log "=== Priority 6B evaluation complete ==="
log "Results: ${RESULTS_DIR}"
log "S3:      s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results/multitask_all/clip_infonce_amfm/"
