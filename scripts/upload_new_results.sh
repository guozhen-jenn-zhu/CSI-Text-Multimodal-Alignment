#!/usr/bin/env bash
#
# Upload Experiment 1 (retrieval + few-shot) and Experiment 3 (held-out SignFi)
# results to S3, then download to local paper_results/.
#
# Run on SageMaker:
#   bash scripts/upload_new_results.sh
#
# After S3 upload, run locally to pull results:
#   bash scripts/upload_new_results.sh --pull-only

set -euo pipefail

S3_RESULTS="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"

PULL_ONLY=0
for arg in "$@"; do
    if [[ "$arg" == "--pull-only" ]]; then
        PULL_ONLY=1
    fi
done

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

if [[ "$PULL_ONLY" == "0" ]]; then
    # ---- Upload Experiment 1: Retrieval + Few-Shot ----
    log "Uploading Experiment 1: retrieval + few-shot (new_evals/)"
    if [ -d "${MODEL_ROOT}/results/new_evals" ]; then
        aws s3 sync "${MODEL_ROOT}/results/new_evals" "${S3_RESULTS}/new_evals" --quiet
        log "  -> Uploaded to ${S3_RESULTS}/new_evals/"
    else
        log "  [WARN] ${MODEL_ROOT}/results/new_evals not found -- skipping"
    fi

    # ---- Upload Experiment 3: Held-Out SignFi ----
    log "Uploading Experiment 3: held-out SignFi classes"
    if [ -d "${MODEL_ROOT}/results/held_out_classes" ]; then
        aws s3 sync "${MODEL_ROOT}/results/held_out_classes" "${S3_RESULTS}/held_out_classes" --quiet
        log "  -> Uploaded to ${S3_RESULTS}/held_out_classes/"
    else
        log "  [WARN] ${MODEL_ROOT}/results/held_out_classes not found -- skipping"
    fi

    log "S3 upload complete."
fi

# ---- Pull to local paper_results/ ----
log "Pulling results to local paper_results/"

mkdir -p "${REPO_ROOT}/paper_results/new_evals"
aws s3 sync "${S3_RESULTS}/new_evals" "${REPO_ROOT}/paper_results/new_evals" --quiet
log "  -> new_evals/ synced"

mkdir -p "${REPO_ROOT}/paper_results/held_out_classes"
aws s3 sync "${S3_RESULTS}/held_out_classes" "${REPO_ROOT}/paper_results/held_out_classes" --quiet
log "  -> held_out_classes/ synced"

log "Done. Results available at:"
echo "  paper_results/new_evals/       (retrieval + few-shot for 42 checkpoints)"
echo "  paper_results/held_out_classes/ (held-out SignFi evaluation)"
