#!/usr/bin/env bash
#
# Upload proximity_fixed evaluation results to S3, then pull to local paper_results/.
#
# Covers all outputs from eval_proximity_fixed.sh:
#   - Stage 1: 6 single-task proximity_fixed evals
#   - Stage 2: Supervised baseline (4-class fixed)
#   - Stage 4: Multitask_all_fixed eval (7 tasks)
#   - Stage 5: Ambient QA fixed
#   - Stage 6: Text granularity ablation (level1, level1_level2)
#
# Run on SageMaker:
#   bash scripts/upload_proximity_fixed_results.sh
#
# After S3 upload, run locally to pull results:
#   bash scripts/upload_proximity_fixed_results.sh --pull-only

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

upload_dir() {
    local local_path="$1"
    local s3_path="$2"
    local label="$3"

    if [[ -d "$local_path" ]]; then
        aws s3 sync "$local_path" "$s3_path" --quiet
        log "  -> Uploaded ${label} to ${s3_path}"
    else
        log "  [WARN] ${local_path} not found -- skipping ${label}"
    fi
}

if [[ "$PULL_ONLY" == "0" ]]; then
    log "=== Uploading proximity_fixed results to S3 ==="

    # Stage 1: 6 single-task evals
    METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
    ENCODER="resnet18_1d"
    for M in "${METHODS[@]}"; do
        upload_dir \
            "${MODEL_ROOT}/results/csibench/proximity_fixed/${M}_${ENCODER}" \
            "${S3_RESULTS}/csibench/proximity_fixed/${M}_${ENCODER}" \
            "proximity_fixed/${M}"
    done

    # Stage 2: Supervised baseline
    upload_dir \
        "${MODEL_ROOT}/results/supervised_baselines/csibench_proximity_fixed" \
        "${S3_RESULTS}/supervised_baselines/csibench_proximity_fixed" \
        "supervised_baselines/csibench_proximity_fixed"

    # Stage 4: Multitask all fixed
    upload_dir \
        "${MODEL_ROOT}/results/multitask_all_fixed" \
        "${S3_RESULTS}/multitask_all_fixed" \
        "multitask_all_fixed"

    # Stage 5: Ambient QA fixed
    upload_dir \
        "${MODEL_ROOT}/results/ambient_qa_fixed" \
        "${S3_RESULTS}/ambient_qa_fixed" \
        "ambient_qa_fixed"

    # Stage 6: Text granularity ablation
    upload_dir \
        "${MODEL_ROOT}/results/ablations/text_level_fixed" \
        "${S3_RESULTS}/ablations/text_level_fixed" \
        "ablations/text_level_fixed"

    log "=== S3 upload complete ==="
fi

# ---- Pull to local paper_results/ ----
log "=== Pulling proximity_fixed results to local paper_results/ ==="

PAPER="${REPO_ROOT}/paper_results"

# Single-task evals
METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
ENCODER="resnet18_1d"
for M in "${METHODS[@]}"; do
    mkdir -p "${PAPER}/proximity_fixed/${M}_${ENCODER}"
    aws s3 sync "${S3_RESULTS}/csibench/proximity_fixed/${M}_${ENCODER}" \
        "${PAPER}/proximity_fixed/${M}_${ENCODER}" --quiet
done
log "  -> proximity_fixed/ (6 methods) synced"

# Supervised baseline
mkdir -p "${PAPER}/supervised_baselines/csibench_proximity_fixed"
aws s3 sync "${S3_RESULTS}/supervised_baselines/csibench_proximity_fixed" \
    "${PAPER}/supervised_baselines/csibench_proximity_fixed" --quiet
log "  -> supervised_baselines/csibench_proximity_fixed/ synced"

# Multitask all fixed
mkdir -p "${PAPER}/multitask_all_fixed"
aws s3 sync "${S3_RESULTS}/multitask_all_fixed" \
    "${PAPER}/multitask_all_fixed" --quiet
log "  -> multitask_all_fixed/ synced"

# Ambient QA fixed
mkdir -p "${PAPER}/ambient_qa_fixed"
aws s3 sync "${S3_RESULTS}/ambient_qa_fixed" \
    "${PAPER}/ambient_qa_fixed" --quiet
log "  -> ambient_qa_fixed/ synced"

# Text granularity ablation
mkdir -p "${PAPER}/ablations/text_level_fixed"
aws s3 sync "${S3_RESULTS}/ablations/text_level_fixed" \
    "${PAPER}/ablations/text_level_fixed" --quiet
log "  -> ablations/text_level_fixed/ synced"

log "=== Done ==="
echo ""
echo "Results available at:"
echo "  paper_results/proximity_fixed/                         (6 methods, single-task)"
echo "  paper_results/supervised_baselines/csibench_proximity_fixed/"
echo "  paper_results/multitask_all_fixed/                     (7-task multitask eval)"
echo "  paper_results/ambient_qa_fixed/                        (ambient QA with fixed proximity)"
echo "  paper_results/ablations/text_level_fixed/              (text granularity ablation)"
