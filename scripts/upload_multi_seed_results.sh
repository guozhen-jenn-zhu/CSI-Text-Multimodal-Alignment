#!/usr/bin/env bash
#
# Upload multi-seed training results to S3 and pull to local paper_results/.
#
# SAFETY:
#   - Uploads only the specific task directories needed (not a blanket sync).
#   - Explicitly EXCLUDES csibench/proximity/ (buggy 3-class from run_multi_seed.sh).
#   - Uses proximity_fixed/ for all proximity results (from run_multi_seed_proximity_fixed.sh).
#   - All multi-seed results go under a dedicated multi_seed/ prefix — no risk of
#     overwriting the original single-seed results at csibench/{task}/, xrf55/, signfi/, etc.
#
# Expected structure on SageMaker (MODEL_ROOT/results/multi_seed/):
#   csibench/{har,fall_detection,motion_source,user_id}/{method}_resnet18_1d/seed_{42,123,456}/
#   csibench/proximity_fixed/{method}_resnet18_1d/seed_{42,123,456}/
#   xrf55/action_recognition/{method}_resnet18_1d/seed_{42,123,456}/
#   signfi/sign_recognition/{method}_resnet18_1d/seed_{42,123,456}/
#   multi_seed_summary.json
#   proximity_fixed_summary.json
#
# Run on SageMaker to upload:
#   bash scripts/upload_multi_seed_results.sh
#
# Run locally to pull from S3:
#   bash scripts/upload_multi_seed_results.sh --pull-only

set -euo pipefail

S3_RESULTS="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

PULL_ONLY=0
for arg in "$@"; do
    if [[ "$arg" == "--pull-only" ]]; then
        PULL_ONLY=1
    fi
done

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
ENCODER="resnet18_1d"
SEEDS=(42 123 456)

# Tasks that are safe to upload (proximity excluded — use proximity_fixed only)
SAFE_TASKS=(
    "csibench:har"
    "csibench:fall_detection"
    "csibench:motion_source"
    "csibench:user_id"
    "csibench:proximity_fixed"
    "xrf55:action_recognition"
    "signfi:sign_recognition"
)

LOCAL_BASE="${MODEL_ROOT}/results/multi_seed"
S3_BASE="${S3_RESULTS}/multi_seed"

upload_task_dir() {
    local local_path="$1"
    local s3_path="$2"
    local label="$3"

    if [[ -d "$local_path" ]]; then
        n_files=$(find "$local_path" -name "metrics.json" | wc -l | tr -d ' ')
        aws s3 sync "$local_path" "$s3_path" --quiet
        log "  -> Uploaded ${label} (${n_files} metrics files)"
    else
        log "  [WARN] Not found: ${local_path} — skipping ${label}"
    fi
}

if [[ "$PULL_ONLY" == "0" ]]; then
    log "=== Uploading multi-seed results to S3 ==="
    log "  Source: ${LOCAL_BASE}"
    log "  Dest:   ${S3_BASE}"

    if [[ ! -d "${LOCAL_BASE}" ]]; then
        log "[ERROR] ${LOCAL_BASE} not found!"
        exit 1
    fi

    # Warn if buggy proximity directory exists
    if [[ -d "${LOCAL_BASE}/csibench/proximity" ]]; then
        log "[INFO] Found csibench/proximity/ (buggy 3-class) — will NOT upload this."
        log "       Only csibench/proximity_fixed/ (corrected 4-class) will be uploaded."
    fi

    # Upload each task directory individually
    for dt_pair in "${SAFE_TASKS[@]}"; do
        IFS=':' read -r dataset task <<< "$dt_pair"
        upload_task_dir \
            "${LOCAL_BASE}/${dataset}/${task}" \
            "${S3_BASE}/${dataset}/${task}" \
            "${dataset}/${task}"
    done

    # Upload summary files
    for summary_file in multi_seed_summary.json proximity_fixed_summary.json; do
        if [[ -f "${LOCAL_BASE}/${summary_file}" ]]; then
            aws s3 cp "${LOCAL_BASE}/${summary_file}" "${S3_BASE}/${summary_file}" --quiet
            log "  -> Uploaded ${summary_file}"
        else
            log "  [WARN] ${summary_file} not found — skipping"
        fi
    done

    # Post-upload verification
    log "Verifying upload..."
    ok=0
    missing=0
    for dt_pair in "${SAFE_TASKS[@]}"; do
        IFS=':' read -r dataset task <<< "$dt_pair"
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                s3_key="${S3_BASE}/${dataset}/${task}/${method}_${ENCODER}/seed_${seed}/metrics.json"
                if aws s3 ls "$s3_key" >/dev/null 2>&1; then
                    ok=$((ok + 1))
                else
                    echo "  [MISS] ${dataset}/${task}/${method}/seed_${seed}"
                    missing=$((missing + 1))
                fi
            done
        done
    done
    log "Verified: ${ok} found, ${missing} missing (expected $((${#METHODS[@]} * ${#SEEDS[@]} * ${#SAFE_TASKS[@]})))"

    log "=== S3 upload complete ==="
fi

# ---- Pull to local paper_results/ ----
log "=== Pulling multi-seed results to local paper_results/multi_seed/ ==="

PAPER="${REPO_ROOT}/paper_results/multi_seed"

# Pull each task separately to avoid pulling any buggy proximity data
for dt_pair in "${SAFE_TASKS[@]}"; do
    IFS=':' read -r dataset task <<< "$dt_pair"
    local_dir="${PAPER}/${dataset}/${task}"
    s3_dir="${S3_BASE}/${dataset}/${task}"
    mkdir -p "$local_dir"
    aws s3 sync "$s3_dir" "$local_dir" --quiet 2>/dev/null || true
done

# Pull summary files
for summary_file in multi_seed_summary.json proximity_fixed_summary.json; do
    aws s3 cp "${S3_BASE}/${summary_file}" "${PAPER}/${summary_file}" --quiet 2>/dev/null || true
done

log "  -> Synced to paper_results/multi_seed/"

# Count and verify
n_metrics=$(find "$PAPER" -name "metrics.json" 2>/dev/null | wc -l | tr -d ' ')
log "  -> Found ${n_metrics} metrics.json files total"

# Verify proximity_fixed specifically
log "Verifying proximity_fixed results..."
prox_ok=0
prox_missing=0
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        f="${PAPER}/csibench/proximity_fixed/${method}_${ENCODER}/seed_${seed}/metrics.json"
        if [[ -f "$f" ]]; then
            prox_ok=$((prox_ok + 1))
        else
            echo "  [MISSING] proximity_fixed/${method}/seed_${seed}"
            prox_missing=$((prox_missing + 1))
        fi
    done
done
log "  -> Proximity fixed: ${prox_ok}/18 found"

# Verify no buggy proximity was pulled
if [[ -d "${PAPER}/csibench/proximity" ]]; then
    log "[WARN] csibench/proximity/ exists locally — this contains buggy 3-class data!"
    log "       Use csibench/proximity_fixed/ for all paper results."
fi

log "=== Done ==="
echo ""
echo "Results available at:"
echo "  paper_results/multi_seed/csibench/{task}/{method}_resnet18_1d/seed_{seed}/"
echo "  paper_results/multi_seed/csibench/proximity_fixed/{method}_resnet18_1d/seed_{seed}/"
echo "  paper_results/multi_seed/xrf55/action_recognition/{method}_resnet18_1d/seed_{seed}/"
echo "  paper_results/multi_seed/signfi/sign_recognition/{method}_resnet18_1d/seed_{seed}/"
echo ""
echo "SAFETY: csibench/proximity/ (buggy 3-class) is intentionally excluded."
echo "        Only csibench/proximity_fixed/ (corrected 4-class) is included."
echo ""
echo "Next: Run the aggregation script to compute mean±std tables."
