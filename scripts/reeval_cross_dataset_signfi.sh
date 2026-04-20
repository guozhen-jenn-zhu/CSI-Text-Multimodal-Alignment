#!/usr/bin/env bash
#
# Re-evaluate cross-dataset transfer: SignFi -> XRF55 (all 6 methods).
#
# The original evaluation had a bug in run_evaluate.py where the
# "signfi -> xrf55_gesture_subset" direction loaded SignFi test data
# instead of XRF55, producing inflated numbers identical to within-dataset
# accuracy.  The fix (in evaluation/cross_task_transfer.py and
# scripts/run_evaluate.py) renames the direction to "signfi -> xrf55"
# and reuses the correct XRF55 data loader.
#
# Stages:
#   eval   - Rerun cross_dataset for 6 methods (SignFi -> XRF55)
#   tables - Regenerate paper tables
#   sync   - Upload corrected results to S3
#   all    - All of the above
#
# Usage (on SageMaker):
#   bash scripts/reeval_cross_dataset_signfi.sh             # run all
#   STAGE=eval bash scripts/reeval_cross_dataset_signfi.sh  # eval only

set -euo pipefail

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"

ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"
STAGE="${STAGE:-all}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

print_cd_metrics() {
    local results_dir="$1"
    local label="$2"
    if [[ -f "${results_dir}/metrics.json" ]]; then
        python3 -c "
import json, sys
with open('${results_dir}/metrics.json') as f:
    m = json.load(f)
cd = m.get('cross_dataset', {})
if not cd:
    print('  [${label}] No cross_dataset results found')
    sys.exit(0)
for pair, res in cd.items():
    if isinstance(res, dict) and 'accuracy' in res:
        print(f'  [${label}] {pair}: acc={res[\"accuracy\"]:.4f}  f1={res.get(\"weighted_f1\", 0):.4f}')
    elif isinstance(res, dict) and 'error' in res:
        print(f'  [${label}] {pair}: ERROR - {res[\"error\"]}')
"
    fi
}

# ------------------------------------------------------------------
# Eval: 6 methods, SignFi -> XRF55
# ------------------------------------------------------------------
stage_eval() {
    log "=== Re-evaluate cross-dataset: SignFi -> XRF55 (6 methods) ==="
    local methods=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
    local task_root="${DATA_ROOT}/signfi/sign_recognition"

    for M in "${methods[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/signfi-sign_recognition-${M}/best.pt"
        local results_dir="${MODEL_ROOT}/results/signfi/sign_recognition/${M}_${ENCODER}"

        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): signfi-sign_recognition-${M}"
            continue
        fi

        log "Evaluating: signfi / ${M} -> xrf55"
        cd "$REPO_ROOT"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=signfi" \
                "task=sign_recognition" \
                "method=${M}" \
                "task_root=${task_root}" \
                "registry_path=${task_root}/pair_registry.json" \
                "text_embeddings_path=${task_root}/text_embeddings_${TEXT_ENCODER}.h5" \
            --eval_protocols cross_dataset \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        print_cd_metrics "$results_dir" "${M}"
    done

    log "Eval complete"
}

# ------------------------------------------------------------------
# Tables: regenerate paper tables
# ------------------------------------------------------------------
stage_tables() {
    log "=== Regenerate paper tables ==="
    cd "$REPO_ROOT"
    python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
        --results_root "${MODEL_ROOT}/results" \
        --output_dir "${MODEL_ROOT}/results/_paper_tables" \
        --stage all

    log "Tables complete: ${MODEL_ROOT}/results/_paper_tables"
}

# ------------------------------------------------------------------
# Sync: upload corrected results to S3
# ------------------------------------------------------------------
stage_sync() {
    log "=== Syncing corrected SignFi cross-dataset results to S3 ==="
    local s3_results="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results"
    local methods=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)

    for M in "${methods[@]}"; do
        local local_dir="${MODEL_ROOT}/results/signfi/sign_recognition/${M}_${ENCODER}"
        if [[ -d "$local_dir" ]]; then
            log "Syncing: ${M}_${ENCODER}"
            aws s3 sync "$local_dir" \
                "${s3_results}/signfi/sign_recognition/${M}_${ENCODER}/" --quiet
        fi
    done

    local tables_dir="${MODEL_ROOT}/results/_paper_tables"
    if [[ -d "$tables_dir" ]]; then
        log "Syncing: _paper_tables"
        aws s3 sync "$tables_dir" \
            "${s3_results}/_paper_tables/" --quiet
    fi

    log "S3 sync complete"
}

# ------------------------------------------------------------------
# Run selected stage(s)
# ------------------------------------------------------------------
case "$STAGE" in
    eval)   stage_eval ;;
    tables) stage_tables ;;
    sync)   stage_sync ;;
    all)
        stage_eval
        stage_tables
        stage_sync
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use eval, tables, sync, or all."
        exit 1
        ;;
esac

log "=== SignFi cross-dataset re-evaluation complete ==="
