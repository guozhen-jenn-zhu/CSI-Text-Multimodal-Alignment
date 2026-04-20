#!/usr/bin/env bash
#
# Rerun SignFi -> XRF55 cross-dataset transfer for all 6 methods.
#
# Bug fix: the previous evaluation accidentally loaded SignFi test data
# as the "target," so the reported cross-dataset numbers were just
# within-dataset SignFi accuracy.  The fix in run_evaluate.py now
# correctly loads XRF55 action_recognition data as the target.
#
# Usage (on SageMaker):
#   bash scripts/reeval_cross_dataset_signfi.sh

set -euo pipefail

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"

ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
TASK_ROOT="${DATA_ROOT}/signfi/sign_recognition"

log "=== SignFi -> XRF55 cross-dataset transfer (6 methods) ==="

for M in "${METHODS[@]}"; do
    CKPT="${MODEL_ROOT}/checkpoints/signfi-sign_recognition-${M}/best.pt"
    RESULTS_DIR="${MODEL_ROOT}/results/signfi/sign_recognition/${M}_${ENCODER}"

    if [[ ! -f "$CKPT" ]]; then
        log "SKIP (no checkpoint): signfi-sign_recognition-${M}"
        continue
    fi

    log "Evaluating: signfi -> xrf55  [method=${M}]"
    cd "$REPO_ROOT"
    python -m scripts.run_evaluate \
        --checkpoint "$CKPT" \
        --config "$CONFIG" \
        --override \
            "dataset=signfi" \
            "task=sign_recognition" \
            "method=${M}" \
            "task_root=${TASK_ROOT}" \
            "registry_path=${TASK_ROOT}/pair_registry.json" \
            "text_embeddings_path=${TASK_ROOT}/text_embeddings_${TEXT_ENCODER}.h5" \
        --eval_protocols cross_dataset \
        --data_root "$DATA_ROOT" \
        --output_dir "$RESULTS_DIR" \
        --no_timestamp

    log "  -> ${RESULTS_DIR}"
done

log "=== Regenerating paper tables ==="
cd "$REPO_ROOT"
python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
    --results_root "${MODEL_ROOT}/results" \
    --output_dir "${MODEL_ROOT}/results/_paper_tables" \
    --stage all

log "Done. Check ${MODEL_ROOT}/results/_paper_tables/ for updated Table 2."
