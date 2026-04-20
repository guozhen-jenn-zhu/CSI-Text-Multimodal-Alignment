#!/usr/bin/env bash
#
# Re-evaluate ALL proximity results with the fixed label_mapping.json code.
#
# Background: The original evaluations used an older _build_label_map that
# auto-remapped labels by sorting unique values, which skipped the absent 0.5m
# class and shifted indices (1.5m->0, 2.5m->1, 3.5m->2) vs the pair_registry
# expectation (0.5m->0, 1.5m->1, 2.5m->2, 3.5m->3).  The fixed code reads
# label_mapping.json so indices stay aligned with text prototypes.
#
# Covers:
#   1. Single-task proximity: 6 methods × resnet18_1d
#   2. Multitask CSI-Bench: clip_infonce × resnet18_1d
#   3. Multitask All: clip_infonce × resnet18_1d
#   4. Multitask All: clip_infonce × amfm
#   5. Ambient-QA proximity (uses multitask_csibench checkpoint)
#   6. Regenerate paper tables
#
# Usage:
#   bash scripts/rerun_proximity_evals.sh          # run all
#   STAGE=1 bash scripts/rerun_proximity_evals.sh  # single-task only
#   STAGE=6 bash scripts/rerun_proximity_evals.sh  # tables only

set -euo pipefail

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"

CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
MT_CSIBENCH_CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_multitask_csibench.yaml"
MT_ALL_CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_multitask_all.yaml"
AMFM_CONFIG="${REPO_ROOT}/configs/experiment/amfm_alignment.yaml"

ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"
STAGE="${STAGE:-all}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

TASK_ROOT="${DATA_ROOT}/csibench/multitask/proximity"
REG_PATH="${TASK_ROOT}/pair_registry.json"
TEXT_EMB="${TASK_ROOT}/text_embeddings_${TEXT_ENCODER}.h5"
REGISTRY_PARENT="${DATA_ROOT}/csibench"

print_metrics() {
    local results_dir="$1"
    if [[ -f "${results_dir}/metrics.json" ]]; then
        python3 -c "
import json
with open('${results_dir}/metrics.json') as f:
    m = json.load(f)
zs = m.get('zero_shot', {})
print(f'  Zero-shot accuracy:    {zs.get(\"accuracy\", \"N/A\")}')
print(f'  Zero-shot weighted F1: {zs.get(\"weighted_f1\", \"N/A\")}')
"
    fi
}

# ------------------------------------------------------------------
# Stage 1: Single-task proximity (6 methods × resnet18_1d)
# ------------------------------------------------------------------
stage_1() {
    log "=== Stage 1: Single-task proximity re-evaluation (6 methods) ==="
    local methods=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)

    for M in "${methods[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/csibench-proximity-${M}/best.pt"
        local results_dir="${MODEL_ROOT}/results/csibench/proximity/${M}_${ENCODER}"

        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): csibench-proximity-${M}"
            continue
        fi

        log "Evaluating proximity with method=${M}"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=csibench" \
                "task=proximity" \
                "method=${M}" \
                "task_root=${TASK_ROOT}" \
                "registry_path=${REG_PATH}" \
                "text_embeddings_path=${TEXT_EMB}" \
            --eval_protocols zero_shot cross_task visualization \
            --registry_dir "$REGISTRY_PARENT" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        log "  -> ${results_dir}"
        print_metrics "$results_dir"
    done
}

# ------------------------------------------------------------------
# Stage 2: Multitask CSI-Bench (clip_infonce × resnet18_1d)
# ------------------------------------------------------------------
stage_2() {
    log "=== Stage 2: Multitask CSI-Bench proximity re-evaluation ==="
    local ckpt="${MODEL_ROOT}/checkpoints/csibench-multitask-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/multitask_csibench/clip_infonce_${ENCODER}/proximity"

    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: Checkpoint not found: ${ckpt}"
        return 1
    fi

    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$MT_CSIBENCH_CONFIG" \
        --override \
            "dataset=csibench" \
            "task=proximity" \
            "task_root=${TASK_ROOT}" \
            "registry_path=${REG_PATH}" \
            "text_embeddings_path=${TEXT_EMB}" \
        --eval_protocols zero_shot cross_task visualization \
        --registry_dir "$REGISTRY_PARENT" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "  -> ${results_dir}"
    print_metrics "$results_dir"
}

# ------------------------------------------------------------------
# Stage 3: Multitask All (clip_infonce × resnet18_1d)
# ------------------------------------------------------------------
stage_3() {
    log "=== Stage 3: Multitask All proximity re-evaluation (resnet18_1d) ==="
    local ckpt="${MODEL_ROOT}/checkpoints/all-multitask-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/multitask_all/clip_infonce_${ENCODER}/proximity"

    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: Checkpoint not found: ${ckpt}"
        return 1
    fi

    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$MT_ALL_CONFIG" \
        --override \
            "dataset=csibench" \
            "task=proximity" \
            "task_root=${TASK_ROOT}" \
            "registry_path=${REG_PATH}" \
            "text_embeddings_path=${TEXT_EMB}" \
        --eval_protocols zero_shot cross_task visualization \
        --registry_dir "$REGISTRY_PARENT" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "  -> ${results_dir}"
    print_metrics "$results_dir"
}

# ------------------------------------------------------------------
# Stage 4: Multitask All (clip_infonce × amfm)
# ------------------------------------------------------------------
stage_4() {
    log "=== Stage 4: Multitask All proximity re-evaluation (AMFM encoder) ==="

    # NOTE: The resnet18_1d and AMFM multitask-all checkpoints share the same
    # directory name (all-multitask-clip_infonce).  If both were trained on the
    # same instance, the last one overwrites best.pt.  Verify this checkpoint
    # corresponds to the AMFM model before running.  If they are stored
    # separately, update AMFM_CKPT below.
    local AMFM_CKPT="${AMFM_CKPT:-${MODEL_ROOT}/checkpoints/all-multitask-clip_infonce/best.pt}"
    local results_dir="${MODEL_ROOT}/results/multitask_all/clip_infonce_amfm/proximity"

    if [[ ! -f "$AMFM_CKPT" ]]; then
        log "ERROR: AMFM checkpoint not found: ${AMFM_CKPT}"
        log "  Set AMFM_CKPT=/path/to/amfm/best.pt if stored separately."
        return 1
    fi

    python -m scripts.run_evaluate \
        --checkpoint "$AMFM_CKPT" \
        --config "$AMFM_CONFIG" \
        --override \
            "dataset=csibench" \
            "task=proximity" \
            "task_root=${TASK_ROOT}" \
            "registry_path=${REG_PATH}" \
            "text_embeddings_path=${TEXT_EMB}" \
        --eval_protocols zero_shot cross_task visualization \
        --registry_dir "$REGISTRY_PARENT" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "  -> ${results_dir}"
    print_metrics "$results_dir"
}

# ------------------------------------------------------------------
# Stage 5: Ambient-QA proximity
# ------------------------------------------------------------------
stage_5() {
    log "=== Stage 5: Ambient-QA proximity re-evaluation ==="
    local ckpt="${MODEL_ROOT}/checkpoints/csibench-multitask-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/ambient_qa/clip_infonce_${ENCODER}"

    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: Checkpoint not found: ${ckpt}"
        return 1
    fi

    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$MT_CSIBENCH_CONFIG" \
        --tasks proximity \
        --eval_protocols ambient_qa \
        --registry_dir "$REGISTRY_PARENT" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "  -> ${results_dir}"
}

# ------------------------------------------------------------------
# Stage 6: Regenerate paper tables
# ------------------------------------------------------------------
stage_6() {
    log "=== Stage 6: Regenerate paper tables ==="
    python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
        --results_root "${MODEL_ROOT}/results" \
        --output_dir "${MODEL_ROOT}/results/_paper_tables" \
        --stage all

    log "  -> ${MODEL_ROOT}/results/_paper_tables"
}

# ------------------------------------------------------------------
# Run selected stage(s)
# ------------------------------------------------------------------
log "Proximity re-evaluation script"
log "  STAGE=${STAGE}"
log "  TASK_ROOT=${TASK_ROOT}"
log "  Verifying label_mapping.json exists..."
if [[ -f "${TASK_ROOT}/metadata/label_mapping.json" ]]; then
    log "  OK: label_mapping.json found"
    python3 -c "
import json
with open('${TASK_ROOT}/metadata/label_mapping.json') as f:
    lm = json.load(f)
print(f'  label_to_idx: {lm[\"label_to_idx\"]}')
print(f'  num_classes:   {lm.get(\"num_classes\", \"N/A\")}')
"
else
    log "  ERROR: label_mapping.json NOT found at ${TASK_ROOT}/metadata/"
    log "  The fix relies on this file. Aborting."
    exit 1
fi

case "$STAGE" in
    1) stage_1 ;;
    2) stage_2 ;;
    3) stage_3 ;;
    4) stage_4 ;;
    5) stage_5 ;;
    6) stage_6 ;;
    all)
        stage_1
        stage_2
        stage_3
        stage_4
        stage_5
        stage_6
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use 1-6 or all."
        exit 1
        ;;
esac

log "Proximity re-evaluation complete."
