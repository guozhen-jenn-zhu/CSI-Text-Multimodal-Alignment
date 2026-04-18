#!/usr/bin/env bash
#
# Rerun missing evaluations for the NeurIPS paper.
#
# Fixes:
#   2A. SigLIP-HAR zero-shot (was missing — only cross_dataset was run)
#   2B. Ambient-QA on multitask checkpoint (metadata path was broken, now fixed)
#   2C. Cross-dataset transfer for all 6 methods × 3 directions (Table 2)
#   2D. Regenerate paper tables
#
# Run on SageMaker after pushing the ambient-QA fix commit.
#
# Usage:
#   bash scripts/rerun_missing_evals.sh          # run all
#   STAGE=2a bash scripts/rerun_missing_evals.sh  # run only SigLIP-HAR

set -euo pipefail

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
MT_CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_multitask_csibench.yaml"

ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"
STAGE="${STAGE:-all}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

# Task root resolution (mirrors train_sagemaker.sh)
declare -A CSIBENCH_STANDALONE_TASKS=(
    [fall_detection]="csibench/fall_detection"
    [motion_source]="csibench/motion_source"
)
declare -A CSIBENCH_MULTITASK_TASKS=(
    [har]="csibench/multitask"
    [proximity]="csibench/multitask"
    [user_id]="csibench/multitask"
)

resolve_task_root() {
    local task_name="$1"
    if [[ -n "${CSIBENCH_STANDALONE_TASKS[$task_name]+x}" ]]; then
        echo "${DATA_ROOT}/${CSIBENCH_STANDALONE_TASKS[$task_name]}"
    elif [[ -n "${CSIBENCH_MULTITASK_TASKS[$task_name]+x}" ]]; then
        echo "${DATA_ROOT}/${CSIBENCH_MULTITASK_TASKS[$task_name]}/${task_name}"
    elif [[ "$task_name" == "sign_recognition" ]]; then
        echo "${DATA_ROOT}/signfi/sign_recognition"
    elif [[ "$task_name" == "action_recognition" ]]; then
        echo "${DATA_ROOT}/xrf55/action_recognition"
    else
        echo "${DATA_ROOT}/csibench/${task_name}"
    fi
}

# ------------------------------------------------------------------
# 2A. SigLIP-HAR re-evaluation
# ------------------------------------------------------------------
stage_2a() {
    log "=== 2A: SigLIP-HAR re-evaluation (zero_shot + cross_task + visualization + cross_dataset) ==="

    local task_root
    task_root=$(resolve_task_root "har")
    local reg_dir="${task_root}"
    local ckpt="${MODEL_ROOT}/checkpoints/csibench-har-siglip/best.pt"
    local results_dir="${MODEL_ROOT}/results/csibench/har/siglip_${ENCODER}"

    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: Checkpoint not found: ${ckpt}"
        return 1
    fi

    cd "$REPO_ROOT"
    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$CONFIG" \
        --override \
            "dataset=csibench" \
            "task=har" \
            "method=siglip" \
            "task_root=${task_root}" \
            "registry_path=${reg_dir}/pair_registry.json" \
            "text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5" \
        --eval_protocols zero_shot cross_task visualization cross_dataset \
        --registry_dir "${DATA_ROOT}/csibench" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "2A complete: ${results_dir}"
}

# ------------------------------------------------------------------
# 2B. Ambient-QA on multitask checkpoint
# ------------------------------------------------------------------
stage_2b() {
    log "=== 2B: Ambient-QA re-run (multitask checkpoint) ==="

    local ckpt="${MODEL_ROOT}/checkpoints/csibench-multitask-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/ambient_qa/clip_infonce_${ENCODER}"

    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: Checkpoint not found: ${ckpt}"
        return 1
    fi

    cd "$REPO_ROOT"
    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$MT_CONFIG" \
        --tasks har proximity user_id \
        --eval_protocols ambient_qa \
        --registry_dir "${DATA_ROOT}/csibench" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "2B complete: ${results_dir}"
}

# ------------------------------------------------------------------
# 2C. Cross-dataset transfer (3 directions × 6 methods)
# ------------------------------------------------------------------
stage_2c() {
    log "=== 2C: Cross-dataset transfer (Table 2) ==="
    local methods=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)

    # Direction 1: CSI-Bench HAR -> XRF55
    log "--- Direction: CSI-Bench HAR -> XRF55 ---"
    for M in "${methods[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/csibench-har-${M}/best.pt"
        local results_dir="${MODEL_ROOT}/results/csibench/har/${M}_${ENCODER}"
        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): csibench-har-${M}"
            continue
        fi
        log "Evaluating: csibench-har-${M} -> xrf55"
        local task_root
        task_root=$(resolve_task_root "har")
        cd "$REPO_ROOT"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=csibench" \
                "task=har" \
                "method=${M}" \
                "task_root=${task_root}" \
                "registry_path=${task_root}/pair_registry.json" \
                "text_embeddings_path=${task_root}/text_embeddings_${TEXT_ENCODER}.h5" \
            --eval_protocols cross_dataset \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp
    done

    # Direction 2: XRF55 -> CSI-Bench HAR
    log "--- Direction: XRF55 -> CSI-Bench HAR ---"
    for M in "${methods[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/xrf55-action_recognition-${M}/best.pt"
        local results_dir="${MODEL_ROOT}/results/xrf55/action_recognition/${M}_${ENCODER}"
        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): xrf55-action_recognition-${M}"
            continue
        fi
        log "Evaluating: xrf55-action_recognition-${M} -> csibench_har"
        local task_root
        task_root=$(resolve_task_root "action_recognition")
        cd "$REPO_ROOT"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=xrf55" \
                "task=action_recognition" \
                "method=${M}" \
                "task_root=${task_root}" \
                "registry_path=${task_root}/pair_registry.json" \
                "text_embeddings_path=${task_root}/text_embeddings_${TEXT_ENCODER}.h5" \
            --eval_protocols cross_dataset \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp
    done

    # Direction 3: SignFi -> XRF55 gesture subset
    log "--- Direction: SignFi -> XRF55 gesture subset ---"
    for M in "${methods[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/signfi-sign_recognition-${M}/best.pt"
        local results_dir="${MODEL_ROOT}/results/signfi/sign_recognition/${M}_${ENCODER}"
        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): signfi-sign_recognition-${M}"
            continue
        fi
        log "Evaluating: signfi-sign_recognition-${M} -> xrf55_gesture"
        local task_root
        task_root=$(resolve_task_root "sign_recognition")
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
    done

    log "2C complete"
}

# ------------------------------------------------------------------
# 2D. Regenerate paper tables
# ------------------------------------------------------------------
stage_2d() {
    log "=== 2D: Regenerate paper tables ==="
    cd "$REPO_ROOT"
    python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
        --results_root "${MODEL_ROOT}/results" \
        --output_dir "${MODEL_ROOT}/results/_paper_tables" \
        --stage all

    log "2D complete: ${MODEL_ROOT}/results/_paper_tables"
}

# ------------------------------------------------------------------
# Run selected stage(s)
# ------------------------------------------------------------------
case "$STAGE" in
    2a) stage_2a ;;
    2b) stage_2b ;;
    2c) stage_2c ;;
    2d) stage_2d ;;
    all)
        stage_2a
        stage_2b
        stage_2c
        stage_2d
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use 2a, 2b, 2c, 2d, or all."
        exit 1
        ;;
esac

log "All requested stages complete."
