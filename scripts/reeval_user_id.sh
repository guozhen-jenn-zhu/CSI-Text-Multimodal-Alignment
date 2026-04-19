#!/usr/bin/env bash
#
# Re-evaluate user_id task after label_mapping fix in data/dataset.py.
#
# The original evaluations had a label index misalignment: CSITextDataset
# assigned contiguous indices via sorted-unique labels (U01->0, U03->1, ...),
# but the pair registry / text prototypes used label_mapping.json indices
# (U01->0, U02->1, U03->2, ...).  With the fix, _build_label_map now uses
# label_mapping.json when available, so indices match.
#
# Stages:
#   p1a  - 6 single-task methods on user_id (zero_shot + cross_task + viz)
#   p1b  - multitask_csibench + multitask_all on user_id
#   p6   - AM-FM encoder on user_id
#   sync - Upload corrected results to S3
#   all  - All of the above
#
# Usage (on SageMaker):
#   bash scripts/reeval_user_id.sh           # run all
#   STAGE=p1a bash scripts/reeval_user_id.sh # only single-task

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

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

# Task root resolution (mirrors train_sagemaker.sh)
resolve_task_root() {
    local task_name="$1"
    case "$task_name" in
        fall_detection) echo "${DATA_ROOT}/csibench/fall_detection" ;;
        motion_source)  echo "${DATA_ROOT}/csibench/motion_source" ;;
        har|proximity|user_id) echo "${DATA_ROOT}/csibench/multitask/${task_name}" ;;
        sign_recognition) echo "${DATA_ROOT}/signfi/sign_recognition" ;;
        action_recognition) echo "${DATA_ROOT}/xrf55/action_recognition" ;;
        *) echo "${DATA_ROOT}/csibench/${task_name}" ;;
    esac
}

print_zs_metrics() {
    local results_dir="$1"
    local label="$2"
    if [[ -f "${results_dir}/metrics.json" ]]; then
        python3 -c "
import json
with open('${results_dir}/metrics.json') as f:
    m = json.load(f)
zs = m.get('zero_shot', {})
print(f'  [${label}] Zero-shot accuracy:    {zs.get(\"accuracy\", \"N/A\")}')
print(f'  [${label}] Zero-shot weighted F1: {zs.get(\"weighted_f1\", \"N/A\")}')
pf1 = zs.get('per_class_f1', {})
if pf1:
    print(f'  [${label}] Per-class F1: {pf1}')
lp = m.get('visualization', {}).get('linear_probe', {})
if lp:
    print(f'  [${label}] Linear probe accuracy: {lp.get(\"accuracy\", \"N/A\")}')
ct = m.get('cross_task', {})
if ct:
    for pair, res in ct.items():
        if isinstance(res, dict) and 'accuracy' in res:
            print(f'  [${label}] Cross-task {pair}: acc={res[\"accuracy\"]:.4f}')
"
    fi
}

# ------------------------------------------------------------------
# P1A: 6 single-task methods on user_id
# ------------------------------------------------------------------
stage_p1a() {
    log "=== P1A: Re-evaluate 6 single-task methods on user_id ==="
    local methods=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
    local task_root
    task_root=$(resolve_task_root "user_id")
    local reg_dir="${task_root}"
    local registry_parent
    registry_parent=$(dirname "$reg_dir")

    for M in "${methods[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/csibench-user_id-${M}/best.pt"
        local results_dir="${MODEL_ROOT}/results/csibench/user_id/${M}_${ENCODER}"

        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): csibench-user_id-${M}"
            continue
        fi

        log "Evaluating: user_id / ${M}"
        cd "$REPO_ROOT"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=csibench" \
                "task=user_id" \
                "method=${M}" \
                "task_root=${task_root}" \
                "registry_path=${reg_dir}/pair_registry.json" \
                "text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5" \
            --eval_protocols zero_shot cross_task visualization \
            --registry_dir "$registry_parent" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        print_zs_metrics "$results_dir" "${M}"
    done

    log "P1A complete"
}

# ------------------------------------------------------------------
# P1B: Multitask models on user_id
# ------------------------------------------------------------------
stage_p1b() {
    log "=== P1B: Re-evaluate multitask models on user_id ==="
    local task_root
    task_root=$(resolve_task_root "user_id")
    local reg_dir="${task_root}"
    local registry_parent
    registry_parent=$(dirname "$reg_dir")

    # -- multitask_csibench --
    local ckpt_csibench="${MODEL_ROOT}/checkpoints/csibench-multitask-clip_infonce/best.pt"
    local results_csibench="${MODEL_ROOT}/results/multitask_csibench/clip_infonce_${ENCODER}/user_id"

    if [[ -f "$ckpt_csibench" ]]; then
        log "Evaluating: multitask_csibench / user_id"
        cd "$REPO_ROOT"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt_csibench" \
            --config "$MT_CSIBENCH_CONFIG" \
            --override \
                "dataset=csibench" \
                "task=user_id" \
                "task_root=${task_root}" \
                "registry_path=${reg_dir}/pair_registry.json" \
                "text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5" \
            --eval_protocols zero_shot cross_task visualization \
            --registry_dir "$registry_parent" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_csibench" \
            --no_timestamp

        print_zs_metrics "$results_csibench" "multitask_csibench"
    else
        log "SKIP (no checkpoint): csibench-multitask-clip_infonce"
    fi

    # -- multitask_all --
    local ckpt_all="${MODEL_ROOT}/checkpoints/all-multitask-clip_infonce/best.pt"
    local results_all="${MODEL_ROOT}/results/multitask_all/clip_infonce_${ENCODER}/user_id"

    if [[ -f "$ckpt_all" ]]; then
        log "Evaluating: multitask_all / user_id"
        cd "$REPO_ROOT"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt_all" \
            --config "$MT_ALL_CONFIG" \
            --override \
                "dataset=csibench" \
                "task=user_id" \
                "task_root=${task_root}" \
                "registry_path=${reg_dir}/pair_registry.json" \
                "text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5" \
            --eval_protocols zero_shot cross_task visualization \
            --registry_dir "$registry_parent" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_all" \
            --no_timestamp

        print_zs_metrics "$results_all" "multitask_all"
    else
        log "SKIP (no checkpoint): all-multitask-clip_infonce"
    fi

    log "P1B complete"
}

# ------------------------------------------------------------------
# P6: AM-FM encoder on user_id
# ------------------------------------------------------------------
stage_p6() {
    log "=== P6: Re-evaluate AM-FM encoder on user_id ==="
    local task_root
    task_root=$(resolve_task_root "user_id")
    local reg_dir="${task_root}"
    local registry_parent
    registry_parent=$(dirname "$reg_dir")

    local ckpt="${MODEL_ROOT}/checkpoints/all-multitask-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/multitask_all/clip_infonce_amfm/user_id"

    if [[ ! -f "$ckpt" ]]; then
        log "SKIP (no checkpoint): AM-FM best.pt"
        return 0
    fi

    log "Evaluating: AM-FM / user_id"
    cd "$REPO_ROOT"
    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$AMFM_CONFIG" \
        --override \
            "csi_encoder.freeze=false" \
            "csi_encoder.encoder_lr_scale=0.1" \
            "dataset=csibench" \
            "task=user_id" \
            "task_root=${task_root}" \
            "registry_path=${reg_dir}/pair_registry.json" \
            "text_embeddings_path=${reg_dir}/text_embeddings_${TEXT_ENCODER}.h5" \
        --eval_protocols zero_shot cross_task visualization \
        --registry_dir "$registry_parent" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    print_zs_metrics "$results_dir" "amfm"
    log "P6 complete"
}

# ------------------------------------------------------------------
# Sync corrected results to S3
# ------------------------------------------------------------------
stage_sync() {
    log "=== Syncing corrected user_id results to S3 ==="
    local s3_results="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results"
    local methods=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)

    for M in "${methods[@]}"; do
        local local_dir="${MODEL_ROOT}/results/csibench/user_id/${M}_${ENCODER}"
        if [[ -d "$local_dir" ]]; then
            log "Syncing: ${M}_${ENCODER}"
            aws s3 sync "$local_dir" \
                "${s3_results}/csibench/user_id/${M}_${ENCODER}/" --quiet
        fi
    done

    for mt in multitask_csibench multitask_all; do
        local local_dir="${MODEL_ROOT}/results/${mt}/clip_infonce_${ENCODER}/user_id"
        if [[ -d "$local_dir" ]]; then
            log "Syncing: ${mt}/clip_infonce_${ENCODER}/user_id"
            aws s3 sync "$local_dir" \
                "${s3_results}/${mt}/clip_infonce_${ENCODER}/user_id/" --quiet
        fi
    done

    local amfm_dir="${MODEL_ROOT}/results/multitask_all/clip_infonce_amfm/user_id"
    if [[ -d "$amfm_dir" ]]; then
        log "Syncing: multitask_all/clip_infonce_amfm/user_id"
        aws s3 sync "$amfm_dir" \
            "${s3_results}/multitask_all/clip_infonce_amfm/user_id/" --quiet
    fi

    log "S3 sync complete"
}

# ------------------------------------------------------------------
# Run selected stage(s)
# ------------------------------------------------------------------
case "$STAGE" in
    p1a)  stage_p1a ;;
    p1b)  stage_p1b ;;
    p6)   stage_p6 ;;
    sync) stage_sync ;;
    all)
        stage_p1a
        stage_p1b
        stage_p6
        stage_sync
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use p1a, p1b, p6, sync, or all."
        exit 1
        ;;
esac

log "=== user_id re-evaluation complete ==="
