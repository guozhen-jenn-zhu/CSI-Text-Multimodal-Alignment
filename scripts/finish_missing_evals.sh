#!/usr/bin/env bash
#
# Finish all incomplete evaluations for the NeurIPS paper.
#
# What this script does:
#
#   Stage 1 (CRITICAL): Restore the original CLIP-based HAR clip_infonce result
#     that was overwritten by Priority 5B (sentence-transformer ablation).
#     Uses epoch_150.pt (resnet18_1d + clip_vit_b32, val_loss=3.5347) since
#     best_ep78_vl3.6398.pt turned out to be a vit_1d checkpoint (P0 trained
#     all 3 encoders sequentially into the same checkpoint dir).
#
#   Stage 2: Evaluate Priority 5B sentence-transformer result into its own
#     separate output directory (so it doesn't overwrite the main benchmark).
#
#   Stage 3: Evaluate Priority 3A text-level ablation checkpoint
#     (last surviving: text_level=level1_level2).
#
#   Stage 4: Evaluate Priority 3B many-to-many ablation checkpoints
#     (last surviving: num_paraphrases=10 for clip_infonce and siglip).
#
#   Stage 5: Evaluate Priority 5C embedding dimension ablation checkpoint
#     (last surviving: projection.dim=512).
#
#   Stage 6: Add missing linear_probe data for clip_infonce_resnet18_1d on
#     fall_detection, motion_source, proximity, user_id.
#
#   Stage 7: Regenerate paper tables from all results.
#
# Usage (on SageMaker):
#   bash scripts/finish_missing_evals.sh              # run all stages
#   STAGE=1 bash scripts/finish_missing_evals.sh      # run only stage 1
#   DRY_RUN=1 bash scripts/finish_missing_evals.sh    # print commands only
#
# Prerequisites:
#   - All checkpoints must be synced to ${MODEL_ROOT}/checkpoints/
#   - Stage 1 auto-downloads the required checkpoint from S3

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
S3_RESULTS="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results"
S3_CHECKPOINTS="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/checkpoints"

ENCODER="resnet18_1d"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log()  { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
warn() { echo -e "\n\033[1;33m[$(date '+%H:%M:%S')] WARNING: $*\033[0m"; }
err()  { echo -e "\n\033[1;31m[$(date '+%H:%M:%S')] ERROR: $*\033[0m"; }

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] $*"
    else
        "$@"
    fi
}

require_checkpoint() {
    local ckpt="$1"
    local desc="$2"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    if [[ ! -f "$ckpt" ]]; then
        err "Checkpoint not found for ${desc}: ${ckpt}"
        return 1
    fi
}

resolve_task_root() {
    local task_name="$1"
    case "$task_name" in
        fall_detection) echo "${DATA_ROOT}/csibench/fall_detection" ;;
        motion_source)  echo "${DATA_ROOT}/csibench/motion_source" ;;
        har)            echo "${DATA_ROOT}/csibench/multitask/har" ;;
        proximity)      echo "${DATA_ROOT}/csibench/multitask/proximity" ;;
        user_id)        echo "${DATA_ROOT}/csibench/multitask/user_id" ;;
        sign_recognition)    echo "${DATA_ROOT}/signfi/sign_recognition" ;;
        action_recognition)  echo "${DATA_ROOT}/xrf55/action_recognition" ;;
        *)              echo "${DATA_ROOT}/csibench/${task_name}" ;;
    esac
}

sync_results() {
    local local_dir="$1"
    local s3_dir="$2"
    if [[ "$DRY_RUN" != "1" ]]; then
        log "Syncing ${local_dir} -> ${s3_dir}"
        aws s3 sync "$local_dir" "$s3_dir" --quiet
    fi
}

# ==================================================================
# Stage 1: Restore original CLIP-based HAR clip_infonce result
# ==================================================================
# The original best resnet18_1d checkpoint (best_ep78_vl3.6398.pt)
# turned out to be vit_1d, not resnet18_1d -- P0 trained all 3 encoders
# sequentially into the same checkpoint directory, overwriting each
# other.  However, epoch_150.pt (val_loss=3.5347) survives as a
# resnet18_1d + clip_vit_b32 checkpoint from the original training.
stage_1() {
    log "=== Stage 1: Restore original CLIP-based HAR clip_infonce_resnet18_1d ==="

    local original_ckpt="${MODEL_ROOT}/checkpoints/csibench-har-clip_infonce/epoch_150.pt"
    local task_root
    task_root=$(resolve_task_root "har")
    local results_dir="${MODEL_ROOT}/results/csibench/har/clip_infonce_${ENCODER}"

    if [[ ! -f "$original_ckpt" ]] && [[ "$DRY_RUN" != "1" ]]; then
        log "Downloading original CLIP resnet18_1d checkpoint (epoch_150) from S3..."
        aws s3 cp \
            "${S3_CHECKPOINTS}/csibench-har-clip_infonce/epoch_150.pt" \
            "$original_ckpt"
    fi

    require_checkpoint "$original_ckpt" "Stage 1 (original CLIP HAR resnet18_1d)" || return 1

    log "Using epoch_150.pt (resnet18_1d + clip_vit_b32, val_loss=3.5347)"
    log "Running full evaluation (zero_shot, cross_task, visualization, cross_dataset)..."
    cd "$REPO_ROOT"
    run_cmd python -m scripts.run_evaluate \
        --checkpoint "$original_ckpt" \
        --config "$CONFIG" \
        --override \
            "dataset=csibench" \
            "task=har" \
            "method=clip_infonce" \
            "csi_encoder.name=${ENCODER}" \
            "text_encoder.name=clip_vit_b32" \
            "task_root=${task_root}" \
            "registry_path=${task_root}/pair_registry.json" \
            "text_embeddings_path=${task_root}/text_embeddings_clip_vit_b32.h5" \
        --eval_protocols zero_shot cross_task visualization cross_dataset \
        --registry_dir "${DATA_ROOT}/csibench" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    sync_results "$results_dir" "${S3_RESULTS}/csibench/har/clip_infonce_${ENCODER}/"
    log "Stage 1 complete: ${results_dir}"
}

# ==================================================================
# Stage 2: Evaluate P5B sentence-transformer as a separate ablation
# ==================================================================
stage_2() {
    log "=== Stage 2: Priority 5B sentence-transformer ablation (separate output) ==="

    local ckpt="${MODEL_ROOT}/checkpoints/csibench-har-clip_infonce/best.pt"
    local task_root
    task_root=$(resolve_task_root "har")
    local results_dir="${MODEL_ROOT}/results/ablations/text_encoder/sentence_transformer"

    # The current best.pt IS the sentence-transformer checkpoint (best_ep27_vl3.5296)
    require_checkpoint "$ckpt" "Stage 2 (sentence-transformer)" || return 1

    log "Running evaluation with sentence_transformer text encoder..."
    cd "$REPO_ROOT"
    run_cmd python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$CONFIG" \
        --override \
            "dataset=csibench" \
            "task=har" \
            "method=clip_infonce" \
            "csi_encoder.name=${ENCODER}" \
            "text_encoder.name=sentence_transformer" \
            "task_root=${task_root}" \
            "registry_path=${task_root}/pair_registry.json" \
            "text_embeddings_path=${task_root}/text_embeddings_sentence_transformer.h5" \
        --eval_protocols zero_shot cross_task visualization \
        --registry_dir "${DATA_ROOT}/csibench" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    sync_results "$results_dir" "${S3_RESULTS}/ablations/text_encoder/sentence_transformer/"
    log "Stage 2 complete: ${results_dir}"
}

# ==================================================================
# Stage 3: Evaluate P3A text-level ablation
# ==================================================================
stage_3() {
    log "=== Stage 3: Priority 3A text-level ablation ==="

    local ckpt="${MODEL_ROOT}/checkpoints/ablation_text_level/csibench-har-clip_infonce/best.pt"
    local task_root
    task_root=$(resolve_task_root "har")
    local results_dir="${MODEL_ROOT}/results/ablations/text_level/level1_level2"

    require_checkpoint "$ckpt" "Stage 3 (text_level)" || return 1

    log "Evaluating text_level ablation checkpoint..."
    log "  NOTE: This checkpoint is from the last text_level training run."
    log "  The checkpoint was trained with text_level=level1_level2 (the last variant)."
    cd "$REPO_ROOT"
    run_cmd python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$CONFIG" \
        --override \
            "dataset=csibench" \
            "task=har" \
            "method=clip_infonce" \
            "csi_encoder.name=${ENCODER}" \
            "text_encoder.name=clip_vit_b32" \
            "text_generation.text_level=level1_level2" \
            "task_root=${task_root}" \
            "registry_path=${task_root}/pair_registry.json" \
            "text_embeddings_path=${task_root}/text_embeddings_clip_vit_b32.h5" \
        --eval_protocols zero_shot visualization \
        --registry_dir "${DATA_ROOT}/csibench" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    sync_results "$results_dir" "${S3_RESULTS}/ablations/text_level/level1_level2/"
    log "Stage 3 complete: ${results_dir}"
}

# ==================================================================
# Stage 4: Evaluate P3B many-to-many paraphrase ablation
# ==================================================================
stage_4() {
    log "=== Stage 4: Priority 3B many-to-many paraphrase ablation ==="

    local task_root
    task_root=$(resolve_task_root "har")

    # 4A: clip_infonce with num_paraphrases=10
    local ckpt_clip="${MODEL_ROOT}/checkpoints/ablation_m2m/csibench-har-clip_infonce/best.pt"
    local results_clip="${MODEL_ROOT}/results/ablations/m2m/clip_infonce_10para"

    if require_checkpoint "$ckpt_clip" "Stage 4A (m2m clip_infonce)"; then
        log "Evaluating m2m ablation: clip_infonce num_paraphrases=10..."
        cd "$REPO_ROOT"
        run_cmd python -m scripts.run_evaluate \
            --checkpoint "$ckpt_clip" \
            --config "$CONFIG" \
            --override \
                "dataset=csibench" \
                "task=har" \
                "method=clip_infonce" \
                "csi_encoder.name=${ENCODER}" \
                "text_encoder.name=clip_vit_b32" \
                "text_generation.num_paraphrases=10" \
                "task_root=${task_root}" \
                "registry_path=${task_root}/pair_registry.json" \
                "text_embeddings_path=${task_root}/text_embeddings_clip_vit_b32.h5" \
            --eval_protocols zero_shot visualization \
            --registry_dir "${DATA_ROOT}/csibench" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_clip" \
            --no_timestamp

        sync_results "$results_clip" "${S3_RESULTS}/ablations/m2m/clip_infonce_10para/"
        log "Stage 4A complete: ${results_clip}"
    fi

    # 4B: siglip with num_paraphrases=10
    local ckpt_siglip="${MODEL_ROOT}/checkpoints/ablation_m2m/csibench-har-siglip/best.pt"
    local results_siglip="${MODEL_ROOT}/results/ablations/m2m/siglip_10para"

    if require_checkpoint "$ckpt_siglip" "Stage 4B (m2m siglip)"; then
        log "Evaluating m2m ablation: siglip num_paraphrases=10..."
        cd "$REPO_ROOT"
        run_cmd python -m scripts.run_evaluate \
            --checkpoint "$ckpt_siglip" \
            --config "$CONFIG" \
            --override \
                "dataset=csibench" \
                "task=har" \
                "method=siglip" \
                "csi_encoder.name=${ENCODER}" \
                "text_encoder.name=clip_vit_b32" \
                "text_generation.num_paraphrases=10" \
                "task_root=${task_root}" \
                "registry_path=${task_root}/pair_registry.json" \
                "text_embeddings_path=${task_root}/text_embeddings_clip_vit_b32.h5" \
            --eval_protocols zero_shot visualization \
            --registry_dir "${DATA_ROOT}/csibench" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_siglip" \
            --no_timestamp

        sync_results "$results_siglip" "${S3_RESULTS}/ablations/m2m/siglip_10para/"
        log "Stage 4B complete: ${results_siglip}"
    fi
}

# ==================================================================
# Stage 5: Evaluate P5C embedding dimension ablation
# ==================================================================
stage_5() {
    log "=== Stage 5: Priority 5C embedding dimension ablation ==="

    local ckpt="${MODEL_ROOT}/checkpoints/ablation_embed_dim/csibench-har-clip_infonce/best.pt"
    local task_root
    task_root=$(resolve_task_root "har")
    local results_dir="${MODEL_ROOT}/results/ablations/embed_dim/dim_512"

    require_checkpoint "$ckpt" "Stage 5 (embed_dim)" || return 1

    # The last training used dim=512 (since 128, 256, 512 were run
    # sequentially and the checkpoint was overwritten each time).
    # We read the actual dim from the checkpoint config at load time.
    log "Evaluating embed_dim ablation checkpoint..."
    log "  NOTE: Checkpoint is from the last embed_dim variant trained (dim=512)."
    cd "$REPO_ROOT"
    run_cmd python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$CONFIG" \
        --override \
            "dataset=csibench" \
            "task=har" \
            "method=clip_infonce" \
            "csi_encoder.name=${ENCODER}" \
            "text_encoder.name=clip_vit_b32" \
            "task_root=${task_root}" \
            "registry_path=${task_root}/pair_registry.json" \
            "text_embeddings_path=${task_root}/text_embeddings_clip_vit_b32.h5" \
        --eval_protocols zero_shot visualization \
        --registry_dir "${DATA_ROOT}/csibench" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    sync_results "$results_dir" "${S3_RESULTS}/ablations/embed_dim/dim_512/"
    log "Stage 5 complete: ${results_dir}"
}

# ==================================================================
# Stage 6: Add missing linear_probe for clip_infonce on 4 tasks
# ==================================================================
stage_6() {
    log "=== Stage 6: Add missing linear_probe for clip_infonce_resnet18_1d ==="

    local tasks=(fall_detection motion_source proximity user_id)
    local datasets=(csibench csibench csibench csibench)

    for i in "${!tasks[@]}"; do
        local task="${tasks[$i]}"
        local ds="${datasets[$i]}"
        local ckpt="${MODEL_ROOT}/checkpoints/csibench-${task}-clip_infonce/best.pt"
        local task_root
        task_root=$(resolve_task_root "$task")
        local results_dir="${MODEL_ROOT}/results/${ds}/${task}/clip_infonce_${ENCODER}"

        if ! require_checkpoint "$ckpt" "Stage 6 (${task} linear_probe)"; then
            continue
        fi

        log "Running visualization (linear_probe + t-SNE) for ${ds}/${task}..."
        cd "$REPO_ROOT"
        run_cmd python -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=${ds}" \
                "task=${task}" \
                "method=clip_infonce" \
                "csi_encoder.name=${ENCODER}" \
                "text_encoder.name=clip_vit_b32" \
                "task_root=${task_root}" \
                "registry_path=${task_root}/pair_registry.json" \
                "text_embeddings_path=${task_root}/text_embeddings_clip_vit_b32.h5" \
            --eval_protocols visualization \
            --registry_dir "${DATA_ROOT}/csibench" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        sync_results "$results_dir" "${S3_RESULTS}/${ds}/${task}/clip_infonce_${ENCODER}/"
        log "  ${task} linear_probe complete"
    done

    log "Stage 6 complete"
}

# ==================================================================
# Stage 7: Regenerate paper tables
# ==================================================================
stage_7() {
    log "=== Stage 7: Regenerate paper tables ==="
    cd "$REPO_ROOT"
    run_cmd python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
        --results_root "${MODEL_ROOT}/results" \
        --output_dir "${MODEL_ROOT}/results/_paper_tables" \
        --stage all

    sync_results "${MODEL_ROOT}/results/_paper_tables" "${S3_RESULTS}/_paper_tables/"
    log "Stage 7 complete"
}

# ==================================================================
# Main
# ==================================================================
log "============================================"
log " Finish Missing Evaluations - NeurIPS Paper"
log "============================================"
log "REPO_ROOT  : ${REPO_ROOT}"
log "DATA_ROOT  : ${DATA_ROOT}"
log "MODEL_ROOT : ${MODEL_ROOT}"
log "STAGE      : ${STAGE}"
log "DRY_RUN    : ${DRY_RUN}"
log ""

case "$STAGE" in
    1) stage_1 ;;
    2) stage_2 ;;
    3) stage_3 ;;
    4) stage_4 ;;
    5) stage_5 ;;
    6) stage_6 ;;
    7) stage_7 ;;
    all)
        stage_1  # CRITICAL: restore CLIP-based HAR first
        stage_2  # sentence-transformer ablation (separate dir)
        stage_3  # text-level ablation
        stage_4  # m2m paraphrase ablation
        stage_5  # embed_dim ablation
        stage_6  # missing linear_probe
        stage_7  # regenerate tables
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use 1-7 or all."
        exit 1
        ;;
esac

log ""
log "============================================"
log " Summary of outputs"
log "============================================"
log ""
log "Stage 1 -> results/csibench/har/clip_infonce_resnet18_1d/     (restored CLIP)"
log "Stage 2 -> results/ablations/text_encoder/sentence_transformer/"
log "Stage 3 -> results/ablations/text_level/level1_level2/"
log "Stage 4 -> results/ablations/m2m/clip_infonce_10para/"
log "           results/ablations/m2m/siglip_10para/"
log "Stage 5 -> results/ablations/embed_dim/dim_512/"
log "Stage 6 -> results/csibench/{task}/clip_infonce_resnet18_1d/  (linear_probe added)"
log "Stage 7 -> results/_paper_tables/                             (regenerated)"
log ""
log "All requested stages complete."
