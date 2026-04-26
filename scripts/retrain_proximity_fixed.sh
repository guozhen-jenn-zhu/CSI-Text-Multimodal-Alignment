#!/usr/bin/env bash
#
# Retrain all proximity models after fixing the 0.5m metadata bug.
#
# This script:
#   Stage 0: Fix metadata + regenerate pair registry + text embeddings
#   Stage 1: Retrain 6 alignment methods (single-task proximity)
#   Stage 2: Retrain supervised baseline (4 classes)
#   Stage 3: Retrain multitask_all (clip_infonce, all 7 tasks with fixed proximity)
#   Stage 4: Text granularity ablation (level1 / level1_level2 on proximity)
#   Stage 5: Retrain multitask_scene (3 scene tasks: har+proximity+user_id)
#
# Checkpoints are saved to csibench-proximity_fixed-{method}/best.pt
# to avoid overwriting the original (buggy) checkpoints.
#
# Usage:
#   bash scripts/retrain_proximity_fixed.sh              # all stages
#   STAGE=0 bash scripts/retrain_proximity_fixed.sh      # metadata fix only
#   STAGE=1 bash scripts/retrain_proximity_fixed.sh      # alignment training only
#   STAGE=2 bash scripts/retrain_proximity_fixed.sh      # supervised baseline only
#   STAGE=3 bash scripts/retrain_proximity_fixed.sh      # multitask_all only
#   STAGE=4 bash scripts/retrain_proximity_fixed.sh      # text granularity only
#   DRY_RUN=1 bash scripts/retrain_proximity_fixed.sh    # print commands only

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-all}"

ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"
METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)

CONFIG_SINGLE="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
CONFIG_MT_ALL="${REPO_ROOT}/configs/experiment/sagemaker_multitask_all.yaml"

TASK_ROOT="${DATA_ROOT}/csibench/multitask/proximity"
REG_PATH="${TASK_ROOT}/pair_registry.json"
TEXT_EMB="${TASK_ROOT}/text_embeddings_${TEXT_ENCODER}.h5"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
log_section() { echo -e "\n\033[1;32m========================================\n $*\n========================================\033[0m"; }

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] $*"
        return 0
    fi
    "$@"
}

# ------------------------------------------------------------------
# Stage 0: Fix metadata + regenerate pair registry + text embeddings
# ------------------------------------------------------------------
stage_0() {
    log_section "Stage 0: Fix proximity metadata and regenerate registry"

    log "Step 0a: Fix metadata CSV and label_mapping.json"
    run_cmd python -m scripts.fix_proximity_metadata \
        --task_root "$TASK_ROOT" \
        --upload_s3

    log "Step 0b: Regenerate pair registry (local metadata → local registry + S3)"
    run_cmd python3 -c "
import sys, json
sys.path.insert(0, '${REPO_ROOT}')
from scripts.generate_all_texts import generate_for_task
from pathlib import Path
import shutil

task_root = '${TASK_ROOT}'
reg_out = generate_for_task(
    dataset_name='csibench',
    task_name='proximity',
    task_root=task_root,
    h5_key='CSI_amps',
    output_dir='${TASK_ROOT}/_gen_tmp',
)
# Move registry to TASK_ROOT so downstream steps find it
dst = Path(task_root) / 'pair_registry.json'
shutil.copy2(str(reg_out), str(dst))
print(f'Registry written to {dst}')
shutil.rmtree('${TASK_ROOT}/_gen_tmp', ignore_errors=True)

# Also upload to S3
try:
    import boto3
    s3 = boto3.client('s3')
    for pfx in [
        'Multimodal_CSI_Text/Dataset/csibench/multitask/proximity',
        'Data/fm_downstream/CSI-Bench/Multitask/H5/ProximityRecognition',
    ]:
        key = f'{pfx}/pair_registry.json'
        s3.upload_file(str(dst), 'rnd-sagemaker', key)
        print(f'Uploaded to s3://rnd-sagemaker/{key}')
except Exception as e:
    print(f'WARNING: S3 upload failed (non-fatal): {e}')
"

    log "Step 0c: Regenerate text embeddings"
    run_cmd python scripts/generate_text_embeddings.py \
        --registry_path "$REG_PATH" \
        --output_path "$TEXT_EMB" \
        --encoder "$TEXT_ENCODER"

    log "Stage 0 complete. Verifying..."
    if [[ "$DRY_RUN" != "1" ]]; then
        python3 -c "
import json
from pathlib import Path

lm = json.load(open('${TASK_ROOT}/metadata/label_mapping.json'))
print(f'  label_mapping: {lm}')

reg = json.load(open('${REG_PATH}'))
classes = reg.get('class_to_texts', {})
for cid in sorted(classes.keys(), key=int):
    n = len(classes[cid])
    print(f'  class {cid}: {n} texts')
print(f'  total samples: {len(reg.get(\"sample_to_texts\", {}))}')
"
    fi
}

# ------------------------------------------------------------------
# Stage 1: Retrain 6 alignment methods
# ------------------------------------------------------------------
stage_1() {
    log_section "Stage 1: Retrain 6 alignment methods on fixed proximity data"

    for M in "${METHODS[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/csibench-proximity_fixed-${M}/best.pt"

        if [[ -f "$ckpt" ]]; then
            log "SKIP (checkpoint exists): csibench-proximity_fixed-${M}"
            continue
        fi

        log "Training proximity_fixed with method=${M} encoder=${ENCODER}"
        run_cmd python -m training.train \
            --config "$CONFIG_SINGLE" \
            --override \
                dataset=csibench \
                task=proximity_fixed \
                "method=${M}" \
                "csi_encoder.name=${ENCODER}" \
                "text_encoder.name=${TEXT_ENCODER}" \
                "task_root=${TASK_ROOT}" \
                "registry_path=${REG_PATH}" \
                "text_embeddings_path=${TEXT_EMB}" \
                "logging.checkpoint_dir=${MODEL_ROOT}/checkpoints"
    done

    log "Stage 1 complete."
}

# ------------------------------------------------------------------
# Stage 2: Retrain supervised baseline (4 classes)
# ------------------------------------------------------------------
stage_2() {
    log_section "Stage 2: Retrain supervised proximity baseline (4 classes)"

    local output_dir="${MODEL_ROOT}/results/supervised_baselines/csibench_proximity_fixed"
    if [[ -f "${output_dir}/supervised_results.json" ]]; then
        log "SKIP (results exist): ${output_dir}"
        return 0
    fi

    run_cmd python -m scripts.train_supervised \
        --task_root "$TASK_ROOT" \
        --num_classes 4 \
        --h5_key CSI_amps \
        --output_dir "$output_dir" \
        --input_channels 1 \
        --target_subcarriers 64 \
        --target_time_steps 500 \
        --embed_dim 256 \
        --dropout 0.3 \
        --batch_size 128 \
        --lr 1e-3 \
        --max_epochs 200 \
        --patience 20 \
        --num_workers 4 \
        --seed 42

    log "Stage 2 complete."
    if [[ -f "${output_dir}/supervised_results.json" ]]; then
        python3 -c "
import json
with open('${output_dir}/supervised_results.json') as f:
    r = json.load(f)
print(f'  Supervised accuracy:  {r[\"test_accuracy_pct\"]:.2f}%')
print(f'  Supervised F1:        {r[\"test_weighted_f1\"]:.4f}')
print(f'  Best epoch:           {r[\"best_epoch\"]}')
"
    fi
}

# ------------------------------------------------------------------
# Stage 3: Multitask All (7 tasks with fixed proximity)
# ------------------------------------------------------------------
stage_3() {
    log_section "Stage 3: Retrain multitask_all (clip_infonce, 7 tasks with fixed proximity)"

    local ckpt="${MODEL_ROOT}/checkpoints/all-multitask_fixed-clip_infonce/best.pt"

    if [[ -f "$ckpt" ]]; then
        log "SKIP (checkpoint exists): all-multitask_fixed-clip_infonce"
        return 0
    fi

    log "Training multitask_all with clip_infonce (includes fixed proximity)..."

    local mt_overrides=(
        method=clip_infonce
        "csi_encoder.name=${ENCODER}"
        "text_encoder.name=${TEXT_ENCODER}"
        task=multitask_fixed
        "logging.checkpoint_dir=${MODEL_ROOT}/checkpoints"
    )

    local mt_tasks=(har proximity user_id fall_detection motion_source sign_recognition action_recognition)
    local mt_task_roots=(
        "${DATA_ROOT}/csibench/multitask/har"
        "${TASK_ROOT}"
        "${DATA_ROOT}/csibench/multitask/user_id"
        "${DATA_ROOT}/csibench/fall_detection"
        "${DATA_ROOT}/csibench/motion_source"
        "${DATA_ROOT}/signfi/sign_recognition"
        "${DATA_ROOT}/xrf55/action_recognition"
    )

    for i in "${!mt_tasks[@]}"; do
        local tr="${mt_task_roots[$i]}"
        mt_overrides+=(
            "multitask.tasks.${i}.registry_path=${tr}/pair_registry.json"
            "multitask.tasks.${i}.text_embeddings_path=${tr}/text_embeddings_${TEXT_ENCODER}.h5"
        )
    done

    run_cmd python -m training.multitask_train \
        --config "$CONFIG_MT_ALL" \
        --override "${mt_overrides[@]}"

    log "Stage 3 complete."
}

# ------------------------------------------------------------------
# Stage 4: Text granularity ablation (level1 / level1+level2)
# ------------------------------------------------------------------
stage_4() {
    log_section "Stage 4: Text granularity ablation on proximity (level1, level1_level2)"

    for level in level1 level1_level2; do
        local ckpt_dir="${MODEL_ROOT}/checkpoints/ablation_text_level_fixed/${level}"
        local ckpt="${ckpt_dir}/csibench-proximity_fixed-clip_infonce/best.pt"

        if [[ -f "$ckpt" ]]; then
            log "SKIP (checkpoint exists): ablation_text_level_fixed/${level}"
            continue
        fi

        log "Training proximity_fixed with text_level=${level}"
        run_cmd python -m training.train \
            --config "$CONFIG_SINGLE" \
            --override \
                dataset=csibench \
                task=proximity_fixed \
                method=clip_infonce \
                "csi_encoder.name=${ENCODER}" \
                "text_encoder.name=${TEXT_ENCODER}" \
                "task_root=${TASK_ROOT}" \
                "registry_path=${REG_PATH}" \
                "text_embeddings_path=${TEXT_EMB}" \
                "text_generation.text_level=${level}" \
                "logging.checkpoint_dir=${ckpt_dir}"
    done

    log "Stage 4 complete."
}

# ------------------------------------------------------------------
# Stage 5: Multitask Scene (3 tasks: har + proximity + user_id, fixed proximity)
# ------------------------------------------------------------------
stage_5() {
    log_section "Stage 5: Retrain multitask_scene (clip_infonce, 3 scene tasks, fixed proximity)"

    local CONFIG_MT="${REPO_ROOT}/configs/experiment/sagemaker_multitask_scene.yaml"
    local ckpt="${MODEL_ROOT}/checkpoints/csibench-scene_fixed-clip_infonce/best.pt"

    if [[ -f "$ckpt" ]]; then
        log "SKIP (checkpoint exists): csibench-scene_fixed-clip_infonce"
        return 0
    fi

    local mt_task_names=(har proximity user_id)
    local mt_task_roots=(
        "${DATA_ROOT}/csibench/multitask/har"
        "${TASK_ROOT}"
        "${DATA_ROOT}/csibench/multitask/user_id"
    )

    local mt_overrides=(
        method=clip_infonce
        "csi_encoder.name=${ENCODER}"
        "text_encoder.name=${TEXT_ENCODER}"
        task=scene_fixed
        "logging.checkpoint_dir=${MODEL_ROOT}/checkpoints"
    )

    for i in "${!mt_task_names[@]}"; do
        local tr="${mt_task_roots[$i]}"
        mt_overrides+=(
            "multitask.tasks.${i}.task_root=${tr}"
            "multitask.tasks.${i}.registry_path=${tr}/pair_registry.json"
            "multitask.tasks.${i}.text_embeddings_path=${tr}/text_embeddings_${TEXT_ENCODER}.h5"
        )
    done

    log "Training multitask_scene with clip_infonce (fixed proximity)..."
    run_cmd python -m training.multitask_train \
        --config "$CONFIG_MT" \
        --override "${mt_overrides[@]}"

    log "Stage 5 complete."
}

# ------------------------------------------------------------------
# Run selected stage(s)
# ------------------------------------------------------------------
log "=== Proximity Fix: Retrain Pipeline ==="
log "  STAGE=${STAGE}"
log "  TASK_ROOT=${TASK_ROOT}"
log "  DRY_RUN=${DRY_RUN}"
log "  ENCODER=${ENCODER}"

case "$STAGE" in
    0) stage_0 ;;
    1) stage_1 ;;
    2) stage_2 ;;
    3) stage_3 ;;
    4) stage_4 ;;
    5) stage_5 ;;
    all)
        stage_0
        stage_1
        stage_2
        stage_3
        stage_4
        stage_5
        ;;
    *)
        echo "Unknown STAGE: ${STAGE}. Use 0, 1, 2, 3, 4, 5, or all."
        exit 1
        ;;
esac

log "=== Retrain pipeline complete ==="
log "Next step: run evaluation with"
log "  bash scripts/eval_proximity_fixed.sh"
