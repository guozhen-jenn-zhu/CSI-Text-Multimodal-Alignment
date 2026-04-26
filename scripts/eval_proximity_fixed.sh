#!/usr/bin/env bash
#
# Evaluate ALL retrained proximity_fixed models.
#
# Covers:
#   Stage 1: Single-task proximity_fixed (6 methods × resnet18_1d)
#            → zero_shot, visualization, retrieval, few_shot
#   Stage 2: Supervised baseline summary
#   Stage 3: Comparison table (old vs fixed)
#   Stage 4: Multitask_all evaluation (7 tasks, proximity uses fixed data)
#   Stage 5: Ambient QA evaluation (using multitask_all_fixed checkpoint)
#   Stage 6: Text granularity ablation eval (level1 / level1_level2)
#   Stage 7: Ambient QA Level 1+2 (multitask_scene_fixed, holistic scene QA)
#
# Results are saved under *_fixed/ dirs to preserve originals untouched.
#
# Usage:
#   bash scripts/eval_proximity_fixed.sh            # all stages
#   STAGE=1 bash scripts/eval_proximity_fixed.sh    # single-task eval only
#   STAGE=2 bash scripts/eval_proximity_fixed.sh    # supervised summary only
#   STAGE=3 bash scripts/eval_proximity_fixed.sh    # tables only
#   STAGE=4 bash scripts/eval_proximity_fixed.sh    # multitask_all eval only
#   STAGE=5 bash scripts/eval_proximity_fixed.sh    # ambient QA (multitask_all) only
#   STAGE=6 bash scripts/eval_proximity_fixed.sh    # text granularity only
#   STAGE=7 bash scripts/eval_proximity_fixed.sh    # ambient QA Level 1+2 only

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
STAGE="${STAGE:-all}"

ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"
METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)

CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
CONFIG_MT_ALL="${REPO_ROOT}/configs/experiment/sagemaker_multitask_all.yaml"
CONFIG_MT_CSIBENCH="${REPO_ROOT}/configs/experiment/sagemaker_multitask_csibench.yaml"
CONFIG_MT_SCENE="${REPO_ROOT}/configs/experiment/sagemaker_multitask_scene.yaml"

TASK_ROOT="${DATA_ROOT}/csibench/multitask/proximity"
REG_PATH="${TASK_ROOT}/pair_registry.json"
TEXT_EMB="${TASK_ROOT}/text_embeddings_${TEXT_ENCODER}.h5"
REGISTRY_PARENT="${DATA_ROOT}/csibench"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
log_section() { echo -e "\n\033[1;32m========================================\n $*\n========================================\033[0m"; }

print_metrics() {
    local results_dir="$1"
    if [[ -f "${results_dir}/metrics.json" ]]; then
        python3 -c "
import json
with open('${results_dir}/metrics.json') as f:
    m = json.load(f)
zs = m.get('zero_shot', {})
lp = m.get('visualization', {}).get('linear_probe', {})
fs = m.get('few_shot', {})
ret = m.get('retrieval', {}).get('csi_to_text', {})
print(f'  Zero-shot accuracy:    {zs.get(\"accuracy\", \"N/A\")}')
print(f'  Zero-shot weighted F1: {zs.get(\"weighted_f1\", \"N/A\")}')
if lp:
    print(f'  Linear probe accuracy: {lp.get(\"accuracy\", \"N/A\")}')
if ret:
    print(f'  Retrieval R@1:         {ret.get(\"recall@1\", \"N/A\")}')
if fs:
    for k, v in sorted(fs.items()):
        if isinstance(v, dict) and 'accuracy_mean' in v:
            print(f'  {k} accuracy:       {v[\"accuracy_mean\"]:.4f} ± {v.get(\"accuracy_std\", 0):.4f}')
"
    fi
}

# ------------------------------------------------------------------
# Stage 1: Single-task proximity_fixed (6 methods)
# ------------------------------------------------------------------
stage_1() {
    log_section "Stage 1: Evaluate 6 single-task proximity_fixed models"

    for M in "${METHODS[@]}"; do
        local ckpt="${MODEL_ROOT}/checkpoints/csibench-proximity_fixed-${M}/best.pt"
        local results_dir="${MODEL_ROOT}/results/csibench/proximity_fixed/${M}_${ENCODER}"

        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): csibench-proximity_fixed-${M}"
            continue
        fi

        log "Evaluating proximity_fixed method=${M}"
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
            --eval_protocols zero_shot visualization retrieval few_shot \
            --registry_dir "$REGISTRY_PARENT" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        log "  -> ${results_dir}"
        print_metrics "$results_dir"
    done
}

# ------------------------------------------------------------------
# Stage 2: Supervised baseline summary
# ------------------------------------------------------------------
stage_2() {
    log_section "Stage 2: Supervised baseline summary"

    local sup_dir="${MODEL_ROOT}/results/supervised_baselines/csibench_proximity_fixed"
    if [[ -f "${sup_dir}/supervised_results.json" ]]; then
        python3 -c "
import json
with open('${sup_dir}/supervised_results.json') as f:
    r = json.load(f)
print('Supervised Proximity (4-class, fixed):')
print(f'  Accuracy:     {r[\"test_accuracy_pct\"]:.2f}%')
print(f'  Weighted F1:  {r[\"test_weighted_f1\"]:.4f}')
print(f'  Best epoch:   {r[\"best_epoch\"]}')
print(f'  Samples:      train={r[\"train_samples\"]} val={r[\"val_samples\"]} test={r[\"test_samples\"]}')
"
    else
        log "Supervised baseline not found at ${sup_dir}"
        log "  Run: STAGE=2 bash scripts/retrain_proximity_fixed.sh"
    fi
}

# ------------------------------------------------------------------
# Stage 3: Comparison table
# ------------------------------------------------------------------
stage_3() {
    log_section "Stage 3: Comparison table (old vs fixed)"

    python3 -c "
import json
from pathlib import Path

model_root = Path('${MODEL_ROOT}/results')
encoder = '${ENCODER}'
methods = ${METHODS[@]+"'${METHODS[0]}' '${METHODS[1]}' '${METHODS[2]}' '${METHODS[3]}' '${METHODS[4]}' '${METHODS[5]}'"}

print()
print(f'{\"Method\":<18} {\"Old ZS%\":<10} {\"Fixed ZS%\":<10} {\"Δ\":<8} {\"Fixed LP%\":<10}')
print('-' * 60)

methods_list = ['clip_infonce', 'siglip', 'language_bind', 'centrobind', 'barlow_twins', 'masked_predict']
for m in methods_list:
    old_f = model_root / 'csibench' / 'proximity' / f'{m}_{encoder}' / 'metrics.json'
    new_f = model_root / 'csibench' / 'proximity_fixed' / f'{m}_{encoder}' / 'metrics.json'

    old_acc = '—'
    new_acc = '—'
    delta = '—'
    lp_acc = '—'

    if old_f.is_file():
        with open(old_f) as f:
            om = json.load(f)
        oa = om.get('zero_shot', {}).get('accuracy')
        if oa is not None:
            old_acc = f'{oa*100:.1f}'

    if new_f.is_file():
        with open(new_f) as f:
            nm = json.load(f)
        na = nm.get('zero_shot', {}).get('accuracy')
        la = nm.get('visualization', {}).get('linear_probe', {}).get('accuracy')
        if na is not None:
            new_acc = f'{na*100:.1f}'
        if la is not None:
            lp_acc = f'{la*100:.1f}'
        if oa is not None and na is not None:
            delta = f'{(na - oa)*100:+.1f}'

    print(f'{m:<18} {old_acc:<10} {new_acc:<10} {delta:<8} {lp_acc:<10}')

# Supervised comparison
old_sup = model_root.parent / 'paper_results' / 'supervised_baselines' / 'csibench_proximity' / 'supervised_results.json'
new_sup = model_root / 'supervised_baselines' / 'csibench_proximity_fixed' / 'supervised_results.json'

print()
old_sa = '—'
new_sa = '—'
if old_sup.is_file():
    with open(old_sup) as f:
        old_sa = f'{json.load(f)[\"test_accuracy_pct\"]:.1f}'
if new_sup.is_file():
    with open(new_sup) as f:
        new_sa = f'{json.load(f)[\"test_accuracy_pct\"]:.1f}'

print(f'{\"Supervised\":<18} {old_sa:<10} {new_sa:<10}')
print()
"
}

# ------------------------------------------------------------------
# Stage 4: Multitask_all evaluation (7 tasks, fixed proximity)
# ------------------------------------------------------------------
stage_4() {
    log_section "Stage 4: Evaluate multitask_all_fixed (clip_infonce, 7 tasks)"

    local ckpt="${MODEL_ROOT}/checkpoints/all-multitask_fixed-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/multitask_all_fixed/clip_infonce_${ENCODER}"

    if [[ ! -f "$ckpt" ]]; then
        log "SKIP (no checkpoint): all-multitask_fixed-clip_infonce"
        log "  Run: STAGE=3 bash scripts/retrain_proximity_fixed.sh"
        return 0
    fi

    log "Evaluating multitask_all_fixed on all 7 tasks..."
    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$CONFIG_MT_ALL" \
        --tasks har proximity user_id fall_detection motion_source sign_recognition action_recognition \
        --eval_protocols zero_shot visualization retrieval few_shot \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "  -> ${results_dir}"
    if [[ -d "$results_dir" ]]; then
        for task_dir in "${results_dir}"/*/; do
            local task_name
            task_name=$(basename "$task_dir")
            log "  Task: ${task_name}"
            print_metrics "$task_dir"
        done
    fi
}

# ------------------------------------------------------------------
# Stage 5: Ambient QA (using multitask_fixed checkpoint)
# ------------------------------------------------------------------
stage_5() {
    log_section "Stage 5: Ambient QA evaluation (fixed proximity)"

    local mt_ckpt="${MODEL_ROOT}/checkpoints/all-multitask_fixed-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/ambient_qa_fixed/clip_infonce_${ENCODER}"

    if [[ ! -f "$mt_ckpt" ]]; then
        log "SKIP (no checkpoint): all-multitask_fixed-clip_infonce"
        log "  Run: STAGE=3 bash scripts/retrain_proximity_fixed.sh"
        return 0
    fi

    log "Running ambient QA on multitask_fixed model..."
    python -m scripts.run_evaluate \
        --checkpoint "$mt_ckpt" \
        --config "$CONFIG_MT_CSIBENCH" \
        --tasks har proximity user_id \
        --eval_protocols zero_shot ambient_qa \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    log "  -> ${results_dir}"
    if [[ -f "${results_dir}/summary.json" ]]; then
        python3 -c "
import json
with open('${results_dir}/summary.json') as f:
    summary = json.load(f)
for task_name, task_res in summary.items():
    if isinstance(task_res, dict):
        aq = task_res.get('ambient_qa', {})
        joint = aq.get('joint_accuracy', 'N/A')
        print(f'  {task_name}: joint_accuracy={joint}')
        per_attr = aq.get('per_attribute', {})
        for attr, attr_res in per_attr.items():
            if isinstance(attr_res, dict):
                print(f'    {attr}: {attr_res.get(\"accuracy\", \"N/A\")}')
"
    fi
}

# ------------------------------------------------------------------
# Stage 6: Text granularity ablation eval (level1 / level1_level2)
# ------------------------------------------------------------------
stage_6() {
    log_section "Stage 6: Text granularity ablation evaluation"

    for level in level1 level1_level2; do
        local ckpt="${MODEL_ROOT}/checkpoints/ablation_text_level_fixed/${level}/csibench-proximity_fixed-clip_infonce/best.pt"
        local results_dir="${MODEL_ROOT}/results/ablations/text_level_fixed/${level}"

        if [[ ! -f "$ckpt" ]]; then
            log "SKIP (no checkpoint): ablation_text_level_fixed/${level}"
            continue
        fi

        log "Evaluating text_level=${level} on proximity"
        python -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=csibench" \
                "task=proximity" \
                "method=clip_infonce" \
                "task_root=${TASK_ROOT}" \
                "registry_path=${REG_PATH}" \
                "text_embeddings_path=${TEXT_EMB}" \
            --eval_protocols zero_shot visualization retrieval few_shot \
            --registry_dir "$REGISTRY_PARENT" \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        log "  -> ${results_dir}"
        print_metrics "$results_dir"
    done
}

# ------------------------------------------------------------------
# Stage 7: Ambient QA (multitask_scene_fixed -- Level 1 + Level 2)
# ------------------------------------------------------------------
stage_7() {
    log_section "Stage 7: Ambient QA Level 1+2 (multitask_scene_fixed)"

    local mt_ckpt="${MODEL_ROOT}/checkpoints/csibench-scene_fixed-clip_infonce/best.pt"
    local results_dir="${MODEL_ROOT}/results/ambient_qa_fixed/multitask_scene/clip_infonce_${ENCODER}"

    if [[ ! -f "$mt_ckpt" ]]; then
        log "SKIP (no checkpoint): csibench-scene_fixed-clip_infonce"
        log "  Run: STAGE=5 bash scripts/retrain_proximity_fixed.sh"
        return 0
    fi

    local eval_tasks=(har proximity user_id)
    for eval_task in "${eval_tasks[@]}"; do
        local task_results="${results_dir}/${eval_task}"
        log "Evaluating ambient QA on task=${eval_task}..."
        python -m scripts.run_evaluate \
            --checkpoint "$mt_ckpt" \
            --config "$CONFIG_MT_SCENE" \
            --override \
                "task=${eval_task}" \
                "multitask.tasks.1.task_root=${TASK_ROOT}" \
            --eval_protocols zero_shot ambient_qa \
            --registry_dir "${DATA_ROOT}/csibench/multitask" \
            --data_root "$DATA_ROOT" \
            --output_dir "$task_results" \
            --no_timestamp

        log "  -> ${task_results}"

        if [[ -f "${task_results}/metrics.json" ]]; then
            python3 -c "
import json
with open('${task_results}/metrics.json') as f:
    m = json.load(f)
aq = m.get('ambient_qa', {})
l1 = aq.get('level1', {})
l2 = aq.get('level2', {})
if l1:
    print('  Level 1 (per-attribute):')
    for attr, res in l1.get('per_attribute', {}).items():
        if isinstance(res, dict):
            print(f'    {attr}: accuracy={res.get(\"accuracy\", \"N/A\"):.4f}')
    print(f'    joint_accuracy={l1.get(\"joint_accuracy\", \"N/A\")}')
if l2:
    print('  Level 2 (composite):')
    print(f'    composite_accuracy={l2.get(\"composite_accuracy\", \"N/A\")}')
    print(f'    top3_accuracy={l2.get(\"top3_accuracy\", \"N/A\")}')
    print(f'    top5_accuracy={l2.get(\"top5_accuracy\", \"N/A\")}')
    print(f'    mrr={l2.get(\"mrr\", \"N/A\")}')
    decomp = l2.get('decomposed_per_attribute', {})
    for attr, acc in decomp.items():
        print(f'    decomposed {attr}: {acc}')
elif 'error' in str(aq):
    print(f'  ERROR: {aq}')
"
        fi
    done
}

# ------------------------------------------------------------------
# Run selected stage(s)
# ------------------------------------------------------------------
log "=== Proximity Fixed: Evaluation Pipeline ==="
log "  STAGE=${STAGE}"
log "  TASK_ROOT=${TASK_ROOT}"

# Verify label_mapping is correct before evaluating
if [[ -f "${TASK_ROOT}/metadata/label_mapping.json" ]]; then
    log "label_mapping.json:"
    python3 -c "
import json
with open('${TASK_ROOT}/metadata/label_mapping.json') as f:
    lm = json.load(f)
print(f'  label_to_idx: {lm[\"label_to_idx\"]}')
print(f'  num_classes:   {lm.get(\"num_classes\", \"N/A\")}')
"
else
    log "WARNING: label_mapping.json NOT found. Run retrain_proximity_fixed.sh stage 0 first."
fi

case "$STAGE" in
    1) stage_1 ;;
    2) stage_2 ;;
    3) stage_3 ;;
    4) stage_4 ;;
    5) stage_5 ;;
    6) stage_6 ;;
    7) stage_7 ;;
    all)
        stage_1
        stage_2
        stage_3
        stage_4
        stage_5
        stage_6
        stage_7
        ;;
    *)
        echo "Unknown STAGE: ${STAGE}. Use 1, 2, 3, 4, 5, 6, 7, or all."
        exit 1
        ;;
esac

log "=== Evaluation pipeline complete ==="
