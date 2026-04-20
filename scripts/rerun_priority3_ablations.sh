#!/usr/bin/env bash
#
# Retrain and evaluate the 7 missing Priority 3 (text generation) ablation
# variants whose checkpoints were overwritten by sequential training into
# shared directories.
#
# Missing experiments:
#   3A: text_level=level1                           (1 variant)
#   3B: clip_infonce  x num_paraphrases={1,3,5}     (3 variants)
#   3B: siglip        x num_paraphrases={1,3,5}     (3 variants)
#
# Each variant now gets its OWN checkpoint directory so they cannot collide.
#
# Usage (on SageMaker):
#   bash scripts/rerun_priority3_ablations.sh            # train + eval all
#   STAGE=3a  bash scripts/rerun_priority3_ablations.sh  # only 3A
#   STAGE=3b  bash scripts/rerun_priority3_ablations.sh  # only 3B
#   DRY_RUN=1 bash scripts/rerun_priority3_ablations.sh  # print commands

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"

ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] $*"
        return 0
    fi
    "$@"
}

TASK_ROOT="${DATA_ROOT}/csibench/multitask/har"
REG_PATH="${TASK_ROOT}/pair_registry.json"
EMB_PATH="${TASK_ROOT}/text_embeddings_${TEXT_ENCODER}.h5"

train_and_eval_variant() {
    local variant_tag="$1"
    local method="$2"
    local ckpt_dir="$3"
    shift 3
    local extra_overrides=("$@")

    local ckpt_subdir="${ckpt_dir}/csibench-har-${method}"
    local ckpt_path="${ckpt_subdir}/best.pt"
    local results_dir="${MODEL_ROOT}/results/ablations/${variant_tag}"

    log "--- Training: ${variant_tag} ---"
    log "  checkpoint_dir: ${ckpt_dir}"
    log "  results_dir:    ${results_dir}"

    run_cmd python -m training.train \
        --config "$CONFIG" \
        --override \
            dataset=csibench \
            task=har \
            "method=${method}" \
            "csi_encoder.name=${ENCODER}" \
            "task_root=${TASK_ROOT}" \
            "registry_path=${REG_PATH}" \
            "text_embeddings_path=${EMB_PATH}" \
            "logging.checkpoint_dir=${ckpt_dir}" \
            "${extra_overrides[@]}"

    if [[ "$DRY_RUN" == "1" ]] || [[ -f "$ckpt_path" ]]; then
        log "--- Evaluating: ${variant_tag} ---"
        run_cmd python -m scripts.run_evaluate \
            --checkpoint "$ckpt_path" \
            --config "$CONFIG" \
            --override \
                dataset=csibench \
                task=har \
                "method=${method}" \
                "csi_encoder.name=${ENCODER}" \
                "task_root=${TASK_ROOT}" \
                "registry_path=${REG_PATH}" \
                "text_embeddings_path=${EMB_PATH}" \
            --eval_protocols zero_shot cross_task visualization \
            --data_root "$DATA_ROOT" \
            --output_dir "$results_dir" \
            --no_timestamp

        log "Results for ${variant_tag}:"
        if [[ -f "${results_dir}/metrics.json" ]]; then
            python3 -c "
import json
with open('${results_dir}/metrics.json') as f:
    m = json.load(f)
zs = m.get('zero_shot', {})
print(f'  Zero-shot accuracy:    {zs.get(\"accuracy\", \"N/A\")}')
print(f'  Zero-shot weighted F1: {zs.get(\"weighted_f1\", \"N/A\")}')
lp = m.get('visualization', {}).get('linear_probe', {})
if lp:
    print(f'  Linear probe accuracy: {lp.get(\"accuracy\", \"N/A\")}')
"
        fi
    else
        log "WARNING: Checkpoint not found at ${ckpt_path} — skipping eval"
    fi
}

# ------------------------------------------------------------------
# 3A: Text granularity — retrain level1 only
# ------------------------------------------------------------------
stage_3a() {
    log "=== 3A: Text Granularity Ablation (level1 only — was overwritten) ==="
    train_and_eval_variant \
        "text_level/level1" \
        "clip_infonce" \
        "${MODEL_ROOT}/checkpoints/ablation_text_level/level1" \
        "text_generation.text_level=level1"
}

# ------------------------------------------------------------------
# 3B: Many-to-many — retrain {1,3,5} paraphrases x {clip_infonce,siglip}
# ------------------------------------------------------------------
stage_3b() {
    log "=== 3B: Many-to-Many Paraphrase Ablation (1,3,5 para — were overwritten) ==="
    for n_para in 1 3 5; do
        for method in clip_infonce siglip; do
            train_and_eval_variant \
                "m2m/${method}_${n_para}para" \
                "$method" \
                "${MODEL_ROOT}/checkpoints/ablation_m2m/${method}_${n_para}para" \
                "text_generation.num_paraphrases=${n_para}"
        done
    done
}

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
generate_summary() {
    log "=== Generating summary of all Priority 3 ablation results ==="
    cd "$REPO_ROOT"
    python3 -c "
import json
from pathlib import Path

results_root = Path('${MODEL_ROOT}/results/ablations')

print()
print('## 3A: Text Granularity Ablation')
print('| Text Level | Zero-Shot Acc | Weighted F1 | Linear Probe |')
print('|------------|---------------|-------------|--------------|')
for level in ['level1', 'level1_level2']:
    mf = results_root / 'text_level' / level / 'metrics.json'
    if mf.exists():
        m = json.load(open(mf))
        zs = m.get('zero_shot', {})
        lp = m.get('visualization', {}).get('linear_probe', {})
        acc = f'{zs[\"accuracy\"]:.1%}' if isinstance(zs.get('accuracy'), float) else 'N/A'
        f1 = f'{zs[\"weighted_f1\"]:.1%}' if isinstance(zs.get('weighted_f1'), float) else 'N/A'
        lpa = f'{lp[\"accuracy\"]:.1%}' if isinstance(lp.get('accuracy'), float) else 'N/A'
        print(f'| {level} | {acc} | {f1} | {lpa} |')
    else:
        print(f'| {level} | -- | -- | -- |')

print()
print('## 3B: Many-to-Many Paraphrase Ablation')
print('| Method | Paraphrases | Zero-Shot Acc | Weighted F1 | Linear Probe |')
print('|--------|-------------|---------------|-------------|--------------|')
for method in ['clip_infonce', 'siglip']:
    for n in [1, 3, 5, 10]:
        mf = results_root / 'm2m' / f'{method}_{n}para' / 'metrics.json'
        if mf.exists():
            m = json.load(open(mf))
            zs = m.get('zero_shot', {})
            lp = m.get('visualization', {}).get('linear_probe', {})
            acc = f'{zs[\"accuracy\"]:.1%}' if isinstance(zs.get('accuracy'), float) else 'N/A'
            f1 = f'{zs[\"weighted_f1\"]:.1%}' if isinstance(zs.get('weighted_f1'), float) else 'N/A'
            lpa = f'{lp[\"accuracy\"]:.1%}' if isinstance(lp.get('accuracy'), float) else 'N/A'
            print(f'| {method} | {n} | {acc} | {f1} | {lpa} |')
        else:
            print(f'| {method} | {n} | -- | -- | -- |')
print()
"
}

# ------------------------------------------------------------------
# Run selected stage(s)
# ------------------------------------------------------------------
case "$STAGE" in
    3a) stage_3a ;;
    3b) stage_3b ;;
    all)
        stage_3a
        stage_3b
        generate_summary
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use 3a, 3b, or all."
        exit 1
        ;;
esac

log "=== Priority 3 rerun complete ==="
