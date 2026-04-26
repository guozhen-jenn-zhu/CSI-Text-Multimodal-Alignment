#!/usr/bin/env bash
#
# Multi-seed evaluation for 6 alignment methods on FIXED proximity (4-class).
#
# Seeds: 42 (reuse from retrain_proximity_fixed.sh), 123, 456.
# Checkpoints:
#   seed 42:  MODEL_ROOT/checkpoints/csibench-proximity_fixed-{method}/best.pt
#   seed N:   MODEL_ROOT/checkpoints/multi_seed/seed_N/csibench-proximity_fixed-{method}/best.pt
# Results:
#   MODEL_ROOT/results/multi_seed/csibench/proximity_fixed/{method}_resnet18_1d/seed_{seed}/metrics.json
# Summary:
#   MODEL_ROOT/results/multi_seed/proximity_fixed_summary.json
#
# Usage:
#   bash scripts/run_multi_seed_proximity_fixed.sh
#   DRY_RUN=1 bash scripts/run_multi_seed_proximity_fixed.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_DIR="${MODEL_ROOT}/results/multi_seed"
ENCODER="resnet18_1d"
TEXT_ENCODER="clip_vit_b32"
METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
ORIG_SEED=42
NEW_SEEDS=(123 456)
ALL_SEEDS=(42 123 456)

CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
TASK_ROOT="${DATA_ROOT}/csibench/multitask/proximity"
REG_PATH="${TASK_ROOT}/pair_registry.json"
TEXT_EMB="${TASK_ROOT}/text_embeddings_${TEXT_ENCODER}.h5"
REGISTRY_PARENT="${DATA_ROOT}/csibench/multitask"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] $*"
        return 0
    fi
    "$@"
}

n_new=$((${#METHODS[@]} * ${#NEW_SEEDS[@]}))
echo "================================================================="
echo "Multi-Seed Proximity (Fixed, 4-class)"
echo "  Methods:  ${METHODS[*]}"
echo "  Seeds:    ${ALL_SEEDS[*]}"
echo "  Reuse seed-${ORIG_SEED}: ${#METHODS[@]} evals"
echo "  New train+eval (seeds ${NEW_SEEDS[*]}): ${n_new} runs"
echo "================================================================="

# ---- Phase 1: Evaluate / copy seed-42 from retrain_proximity_fixed ----
log "Phase 1: Seed-${ORIG_SEED} results"

for method in "${METHODS[@]}"; do
    seed42_out="${RESULTS_DIR}/csibench/proximity_fixed/${method}_${ENCODER}/seed_${ORIG_SEED}"
    ckpt42="${MODEL_ROOT}/checkpoints/csibench-proximity_fixed-${method}/best.pt"

    if [[ -f "${seed42_out}/metrics.json" ]]; then
        echo "  [exists] ${method} seed-${ORIG_SEED}"
        continue
    fi

    # Check if seed-42 was already evaluated by eval_proximity_fixed.sh
    eval_results="${MODEL_ROOT}/results/csibench/proximity_fixed/${method}_${ENCODER}"
    if [[ -f "${eval_results}/metrics.json" ]]; then
        log "  Copying existing seed-42 results for ${method}"
        mkdir -p "${seed42_out}"
        cp "${eval_results}/metrics.json" "${seed42_out}/metrics.json"
        [[ -f "${eval_results}/tsne_embeddings.png" ]] && cp "${eval_results}/tsne_embeddings.png" "${seed42_out}/tsne_embeddings.png"
        echo "  [copied] ${method} seed-${ORIG_SEED}"
        continue
    fi

    # Otherwise, evaluate the seed-42 checkpoint directly
    if [[ ! -f "$ckpt42" ]]; then
        echo "  [WARN] No seed-42 checkpoint: ${ckpt42}"
        continue
    fi

    log "  Evaluating seed-42 for ${method}"
    mkdir -p "${seed42_out}"
    run_cmd python3 -m scripts.run_evaluate \
        --checkpoint "${ckpt42}" \
        --config "${CONFIG}" \
        --override \
            dataset=csibench \
            task=proximity \
            "method=${method}" \
            "csi_encoder.name=${ENCODER}" \
            "task_root=${TASK_ROOT}" \
            "registry_path=${REG_PATH}" \
            "text_embeddings_path=${TEXT_EMB}" \
        --eval_protocols zero_shot retrieval few_shot visualization \
        --output_dir "${seed42_out}" \
        --no_timestamp \
        --data_root "${DATA_ROOT}" \
        --registry_dir "${REGISTRY_PARENT}"

    echo "  [evaluated] ${method} seed-${ORIG_SEED}"
done

# ---- Phase 2: Train + evaluate new seeds ----
log "Phase 2: Training new seeds (${NEW_SEEDS[*]})"

for method in "${METHODS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        run_id="proximity_fixed-${method}-seed${seed}"
        ckpt_base="${MODEL_ROOT}/checkpoints/multi_seed/seed_${seed}"
        ckpt_path="${ckpt_base}/csibench-proximity_fixed-${method}/best.pt"
        out_dir="${RESULTS_DIR}/csibench/proximity_fixed/${method}_${ENCODER}/seed_${seed}"

        log "${run_id}"

        if [[ -f "${out_dir}/metrics.json" ]]; then
            echo "  -> Results exist, skipping."
            continue
        fi

        mkdir -p "${out_dir}"

        # Train
        if [[ ! -f "${ckpt_path}" ]]; then
            log "  Training ${method} seed=${seed}..."
            run_cmd python3 -m training.train \
                --config "${CONFIG}" \
                --override \
                    dataset=csibench \
                    task=proximity_fixed \
                    "method=${method}" \
                    "csi_encoder.name=${ENCODER}" \
                    "text_encoder.name=${TEXT_ENCODER}" \
                    "training.seed=${seed}" \
                    "training.max_epochs=200" \
                    "training.early_stopping_patience=20" \
                    "task_root=${TASK_ROOT}" \
                    "registry_path=${REG_PATH}" \
                    "text_embeddings_path=${TEXT_EMB}" \
                    "logging.checkpoint_dir=${ckpt_base}"
        else
            echo "  Checkpoint exists: ${ckpt_path}"
        fi

        # Evaluate
        if [[ ! -f "${ckpt_path}" && "$DRY_RUN" != "1" ]]; then
            echo "  [WARN] Checkpoint not found after training: ${ckpt_path}"
            continue
        fi

        log "  Evaluating ${method} seed=${seed}..."
        run_cmd python3 -m scripts.run_evaluate \
            --checkpoint "${ckpt_path}" \
            --config "${CONFIG}" \
            --override \
                dataset=csibench \
                task=proximity \
                "method=${method}" \
                "csi_encoder.name=${ENCODER}" \
                "task_root=${TASK_ROOT}" \
                "registry_path=${REG_PATH}" \
                "text_embeddings_path=${TEXT_EMB}" \
            --eval_protocols zero_shot retrieval few_shot visualization \
            --output_dir "${out_dir}" \
            --no_timestamp \
            --data_root "${DATA_ROOT}" \
            --registry_dir "${REGISTRY_PARENT}"

        echo "  -> Saved to ${out_dir}"
    done
done

# ---- Phase 3: Aggregate ----
log "Phase 3: Aggregating multi-seed results"

run_cmd python3 - <<'PYEOF'
import json
import numpy as np
from pathlib import Path
import os

results_dir = Path(os.environ.get("MODEL_ROOT", "/home/sagemaker-user/Multimodal_CSI_Text/Model")) / "results" / "multi_seed"
methods = ["clip_infonce", "siglip", "language_bind", "centrobind", "barlow_twins", "masked_predict"]
seeds = [42, 123, 456]
encoder = "resnet18_1d"

summary = {}
print("\n===== Proximity Fixed: Multi-Seed Summary =====\n")
print(f"{'Method':<20} {'ZS Acc (mean±std)':<22} {'Seeds':>6}")
print("-" * 52)

for method in methods:
    accs = []
    for seed in seeds:
        mf = results_dir / "csibench" / "proximity_fixed" / f"{method}_{encoder}" / f"seed_{seed}" / "metrics.json"
        if mf.exists():
            with open(mf) as f:
                data = json.load(f)
            acc = data.get("zero_shot", {}).get("accuracy")
            if acc is not None:
                accs.append(acc)

    key = f"csibench/proximity_fixed/{method}"
    if accs:
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        summary[key] = {
            "mean": mean_acc,
            "std": std_acc,
            "values": accs,
            "n_seeds": len(accs),
        }
        print(f"{method:<20} {mean_acc*100:.1f} ± {std_acc*100:.1f}%{len(accs):>10}")
    else:
        print(f"{method:<20} {'N/A':<22} {'0':>6}")

out = results_dir / "proximity_fixed_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out}")
PYEOF

log "Multi-seed proximity fixed pipeline complete."
