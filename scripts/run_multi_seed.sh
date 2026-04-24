#!/usr/bin/env bash
#
# Run all 6 methods on main benchmark with multiple seeds for error bars.
#
# This script evaluates all 6 alignment methods across all 7 tasks with
# 3 seeds each, producing variance estimates for every number in Table 1.
# Seed-42 results are reused from the original training runs (Phase 1);
# only seeds 123 and 456 require new training (Phase 2).
#
# Total new training runs: 6 methods x 7 tasks x 2 new seeds = 84
#
# Usage:
#   bash scripts/run_multi_seed.sh
#
# Environment:
#   Designed for SageMaker g5 instances.
#   Uses the same training pipeline as train_sagemaker.sh.

set -euo pipefail

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
RESULTS_DIR="${MODEL_ROOT}/results/multi_seed"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
NEW_SEEDS=(123 456)
ALL_SEEDS=(42 123 456)
ENCODER="resnet18_1d"
ORIG_SEED=42

DATASET_TASKS=(
    "csibench:har"
    "csibench:fall_detection"
    "csibench:motion_source"
    "csibench:proximity"
    "csibench:user_id"
    "xrf55:action_recognition"
    "signfi:sign_recognition"
)

n_new=$((${#METHODS[@]} * ${#DATASET_TASKS[@]} * ${#NEW_SEEDS[@]}))
n_reuse=$((${#METHODS[@]} * ${#DATASET_TASKS[@]}))
echo "================================================================="
echo "Multi-Seed Benchmark"
echo "  Reuse existing seed-${ORIG_SEED} results: ${n_reuse} tasks"
echo "  New training runs (seeds ${NEW_SEEDS[*]}): ${n_new} tasks"
echo "================================================================="

# ---- Phase 1: Reuse seed-42 from original training runs ----
echo ""
echo "--- Phase 1: Linking existing seed-${ORIG_SEED} results ---"

for method in "${METHODS[@]}"; do
    for dt_pair in "${DATASET_TASKS[@]}"; do
        IFS=':' read -r dataset task <<< "$dt_pair"
        orig_results="${MODEL_ROOT}/results/${dataset}/${task}/${method}_${ENCODER}"
        seed42_out="${RESULTS_DIR}/${dataset}/${task}/${method}_${ENCODER}/seed_${ORIG_SEED}"

        if [ -f "${seed42_out}/metrics.json" ]; then
            echo "  [exists] ${dataset}/${task}/${method} seed-${ORIG_SEED}"
            continue
        fi

        if [ ! -f "${orig_results}/metrics.json" ]; then
            echo "  [WARN] Original results missing: ${orig_results}/metrics.json"
            continue
        fi

        mkdir -p "${seed42_out}"
        cp "${orig_results}/metrics.json" "${seed42_out}/metrics.json"
        if [ -f "${orig_results}/tsne_embeddings.png" ]; then
            cp "${orig_results}/tsne_embeddings.png" "${seed42_out}/tsne_embeddings.png"
        fi
        echo "  [copied] ${dataset}/${task}/${method} seed-${ORIG_SEED}"
    done
done

# ---- Phase 2: Train + evaluate new seeds ----
echo ""
echo "--- Phase 2: Training new seeds ---"

for method in "${METHODS[@]}"; do
    for dt_pair in "${DATASET_TASKS[@]}"; do
        IFS=':' read -r dataset task <<< "$dt_pair"
        for seed in "${NEW_SEEDS[@]}"; do
            run_id="${dataset}-${task}-${method}-seed${seed}"
            ckpt_base="${MODEL_ROOT}/checkpoints/multi_seed/seed_${seed}"
            ckpt_path="${ckpt_base}/${dataset}-${task}-${method}/best.pt"
            out_dir="${RESULTS_DIR}/${dataset}/${task}/${method}_${ENCODER}/seed_${seed}"

            echo ""
            echo "----- ${run_id} -----"

            if [ -f "${out_dir}/metrics.json" ]; then
                echo "  -> Already exists, skipping."
                continue
            fi

            mkdir -p "${out_dir}"

            # Determine config, task_root, and registry parent
            CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
            if [ "$dataset" = "csibench" ]; then
                case "$task" in
                    har|proximity|user_id)
                        TASK_ROOT="${DATA_ROOT}/csibench/multitask/${task}"
                        REGISTRY_PARENT="${DATA_ROOT}/csibench/multitask"
                        ;;
                    *)
                        TASK_ROOT="${DATA_ROOT}/csibench/${task}"
                        REGISTRY_PARENT="${DATA_ROOT}/csibench"
                        ;;
                esac
            elif [ "$dataset" = "signfi" ]; then
                TASK_ROOT="${DATA_ROOT}/signfi/sign_recognition"
                REGISTRY_PARENT="${DATA_ROOT}/signfi"
            elif [ "$dataset" = "xrf55" ]; then
                TASK_ROOT="${DATA_ROOT}/xrf55/action_recognition"
                REGISTRY_PARENT="${DATA_ROOT}/xrf55"
            fi

            # Train (logging.checkpoint_dir is the correct config key;
            # the trainer auto-appends {dataset}-{task}-{method}/)
            python3 -m training.train \
                --config "${CONFIG}" \
                --override \
                    dataset="${dataset}" \
                    task="${task}" \
                    method="${method}" \
                    csi_encoder.name="${ENCODER}" \
                    training.seed="${seed}" \
                    training.max_epochs=200 \
                    training.patience=20 \
                    logging.checkpoint_dir="${ckpt_base}" \
                    task_root="${TASK_ROOT}"

            # Evaluate
            python3 -m scripts.run_evaluate \
                --checkpoint "${ckpt_path}" \
                --config "${CONFIG}" \
                --override \
                    dataset="${dataset}" \
                    task="${task}" \
                    method="${method}" \
                    csi_encoder.name="${ENCODER}" \
                    task_root="${TASK_ROOT}" \
                --eval_protocols zero_shot retrieval few_shot visualization \
                --output_dir "${out_dir}" \
                --no_timestamp \
                --data_root "${DATA_ROOT}" \
                --registry_dir "${REGISTRY_PARENT}"

            echo "  -> Saved to ${out_dir}"
        done
    done
done

echo ""
echo "================================================================="
echo "Aggregating multi-seed results..."
echo "================================================================="

python3 - <<PYEOF
import json
import numpy as np
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")
methods = ["clip_infonce", "siglip", "language_bind", "centrobind", "barlow_twins", "masked_predict"]
seeds = [42, 123, 456]
tasks = [
    ("csibench", "har"),
    ("csibench", "fall_detection"),
    ("csibench", "motion_source"),
    ("csibench", "proximity"),
    ("csibench", "user_id"),
    ("xrf55", "action_recognition"),
    ("signfi", "sign_recognition"),
]
encoder = "resnet18_1d"

summary = {}
for dataset, task in tasks:
    for method in methods:
        accs = []
        for seed in seeds:
            mf = results_dir / dataset / task / f"{method}_{encoder}" / f"seed_{seed}" / "metrics.json"
            if mf.exists():
                with open(mf) as f:
                    data = json.load(f)
                acc = data.get("zero_shot", {}).get("accuracy")
                if acc is not None:
                    accs.append(acc)
        if accs:
            key = f"{dataset}/{task}/{method}"
            summary[key] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "values": accs,
                "n_seeds": len(accs),
            }
            print(f"  {key}: {np.mean(accs)*100:.1f} +/- {np.std(accs)*100:.1f}%")

out = results_dir / "multi_seed_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out}")
PYEOF

echo "Done."
