#!/usr/bin/env bash
#
# Re-run multi-seed for motion_source with the correct 4-class label mapping.
#
# ROOT CAUSE: label_mapping.json at csibench/motion_source/metadata/ is the old
# 2-class binary mapping (Fan/IRobot/Pet → 0 "Nonhuman", Human → 1), while the
# pair_registry and text templates use 4 fine-grained classes (Fan=0, Human=1,
# IRobot=2, Pet=3). Seeds 123/456 picked up this 2-class mapping, collapsing
# classes 2+3 into class 0.
#
# APPROACH (non-destructive — follows the proximity_fixed pattern):
#   - Creates a new data directory csibench/motion_source_fourclass/ with the
#     correct label_mapping.json and symlinks to the original data/splits.
#   - Trains using task=motion_source_fourclass so checkpoints are saved to a
#     separate path (csibench-motion_source_fourclass-{method}/best.pt).
#   - Results saved to multi_seed/csibench/motion_source_fourclass/.
#   - Original results at multi_seed/csibench/motion_source/ are UNTOUCHED.
#   - Original checkpoints at multi_seed/seed_N/csibench-motion_source-{method}/
#     are UNTOUCHED.
#
# Checkpoints (new):
#   seed N:  MODEL_ROOT/checkpoints/multi_seed/seed_N/csibench-motion_source_fourclass-{method}/best.pt
# Results (new):
#   MODEL_ROOT/results/multi_seed/csibench/motion_source_fourclass/{method}_resnet18_1d/seed_{seed}/metrics.json
# Old results (preserved):
#   MODEL_ROOT/results/multi_seed/csibench/motion_source/{method}_resnet18_1d/seed_{seed}/metrics.json
#
# Usage:
#   bash scripts/run_multi_seed_motion_source_fixed.sh
#   DRY_RUN=1 bash scripts/run_multi_seed_motion_source_fixed.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_DIR="${MODEL_ROOT}/results/multi_seed"
ENCODER="resnet18_1d"
METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
ORIG_SEED=42
NEW_SEEDS=(123 456)
ALL_SEEDS=(42 123 456)

CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"

# Original data directory (UNTOUCHED)
TASK_ROOT_ORIG="${DATA_ROOT}/csibench/motion_source"
# New data directory with fourclass label_mapping.json
TASK_ROOT_FIXED="${DATA_ROOT}/csibench/motion_source_fourclass"
REGISTRY_PARENT="${DATA_ROOT}/csibench"

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
echo "Multi-Seed Motion Source (4-class, non-destructive)"
echo "  Methods:       ${METHODS[*]}"
echo "  Seeds:         ${ALL_SEEDS[*]}"
echo "  Reuse seed-42: ${#METHODS[@]} evals (from original 4-class runs)"
echo "  New train+eval (seeds ${NEW_SEEDS[*]}): ${n_new} runs"
echo ""
echo "  Original data: ${TASK_ROOT_ORIG}  (UNTOUCHED)"
echo "  Fixed data:    ${TASK_ROOT_FIXED}  (new, symlinked)"
echo "  Old results:   multi_seed/csibench/motion_source/     (PRESERVED)"
echo "  New results:   multi_seed/csibench/motion_source_fourclass/"
echo "================================================================="

# ---- Phase 0: Create motion_source_fourclass data directory ----
log "Phase 0: Setting up motion_source_fourclass data directory"

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "${TASK_ROOT_FIXED}/metadata"

    # Copy the correct 4-class label_mapping.json
    if [[ -f "${TASK_ROOT_ORIG}/metadata/label_mapping_fourclass.json" ]]; then
        cp "${TASK_ROOT_ORIG}/metadata/label_mapping_fourclass.json" \
           "${TASK_ROOT_FIXED}/metadata/label_mapping.json"
        log "  -> Copied label_mapping_fourclass.json as label_mapping.json"
    else
        log "  [ERROR] label_mapping_fourclass.json not found at ${TASK_ROOT_ORIG}/metadata/"
        log "  Downloading from S3..."
        aws s3 cp "s3://rnd-sagemaker/Multimodal_CSI_Text/Dataset/csibench/motion_source/metadata/label_mapping_fourclass.json" \
            "${TASK_ROOT_FIXED}/metadata/label_mapping.json"
    fi

    # Symlink the sample_metadata.csv
    ln -sf "${TASK_ROOT_ORIG}/metadata/sample_metadata.csv" \
           "${TASK_ROOT_FIXED}/metadata/sample_metadata.csv" 2>/dev/null || \
        cp "${TASK_ROOT_ORIG}/metadata/sample_metadata.csv" \
           "${TASK_ROOT_FIXED}/metadata/sample_metadata.csv"

    # Symlink splits directory
    if [[ -d "${TASK_ROOT_ORIG}/splits" ]] && [[ ! -e "${TASK_ROOT_FIXED}/splits" ]]; then
        ln -sf "${TASK_ROOT_ORIG}/splits" "${TASK_ROOT_FIXED}/splits"
    fi

    # Symlink pair_registry.json
    if [[ -f "${TASK_ROOT_ORIG}/pair_registry.json" ]] && [[ ! -e "${TASK_ROOT_FIXED}/pair_registry.json" ]]; then
        ln -sf "${TASK_ROOT_ORIG}/pair_registry.json" "${TASK_ROOT_FIXED}/pair_registry.json"
    fi

    # Symlink text embeddings
    for emb_file in "${TASK_ROOT_ORIG}"/text_embeddings_*.h5; do
        if [[ -f "$emb_file" ]]; then
            bn="$(basename "$emb_file")"
            if [[ ! -e "${TASK_ROOT_FIXED}/${bn}" ]]; then
                ln -sf "$emb_file" "${TASK_ROOT_FIXED}/${bn}"
            fi
        fi
    done

    # Symlink data subdirectories (sub_Fan, sub_Human, sub_IRobot, sub_Pet)
    for subdir in "${TASK_ROOT_ORIG}"/sub_*; do
        if [[ -d "$subdir" ]]; then
            bn="$(basename "$subdir")"
            if [[ ! -e "${TASK_ROOT_FIXED}/${bn}" ]]; then
                ln -sf "$subdir" "${TASK_ROOT_FIXED}/${bn}"
            fi
        fi
    done

    # Verify the label mapping
    log "  Verification:"
    python3 -c "
import json
with open('${TASK_ROOT_FIXED}/metadata/label_mapping.json') as f:
    lm = json.load(f)
print(f'    num_classes: {lm[\"num_classes\"]}')
print(f'    label_to_idx: {lm[\"label_to_idx\"]}')
assert lm['num_classes'] == 4, f'Expected 4 classes, got {lm[\"num_classes\"]}'
assert len(lm['label_to_idx']) == 4, f'Expected 4 labels, got {len(lm[\"label_to_idx\"])}'
print('    -> OK: 4-class mapping verified')
"

    log "  Directory listing:"
    ls -la "${TASK_ROOT_FIXED}/" | head -15
else
    echo "[DRY RUN] Would create ${TASK_ROOT_FIXED}/ with symlinks"
fi

# ---- Phase 1: Copy seed-42 from original 4-class runs ----
log "Phase 1: Seed-${ORIG_SEED} results (copy from original 4-class runs)"

for method in "${METHODS[@]}"; do
    seed42_out="${RESULTS_DIR}/csibench/motion_source_fourclass/${method}_${ENCODER}/seed_${ORIG_SEED}"
    ckpt42="${MODEL_ROOT}/checkpoints/csibench-motion_source-${method}/best.pt"

    if [[ -f "${seed42_out}/metrics.json" ]]; then
        echo "  [exists] ${method} seed-${ORIG_SEED}"
        continue
    fi

    # Copy from original results (these are known to be 4-class valid)
    orig_results="${MODEL_ROOT}/results/csibench/motion_source/${method}_${ENCODER}"
    if [[ -f "${orig_results}/metrics.json" ]]; then
        log "  Copying original 4-class results for ${method}"
        run_cmd mkdir -p "${seed42_out}"
        run_cmd cp "${orig_results}/metrics.json" "${seed42_out}/metrics.json"
        if [[ -f "${orig_results}/tsne_embeddings.png" ]]; then
            run_cmd cp "${orig_results}/tsne_embeddings.png" "${seed42_out}/tsne_embeddings.png"
        fi
        echo "  [copied] ${method} seed-${ORIG_SEED}"
        continue
    fi

    # Fallback: re-evaluate seed-42 checkpoint with the fixed mapping
    if [[ ! -f "$ckpt42" ]]; then
        echo "  [WARN] No seed-42 checkpoint or results: ${method}"
        continue
    fi

    log "  Re-evaluating seed-42 for ${method} with 4-class mapping"
    run_cmd mkdir -p "${seed42_out}"
    run_cmd python3 -m scripts.run_evaluate \
        --checkpoint "${ckpt42}" \
        --config "${CONFIG}" \
        --override \
            dataset=csibench \
            task=motion_source \
            "method=${method}" \
            "csi_encoder.name=${ENCODER}" \
            "task_root=${TASK_ROOT_FIXED}" \
        --eval_protocols zero_shot retrieval few_shot visualization \
        --output_dir "${seed42_out}" \
        --no_timestamp \
        --data_root "${DATA_ROOT}" \
        --registry_dir "${REGISTRY_PARENT}"

    echo "  [evaluated] ${method} seed-${ORIG_SEED}"
done

# ---- Phase 2: Train + evaluate new seeds (separate checkpoint paths) ----
log "Phase 2: Training new seeds (${NEW_SEEDS[*]}) with 4-class mapping"

for method in "${METHODS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        run_id="motion_source_fourclass-${method}-seed${seed}"
        ckpt_base="${MODEL_ROOT}/checkpoints/multi_seed/seed_${seed}"
        ckpt_path="${ckpt_base}/csibench-motion_source_fourclass-${method}/best.pt"
        out_dir="${RESULTS_DIR}/csibench/motion_source_fourclass/${method}_${ENCODER}/seed_${seed}"

        log "${run_id}"

        if [[ -f "${out_dir}/metrics.json" ]]; then
            echo "  -> Results exist, skipping."
            continue
        fi

        run_cmd mkdir -p "${out_dir}"

        # Train with the fourclass data directory
        if [[ ! -f "${ckpt_path}" ]]; then
            log "  Training ${method} seed=${seed}..."
            run_cmd python3 -m training.train \
                --config "${CONFIG}" \
                --override \
                    dataset=csibench \
                    task=motion_source_fourclass \
                    "method=${method}" \
                    "csi_encoder.name=${ENCODER}" \
                    "training.seed=${seed}" \
                    training.max_epochs=200 \
                    training.patience=20 \
                    "task_root=${TASK_ROOT_FIXED}" \
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
                task=motion_source \
                "method=${method}" \
                "csi_encoder.name=${ENCODER}" \
                "task_root=${TASK_ROOT_FIXED}" \
            --eval_protocols zero_shot retrieval few_shot visualization \
            --output_dir "${out_dir}" \
            --no_timestamp \
            --data_root "${DATA_ROOT}" \
            --registry_dir "${REGISTRY_PARENT}"

        echo "  -> Saved to ${out_dir}"
    done
done

# ---- Phase 3: Validate and aggregate ----
log "Phase 3: Validating and aggregating"

python3 - <<PYEOF
import json
import numpy as np
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")
methods = ["clip_infonce", "siglip", "language_bind", "centrobind", "barlow_twins", "masked_predict"]
seeds = [42, 123, 456]
encoder = "${ENCODER}"

print("\n===== Motion Source (4-class, motion_source_fourclass): Validation =====\n")

all_ok = True
for method in methods:
    for seed in seeds:
        mf = results_dir / "csibench" / "motion_source_fourclass" / f"{method}_{encoder}" / f"seed_{seed}" / "metrics.json"
        if not mf.exists():
            print(f"  [MISSING] {method} seed_{seed}")
            all_ok = False
            continue
        with open(mf) as f:
            data = json.load(f)
        cm = data["zero_shot"]["confusion_matrix"]
        row_sums = [sum(row) for row in cm]
        nonzero_rows = sum(1 for s in row_sums if s > 0)
        acc = data["zero_shot"]["accuracy"]
        if nonzero_rows != 4:
            print(f"  [BAD] {method} seed_{seed}: only {nonzero_rows} classes (row_sums={row_sums})")
            all_ok = False
        else:
            print(f"  [OK]  {method} seed_{seed}: acc={acc*100:.1f}%, 4-class CM verified")

print("\n===== Motion Source (4-class): Multi-Seed Summary =====\n")
print(f"{'Method':<20} {'ZS Acc (mean+/-std)':<22} {'Seeds':>6}")
print("-" * 56)

summary = {}
for method in methods:
    accs = []
    for seed in seeds:
        mf = results_dir / "csibench" / "motion_source_fourclass" / f"{method}_{encoder}" / f"seed_{seed}" / "metrics.json"
        if mf.exists():
            with open(mf) as f:
                data = json.load(f)
            acc = data.get("zero_shot", {}).get("accuracy")
            if acc is not None:
                accs.append(acc)

    key = f"csibench/motion_source_fourclass/{method}"
    if accs:
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        summary[key] = {
            "mean": mean_acc,
            "std": std_acc,
            "values": accs,
            "n_seeds": len(accs),
        }
        vals_str = ", ".join(f"{a*100:.1f}" for a in accs)
        print(f"{method:<20} {mean_acc*100:.1f} +/- {std_acc*100:.1f}%{len(accs):>10}  [{vals_str}]")
    else:
        print(f"{method:<20} {'N/A':<22} {'0':>6}")

out = results_dir / "motion_source_fourclass_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out}")

if all_ok:
    print("\nAll 18 metrics files validated (4-class, consistent CMs)")
else:
    print("\nSome results are missing or invalid -- check above")
PYEOF

# ---- Phase 4: Upload fixed results to S3 ----
log "Phase 4: Uploading motion_source_fourclass results to S3"

S3_RESULTS="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results"

for method in "${METHODS[@]}"; do
    for seed in "${ALL_SEEDS[@]}"; do
        local_dir="${RESULTS_DIR}/csibench/motion_source_fourclass/${method}_${ENCODER}/seed_${seed}"
        s3_dir="${S3_RESULTS}/multi_seed/csibench/motion_source_fourclass/${method}_${ENCODER}/seed_${seed}"
        if [[ -d "$local_dir" ]]; then
            run_cmd aws s3 sync "$local_dir" "$s3_dir" --quiet
        fi
    done
done

# Upload summary
local_summary="${RESULTS_DIR}/motion_source_fourclass_summary.json"
s3_summary="${S3_RESULTS}/multi_seed/motion_source_fourclass_summary.json"
if [[ -f "$local_summary" ]]; then
    run_cmd aws s3 cp "$local_summary" "$s3_summary" --quiet
fi

log "  -> Uploaded to ${S3_RESULTS}/multi_seed/csibench/motion_source_fourclass/"

log "Multi-seed motion source (4-class) complete."
echo ""
echo "Nothing was deleted or overwritten:"
echo "  Old buggy results preserved at: multi_seed/csibench/motion_source/"
echo "  Old buggy checkpoints at:       multi_seed/seed_N/csibench-motion_source-{method}/"
echo "  New fixed results at:           multi_seed/csibench/motion_source_fourclass/"
echo "  New fixed checkpoints at:       multi_seed/seed_N/csibench-motion_source_fourclass-{method}/"
echo ""
echo "Next steps:"
echo "  1. Pull results locally: bash scripts/upload_multi_seed_results.sh --pull-only"
echo "  2. Re-run aggregation and update RESULTS_STATUS.md / RESULTS_ORGANIZED.md"
