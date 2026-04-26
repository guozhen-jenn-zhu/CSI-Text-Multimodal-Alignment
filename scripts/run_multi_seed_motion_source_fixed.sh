#!/usr/bin/env bash
#
# Fix and re-run multi-seed for motion_source (4-class).
#
# ROOT CAUSE: label_mapping.json on S3 is the old 2-class binary mapping
# (Fan/IRobot/Pet → 0 "Nonhuman", Human → 1), while the pair_registry
# and text templates use 4 fine-grained classes (Fan=0, Human=1, IRobot=2,
# Pet=3). Seeds 123/456 picked up this 2-class mapping, collapsing classes
# 2+3 into class 0. Seed-42 was copied from original runs that used the
# correct mapping (either via sorted-unique fallback or fourclass file).
#
# FIX: Replace label_mapping.json with label_mapping_fourclass.json
# (already exists on S3), then re-train seeds 123/456 and re-evaluate all 3.
#
# Checkpoints:
#   seed 42:  MODEL_ROOT/checkpoints/csibench-motion_source-{method}/best.pt
#   seed N:   MODEL_ROOT/checkpoints/multi_seed/seed_N/csibench-motion_source-{method}/best.pt
# Results:
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
TASK_ROOT="${DATA_ROOT}/csibench/motion_source"
REGISTRY_PARENT="${DATA_ROOT}/csibench"
S3_METADATA="s3://rnd-sagemaker/Multimodal_CSI_Text/Dataset/csibench/motion_source/metadata"

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
echo "Multi-Seed Motion Source Fix (4-class)"
echo "  Methods:  ${METHODS[*]}"
echo "  Seeds:    ${ALL_SEEDS[*]}"
echo "  Reuse seed-${ORIG_SEED}: ${#METHODS[@]} evals (from original runs)"
echo "  New train+eval (seeds ${NEW_SEEDS[*]}): ${n_new} runs"
echo "================================================================="

# ---- Phase 0: Fix label_mapping.json ----
log "Phase 0: Fixing label_mapping.json (2-class → 4-class)"

LM_LOCAL="${TASK_ROOT}/metadata/label_mapping.json"
LM_FOURCLASS="${TASK_ROOT}/metadata/label_mapping_fourclass.json"
LM_BACKUP="${TASK_ROOT}/metadata/label_mapping_2class_backup.json"

if [[ -f "$LM_LOCAL" ]]; then
    NUM_CLS=$(python3 -c "import json; print(json.load(open('${LM_LOCAL}'))['num_classes'])")
    if [[ "$NUM_CLS" == "2" ]]; then
        log "  Current label_mapping.json is 2-class. Backing up and replacing."
        run_cmd cp "$LM_LOCAL" "$LM_BACKUP"

        if [[ -f "$LM_FOURCLASS" ]]; then
            run_cmd cp "$LM_FOURCLASS" "$LM_LOCAL"
            log "  -> Replaced with label_mapping_fourclass.json"
        else
            log "  -> label_mapping_fourclass.json not found locally. Downloading from S3."
            run_cmd aws s3 cp "${S3_METADATA}/label_mapping_fourclass.json" "$LM_LOCAL"
        fi
    elif [[ "$NUM_CLS" == "4" ]]; then
        log "  label_mapping.json already has 4 classes. No fix needed."
    else
        log "  [WARN] Unexpected num_classes=${NUM_CLS} in label_mapping.json"
    fi
else
    log "  label_mapping.json not found locally. Downloading fourclass version from S3."
    mkdir -p "$(dirname "$LM_LOCAL")"
    run_cmd aws s3 cp "${S3_METADATA}/label_mapping_fourclass.json" "$LM_LOCAL"
fi

# Verify
if [[ "$DRY_RUN" != "1" ]] && [[ -f "$LM_LOCAL" ]]; then
    log "  Verification:"
    python3 -c "
import json
with open('${LM_LOCAL}') as f:
    lm = json.load(f)
print(f'    num_classes: {lm[\"num_classes\"]}')
print(f'    label_to_idx: {lm[\"label_to_idx\"]}')
assert lm['num_classes'] == 4, f'Expected 4 classes, got {lm[\"num_classes\"]}'
assert len(lm['label_to_idx']) == 4, f'Expected 4 labels, got {len(lm[\"label_to_idx\"])}'
print('    -> OK: 4-class mapping verified')
"
fi

# Also upload the fix to S3 so future runs use it
log "  Uploading fixed label_mapping.json to S3"
run_cmd aws s3 cp "$LM_LOCAL" "${S3_METADATA}/label_mapping.json"
run_cmd aws s3 cp "$LM_BACKUP" "${S3_METADATA}/label_mapping_2class_backup.json" 2>/dev/null || true

# ---- Phase 1: Evaluate / copy seed-42 from original runs ----
log "Phase 1: Seed-${ORIG_SEED} results"

for method in "${METHODS[@]}"; do
    seed42_out="${RESULTS_DIR}/csibench/motion_source/${method}_${ENCODER}/seed_${ORIG_SEED}"
    ckpt42="${MODEL_ROOT}/checkpoints/csibench-motion_source-${method}/best.pt"

    # Check if existing seed-42 results are valid (4-class)
    if [[ -f "${seed42_out}/metrics.json" ]]; then
        n_cm_rows=$(python3 -c "
import json
with open('${seed42_out}/metrics.json') as f:
    data = json.load(f)
cm = data['zero_shot']['confusion_matrix']
# Count rows with nonzero sum
nonzero = sum(1 for row in cm if sum(row) > 0)
print(nonzero)
")
        if [[ "$n_cm_rows" == "4" ]]; then
            echo "  [exists, 4-class OK] ${method} seed-${ORIG_SEED}"
            continue
        else
            log "  [STALE] ${method} seed-${ORIG_SEED} has ${n_cm_rows}-class CM. Re-evaluating."
        fi
    fi

    # Try to copy from original results
    orig_results="${MODEL_ROOT}/results/csibench/motion_source/${method}_${ENCODER}"
    if [[ -f "${orig_results}/metrics.json" ]]; then
        n_cm_rows=$(python3 -c "
import json
with open('${orig_results}/metrics.json') as f:
    data = json.load(f)
cm = data['zero_shot']['confusion_matrix']
nonzero = sum(1 for row in cm if sum(row) > 0)
print(nonzero)
")
        if [[ "$n_cm_rows" == "4" ]]; then
            log "  Copying valid 4-class seed-42 results for ${method}"
            mkdir -p "${seed42_out}"
            cp "${orig_results}/metrics.json" "${seed42_out}/metrics.json"
            [[ -f "${orig_results}/tsne_embeddings.png" ]] && cp "${orig_results}/tsne_embeddings.png" "${seed42_out}/tsne_embeddings.png"
            echo "  [copied] ${method} seed-${ORIG_SEED}"
            continue
        fi
    fi

    # Otherwise, re-evaluate the seed-42 checkpoint with the fixed mapping
    if [[ ! -f "$ckpt42" ]]; then
        echo "  [WARN] No seed-42 checkpoint: ${ckpt42}"
        continue
    fi

    log "  Re-evaluating seed-42 for ${method} with 4-class mapping"
    mkdir -p "${seed42_out}"
    run_cmd python3 -m scripts.run_evaluate \
        --checkpoint "${ckpt42}" \
        --config "${CONFIG}" \
        --override \
            dataset=csibench \
            task=motion_source \
            "method=${method}" \
            "csi_encoder.name=${ENCODER}" \
            "task_root=${TASK_ROOT}" \
        --eval_protocols zero_shot retrieval few_shot visualization \
        --output_dir "${seed42_out}" \
        --no_timestamp \
        --data_root "${DATA_ROOT}" \
        --registry_dir "${REGISTRY_PARENT}"

    echo "  [evaluated] ${method} seed-${ORIG_SEED}"
done

# ---- Phase 2: Delete stale seeds 123/456 results, re-train + evaluate ----
log "Phase 2: Training new seeds (${NEW_SEEDS[*]}) with 4-class mapping"

for method in "${METHODS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        run_id="motion_source-${method}-seed${seed}"
        ckpt_base="${MODEL_ROOT}/checkpoints/multi_seed/seed_${seed}"
        ckpt_path="${ckpt_base}/csibench-motion_source-${method}/best.pt"
        out_dir="${RESULTS_DIR}/csibench/motion_source/${method}_${ENCODER}/seed_${seed}"

        log "${run_id}"

        # Remove stale results from buggy 2-class run
        if [[ -f "${out_dir}/metrics.json" ]]; then
            n_cm_rows=$(python3 -c "
import json
with open('${out_dir}/metrics.json') as f:
    data = json.load(f)
cm = data['zero_shot']['confusion_matrix']
nonzero = sum(1 for row in cm if sum(row) > 0)
print(nonzero)
")
            if [[ "$n_cm_rows" == "4" ]]; then
                echo "  -> Valid 4-class results exist, skipping."
                continue
            else
                log "  -> Removing stale ${n_cm_rows}-class results"
                rm -f "${out_dir}/metrics.json" "${out_dir}/tsne_embeddings.png"
            fi
        fi

        # Remove stale checkpoint (trained with 2-class mapping)
        if [[ -f "${ckpt_path}" ]]; then
            log "  -> Removing stale 2-class checkpoint: ${ckpt_path}"
            rm -f "${ckpt_path}"
        fi

        mkdir -p "${out_dir}"

        # Train with fixed 4-class mapping
        log "  Training ${method} seed=${seed}..."
        run_cmd python3 -m training.train \
            --config "${CONFIG}" \
            --override \
                dataset=csibench \
                task=motion_source \
                "method=${method}" \
                "csi_encoder.name=${ENCODER}" \
                "training.seed=${seed}" \
                "task_root=${TASK_ROOT}" \
                "logging.checkpoint_dir=${ckpt_base}"

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
                "task_root=${TASK_ROOT}" \
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

print("\n===== Motion Source (4-class fixed): Validation =====\n")

all_ok = True
for method in methods:
    for seed in seeds:
        mf = results_dir / "csibench" / "motion_source" / f"{method}_{encoder}" / f"seed_{seed}" / "metrics.json"
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

print("\n===== Motion Source: Multi-Seed Summary =====\n")
print(f"{'Method':<20} {'ZS Acc (mean±std)':<22} {'Seeds':>6}")
print("-" * 52)

summary = {}
for method in methods:
    accs = []
    for seed in seeds:
        mf = results_dir / "csibench" / "motion_source" / f"{method}_{encoder}" / f"seed_{seed}" / "metrics.json"
        if mf.exists():
            with open(mf) as f:
                data = json.load(f)
            acc = data.get("zero_shot", {}).get("accuracy")
            if acc is not None:
                accs.append(acc)

    key = f"csibench/motion_source/{method}"
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
        print(f"{method:<20} {mean_acc*100:.1f} ± {std_acc*100:.1f}%{len(accs):>10}  [{vals_str}]")
    else:
        print(f"{method:<20} {'N/A':<22} {'0':>6}")

out = results_dir / "motion_source_fixed_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out}")

if all_ok:
    print("\n✓ All 18 metrics files validated (4-class, consistent CMs)")
else:
    print("\n✗ Some results are missing or invalid — check above")
PYEOF

# ---- Phase 4: Upload fixed results to S3 ----
log "Phase 4: Uploading fixed motion_source results to S3"

S3_RESULTS="s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results"

for method in "${METHODS[@]}"; do
    for seed in "${ALL_SEEDS[@]}"; do
        local_dir="${RESULTS_DIR}/csibench/motion_source/${method}_${ENCODER}/seed_${seed}"
        s3_dir="${S3_RESULTS}/multi_seed/csibench/motion_source/${method}_${ENCODER}/seed_${seed}"
        if [[ -d "$local_dir" ]]; then
            run_cmd aws s3 sync "$local_dir" "$s3_dir" --quiet
        fi
    done
done

# Upload summary
s3_summary="${S3_RESULTS}/multi_seed/motion_source_fixed_summary.json"
local_summary="${RESULTS_DIR}/motion_source_fixed_summary.json"
if [[ -f "$local_summary" ]]; then
    run_cmd aws s3 cp "$local_summary" "$s3_summary" --quiet
fi

log "  -> Uploaded to ${S3_RESULTS}/multi_seed/csibench/motion_source/"

log "Multi-seed motion source fix complete."
echo ""
echo "Next steps:"
echo "  1. Pull results locally: bash scripts/upload_multi_seed_results.sh --pull-only"
echo "  2. Re-run aggregation and update RESULTS_STATUS.md / RESULTS_ORGANIZED.md"
