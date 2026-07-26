#!/usr/bin/env bash
#
# Cross-domain evaluation of the supervised CSI-only baselines.
#
# Scores the already-trained ResNet-18 1D + linear-head checkpoints from
# train_supervised_baselines.sh on the held-out domain splits, giving the
# CSI-only counterpart to scripts/run_cross_domain_eval.sh.  Running both and
# then scripts/compare_aligned_vs_supervised.py answers whether CSI-text
# alignment generalises to unseen environments and devices better than plain
# supervised training.
#
# Usage:
#   # All five CSI-Bench tasks
#   bash scripts/run_supervised_cross_domain.sh
#
#   # A single task, or several (comma-separated)
#   TASK=har bash scripts/run_supervised_cross_domain.sh
#   TASK=har,proximity bash scripts/run_supervised_cross_domain.sh
#
#   # Fetch any missing checkpoint from S3 first
#   SYNC_S3=1 bash scripts/run_supervised_cross_domain.sh
#
#   # Only one domain axis
#   SPLITS="test cross_env" bash scripts/run_supervised_cross_domain.sh
#
#   # Check checkpoint arity, label map and split files without running the model
#   PREFLIGHT=1 bash scripts/run_supervised_cross_domain.sh
#
#   # Point at a local checkout / dataset instead of the SageMaker layout
#   REPO_ROOT=$PWD DATA_ROOT=/data/Dataset MODEL_ROOT=/data/Model \
#       bash scripts/run_supervised_cross_domain.sh
#
# Note: proximity uses the csibench_proximity_fixed baseline (corrected 4-class
# metadata), matching the model tag that run_cross_domain_eval.sh scores on the
# aligned side; the data still comes from multitask/proximity.

set -euo pipefail

# ------------------------------------------------------------------
# Configuration (override via environment variables)
# ------------------------------------------------------------------
DATASET="${DATASET:-csibench}"
TASK="${TASK:-all}"
SPLITS="${SPLITS:-test cross_env cross_device}"

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
PY_BIN="${PY_BIN:-python}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Supervised baselines live beside the aligned results, one directory per
# dataset_task pair (see train_supervised_baselines.sh).
SUP_ROOT="${SUP_ROOT:-${MODEL_ROOT}/results/supervised_baselines}"
S3_SUP="${S3_SUP:-s3://rnd-sagemaker/Multimodal_CSI_Text/Model/results/supervised_baselines}"
SYNC_S3="${SYNC_S3:-0}"
PREFLIGHT="${PREFLIGHT:-0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ALL_CSIBENCH_TASKS=(har proximity user_id fall_detection motion_source)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

# Mirrors resolve_task_root() in run_cross_domain_eval.sh so both sides read the
# same data. A case statement keeps this runnable under bash 3.2 (stock macOS).
resolve_task_root() {
    case "$1" in
        fall_detection|motion_source)
            echo "${DATA_ROOT}/csibench/$1" ;;
        har|proximity|user_id)
            echo "${DATA_ROOT}/csibench/multitask/$1" ;;
        *)
            echo "${DATA_ROOT}/${DATASET}/$1" ;;
    esac
}

# Proximity was retrained on corrected 4-class metadata under a _fixed tag; only
# the checkpoint/results names change, the data stays in multitask/proximity.
resolve_model_tag() {
    case "$1" in
        proximity) echo "${PROXIMITY_TAG:-proximity_fixed}" ;;
        *)         echo "$1" ;;
    esac
}

read -ra SPLIT_LIST <<< "$SPLITS"

# Reports what the evaluator would load, so a class-count or split-file problem
# surfaces before any GPU time is spent.
preflight_task() {
    local ckpt_dir="$1" task_root="$2"
    "$PY_BIN" - "$ckpt_dir" "$task_root" "${SPLIT_LIST[@]}" <<'PY'
import json
import sys
from pathlib import Path

import torch

ckpt_dir, task_root = Path(sys.argv[1]), Path(sys.argv[2])
splits = sys.argv[3:]

state = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
ckpt_classes = int(state["classifier.weight"].shape[0])
print(f"  checkpoint classes: {ckpt_classes}")

lm_path = task_root / "metadata" / "label_mapping.json"
if lm_path.is_file():
    lm = json.loads(lm_path.read_text())
    lm_classes = int(lm.get("num_classes", len(lm.get("label_to_idx", {}))))
    verdict = "OK" if lm_classes == ckpt_classes else "MISMATCH"
    print(f"  label_mapping.json: {lm_classes} classes [{verdict}]")
    if verdict == "MISMATCH":
        print("    The checkpoint was trained on a different class set than this "
              "task_root; check the model tag.")
else:
    print(f"  label_mapping.json: MISSING at {lm_path}")
    print("    Evaluation will refuse to run: class indices would be rebuilt "
          "per split and could misalign.")

for split in splits:
    found = next(
        (
            p for p in (
                task_root / "splits" / f"{split}_id.json",
                task_root / "splits" / f"{split}_ids.json",
                task_root / "splits" / f"{split}.json",
                task_root / "splits" / f"test_{split}.json",
            )
            if p.is_file()
        ),
        None,
    )
    print(f"  split {split}: {found if found else 'NOT FOUND (will be skipped)'}")
PY
}

# ------------------------------------------------------------------
# Evaluate one supervised baseline on the cross-domain splits
# ------------------------------------------------------------------
evaluate_supervised() {
    local task_name="$1"

    local task_root model_tag ckpt_dir results_dir
    task_root=$(resolve_task_root "$task_name")
    model_tag=$(resolve_model_tag "$task_name")
    ckpt_dir="${SUP_ROOT}/${DATASET}_${model_tag}"
    results_dir="${ckpt_dir}/cross_domain_eval"

    if [[ "$SYNC_S3" == "1" && ! -f "${ckpt_dir}/best.pt" ]]; then
        log "Fetching supervised checkpoint from S3: ${DATASET}_${model_tag}"
        mkdir -p "$ckpt_dir"
        aws s3 sync "${S3_SUP}/${DATASET}_${model_tag}" "$ckpt_dir" --quiet
    fi

    if [[ ! -f "${ckpt_dir}/best.pt" ]]; then
        log "SKIP ${task_name}: no supervised checkpoint at ${ckpt_dir}/best.pt"
        echo "  Train it with: DATASET=csibench bash scripts/train_supervised_baselines.sh"
        echo "  or fetch it with: SYNC_S3=1 bash scripts/run_supervised_cross_domain.sh"
        if [[ "$model_tag" != "$task_name" ]]; then
            echo "  To score the older ${DATASET}_${task_name} baseline instead:"
            echo "    PROXIMITY_TAG=${task_name} bash scripts/run_supervised_cross_domain.sh"
        fi
        return 1
    fi

    log "Supervised cross-domain eval: ${DATASET}/${task_name}"
    echo "  checkpoint: ${ckpt_dir}/best.pt"
    if [[ "$model_tag" != "$task_name" ]]; then
        echo "  model tag:  ${model_tag} (data stays on task '${task_name}')"
    fi
    echo "  task_root:  ${task_root}"
    echo "  splits:     ${SPLIT_LIST[*]}"
    echo "  results:    ${results_dir}"

    cd "$REPO_ROOT"

    if [[ "$PREFLIGHT" == "1" ]]; then
        preflight_task "$ckpt_dir" "$task_root"
        return 0
    fi

    "$PY_BIN" -m scripts.eval_supervised_cross_domain \
        --checkpoint_dir "$ckpt_dir" \
        --task_root "$task_root" \
        --splits "${SPLIT_LIST[@]}" \
        --output_dir "$results_dir" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"

    if [[ -f "${results_dir}/metrics.json" ]]; then
        python3 - "$results_dir" "$task_name" <<'PY'
import json
import sys

results_dir, task = sys.argv[1], sys.argv[2]
with open(f"{results_dir}/metrics.json") as f:
    metrics = json.load(f)

iid = (metrics.get("iid") or {}).get("accuracy")
if isinstance(iid, float):
    print(f"  [{task}] IID test accuracy:  {iid:.4f}")

check = metrics.get("iid_reproduction")
if check:
    verdict = "OK" if check["consistent"] else "FAILED"
    print(
        f"  [{task}] IID reproduction [{verdict}]: recorded "
        f"{check['recorded_at_training']:.4f}, delta {check['delta']:+.4f}"
    )

for split, res in sorted((metrics.get("cross_domain") or {}).items()):
    if not isinstance(res, dict):
        continue
    acc = res.get("accuracy")
    if acc is None:
        print(f"  [{task}] {split}: {res.get('skipped') or res.get('error')}")
        continue
    delta = f"  (delta {acc - iid:+.4f})" if isinstance(iid, float) else ""
    missing = res.get("num_missing_files") or 0
    dropped = f"  [{missing} H5 files missing]" if missing else ""
    print(
        f"  [{task}] {split}: accuracy={acc:.4f}  "
        f"f1={res.get('weighted_f1', 0):.4f}  n={res.get('num_samples', '?')}"
        f"{delta}{dropped}"
    )
PY
    fi
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    local tasks=()
    if [[ "$TASK" == "all" ]]; then
        tasks=("${ALL_CSIBENCH_TASKS[@]}")
    elif [[ "$TASK" == *","* ]]; then
        IFS=',' read -ra tasks <<< "$TASK"
    else
        tasks=("$TASK")
    fi

    log "Supervised cross-domain evaluation over ${#tasks[@]} task(s): ${tasks[*]}"

    local succeeded=() failed=()
    for t in "${tasks[@]}"; do
        if evaluate_supervised "$t"; then
            succeeded+=("$t")
        else
            failed+=("$t")
        fi
    done

    if [[ "$PREFLIGHT" == "1" ]]; then
        log "Preflight done. Checked: ${succeeded[*]:-none}"
        if [[ ${#failed[@]} -gt 0 ]]; then
            echo "  Missing checkpoints: ${failed[*]}"
        fi
        echo ""
        echo "Rerun without PREFLIGHT=1 to score these checkpoints."
        return 0
    fi

    log "Done. Evaluated: ${succeeded[*]:-none}"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "  Skipped/failed: ${failed[*]}"
    fi
    echo ""
    echo "Next: build the comparison table with"
    echo "  python -m scripts.compare_aligned_vs_supervised"
}

main "$@"
