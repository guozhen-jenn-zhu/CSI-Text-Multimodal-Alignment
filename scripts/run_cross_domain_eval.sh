#!/usr/bin/env bash
#
# Cross-domain (cross-device / cross-environment) evaluation for CSI-Bench.
#
# Evaluates already-trained checkpoints on the held-out domain splits
# (splits/cross_env.json and splits/cross_device.json) without retraining.
# Comparing these against the IID test accuracy quantifies how much each
# model degrades on unseen environments and unseen device hardware.
#
# Usage:
#   # All five CSI-Bench tasks with the default method/encoder
#   bash scripts/run_cross_domain_eval.sh
#
#   # A single task
#   TASK=har bash scripts/run_cross_domain_eval.sh
#
#   # Several tasks (comma-separated)
#   TASK=har,proximity bash scripts/run_cross_domain_eval.sh
#
#   # A different alignment method / encoder
#   METHOD=siglip ENCODER=patchtst bash scripts/run_cross_domain_eval.sh
#
#   # Include the IID zero-shot number in the same run so summary.md
#   # can report the delta (otherwise only the domain splits are scored)
#   WITH_IID=1 bash scripts/run_cross_domain_eval.sh
#
#   # Only one domain axis
#   SPLITS="cross_env" bash scripts/run_cross_domain_eval.sh
#
#   # Also adapt on a few labelled target-domain samples and report how much
#   # accuracy recovers at K = 1/5/10/20 shots
#   FEW_SHOT=1 bash scripts/run_cross_domain_eval.sh
#
#   # Cheaper sweep: projection head only, fewer shots and trials
#   FEW_SHOT=1 SHOTS="5 20" FT_SCOPES="projection" FT_TRIALS=3 \
#       bash scripts/run_cross_domain_eval.sh
#
#   # Point at a local checkout / dataset instead of the SageMaker layout
#   REPO_ROOT=$PWD DATA_ROOT=/data/Dataset MODEL_ROOT=/data/Model \
#       bash scripts/run_cross_domain_eval.sh

set -euo pipefail

# ------------------------------------------------------------------
# Configuration (override via environment variables)
# ------------------------------------------------------------------
DATASET="${DATASET:-csibench}"
TASK="${TASK:-all}"
METHOD="${METHOD:-clip_infonce}"
ENCODER="${ENCODER:-resnet18_1d}"
TEXT_ENCODER="${TEXT_ENCODER:-clip_vit_b32}"
SPLITS="${SPLITS:-cross_env cross_device}"
WITH_IID="${WITH_IID:-0}"

# Few-shot fine-tuning on the held-out domains (off by default: cost scales
# with shots x scopes x trials x splits).
FEW_SHOT="${FEW_SHOT:-0}"
SHOTS="${SHOTS:-1 5 10 20}"
FT_SCOPES="${FT_SCOPES:-projection encoder}"
FT_EPOCHS="${FT_EPOCHS:-100}"
FT_TRIALS="${FT_TRIALS:-5}"

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml}"

# Match the checkpoint / results naming used by train_sagemaker.sh so this
# script can pick up runs produced by the standard pipeline.
RUN_TAG="${RUN_TAG:-${VARIANT:-}}"
if [[ -n "$RUN_TAG" ]]; then
    CKPT_SUFFIX="-${RUN_TAG}"
    RESULTS_SUFFIX="_${RUN_TAG}"
else
    CKPT_SUFFIX=""
    RESULTS_SUFFIX=""
fi
EVAL_REGISTRY_FILE="${EVAL_REGISTRY_FILE:-pair_registry.json}"
EVAL_EMB_FILE="text_embeddings_${TEXT_ENCODER}.h5"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ALL_CSIBENCH_TASKS=(har proximity user_id fall_detection motion_source)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }

# Mirrors resolve_task_root() in train_sagemaker.sh. A case statement keeps
# this runnable under bash 3.2 (stock macOS) as well as SageMaker's bash 5.
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

read -ra SPLIT_LIST <<< "$SPLITS"
read -ra SHOT_LIST <<< "$SHOTS"
read -ra SCOPE_LIST <<< "$FT_SCOPES"

# ------------------------------------------------------------------
# Evaluate one task on the cross-domain splits
# ------------------------------------------------------------------
evaluate_cross_domain() {
    local task_name="$1"

    local task_root reg_dir reg_path ckpt_path results_dir
    task_root=$(resolve_task_root "$task_name")
    reg_dir="$task_root"
    reg_path="${reg_dir}/${EVAL_REGISTRY_FILE}"
    ckpt_path="${MODEL_ROOT}/checkpoints/${DATASET}-${task_name}-${METHOD}${CKPT_SUFFIX}/best.pt"
    results_dir="${MODEL_ROOT}/results/${DATASET}/${task_name}/${METHOD}_${ENCODER}${RESULTS_SUFFIX}"

    if [[ ! -f "$ckpt_path" ]]; then
        log "SKIP ${task_name}: checkpoint not found at ${ckpt_path}"
        return 1
    fi

    local protocols=(cross_domain)
    if [[ "$WITH_IID" == "1" ]]; then
        protocols=(zero_shot cross_domain)
    fi

    local few_shot_args=()
    if [[ "$FEW_SHOT" == "1" ]]; then
        protocols+=(cross_domain_few_shot)
        few_shot_args=(
            --cross_domain_shots "${SHOT_LIST[@]}"
            --cross_domain_ft_scopes "${SCOPE_LIST[@]}"
            --cross_domain_ft_epochs "$FT_EPOCHS"
            --cross_domain_ft_trials "$FT_TRIALS"
        )
    fi

    log "Cross-domain eval: ${DATASET}/${task_name}"
    echo "  checkpoint: ${ckpt_path}"
    echo "  task_root:  ${task_root}"
    echo "  splits:     ${SPLIT_LIST[*]}"
    echo "  protocols:  ${protocols[*]}"
    if [[ "$FEW_SHOT" == "1" ]]; then
        echo "  shots:      ${SHOT_LIST[*]}  scopes: ${SCOPE_LIST[*]}  trials: ${FT_TRIALS}"
    fi
    echo "  results:    ${results_dir}"

    cd "$REPO_ROOT"
    python -m scripts.run_evaluate \
        --checkpoint "$ckpt_path" \
        --config "$CONFIG" \
        --override \
            "task_root=${task_root}" \
            "registry_path=${reg_path}" \
            "text_embeddings_path=${reg_dir}/${EVAL_EMB_FILE}" \
        --eval_protocols "${protocols[@]}" \
        --cross_domain_splits "${SPLIT_LIST[@]}" \
        "${few_shot_args[@]+"${few_shot_args[@]}"}" \
        --registry_dir "$(dirname "$reg_dir")" \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    if [[ -f "${results_dir}/metrics.json" ]]; then
        python3 - "$results_dir" "$task_name" <<'PY'
import json
import sys

results_dir, task = sys.argv[1], sys.argv[2]
with open(f"{results_dir}/metrics.json") as f:
    metrics = json.load(f)

iid = metrics.get("zero_shot", {}).get("accuracy")
if isinstance(iid, float):
    print(f"  [{task}] IID test accuracy:  {iid:.4f}")

for split, res in sorted(metrics.get("cross_domain", {}).items()):
    if not isinstance(res, dict):
        continue
    acc = res.get("accuracy")
    if acc is None:
        print(f"  [{task}] {split}: {res.get('skipped') or res.get('error')}")
        continue
    delta = f"  (delta {acc - iid:+.4f})" if isinstance(iid, float) else ""
    print(
        f"  [{task}] {split}: accuracy={acc:.4f}  "
        f"f1={res.get('weighted_f1', 0):.4f}  n={res.get('num_samples', '?')}{delta}"
    )

for split, res in sorted(metrics.get("cross_domain_few_shot", {}).items()):
    if not isinstance(res, dict):
        continue
    ft = res.get("finetune")
    if not ft:
        print(f"  [{task}] {split} few-shot: {res.get('skipped') or res.get('error')}")
        continue
    base = (res.get("zero_shot_baseline") or {}).get("accuracy")
    base_str = f"{base:.4f}" if isinstance(base, float) else "n/a"
    for scope, per_k in sorted(ft.items()):
        entries = sorted(per_k.values(), key=lambda m: m["k"])
        cells = "  ".join(
            f"K={m['k']}:{m['accuracy_mean']:.4f}" for m in entries
        )
        print(f"  [{task}] {split} few-shot [{scope}]: K=0:{base_str}  {cells}")
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

    log "Cross-domain evaluation over ${#tasks[@]} task(s): ${tasks[*]}"

    local succeeded=() failed=()
    for t in "${tasks[@]}"; do
        if evaluate_cross_domain "$t"; then
            succeeded+=("$t")
        else
            failed+=("$t")
        fi
    done

    log "Done. Evaluated: ${succeeded[*]:-none}"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "  Skipped/failed: ${failed[*]}"
    fi
}

main "$@"
