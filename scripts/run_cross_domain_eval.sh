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
#
#   # Show which checkpoint file each task resolves to, without evaluating
#   PREFLIGHT=1 bash scripts/run_cross_domain_eval.sh
#
# Checkpoint selection: several checkpoint directories contain weights from
# more than one training run, so best.pt is not always the model behind the
# published numbers -- csibench-har-clip_infonce/best.pt is the Priority 5B
# sentence-transformer ablation (90.2% zero-shot on HAR), while the benchmark
# 98.7% comes from epoch_150_p0_clip.pt (resnet18_1d + clip_vit_b32).  The file
# is therefore chosen by scripts/resolve_checkpoint.py, which reads the config
# embedded in each candidate and rejects any that was not trained with
# ENCODER + TEXT_ENCODER + METHOD.  Set CKPT_FILE to force a specific filename.
#
# Note: the proximity task uses the retrained proximity_fixed checkpoints
# (corrected 4-class metadata); see resolve_model_tag below.

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

# Checkpoint selection. CKPT_FILE forces a filename inside the run directory;
# otherwise the file is verified against ENCODER/TEXT_ENCODER/METHOD. IID_TOL is
# how far the freshly measured IID accuracy may drift from the value already
# recorded in metrics.json before the run is flagged as inconsistent.
CKPT_FILE="${CKPT_FILE:-}"
IID_TOL="${IID_TOL:-0.005}"
PREFLIGHT="${PREFLIGHT:-0}"

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
PY_BIN="${PY_BIN:-python}"

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

# Some checkpoints are tagged differently from their data folder. The proximity
# models were retrained on corrected 4-class metadata by
# retrain_proximity_fixed.sh and saved as csibench-proximity_fixed-<method>;
# the data still lives in multitask/proximity, so only the checkpoint and
# results names change. Set PROXIMITY_TAG=proximity to score the older models.
resolve_model_tag() {
    case "$1" in
        proximity) echo "${PROXIMITY_TAG:-proximity_fixed}" ;;
        *)         echo "$1" ;;
    esac
}

# Filenames known to hold the weights behind the published numbers, for
# directories where a later run replaced best.pt. resolve_checkpoint.py would
# reject the wrong file anyway; naming it here keeps the choice explicit and
# gives it priority over any duplicate with an identical config.
known_ckpt_file() {
    case "$1-$2" in
        har-clip_infonce) echo "epoch_150_p0_clip.pt" ;;
        *)                echo "" ;;
    esac
}

# Resolve the checkpoint file for a run directory, verifying that whatever is
# on disk was trained with the encoder / text encoder / method being reported.
resolve_ckpt_path() {
    local ckpt_dir="$1" model_tag="$2"
    local prefer_args=() known
    if [[ -n "$CKPT_FILE" ]]; then
        prefer_args+=(--prefer "$CKPT_FILE")
    fi
    known=$(known_ckpt_file "$model_tag" "$METHOD")
    if [[ -n "$known" ]]; then
        prefer_args+=(--prefer "$known")
    fi

    "$PY_BIN" -m scripts.resolve_checkpoint \
        --checkpoint_dir "$ckpt_dir" \
        --encoder "$ENCODER" \
        --text_encoder "$TEXT_ENCODER" \
        --method "$METHOD" \
        --task "$model_tag" \
        "${prefer_args[@]+"${prefer_args[@]}"}"
}

read -ra SPLIT_LIST <<< "$SPLITS"
read -ra SHOT_LIST <<< "$SHOTS"
read -ra SCOPE_LIST <<< "$FT_SCOPES"

# ------------------------------------------------------------------
# Evaluate one task on the cross-domain splits
# ------------------------------------------------------------------
evaluate_cross_domain() {
    local task_name="$1"

    local task_root reg_dir reg_path ckpt_dir ckpt_path results_dir model_tag
    task_root=$(resolve_task_root "$task_name")
    model_tag=$(resolve_model_tag "$task_name")
    reg_dir="$task_root"
    reg_path="${reg_dir}/${EVAL_REGISTRY_FILE}"
    ckpt_dir="${MODEL_ROOT}/checkpoints/${DATASET}-${model_tag}-${METHOD}${CKPT_SUFFIX}"
    results_dir="${MODEL_ROOT}/results/${DATASET}/${model_tag}/${METHOD}_${ENCODER}${RESULTS_SUFFIX}"

    if [[ ! -d "$ckpt_dir" ]]; then
        log "SKIP ${task_name}: no checkpoint directory at ${ckpt_dir}"
        return 1
    fi

    cd "$REPO_ROOT"
    log "Cross-domain eval: ${DATASET}/${task_name}"
    if ! ckpt_path=$(resolve_ckpt_path "$ckpt_dir" "$model_tag"); then
        log "SKIP ${task_name}: no ${ENCODER} + ${TEXT_ENCODER} checkpoint in ${ckpt_dir}"
        return 1
    fi

    if [[ "$PREFLIGHT" == "1" ]]; then
        echo "  checkpoint: ${ckpt_path}"
        echo "  task_root:  ${task_root}"
        echo "  results:    ${results_dir}"
        return 0
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

    echo "  checkpoint: ${ckpt_path}"
    if [[ "$model_tag" != "$task_name" ]]; then
        echo "  model tag:  ${model_tag} (data stays on task '${task_name}')"
    fi
    echo "  task_root:  ${task_root}"
    echo "  splits:     ${SPLIT_LIST[*]}"
    echo "  protocols:  ${protocols[*]}"
    if [[ "$FEW_SHOT" == "1" ]]; then
        echo "  shots:      ${SHOT_LIST[*]}  scopes: ${SCOPE_LIST[*]}  trials: ${FT_TRIALS}"
    fi
    echo "  results:    ${results_dir}"

    # Recomputing zero_shot overwrites the IID entry of an existing metrics.json,
    # so keep the published value around to compare against afterwards: a drift
    # means this run scored a different model than the reported one.
    local ref_iid="" backup="${results_dir}/metrics.pre_cross_domain.json"
    if [[ "$WITH_IID" == "1" && -f "${results_dir}/metrics.json" ]]; then
        ref_iid=$(python3 -c '
import json, sys
zs = json.load(open(sys.argv[1])).get("zero_shot") or {}
acc = zs.get("accuracy")
print(acc if isinstance(acc, (int, float)) else "")
' "${results_dir}/metrics.json")
        if [[ -n "$ref_iid" ]]; then
            cp "${results_dir}/metrics.json" "$backup"
        fi
    fi

    "$PY_BIN" -m scripts.run_evaluate \
        --checkpoint "$ckpt_path" \
        --config "$CONFIG" \
        --override \
            "dataset=${DATASET}" \
            "task=${task_name}" \
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
        python3 - "$results_dir" "$task_name" "$ref_iid" "$backup" "$IID_TOL" <<'PY'
import json
import os
import sys

results_dir, task, ref_iid, backup, tol = sys.argv[1:6]
with open(f"{results_dir}/metrics.json") as f:
    metrics = json.load(f)

iid = metrics.get("zero_shot", {}).get("accuracy")
if isinstance(iid, float):
    print(f"  [{task}] IID test accuracy:  {iid:.4f}")

if ref_iid and isinstance(iid, float):
    ref = float(ref_iid)
    if abs(ref - iid) > float(tol):
        print(
            f"  [{task}] WARNING: IID accuracy moved {ref:.4f} -> {iid:.4f} "
            f"(> {float(tol):.4f}).\n"
            f"  [{task}]          This checkpoint is not the one behind the "
            f"previously reported number.\n"
            f"  [{task}]          Previous metrics kept at {backup}; inspect the "
            f"checkpoint directory with\n"
            f"  [{task}]          python -m scripts.resolve_checkpoint "
            f"--checkpoint_dir <dir> --info"
        )
    else:
        print(f"  [{task}] IID matches previously recorded {ref:.4f}")
        if os.path.exists(backup):
            os.remove(backup)
elif backup and os.path.exists(backup):
    os.remove(backup)

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

    if [[ "$PREFLIGHT" == "1" ]]; then
        log "Preflight: resolving checkpoints for ${#tasks[@]} task(s): ${tasks[*]}"
    else
        log "Cross-domain evaluation over ${#tasks[@]} task(s): ${tasks[*]}"
    fi

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
