#!/usr/bin/env bash
#
# Run many-to-many retrieval evaluation (mAP / nDCG / R-Precision / MRR over
# the full caption gallery) on all existing trained checkpoints.
#
# This is a post-hoc re-scoring pass: no retraining, one CSI encode per
# checkpoint. Results land in a SEPARATE directory
# ($MODEL_ROOT/results/m2m_retrieval/) by default so nothing is overwritten.
# Use --merge-in-place to merge the new metrics into the original results dirs.
#
# Check first that every checkpoint, registry and split actually loads:
#   python scripts/preflight_m2m.py
#
# Usage:
#   # Default: write to m2m_retrieval/ (safe, no overwrites)
#   bash scripts/run_m2m_retrieval_evals.sh
#
#   # Merge into original results (backs up metrics.json first)
#   bash scripts/run_m2m_retrieval_evals.sh --merge-in-place
#
#   # Dry run
#   DRY_RUN=1 bash scripts/run_m2m_retrieval_evals.sh
#
#   # Single task / method
#   ONLY_TASKS="csibench:har" ONLY_METHODS="clip_infonce" bash scripts/run_m2m_retrieval_evals.sh
#
# Environment:
#   Designed for SageMaker g5 instances.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
DRY_RUN="${DRY_RUN:-0}"

MERGE_IN_PLACE=0
for arg in "$@"; do
    if [[ "$arg" == "--merge-in-place" ]]; then
        MERGE_IN_PLACE=1
    fi
done

CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ALL_METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
ENCODER="resnet18_1d"
TEXT_ENCODER="${TEXT_ENCODER:-clip_vit_b32}"

DATASET_TASKS=(
    "csibench:har"
    "csibench:fall_detection"
    "csibench:motion_source"
    "csibench:proximity"
    "csibench:user_id"
    "xrf55:action_recognition"
    "signfi:sign_recognition"
)

# Optional space-separated filters.
if [[ -n "${ONLY_METHODS:-}" ]]; then
    read -r -a ALL_METHODS <<< "${ONLY_METHODS}"
fi
if [[ -n "${ONLY_TASKS:-}" ]]; then
    read -r -a DATASET_TASKS <<< "${ONLY_TASKS}"
fi

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
log_warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }

resolve_task_root() {
    local dataset="$1" task="$2"
    if [ "$dataset" = "csibench" ]; then
        case "$task" in
            har|proximity|user_id)
                echo "${DATA_ROOT}/csibench/multitask/${task}"
                ;;
            *)
                echo "${DATA_ROOT}/csibench/${task}"
                ;;
        esac
    elif [ "$dataset" = "signfi" ]; then
        echo "${DATA_ROOT}/signfi/sign_recognition"
    elif [ "$dataset" = "xrf55" ]; then
        echo "${DATA_ROOT}/xrf55/action_recognition"
    fi
}

resolve_registry_parent() {
    local dataset="$1" task="$2"
    if [ "$dataset" = "csibench" ]; then
        case "$task" in
            har|proximity|user_id)
                echo "${DATA_ROOT}/csibench/multitask"
                ;;
            *)
                echo "${DATA_ROOT}/csibench"
                ;;
        esac
    elif [ "$dataset" = "signfi" ]; then
        echo "${DATA_ROOT}/signfi"
    elif [ "$dataset" = "xrf55" ]; then
        echo "${DATA_ROOT}/xrf55"
    fi
}

# Some checkpoint directories hold weights from more than one training run:
# the trainer appends only {dataset}-{task}-{method}, so an ablation with a
# different text encoder overwrites best.pt in the same folder.  In
# csibench-har-clip_infonce/ the surviving best.pt is the sentence-transformer
# ablation (90.2% zero-shot), while the published 98.7% comes from
# epoch_150_p0_clip.pt.  Naming it here gives it priority; resolve_checkpoint.py
# verifies the embedded config either way.
known_ckpt_file() {
    case "$1-$2" in
        har-clip_infonce) echo "epoch_150_p0_clip.pt" ;;
        *)                echo "" ;;
    esac
}

# Print the .pt in a run directory whose embedded config matches the encoder /
# text encoder / method being reported, rather than trusting the filename.
resolve_ckpt_path() {
    local ckpt_dir="$1" task_name="$2" method="$3"
    local prefer_args=() known
    known=$(known_ckpt_file "$task_name" "$method")
    if [[ -n "$known" ]]; then
        prefer_args+=(--prefer "$known")
    fi

    python3 -m scripts.resolve_checkpoint \
        --checkpoint_dir "$ckpt_dir" \
        --encoder "$ENCODER" \
        --text_encoder "$TEXT_ENCODER" \
        --method "$method" \
        --task "$task_name" \
        "${prefer_args[@]+"${prefer_args[@]}"}"
}

n_total=0
n_skipped=0
n_evaluated=0
n_failed=0

if [[ "$MERGE_IN_PLACE" == "1" ]]; then
    log "Mode: MERGE-IN-PLACE (will back up and merge into original results)"
else
    log "Mode: SEPARATE OUTPUT (results go to ${MODEL_ROOT}/results/m2m_retrieval/)"
fi

echo "================================================================="
echo "Running many-to-many retrieval on existing checkpoints"
echo "Methods: ${ALL_METHODS[*]}"
echo "Tasks: ${#DATASET_TASKS[@]}"
echo "================================================================="

for dt_pair in "${DATASET_TASKS[@]}"; do
    IFS=':' read -r dataset task <<< "$dt_pair"
    TASK_ROOT=$(resolve_task_root "$dataset" "$task")
    REGISTRY_PARENT=$(resolve_registry_parent "$dataset" "$task")
    REGISTRY_FILE="${REGISTRY_PARENT}/${task}/pair_registry.json"

    if [[ ! -f "$REGISTRY_FILE" ]]; then
        log_warn "Registry not found: ${REGISTRY_FILE} -- ${dataset}/${task} will be skipped"
    fi

    for method in "${ALL_METHODS[@]}"; do
        n_total=$((n_total + 1))
        ckpt_dir="${MODEL_ROOT}/checkpoints/${dataset}-${task}-${method}"

        if [[ ! -d "$ckpt_dir" ]]; then
            log_warn "Checkpoint dir not found: ${ckpt_dir} -- skipping"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        if ! ckpt=$(resolve_ckpt_path "$ckpt_dir" "$task" "$method"); then
            log_warn "No checkpoint in ${ckpt_dir} matches ${ENCODER}/${TEXT_ENCODER}/${method} -- skipping"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        original_results_dir="${MODEL_ROOT}/results/${dataset}/${task}/${method}_${ENCODER}"

        if [[ "$MERGE_IN_PLACE" == "1" ]]; then
            out_dir="${original_results_dir}"
            if [[ -f "${out_dir}/metrics.json" ]]; then
                backup="${out_dir}/metrics.json.bak.$(date +%Y%m%d_%H%M%S)"
                cp "${out_dir}/metrics.json" "$backup"
                log "Backed up ${out_dir}/metrics.json -> $(basename "$backup")"
            fi
        else
            out_dir="${MODEL_ROOT}/results/m2m_retrieval/${dataset}/${task}/${method}_${ENCODER}"
        fi

        log "Evaluating: ${dataset}/${task}/${method}"

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[DRY RUN] python3 -m scripts.run_evaluate \\"
            echo "    --checkpoint ${ckpt} \\"
            echo "    --eval_protocols retrieval_m2m \\"
            echo "    --registry_dir ${REGISTRY_PARENT} \\"
            echo "    --output_dir ${out_dir}"
            n_evaluated=$((n_evaluated + 1))
            continue
        fi

        mkdir -p "$out_dir"

        if python3 -m scripts.run_evaluate \
            --checkpoint "$ckpt" \
            --config "$CONFIG" \
            --override \
                "dataset=${dataset}" \
                "task=${task}" \
                "method=${method}" \
                "csi_encoder.name=${ENCODER}" \
                "task_root=${TASK_ROOT}" \
            --eval_protocols retrieval_m2m \
            --output_dir "$out_dir" \
            --no_timestamp \
            --data_root "$DATA_ROOT" \
            --registry_dir "$REGISTRY_PARENT"
        then
            n_evaluated=$((n_evaluated + 1))
            log "  -> Saved to ${out_dir}"
        else
            n_failed=$((n_failed + 1))
            log_warn "  -> FAILED: ${dataset}/${task}/${method}"
        fi
    done
done

echo ""
echo "================================================================="
echo "Done: ${n_evaluated} evaluated, ${n_failed} failed, ${n_skipped} skipped (no checkpoint), ${n_total} total"
echo "================================================================="

if [[ "$DRY_RUN" != "1" && "$MERGE_IN_PLACE" != "1" ]]; then
    echo ""
    echo "Next: aggregate the run into paper tables"
    echo "  python3 scripts/aggregate_m2m_retrieval.py \\"
    echo "      --results_root ${MODEL_ROOT}/results/m2m_retrieval \\"
    echo "      --output_dir paper_results/m2m_retrieval"
fi
