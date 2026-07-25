#!/usr/bin/env bash
#
# Unseen-composition experiment for the compositional scene claim.
#
# Answers the reviewer question of whether Level 2 ambient QA reflects genuine
# composition or memorised co-occurrence, in four stages:
#
#   audit    How factorial is the co-labeled subset? (no training)
#   splits   Build the held-out-composition task roots (no training)
#   train    Retrain the 3-task scene model on HOC-strict and HOC-text
#   analyze  Zero-training controls + seen/unseen breakdown for every model
#
# Usage:
#   bash scripts/run_composition_experiment.sh                 # all stages
#   STAGES=audit,splits bash scripts/run_composition_experiment.sh
#   PARALLEL=1 bash scripts/run_composition_experiment.sh      # both trainings at once
#   HOLDOUT_FRACTION=0.3 MIN_TEST_SAMPLES=15 bash scripts/run_composition_experiment.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"

METHOD="${METHOD:-clip_infonce}"
ENCODER="${ENCODER:-resnet18_1d}"
TEXT_ENCODER="${TEXT_ENCODER:-clip_vit_b32}"

STAGES="${STAGES:-audit,splits,train,analyze}"
PARALLEL="${PARALLEL:-0}"

HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.25}"
MIN_TEST_SAMPLES="${MIN_TEST_SAMPLES:-20}"
SEED="${SEED:-42}"

BASE_CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_multitask_scene.yaml"
HOC_CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_multitask_scene_hoc.yaml"
HOC_TEXT_CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_multitask_scene_hoc_text.yaml"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/paper_results/composition}"
MANIFEST="${RESULTS_DIR}/composition_split.json"

BASE_CKPT="${BASE_CKPT:-${MODEL_ROOT}/checkpoints/csibench-multitask-${METHOD}/best.pt}"
HOC_CKPT="${MODEL_ROOT}/checkpoints/csibench-multitask_scene_hoc-${METHOD}/best.pt"
HOC_TEXT_CKPT="${MODEL_ROOT}/checkpoints/csibench-multitask_scene_hoc_text-${METHOD}/best.pt"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

detect_gpus() {
    local count
    count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null) || true
    [[ -z "$count" || "$count" -le 0 ]] && count=1
    echo "$count"
}

# ------------------------------------------------------------------
# Stage: composition audit
# ------------------------------------------------------------------
stage_audit() {
    log "Stage audit: how factorial is the co-labeled subset?"
    cd "$REPO_ROOT"
    python -m scripts.analyze_composition_structure \
        --config "$BASE_CONFIG" \
        --output_dir "$RESULTS_DIR"
}

# ------------------------------------------------------------------
# Stage: held-out-composition splits
# ------------------------------------------------------------------
stage_splits() {
    log "Stage splits: building HOC-strict and HOC-text task roots"
    cd "$REPO_ROOT"
    python -m scripts.build_composition_splits \
        --config "$BASE_CONFIG" \
        --mode both \
        --holdout_fraction "$HOLDOUT_FRACTION" \
        --min_test_samples "$MIN_TEST_SAMPLES" \
        --seed "$SEED" \
        --manifest_dir "$RESULTS_DIR" \
        --force
}

# ------------------------------------------------------------------
# Stage: retraining
# ------------------------------------------------------------------
train_variant() {
    # $1 = config, $2 = log tag, $3 = gpu list (empty for all), $4 = master port
    local config="$1" tag="$2" gpus="$3" port="$4"
    local num_gpus

    cd "$REPO_ROOT"
    if [[ -n "$gpus" ]]; then
        export CUDA_VISIBLE_DEVICES="$gpus"
        num_gpus=$(awk -F',' '{print NF}' <<< "$gpus")
    else
        num_gpus=$(detect_gpus)
    fi

    local overrides=(
        "method=${METHOD}"
        "csi_encoder.name=${ENCODER}"
        "text_encoder.name=${TEXT_ENCODER}"
        "logging.checkpoint_dir=${MODEL_ROOT}/checkpoints"
    )

    log "Training ${tag}: ${num_gpus} GPU(s)${gpus:+ (CUDA_VISIBLE_DEVICES=${gpus})}"
    if [[ "$num_gpus" -gt 1 ]]; then
        python -m torch.distributed.run \
            --nproc_per_node "$num_gpus" \
            --master_port "$port" \
            -m training.multitask_train \
            --config "$config" \
            --override "${overrides[@]}"
    else
        python -m training.multitask_train \
            --config "$config" \
            --override "${overrides[@]}"
    fi
}

stage_train() {
    if [[ ! -f "$MANIFEST" ]]; then
        log "ERROR: ${MANIFEST} not found. Run the splits stage first."
        exit 1
    fi

    local total_gpus
    total_gpus=$(detect_gpus)

    if [[ "$PARALLEL" == "1" && "$total_gpus" -ge 2 ]]; then
        local half=$((total_gpus / 2))
        local first second
        first=$(seq -s, 0 $((half - 1)))
        second=$(seq -s, "$half" $((total_gpus - 1)))
        log "Stage train: HOC-strict and HOC-text in parallel on ${total_gpus} GPUs"
        ( train_variant "$HOC_CONFIG" "hoc-strict" "$first" 29500 ) &
        local pid_strict=$!
        ( train_variant "$HOC_TEXT_CONFIG" "hoc-text" "$second" 29600 ) &
        local pid_text=$!
        wait "$pid_strict" "$pid_text"
    else
        log "Stage train: HOC-strict then HOC-text"
        train_variant "$HOC_CONFIG" "hoc-strict" "" 29500
        train_variant "$HOC_TEXT_CONFIG" "hoc-text" "" 29500
    fi
}

# ------------------------------------------------------------------
# Stage: analysis
# ------------------------------------------------------------------
analyze_model() {
    # $1 = name, $2 = checkpoint, $3 = config (population the model was trained on)
    local name="$1" ckpt="$2" config="$3"
    if [[ ! -f "$ckpt" ]]; then
        log "Skipping ${name}: checkpoint not found at ${ckpt}"
        return
    fi

    log "Analyzing ${name}"
    cd "$REPO_ROOT"

    # Controls and the seen/unseen breakdown are always measured on the
    # baseline test population so numbers are comparable across models.
    python -m scripts.run_composition_analysis \
        --checkpoint "$ckpt" \
        --config "$BASE_CONFIG" \
        --registry_dir "${DATA_ROOT}/csibench/multitask" \
        --held_out "$MANIFEST" \
        --output_dir "${RESULTS_DIR}/${name}"

    # Standard ambient QA numbers (Level 1 + Level 2) for the same model.
    python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$config" \
        --eval_protocols ambient_qa \
        --registry_dir "${DATA_ROOT}/csibench/multitask" \
        --data_root "$DATA_ROOT" \
        --output_dir "${RESULTS_DIR}/${name}/ambient_qa" \
        --no_timestamp
}

stage_analyze() {
    if [[ ! -f "$MANIFEST" ]]; then
        log "ERROR: ${MANIFEST} not found. Run the splits stage first."
        exit 1
    fi
    analyze_model "baseline"   "$BASE_CKPT"     "$BASE_CONFIG"
    analyze_model "hoc_strict" "$HOC_CKPT"      "$BASE_CONFIG"
    analyze_model "hoc_text"   "$HOC_TEXT_CKPT" "$BASE_CONFIG"

    log "Building the paper tables"
    cd "$REPO_ROOT"
    python -m scripts.summarize_composition_results \
        --results_dir "$RESULTS_DIR" \
        --paper_dir "${REPO_ROOT}/paper/latex/generated"
}

# ------------------------------------------------------------------
main() {
    log "=== Unseen-composition experiment | stages: ${STAGES} ==="
    mkdir -p "$RESULTS_DIR"

    if has_stage audit;   then stage_audit;   fi
    if has_stage splits;  then stage_splits;  fi
    if has_stage train;   then stage_train;   fi
    if has_stage analyze; then stage_analyze; fi

    log "=== Done. Results under ${RESULTS_DIR} ==="
}

main "$@"
