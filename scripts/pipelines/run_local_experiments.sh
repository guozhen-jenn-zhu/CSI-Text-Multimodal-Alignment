#!/usr/bin/env bash
#
# Local Experiment Runner
# =======================
# Local equivalent of scripts/pipelines/run_neurips_experiments.sh -- orchestrates
# the same priority levels but without any SageMaker / AWS S3 dependencies:
#
#   Priority 0:  Encoder preliminary (3 runs)
#   Priority 1A: Main benchmark (6 methods x best encoder x 3 datasets)
#   Priority 1B: Multitask training
#   Priority 2:  Cross-dataset / cross-task transfer (eval-only)
#   Priority 3:  Text generation ablations
#   Priority 4:  Ambient QA (eval-only)
#   Priority 5:  Architecture ablations
#   Priority 6:  AM-FM pretrained encoder
#   Priority 7:  Analysis & visualization (eval-only)
#
# Training is delegated to scripts/pipelines/train_local.sh, which auto-detects
# GPUs and falls back to CPU when none are present.
#
# Usage:
#   bash scripts/pipelines/run_local_experiments.sh
#   PRIORITY=0 bash scripts/pipelines/run_local_experiments.sh
#   PRIORITY=0,1,2 bash scripts/pipelines/run_local_experiments.sh
#   DRY_RUN=1 bash scripts/pipelines/run_local_experiments.sh
#   BEST_ENCODER=resnet18_1d PRIORITY=1 bash scripts/pipelines/run_local_experiments.sh
#   SKIP_EXISTING=0 bash scripts/pipelines/run_local_experiments.sh
#   BATCH_SIZE=32 MAX_EPOCHS=20 bash scripts/pipelines/run_local_experiments.sh
#
# Pointing at your data:
#   DATA_ROOT=/d/Multimodal_CSI_Text/Dataset \
#   MODEL_ROOT=/d/Multimodal_CSI_Text/Model \
#       bash scripts/pipelines/run_local_experiments.sh

set -euo pipefail

# ------------------------------------------------------------------
# Configuration (env-overridable)
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR is scripts/pipelines/, so REPO_ROOT is two levels up.
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/runs}"

DRY_RUN="${DRY_RUN:-0}"
PRIORITY="${PRIORITY:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NUM_GPUS="${NUM_GPUS:--1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"

BEST_ENCODER="${BEST_ENCODER:-}"
ENCODER_PRELIMINARY_FILE="${MODEL_ROOT}/results/_encoder_preliminary_best.txt"

ALL_METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
ALL_ENCODERS=(resnet18_1d patchtst vit_1d)

CONFIG_SINGLE="${REPO_ROOT}/configs/experiment/default.yaml"
CONFIG_MT_CSIBENCH="${REPO_ROOT}/configs/experiment/sagemaker_multitask_csibench.yaml"
CONFIG_MT_ALL="${REPO_ROOT}/configs/experiment/sagemaker_multitask_all.yaml"
CONFIG_AMFM="${REPO_ROOT}/configs/experiment/amfm_alignment.yaml"

LOG_DIR="${MODEL_ROOT}/logs"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/pipelines/train_local.sh"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() { echo -e "\n\033[1;34m[$(date '+%Y-%m-%d %H:%M:%S')] $*\033[0m"; }
log_section() { echo -e "\n\033[1;32m========================================\n $*\n========================================\033[0m"; }
log_warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
log_error() { echo -e "\033[1;31m[ERROR] $*\033[0m"; }

# Translate a path to a form that native Windows tools (incl. Anaconda Python)
# can open. Under MSYS / Git Bash a path like "/d/foo" needs to be translated
# to "D:/foo" because Python on Windows otherwise interprets the leading slash
# as drive-relative. On Linux/macOS this is a no-op.
#
# We prefer the explicit /<drive>/... regex over `cygpath -m` because some
# MSYS installs (notably the one bundled with Anaconda's Git Bash) treat
# /d/... as a path *inside* the MSYS root rather than as a Windows drive
# reference, so `cygpath -m /d/foo` returns
# "C:/Users/.../anaconda3/Library/d/foo" instead of "D:/foo". cygpath is
# only consulted as a last resort for paths the regex can't recognise.
to_winpath() {
    local p="$1"
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*) ;;
        *) echo "$p"; return ;;
    esac

    local drive
    if [[ "$p" =~ ^/([a-zA-Z])/(.*)$ ]]; then
        drive=$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:lower:]' '[:upper:]')
        echo "${drive}:/${BASH_REMATCH[2]}"
        return
    elif [[ "$p" =~ ^/([a-zA-Z])$ ]]; then
        drive=$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:lower:]' '[:upper:]')
        echo "${drive}:/"
        return
    fi

    if command -v cygpath >/dev/null 2>&1; then
        local out
        out=$(cygpath -m "$p" 2>/dev/null) && { echo "$out"; return; }
    fi
    echo "$p"
}

# Resolve a Python interpreter. Prefer $PYTHON_BIN, else `python`, else `python3`.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        log_error "No Python interpreter found. Install Python 3.10+ or set PYTHON_BIN."
        exit 1
    fi
fi
export PYTHON_BIN

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] $*"
        return 0
    fi
    "$@"
}

should_run() {
    local level="$1"
    if [[ "$PRIORITY" == "all" ]]; then
        return 0
    fi
    IFS=',' read -ra levels <<< "$PRIORITY"
    for l in "${levels[@]}"; do
        [[ "$l" == "$level" ]] && return 0
    done
    return 1
}

checkpoint_exists() {
    [[ "$SKIP_EXISTING" == "1" ]] || return 1
    local dataset="$1" task="$2" method="$3"
    [[ -f "${MODEL_ROOT}/checkpoints/${dataset}-${task}-${method}/best.pt" ]]
}

result_exists() {
    [[ "$SKIP_EXISTING" == "1" ]] || return 1
    local dataset="$1" task="$2" method="$3" encoder="$4"
    [[ -f "${MODEL_ROOT}/results/${dataset}/${task}/${method}_${encoder}/metrics.json" ]]
}

train_single() {
    local dataset="$1" task="$2" method="$3" encoder="$4"
    if checkpoint_exists "$dataset" "$task" "$method"; then
        log "SKIP (checkpoint exists): ${dataset}-${task}-${method}"
        return 0
    fi
    log "Training: dataset=${dataset} task=${task} method=${method} encoder=${encoder}"
    run_cmd env \
        REPO_ROOT="$REPO_ROOT" \
        DATA_ROOT="$DATA_ROOT" \
        MODEL_ROOT="$MODEL_ROOT" \
        DATASET="$dataset" \
        TASK="$task" \
        METHOD="$method" \
        ENCODER="$encoder" \
        NUM_GPUS="$NUM_GPUS" \
        BATCH_SIZE="$BATCH_SIZE" \
        MAX_EPOCHS="$MAX_EPOCHS" \
        bash "$TRAIN_SCRIPT"
}

evaluate_single() {
    local dataset="$1" task="$2" method="$3" encoder="$4"
    local protocols="${5:-zero_shot cross_task visualization}"

    local ckpt="${MODEL_ROOT}/checkpoints/${dataset}-${task}-${method}/best.pt"
    local results_dir="${MODEL_ROOT}/results/${dataset}/${task}/${method}_${encoder}"

    if result_exists "$dataset" "$task" "$method" "$encoder"; then
        log "SKIP (results exist): ${results_dir}"
        return 0
    fi

    if [[ ! -f "$ckpt" ]] && [[ "$DRY_RUN" != "1" ]]; then
        log_warn "Checkpoint not found: ${ckpt} -- skipping evaluation"
        return 0
    fi

    log "Evaluating: ${dataset}/${task} method=${method} encoder=${encoder}"
    run_cmd "$PYTHON_BIN" -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$CONFIG_SINGLE" \
        --override \
            "dataset=${dataset}" \
            "task=${task}" \
            "method=${method}" \
            "csi_encoder.name=${encoder}" \
            "dataset_config.data_root=${DATA_ROOT}/${dataset}" \
            "s3.enabled=false" \
        --eval_protocols $protocols \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp
}

get_best_encoder() {
    if [[ -n "$BEST_ENCODER" ]]; then
        echo "$BEST_ENCODER"; return
    fi
    if [[ -f "$ENCODER_PRELIMINARY_FILE" ]]; then
        cat "$ENCODER_PRELIMINARY_FILE"; return
    fi
    echo "resnet18_1d"
}

# ------------------------------------------------------------------
# Priority 0: Encoder Preliminary
#
# Trains clip_infonce with each candidate encoder on csibench/har and then
# evaluates each one (zero_shot + visualization protocols) so we have
# accuracy / weighted-F1 / linear-probe numbers to pick the best encoder.
#
# Each encoder gets its own checkpoint subdir
# (${MODEL_ROOT}/checkpoints/encoder_preliminary/${encoder}/...) because the
# trainer's run name is ${dataset}-${task}-${method} -- without per-encoder
# isolation, runs would silently overwrite each other.
# ------------------------------------------------------------------
priority_0() {
    log_section "PRIORITY 0: Encoder Preliminary (clip_infonce x 3 encoders on HAR)"

    for encoder in "${ALL_ENCODERS[@]}"; do
        local enc_ckpt_parent="${MODEL_ROOT}/checkpoints/encoder_preliminary/${encoder}"
        local enc_ckpt="${enc_ckpt_parent}/csibench-har-clip_infonce/best.pt"
        local enc_results_dir="${MODEL_ROOT}/results/csibench/har/clip_infonce_${encoder}"

        # ---- Train per-encoder
        if [[ "$SKIP_EXISTING" == "1" && -f "$enc_ckpt" ]]; then
            log "SKIP train (checkpoint exists): ${enc_ckpt}"
        else
            log "Training: dataset=csibench task=har method=clip_infonce encoder=${encoder}"
            run_cmd env \
                REPO_ROOT="$REPO_ROOT" \
                DATA_ROOT="$DATA_ROOT" \
                MODEL_ROOT="$MODEL_ROOT" \
                DATASET=csibench \
                TASK=har \
                METHOD=clip_infonce \
                ENCODER="$encoder" \
                NUM_GPUS="$NUM_GPUS" \
                BATCH_SIZE="$BATCH_SIZE" \
                MAX_EPOCHS="$MAX_EPOCHS" \
                EXTRA_OVERRIDES="logging.checkpoint_dir=${enc_ckpt_parent}" \
                bash "$TRAIN_SCRIPT"
        fi

        # ---- Evaluate per-encoder
        if [[ "$SKIP_EXISTING" == "1" && -f "${enc_results_dir}/metrics.json" ]]; then
            log "SKIP eval (results exist): ${enc_results_dir}/metrics.json"
        elif [[ -f "$enc_ckpt" || "$DRY_RUN" == "1" ]]; then
            log "Evaluating: encoder=${encoder}"
            run_cmd "$PYTHON_BIN" -m scripts.run_evaluate \
                --checkpoint "$enc_ckpt" \
                --config "$CONFIG_SINGLE" \
                --override \
                    "dataset=csibench" \
                    "task=har" \
                    "method=clip_infonce" \
                    "csi_encoder.name=${encoder}" \
                    "dataset_config.data_root=${DATA_ROOT}/csibench" \
                    "s3.enabled=false" \
                --eval_protocols zero_shot visualization \
                --data_root "$DATA_ROOT" \
                --output_dir "$enc_results_dir" \
                --no_timestamp
        else
            log_warn "No checkpoint at ${enc_ckpt} -- skipping eval for ${encoder}"
        fi
    done

    if [[ "$DRY_RUN" == "1" ]]; then
        log "[DRY RUN] Would compare results and pick best encoder"
        return
    fi

    mkdir -p "$(dirname "$ENCODER_PRELIMINARY_FILE")"
    cd "$REPO_ROOT"

    # Pass paths to Python in Windows-friendly form, otherwise on MSYS / Git
    # Bash, Python on Windows treats /d/foo as drive-relative and the file
    # lookup fails. We capture the chosen encoder from Python's stdout and
    # write the marker file from Bash itself (Bash handles MinGW paths fine).
    local results_root_win
    results_root_win=$(to_winpath "${MODEL_ROOT}/results/csibench/har")

    local best_enc
    best_enc=$(RESULTS_ROOT="$results_root_win" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])
encoders = ["resnet18_1d", "patchtst", "vit_1d"]
best_acc, best_enc = -1.0, "resnet18_1d"

print("", file=sys.stderr)
print("=== Encoder Preliminary Results ===", file=sys.stderr)
print(f'{"Encoder":<15} {"Accuracy":<12} {"Weighted F1":<12} {"Linear Probe":<12}', file=sys.stderr)
print("-" * 55, file=sys.stderr)
for enc in encoders:
    mf = results_root / f"clip_infonce_{enc}" / "metrics.json"
    if not mf.is_file():
        print(f"{enc:<15} NOT FOUND", file=sys.stderr)
        continue
    with open(mf) as f:
        m = json.load(f)
    zs = m.get("zero_shot", {})
    acc = zs.get("accuracy", 0)
    f1 = zs.get("weighted_f1", 0)
    lp = m.get("visualization", {}).get("linear_probe", {}).get("accuracy", 0)
    print(f"{enc:<15} {acc:<12.4f} {f1:<12.4f} {lp:<12.4f}", file=sys.stderr)
    if acc > best_acc:
        best_acc, best_enc = acc, enc

print("", file=sys.stderr)
print(f"Best encoder: {best_enc} (accuracy={best_acc:.4f})", file=sys.stderr)
print(best_enc)
PY
)

    if [[ -n "$best_enc" ]]; then
        echo "$best_enc" > "$ENCODER_PRELIMINARY_FILE"
        log "Best encoder saved to: ${ENCODER_PRELIMINARY_FILE}"
    else
        log_error "Failed to determine best encoder."
        return 1
    fi
}

# ------------------------------------------------------------------
# Priority 1A: Main Benchmark
# ------------------------------------------------------------------
priority_1a() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 1A: Main Benchmark (6 methods x ${encoder})"

    local tasks_csibench=(har fall_detection motion_source proximity user_id)
    local tasks_xrf55=(action_recognition)
    local tasks_signfi=(sign_recognition)

    for method in "${ALL_METHODS[@]}"; do
        for task in "${tasks_csibench[@]}"; do
            train_single csibench "$task" "$method" "$encoder"
        done
        for task in "${tasks_xrf55[@]}"; do
            train_single xrf55 "$task" "$method" "$encoder"
        done
        for task in "${tasks_signfi[@]}"; do
            train_single signfi "$task" "$method" "$encoder"
        done
    done

    log "Generating benchmark summary..."
    if [[ "$DRY_RUN" != "1" ]]; then
        cd "$REPO_ROOT"
        "$PYTHON_BIN" "${REPO_ROOT}/scripts/paper_experiments/aggregate_neurips_results.py" \
            --results_root "${MODEL_ROOT}/results" \
            --output_dir "${MODEL_ROOT}/results/_paper_tables" \
            --stage main_benchmark
    fi
}

# ------------------------------------------------------------------
# Priority 1B: Multitask Training
# ------------------------------------------------------------------
priority_1b() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 1B: Multitask Training (encoder=${encoder})"

    local best_method="clip_infonce"

    if checkpoint_exists csibench multitask "$best_method"; then
        log "SKIP (checkpoint exists): csibench-multitask-${best_method}"
    else
        log "Training multitask CSI-Bench..."
        # NOTE: pass the parent checkpoints dir; the trainer appends
        # ${dataset}-${task}-${method} (e.g. csibench-multitask-clip_infonce)
        # automatically. Pre-appending it here would double the path.
        run_cmd bash -c "
            cd '${REPO_ROOT}' &&
            '${PYTHON_BIN}' -m training.multitask_train \
                --config '${CONFIG_MT_CSIBENCH}' \
                --override \
                    csi_encoder.name=${encoder} \
                    method=${best_method} \
                    dataset_config.data_root=${DATA_ROOT}/csibench \
                    logging.checkpoint_dir=${MODEL_ROOT}/checkpoints \
                    s3.enabled=false
        "
    fi

    if checkpoint_exists all multitask "$best_method"; then
        log "SKIP (checkpoint exists): all-multitask-${best_method}"
    else
        log "Training multitask ALL (CSI-Bench + XRF55 + SignFi)..."
        run_cmd bash -c "
            cd '${REPO_ROOT}' &&
            '${PYTHON_BIN}' -m training.multitask_train \
                --config '${CONFIG_MT_ALL}' \
                --override \
                    csi_encoder.name=${encoder} \
                    method=${best_method} \
                    dataset_config.data_root=${DATA_ROOT} \
                    logging.checkpoint_dir=${MODEL_ROOT}/checkpoints \
                    s3.enabled=false
        "
    fi
}

# ------------------------------------------------------------------
# Priority 2A: Cross-Dataset Transfer
# ------------------------------------------------------------------
priority_2a() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 2A: Cross-Dataset Transfer"

    local methods_to_eval=(clip_infonce siglip language_bind)
    for method in "${methods_to_eval[@]}"; do
        evaluate_single csibench har "$method" "$encoder" "cross_dataset"
        evaluate_single xrf55 action_recognition "$method" "$encoder" "cross_dataset"
        evaluate_single signfi sign_recognition "$method" "$encoder" "cross_dataset"
    done

    if [[ "$DRY_RUN" != "1" ]]; then
        cd "$REPO_ROOT"
        "$PYTHON_BIN" "${REPO_ROOT}/scripts/paper_experiments/aggregate_neurips_results.py" \
            --results_root "${MODEL_ROOT}/results" \
            --output_dir "${MODEL_ROOT}/results/_paper_tables" \
            --stage cross_dataset
    fi
}

# ------------------------------------------------------------------
# Priority 2B: Cross-Task Transfer
# ------------------------------------------------------------------
priority_2b() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 2B: Cross-Task Transfer"

    local methods_to_eval=(clip_infonce siglip language_bind)
    for method in "${methods_to_eval[@]}"; do
        for task in har fall_detection motion_source proximity user_id; do
            evaluate_single csibench "$task" "$method" "$encoder" "cross_task"
        done
    done

    local mt_ckpt="${MODEL_ROOT}/checkpoints/csibench-multitask-clip_infonce/best.pt"
    local mt_results="${MODEL_ROOT}/results/multitask_csibench/clip_infonce_${encoder}"
    if [[ "$SKIP_EXISTING" == "1" && -f "${mt_results}/summary.json" ]]; then
        log "SKIP (results exist): ${mt_results}"
    elif [[ -f "$mt_ckpt" ]] || [[ "$DRY_RUN" == "1" ]]; then
        log "Cross-task eval on multitask model..."
        run_cmd "$PYTHON_BIN" -m scripts.run_evaluate \
            --checkpoint "$mt_ckpt" \
            --config "$CONFIG_MT_CSIBENCH" \
            --override "dataset_config.data_root=${DATA_ROOT}/csibench" "s3.enabled=false" \
            --tasks har fall_detection motion_source proximity user_id \
            --eval_protocols zero_shot cross_task visualization \
            --data_root "$DATA_ROOT" \
            --output_dir "$mt_results" \
            --no_timestamp
    fi

    if [[ "$DRY_RUN" != "1" ]]; then
        cd "$REPO_ROOT"
        "$PYTHON_BIN" "${REPO_ROOT}/scripts/paper_experiments/aggregate_neurips_results.py" \
            --results_root "${MODEL_ROOT}/results" \
            --output_dir "${MODEL_ROOT}/results/_paper_tables" \
            --stage cross_task
    fi
}

# ------------------------------------------------------------------
# Priority 3: Text Generation Ablations
# ------------------------------------------------------------------
priority_3() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 3: Text Generation Ablations"

    log "3A: Text granularity ablation"
    for level in level1 level1_level2; do
        log "  Training with text_level=${level}"
        run_cmd env \
            DATASET=csibench TASK=har METHOD=clip_infonce ENCODER="$encoder" \
            REPO_ROOT="$REPO_ROOT" DATA_ROOT="$DATA_ROOT" MODEL_ROOT="$MODEL_ROOT" \
            NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" MAX_EPOCHS="$MAX_EPOCHS" \
            EXTRA_OVERRIDES="text_generation.text_level=${level} logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/ablation_text_level/${level}" \
            bash "$TRAIN_SCRIPT"
    done

    log "3B: Many-to-many ratio ablation (InfoNCE vs SigLIP)"
    for n_paraphrases in 1 3 5 10; do
        for method in clip_infonce siglip; do
            log "  Training: method=${method} num_paraphrases=${n_paraphrases}"
            run_cmd env \
                DATASET=csibench TASK=har METHOD="$method" ENCODER="$encoder" \
                REPO_ROOT="$REPO_ROOT" DATA_ROOT="$DATA_ROOT" MODEL_ROOT="$MODEL_ROOT" \
                NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" MAX_EPOCHS="$MAX_EPOCHS" \
                EXTRA_OVERRIDES="text_generation.num_paraphrases=${n_paraphrases} logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/ablation_m2m/${method}_${n_paraphrases}para" \
                bash "$TRAIN_SCRIPT"
        done
    done
}

# ------------------------------------------------------------------
# Priority 4: Ambient Scene QA
# ------------------------------------------------------------------
priority_4() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 4: Ambient Scene QA"

    local mt_ckpt="${MODEL_ROOT}/checkpoints/csibench-multitask-clip_infonce/best.pt"
    if [[ ! -f "$mt_ckpt" ]] && [[ "$DRY_RUN" != "1" ]]; then
        log_warn "Multitask checkpoint not found. Run Priority 1B first."
        return 0
    fi

    log "Running ambient QA on multitask CSI-Bench model..."
    local results_dir="${MODEL_ROOT}/results/ambient_qa/clip_infonce_${encoder}"
    run_cmd "$PYTHON_BIN" -m scripts.run_evaluate \
        --checkpoint "$mt_ckpt" \
        --config "$CONFIG_MT_CSIBENCH" \
        --override "dataset_config.data_root=${DATA_ROOT}/csibench" "s3.enabled=false" \
        --tasks har proximity user_id \
        --eval_protocols zero_shot ambient_qa \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp

    for method in siglip language_bind; do
        local alt_ckpt="${MODEL_ROOT}/checkpoints/csibench-multitask-${method}/best.pt"
        if [[ -f "$alt_ckpt" ]] || [[ "$DRY_RUN" == "1" ]]; then
            log "Ambient QA with method=${method}..."
            local alt_dir="${MODEL_ROOT}/results/ambient_qa/${method}_${encoder}"
            run_cmd "$PYTHON_BIN" -m scripts.run_evaluate \
                --checkpoint "$alt_ckpt" \
                --config "$CONFIG_MT_CSIBENCH" \
                --override "method=${method}" "dataset_config.data_root=${DATA_ROOT}/csibench" "s3.enabled=false" \
                --tasks har proximity user_id \
                --eval_protocols zero_shot ambient_qa \
                --data_root "$DATA_ROOT" \
                --output_dir "$alt_dir" \
                --no_timestamp
        fi
    done
}

# ------------------------------------------------------------------
# Priority 5: Architecture Ablations
# ------------------------------------------------------------------
priority_5() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 5: Architecture Ablations"

    log "5B: Text encoder ablation (CLIP vs sentence-transformers)"
    run_cmd env \
        DATASET=csibench TASK=har METHOD=clip_infonce ENCODER="$encoder" \
        TEXT_ENCODER=sentence_transformer \
        REPO_ROOT="$REPO_ROOT" DATA_ROOT="$DATA_ROOT" MODEL_ROOT="$MODEL_ROOT" \
        NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" MAX_EPOCHS="$MAX_EPOCHS" \
        EXTRA_OVERRIDES="logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/ablation_text_encoder/sentence_transformer" \
        bash "$TRAIN_SCRIPT"

    log "5C: Embedding dimension ablation"
    for dim in 128 256 512; do
        log "  Training with projection.dim=${dim}"
        run_cmd env \
            DATASET=csibench TASK=har METHOD=clip_infonce ENCODER="$encoder" \
            REPO_ROOT="$REPO_ROOT" DATA_ROOT="$DATA_ROOT" MODEL_ROOT="$MODEL_ROOT" \
            NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" MAX_EPOCHS="$MAX_EPOCHS" \
            EXTRA_OVERRIDES="projection.dim=${dim} projection.hidden_dim=$((dim * 2)) logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/ablation_embed_dim/dim_${dim}" \
            bash "$TRAIN_SCRIPT"
    done
}

# ------------------------------------------------------------------
# Priority 6: AM-FM Pretrained Encoder
# ------------------------------------------------------------------
priority_6() {
    log_section "PRIORITY 6: AM-FM Pretrained Encoder"

    log "6A: Frozen AM-FM encoder (train projection only)"
    run_cmd bash -c "
        cd '${REPO_ROOT}' &&
        '${PYTHON_BIN}' -m training.multitask_train \
            --config '${CONFIG_AMFM}' \
            --override \
                dataset_config.data_root=${DATA_ROOT} \
                logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/amfm/frozen \
                s3.enabled=false
    "

    log "6B: Fine-tuned AM-FM encoder (encoder_lr_scale=0.1)"
    run_cmd bash -c "
        cd '${REPO_ROOT}' &&
        '${PYTHON_BIN}' -m training.multitask_train \
            --config '${CONFIG_AMFM}' \
            --override \
                csi_encoder.freeze=false \
                csi_encoder.encoder_lr_scale=0.1 \
                dataset_config.data_root=${DATA_ROOT} \
                logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/amfm/finetuned \
                s3.enabled=false
    "
}

# ------------------------------------------------------------------
# Priority 7: Analysis & Visualization
# ------------------------------------------------------------------
priority_7() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 7: Analysis & Visualization"

    local best_method="clip_infonce"
    for task in har fall_detection motion_source; do
        evaluate_single csibench "$task" "$best_method" "$encoder" "zero_shot visualization"
    done
    evaluate_single xrf55 action_recognition "$best_method" "$encoder" "zero_shot visualization"
    evaluate_single signfi sign_recognition "$best_method" "$encoder" "zero_shot visualization"

    log "Generating final paper tables and figures..."
    if [[ "$DRY_RUN" != "1" ]]; then
        cd "$REPO_ROOT"
        "$PYTHON_BIN" "${REPO_ROOT}/scripts/paper_experiments/aggregate_neurips_results.py" \
            --results_root "${MODEL_ROOT}/results" \
            --output_dir "${MODEL_ROOT}/results/_paper_tables" \
            --stage all
    fi
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    log_section "Local Experiment Runner"
    log "REPO_ROOT:     ${REPO_ROOT}"
    log "DATA_ROOT:     ${DATA_ROOT}"
    log "MODEL_ROOT:    ${MODEL_ROOT}"
    log "PRIORITY:      ${PRIORITY}"
    log "DRY_RUN:       ${DRY_RUN}"
    log "SKIP_EXISTING: ${SKIP_EXISTING}"
    log "NUM_GPUS:      ${NUM_GPUS} (-1 = auto-detect)"
    log "BATCH_SIZE:    ${BATCH_SIZE}"
    log "MAX_EPOCHS:    ${MAX_EPOCHS}"
    if [[ -n "$BEST_ENCODER" ]]; then
        log "BEST_ENCODER (override): ${BEST_ENCODER}"
    fi

    if [[ ! -f "$TRAIN_SCRIPT" ]]; then
        log_error "train_local.sh not found at ${TRAIN_SCRIPT}"
        exit 1
    fi

    mkdir -p "$LOG_DIR"

    if should_run 0; then priority_0 2>&1 | tee "${LOG_DIR}/priority_0.log"; fi
    if should_run 1; then
        priority_1a 2>&1 | tee "${LOG_DIR}/priority_1a.log"
        priority_1b 2>&1 | tee "${LOG_DIR}/priority_1b.log"
    fi
    if should_run 2; then
        priority_2a 2>&1 | tee "${LOG_DIR}/priority_2a.log"
        priority_2b 2>&1 | tee "${LOG_DIR}/priority_2b.log"
    fi
    if should_run 3; then priority_3 2>&1 | tee "${LOG_DIR}/priority_3.log"; fi
    if should_run 4; then priority_4 2>&1 | tee "${LOG_DIR}/priority_4.log"; fi
    if should_run 5; then priority_5 2>&1 | tee "${LOG_DIR}/priority_5.log"; fi
    if should_run 6; then priority_6 2>&1 | tee "${LOG_DIR}/priority_6.log"; fi
    if should_run 7; then priority_7 2>&1 | tee "${LOG_DIR}/priority_7.log"; fi

    log_section "ALL REQUESTED PRIORITIES COMPLETE"
}

main "$@"
