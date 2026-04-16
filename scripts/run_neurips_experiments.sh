#!/usr/bin/env bash
#
# NeurIPS 2026 Experiment Runner
# ==============================
# Orchestrates all experiments from the NeurIPS Experiment Roadmap:
#   Priority 0: Encoder preliminary (3 runs)
#   Priority 1A: Main benchmark (6 methods x chosen encoder x 3 datasets)
#   Priority 1B: Multitask training
#   Priority 2: Cross-dataset/task transfer (eval-only)
#   Priority 3: Text generation ablations
#   Priority 4: Ambient QA (eval-only)
#   Priority 5: Architecture ablations
#   Priority 6: AM-FM pretrained encoder
#   Priority 7: Analysis & visualization (eval-only)
#
# Usage:
#   # Run all experiments sequentially
#   bash scripts/run_neurips_experiments.sh
#
#   # Run a specific priority level
#   PRIORITY=0 bash scripts/run_neurips_experiments.sh
#
#   # Run priorities 0 through 2
#   PRIORITY=0,1,2 bash scripts/run_neurips_experiments.sh
#
#   # Dry run (print commands without executing)
#   DRY_RUN=1 bash scripts/run_neurips_experiments.sh
#
#   # Override the best encoder (skip Priority 0)
#   BEST_ENCODER=patchtst PRIORITY=1 bash scripts/run_neurips_experiments.sh

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/sagemaker-user/CSI-Text-Multimodal-Alignment}"
DATA_ROOT="${DATA_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-/home/sagemaker-user/Multimodal_CSI_Text/Model}"
DRY_RUN="${DRY_RUN:-0}"
PRIORITY="${PRIORITY:-all}"

BEST_ENCODER="${BEST_ENCODER:-}"
ENCODER_PRELIMINARY_FILE="${MODEL_ROOT}/results/_encoder_preliminary_best.txt"

ALL_METHODS=(clip_infonce siglip language_bind centrobind barlow_twins masked_predict)
ALL_ENCODERS=(resnet18_1d patchtst vit_1d)

CONFIG_SINGLE="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"
CONFIG_MT_CSIBENCH="${REPO_ROOT}/configs/experiment/sagemaker_multitask_csibench.yaml"
CONFIG_MT_ALL="${REPO_ROOT}/configs/experiment/sagemaker_multitask_all.yaml"
CONFIG_AMFM="${REPO_ROOT}/configs/experiment/amfm_alignment.yaml"

LOG_DIR="${MODEL_ROOT}/results/_logs"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() { echo -e "\n\033[1;34m[$(date '+%Y-%m-%d %H:%M:%S')] $*\033[0m"; }
log_section() { echo -e "\n\033[1;32m========================================\n $*\n========================================\033[0m"; }
log_warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
log_error() { echo -e "\033[1;31m[ERROR] $*\033[0m"; }

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
        if [[ "$l" == "$level" ]]; then
            return 0
        fi
    done
    return 1
}

train_single() {
    local dataset="$1" task="$2" method="$3" encoder="$4"
    local extra_overrides=("${@:5}")
    log "Training: dataset=${dataset} task=${task} method=${method} encoder=${encoder}"
    run_cmd env \
        DATASET="$dataset" \
        TASK="$task" \
        METHOD="$method" \
        ENCODER="$encoder" \
        bash "${REPO_ROOT}/scripts/train_sagemaker.sh"
}

evaluate_single() {
    local dataset="$1" task="$2" method="$3" encoder="$4"
    local protocols="${5:-zero_shot cross_task visualization}"

    local ckpt="${MODEL_ROOT}/checkpoints/${dataset}-${task}-${method}/best.pt"
    local results_dir="${MODEL_ROOT}/results/${dataset}/${task}/${method}_${encoder}"

    if [[ ! -f "$ckpt" ]] && [[ "$DRY_RUN" != "1" ]]; then
        log_warn "Checkpoint not found: ${ckpt} — skipping evaluation"
        return 0
    fi

    log "Evaluating: ${dataset}/${task} method=${method} encoder=${encoder}"
    run_cmd python -m scripts.run_evaluate \
        --checkpoint "$ckpt" \
        --config "$CONFIG_SINGLE" \
        --override \
            "dataset=${dataset}" \
            "task=${task}" \
            "method=${method}" \
            "csi_encoder.name=${encoder}" \
        --eval_protocols $protocols \
        --data_root "$DATA_ROOT" \
        --output_dir "$results_dir" \
        --no_timestamp
}

get_best_encoder() {
    if [[ -n "$BEST_ENCODER" ]]; then
        echo "$BEST_ENCODER"
        return
    fi
    if [[ -f "$ENCODER_PRELIMINARY_FILE" ]]; then
        cat "$ENCODER_PRELIMINARY_FILE"
        return
    fi
    echo "resnet18_1d"
}

# ------------------------------------------------------------------
# Priority 0: Quick Encoder Preliminary
# ------------------------------------------------------------------
priority_0() {
    log_section "PRIORITY 0: Encoder Preliminary (clip_infonce x 3 encoders on HAR)"

    for encoder in "${ALL_ENCODERS[@]}"; do
        train_single csibench har clip_infonce "$encoder"
    done

    log "Comparing encoder results..."
    if [[ "$DRY_RUN" == "1" ]]; then
        log "[DRY RUN] Would compare results and pick best encoder"
        return
    fi

    mkdir -p "$(dirname "$ENCODER_PRELIMINARY_FILE")"
    cd "$REPO_ROOT"

    python3 -c "
import json
from pathlib import Path

results_root = Path('${MODEL_ROOT}/results/csibench/har')
encoders = ['resnet18_1d', 'patchtst', 'vit_1d']
best_acc, best_enc = -1, 'resnet18_1d'

print('\\n=== Encoder Preliminary Results ===')
print(f'{\"Encoder\":<15} {\"Accuracy\":<12} {\"Weighted F1\":<12} {\"Linear Probe\":<12}')
print('-' * 55)

for enc in encoders:
    mf = results_root / f'clip_infonce_{enc}' / 'metrics.json'
    if not mf.is_file():
        print(f'{enc:<15} NOT FOUND')
        continue
    with open(mf) as f:
        m = json.load(f)
    zs = m.get('zero_shot', {})
    acc = zs.get('accuracy', 0)
    f1 = zs.get('weighted_f1', 0)
    lp = m.get('visualization', {}).get('linear_probe', {}).get('accuracy', 0)
    print(f'{enc:<15} {acc:<12.4f} {f1:<12.4f} {lp:<12.4f}')
    if acc > best_acc:
        best_acc, best_enc = acc, enc

print(f'\\nBest encoder: {best_enc} (accuracy={best_acc:.4f})')

with open('${ENCODER_PRELIMINARY_FILE}', 'w') as f:
    f.write(best_enc)
"
    log "Best encoder saved to: ${ENCODER_PRELIMINARY_FILE}"
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
        python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
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

    log "Training multitask CSI-Bench..."
    run_cmd env \
        TASK=multitask_csibench \
        METHOD="$best_method" \
        ENCODER="$encoder" \
        bash "${REPO_ROOT}/scripts/train_sagemaker.sh"

    log "Training multitask ALL (CSI-Bench + XRF55 + SignFi)..."
    run_cmd env \
        TASK=multitask_all \
        METHOD="$best_method" \
        ENCODER="$encoder" \
        bash "${REPO_ROOT}/scripts/train_sagemaker.sh"
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
        log "Cross-dataset eval: csibench/har -> xrf55 (method=${method})"
        evaluate_single csibench har "$method" "$encoder" "cross_dataset"

        log "Cross-dataset eval: xrf55/action_recognition -> csibench_har (method=${method})"
        evaluate_single xrf55 action_recognition "$method" "$encoder" "cross_dataset"

        log "Cross-dataset eval: signfi/sign_recognition -> xrf55_gesture (method=${method})"
        evaluate_single signfi sign_recognition "$method" "$encoder" "cross_dataset"
    done

    if [[ "$DRY_RUN" != "1" ]]; then
        cd "$REPO_ROOT"
        python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
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
    if [[ -f "$mt_ckpt" ]] || [[ "$DRY_RUN" == "1" ]]; then
        log "Cross-task eval on multitask model..."
        local mt_results="${MODEL_ROOT}/results/multitask_csibench/clip_infonce_${encoder}"
        run_cmd python -m scripts.run_evaluate \
            --checkpoint "$mt_ckpt" \
            --config "$CONFIG_MT_CSIBENCH" \
            --tasks har fall_detection motion_source proximity user_id \
            --eval_protocols zero_shot cross_task visualization \
            --data_root "$DATA_ROOT" \
            --output_dir "$mt_results" \
            --no_timestamp
    fi

    if [[ "$DRY_RUN" != "1" ]]; then
        cd "$REPO_ROOT"
        python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
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
            bash -c "
                cd ${REPO_ROOT}
                python -m training.train \
                    --config ${CONFIG_SINGLE} \
                    --override \
                        dataset=csibench task=har method=clip_infonce \
                        csi_encoder.name=${encoder} \
                        text_generation.text_level=${level} \
                        logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/ablation_text_level
            "
    done

    log "3B: Many-to-many ratio ablation (InfoNCE vs SigLIP)"
    for n_paraphrases in 1 3 5 10; do
        for method in clip_infonce siglip; do
            log "  Training: method=${method} num_paraphrases=${n_paraphrases}"
            run_cmd env \
                DATASET=csibench TASK=har METHOD="$method" ENCODER="$encoder" \
                bash -c "
                    cd ${REPO_ROOT}
                    python -m training.train \
                        --config ${CONFIG_SINGLE} \
                        --override \
                            dataset=csibench task=har method=${method} \
                            csi_encoder.name=${encoder} \
                            text_generation.num_paraphrases=${n_paraphrases} \
                            logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/ablation_m2m
                "
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
    run_cmd python -m scripts.run_evaluate \
        --checkpoint "$mt_ckpt" \
        --config "$CONFIG_MT_CSIBENCH" \
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
            run_cmd python -m scripts.run_evaluate \
                --checkpoint "$alt_ckpt" \
                --config "$CONFIG_MT_CSIBENCH" \
                --override "method=${method}" \
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
        bash "${REPO_ROOT}/scripts/train_sagemaker.sh"

    log "5C: Embedding dimension ablation"
    for dim in 128 256 512; do
        log "  Training with projection.dim=${dim}"
        run_cmd env \
            DATASET=csibench TASK=har METHOD=clip_infonce ENCODER="$encoder" \
            bash -c "
                cd ${REPO_ROOT}
                python -m training.train \
                    --config ${CONFIG_SINGLE} \
                    --override \
                        dataset=csibench task=har method=clip_infonce \
                        csi_encoder.name=${encoder} \
                        projection.dim=${dim} \
                        projection.hidden_dim=$((dim * 2)) \
                        logging.checkpoint_dir=${MODEL_ROOT}/checkpoints/ablation_embed_dim
            "
    done
}

# ------------------------------------------------------------------
# Priority 6: AM-FM Pretrained Encoder
# ------------------------------------------------------------------
priority_6() {
    log_section "PRIORITY 6: AM-FM Pretrained Encoder"

    log "6A: Frozen AM-FM encoder (train projection only)"
    run_cmd bash -c "
        cd ${REPO_ROOT}
        python -m training.multitask_train \
            --config ${CONFIG_AMFM}
    "

    log "6B: Fine-tuned AM-FM encoder (encoder_lr_scale=0.1)"
    run_cmd bash -c "
        cd ${REPO_ROOT}
        python -m training.multitask_train \
            --config ${CONFIG_AMFM} \
            --override \
                csi_encoder.freeze=false \
                csi_encoder.encoder_lr_scale=0.1
    "
}

# ------------------------------------------------------------------
# Priority 7: Analysis & Visualization
# ------------------------------------------------------------------
priority_7() {
    local encoder
    encoder=$(get_best_encoder)
    log_section "PRIORITY 7: Analysis & Visualization"

    log "Running visualization + linear probe on best models..."
    local tasks=(har fall_detection motion_source)
    local datasets=(csibench csibench csibench)
    local best_method="clip_infonce"

    for i in "${!tasks[@]}"; do
        local t="${tasks[$i]}"
        local d="${datasets[$i]}"
        evaluate_single "$d" "$t" "$best_method" "$encoder" "zero_shot visualization"
    done

    evaluate_single xrf55 action_recognition "$best_method" "$encoder" "zero_shot visualization"
    evaluate_single signfi sign_recognition "$best_method" "$encoder" "zero_shot visualization"

    log "Generating final paper tables and figures..."
    if [[ "$DRY_RUN" != "1" ]]; then
        cd "$REPO_ROOT"
        python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" \
            --results_root "${MODEL_ROOT}/results" \
            --output_dir "${MODEL_ROOT}/results/_paper_tables" \
            --stage all
    fi
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    log_section "NeurIPS 2026 Experiment Runner"
    log "REPO_ROOT:  ${REPO_ROOT}"
    log "DATA_ROOT:  ${DATA_ROOT}"
    log "MODEL_ROOT: ${MODEL_ROOT}"
    log "PRIORITY:   ${PRIORITY}"
    log "DRY_RUN:    ${DRY_RUN}"
    if [[ -n "$BEST_ENCODER" ]]; then
        log "BEST_ENCODER (override): ${BEST_ENCODER}"
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
