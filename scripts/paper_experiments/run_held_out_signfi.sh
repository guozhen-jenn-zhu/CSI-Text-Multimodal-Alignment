#!/usr/bin/env bash
#
# Evaluate zero-shot generalization to held-out SignFi classes.
#
# By default, runs ONLY the evaluation step (Step 3) with an existing
# checkpoint.  Pass --retrain to also run Steps 1-2 (split + train).
#
# Training mirrors the standard train_sagemaker.sh pipeline: auto-detects
# GPUs, uses torchrun for multi-GPU, and passes registry + text embeddings
# explicitly.
#
# Usage:
#   # Evaluate only (recommended -- uses existing checkpoint):
#   bash scripts/paper_experiments/run_held_out_signfi.sh
#
#   # Full pipeline (split + train + evaluate):
#   bash scripts/paper_experiments/run_held_out_signfi.sh --retrain
#
#   # Retrain with fewer GPUs (more gradient steps per epoch):
#   bash scripts/paper_experiments/run_held_out_signfi.sh --retrain --gpus=2

set -euo pipefail

RETRAIN=false
MAX_GPUS=""
for arg in "$@"; do
    case "$arg" in
        --retrain) RETRAIN=true ;;
        --gpus=*) MAX_GPUS="${arg#*=}" ;;
    esac
done

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
SIGNFI_ROOT="${DATA_ROOT}/signfi/sign_recognition"
REGISTRY_PATH="${SIGNFI_ROOT}/pair_registry.json"
TEXT_ENCODER="${TEXT_ENCODER:-clip_vit_b32}"
TEXT_EMB_PATH="${SIGNFI_ROOT}/text_embeddings_${TEXT_ENCODER}.h5"
OUTPUT_DIR="${MODEL_ROOT}/results/held_out_classes_fixed"
CKPT_BASE="${MODEL_ROOT}/checkpoints/held_out_signfi"
CKPT_PATH="${CKPT_BASE}/signfi-sign_recognition-clip_infonce/best.pt"
CONFIG="${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml"

# Use the ORIGINAL split (class IDs 0-275 are identical regardless of
# registry vs template ordering, so the existing split is valid).
SPLIT_PATH="${MODEL_ROOT}/results/held_out_classes/held_out_class_split.json"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

detect_gpus() {
    local count
    count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null) || true
    if [[ -z "$count" ]] || [[ "$count" -le 0 ]]; then
        count=$(nvidia-smi -L 2>/dev/null | wc -l) || true
    fi
    if [[ -z "$count" ]] || [[ "$count" -le 0 ]]; then
        count=1
    fi
    echo "$count"
}

NUM_GPUS=$(detect_gpus)
if [[ -n "${MAX_GPUS}" ]] && [[ "${MAX_GPUS}" -lt "${NUM_GPUS}" ]]; then
    NUM_GPUS="${MAX_GPUS}"
fi

echo "================================================================="
echo "Held-Out Classes Experiment: SignFi 200-seen / 76-held-out"
echo "  (Fixed: uses pair registry for consistent class IDs)"
echo "  Retrain: ${RETRAIN}  |  GPUs: ${NUM_GPUS}"
echo "================================================================="

if [ "${RETRAIN}" = true ]; then
    # Step 1: Generate class split (using pair registry for correct class IDs)
    echo ""
    echo "--- Step 1: Generating class split ---"
    SPLIT_PATH="${OUTPUT_DIR}/held_out_class_split.json"
    python3 -m scripts.paper_experiments.run_held_out_classes \
        --stage split \
        --signfi_root "${SIGNFI_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --registry_path "${REGISTRY_PATH}"

    # Step 2: Train on 200-class subset
    # Uses the same pipeline as train_sagemaker.sh: multi-GPU, explicit
    # registry + text embeddings, matching the full model's training setup.
    echo ""
    echo "--- Step 2: Training on 200-class subset ---"

    if [ ! -f "${REGISTRY_PATH}" ]; then
        echo "ERROR: Pair registry not found: ${REGISTRY_PATH}"
        exit 1
    fi
    if [ ! -f "${TEXT_EMB_PATH}" ]; then
        echo "Generating text embeddings..."
        python3 -m scripts.generate_text_embeddings \
            --registry_path "${REGISTRY_PATH}" \
            --output_path "${TEXT_EMB_PATH}" \
            --encoder "${TEXT_ENCODER}" \
            --batch_size 256
    fi

    SEEN_CLASSES=$(python3 -c "
import json
with open('${SPLIT_PATH}') as f:
    split = json.load(f)
print(','.join(str(c) for c in split['seen_class_ids']))
")

    OVERRIDES=(
        "dataset=signfi"
        "task=sign_recognition"
        "method=clip_infonce"
        "csi_encoder.name=resnet18_1d"
        "text_encoder.name=${TEXT_ENCODER}"
        "task_root=${SIGNFI_ROOT}"
        "registry_path=${REGISTRY_PATH}"
        "text_embeddings_path=${TEXT_EMB_PATH}"
        "training.batch_size=128"
        "training.max_epochs=200"
        "training.learning_rate=1e-3"
        "training.num_workers=4"
        "training.seed=42"
        "logging.checkpoint_dir=${CKPT_BASE}"
        "held_out_classes.enabled=true"
        "held_out_classes.seen_class_ids=${SEEN_CLASSES}"
    )

    echo "  GPUs: ${NUM_GPUS}  |  effective batch_size: $((128 * NUM_GPUS))"
    echo "  registry:   ${REGISTRY_PATH}"
    echo "  embeddings: ${TEXT_EMB_PATH}"

    VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

    if [[ "${NUM_GPUS}" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
        python3 -m torch.distributed.run \
            --nproc_per_node "${NUM_GPUS}" \
            --master_port 29500 \
            -m training.train \
            --config "${CONFIG}" \
            --override "${OVERRIDES[@]}"
    else
        CUDA_VISIBLE_DEVICES="0" \
        python3 -m training.train \
            --config "${CONFIG}" \
            --override "${OVERRIDES[@]}"
    fi

    echo "Training complete."
fi

# Step 3: Evaluate on held-out classes (using pair registry for correct class texts)
echo ""
echo "--- Step 3: Evaluating on held-out classes ---"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Split:      ${SPLIT_PATH}"
echo "  Registry:   ${REGISTRY_PATH}"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CKPT_PATH}"
    echo "  Run with --retrain to train a new model, or set CKPT_PATH."
    exit 1
fi

python3 -m scripts.paper_experiments.run_held_out_classes \
    --stage evaluate \
    --signfi_root "${SIGNFI_ROOT}" \
    --checkpoint "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --registry_path "${REGISTRY_PATH}"

echo ""
echo "================================================================="
echo "Done. Results saved to ${OUTPUT_DIR}"
echo "================================================================="
