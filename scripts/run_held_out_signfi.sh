#!/usr/bin/env bash
#
# Train and evaluate zero-shot generalization to held-out SignFi classes.
#
# Splits 276 sign language classes into 200 seen + 76 unseen,
# trains CLIP InfoNCE alignment on the 200-class subset, then
# evaluates zero-shot classification on the 76 held-out classes.
#
# Usage:
#   bash scripts/run_held_out_signfi.sh

set -euo pipefail

REPO_ROOT="/home/sagemaker-user/CSI-Text-Multimodal-Alignment"
DATA_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Dataset"
MODEL_ROOT="/home/sagemaker-user/Multimodal_CSI_Text/Model"
SIGNFI_ROOT="${DATA_ROOT}/signfi/sign_recognition"
OUTPUT_DIR="${MODEL_ROOT}/results/held_out_classes"
CKPT_BASE="${MODEL_ROOT}/checkpoints/held_out_signfi"
CKPT_PATH="${CKPT_BASE}/signfi-sign_recognition-clip_infonce/best.pt"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "================================================================="
echo "Held-Out Classes Experiment: SignFi 200-seen / 76-held-out"
echo "================================================================="

# Step 1: Generate class split
echo ""
echo "--- Step 1: Generating class split ---"
python3 -m scripts.run_held_out_classes \
    --stage split \
    --signfi_root "${SIGNFI_ROOT}" \
    --output_dir "${OUTPUT_DIR}"

# Step 2: Train on 200-class subset
# We need to filter the training data to only include the 200 seen classes.
# This is done by creating a modified label_mapping.json and filtering
# the pair registry.
echo ""
echo "--- Step 2: Training on 200-class subset ---"

SEEN_CLASSES=$(python3 -c "
import json
with open('${OUTPUT_DIR}/held_out_class_split.json') as f:
    split = json.load(f)
print(','.join(str(c) for c in split['seen_class_ids']))
")

python3 -m training.train \
    --config "${REPO_ROOT}/configs/experiment/sagemaker_g5.yaml" \
    --override \
        dataset=signfi \
        task=sign_recognition \
        method=clip_infonce \
        csi_encoder.name=resnet18_1d \
        training.max_epochs=200 \
        training.patience=20 \
        training.seed=42 \
        logging.checkpoint_dir="${CKPT_BASE}" \
        task_root="${SIGNFI_ROOT}" \
        held_out_classes.enabled=true \
        "held_out_classes.seen_class_ids=${SEEN_CLASSES}"

# Step 3: Evaluate on held-out classes
echo ""
echo "--- Step 3: Evaluating on held-out classes ---"
python3 -m scripts.run_held_out_classes \
    --stage evaluate \
    --signfi_root "${SIGNFI_ROOT}" \
    --checkpoint "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --split_path "${OUTPUT_DIR}/held_out_class_split.json"

echo ""
echo "================================================================="
echo "Done. Results saved to ${OUTPUT_DIR}"
echo "================================================================="
