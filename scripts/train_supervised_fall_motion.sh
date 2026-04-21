#!/usr/bin/env bash
#
# Train supervised ResNet-18 baselines for Fall Detection and Motion Source.
# These are the two CSI-Bench standalone tasks that were previously missing
# controlled supervised baselines (old numbers were from Yang et al. 2025).
#
# Usage (SageMaker):
#   bash scripts/train_supervised_fall_motion.sh
#
# Usage (custom paths):
#   DATA_ROOT=/path/to/data bash scripts/train_supervised_fall_motion.sh
#
# After training, results will be in:
#   paper_results/supervised_baselines/csibench_fall_detection/
#   paper_results/supervised_baselines/csibench_motion_source/
#
# Then update the paper:
#   1. Read supervised_results.json from each output directory
#   2. Update Table 1 in paper/draft/06_experiments.md with the new numbers
#   3. Remove the $^\dagger$ markers from Fall Detection and Motion Source
#   4. Update Finding 1, Discussion (07), and Conclusion (08) accordingly

set -euo pipefail

DATASET=csibench_standalone exec bash "$(dirname "$0")/train_supervised_baselines.sh"
