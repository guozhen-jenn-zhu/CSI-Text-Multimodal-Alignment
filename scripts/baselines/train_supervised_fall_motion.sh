#!/usr/bin/env bash
#
# Train supervised ResNet-18 baselines for Fall Detection and Motion Source.
# Convenience wrapper around train_supervised_baselines.sh restricted to the
# two CSI-Bench standalone tasks.
#
# Usage:
#   bash scripts/baselines/train_supervised_fall_motion.sh
#
# Usage (custom paths):
#   DATA_ROOT=/path/to/data bash scripts/baselines/train_supervised_fall_motion.sh
#
# Results are written to:
#   paper_results/supervised_baselines/csibench_fall_detection/
#   paper_results/supervised_baselines/csibench_motion_source/

set -euo pipefail

DATASET=csibench_standalone exec bash "$(dirname "$0")/train_supervised_baselines.sh"
