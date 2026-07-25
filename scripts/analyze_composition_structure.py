"""Audit the attribute-combination structure of the co-labeled CSI-Bench subset.

Level 2 ambient QA scores a CSI embedding against 140 composite scene
descriptions (5 activities x 4 proximities x 7 users).  That number is only
meaningful if the 140 combinations are genuinely in play.  This script joins
the har / proximity / user_id multitask metadata on their shared sample IDs and
reports how many combinations actually occur, how strongly the attributes are
coupled, and what a predictor that knows only the combination prior can score.

Usage
-----
PYTHONPATH=. python scripts/analyze_composition_structure.py \
    --config configs/experiment/sagemaker_multitask_scene.yaml \
    --output_dir paper_results/composition
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from evaluation.composition import (
    audit_composition,
    format_audit_markdown,
    load_scene_attribute_table,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit attribute-combination structure of the co-labeled subset",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/experiment/sagemaker_multitask_scene.yaml",
        help="Experiment config providing multitask.tasks entries.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="paper_results/composition",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
    )
    parser.add_argument(
        "--skip_file_check", action="store_true",
        help="Do not drop samples whose H5 file is missing (faster, less exact).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task_entries = config.get("multitask", {}).get("tasks", [])
    if not task_entries:
        raise SystemExit(
            f"{args.config} has no multitask.tasks entries; "
            "use a multitask scene config."
        )

    table = load_scene_attribute_table(
        task_entries,
        splits=tuple(args.splits),
        require_valid_files=not args.skip_file_check,
    )
    audit = audit_composition(table)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "composition_audit.json"
    with open(json_path, "w") as f:
        json.dump(audit, f, indent=2)

    md_path = out_dir / "composition_audit.md"
    with open(md_path, "w") as f:
        f.write(format_audit_markdown(audit))

    print(format_audit_markdown(audit))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
