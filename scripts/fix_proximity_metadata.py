#!/usr/bin/env python
"""Fix proximity metadata for CSI-Bench: correct 0.5m labels.

The CSI-Bench multitask proximity task has a metadata bug where samples
recorded at 0.5m distance (file paths containing 'walking-case1') have
incorrect ``label`` and ``distance`` values.  This script:

  1. Loads the proximity ``sample_metadata.csv`` (local or from S3).
  2. Identifies rows whose ``file_path`` contains the 0.5m indicator
     (default: ``walking-case1``).
  3. Overwrites their ``label`` and ``distance`` to ``0.5``.
  4. Writes/updates ``label_mapping.json`` to the 4-class scheme
     {0.5->0, 1.5->1, 2.5->2, 3.5->3}.
  5. Backs up original files before overwriting.
  6. Optionally uploads the fixed files back to S3.

Usage (local data, e.g. SageMaker):
    python scripts/fix_proximity_metadata.py \
        --task_root /home/sagemaker-user/Multimodal_CSI_Text/Dataset/csibench/multitask/proximity

Upload fixes to both source and destination S3 prefixes:
    python scripts/fix_proximity_metadata.py \
        --task_root /home/sagemaker-user/Multimodal_CSI_Text/Dataset/csibench/multitask/proximity \
        --upload_s3

Dry-run (report changes without writing):
    python scripts/fix_proximity_metadata.py \
        --task_root /path/to/proximity \
        --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CORRECT_LABEL_MAPPING = {
    "label_to_idx": {"0.5": 0, "1.5": 1, "2.5": 2, "3.5": 3},
    "idx_to_label": {"0": "0.5", "1": "1.5", "2": "2.5", "3": "3.5"},
    "num_classes": 4,
}

DEFAULT_PATTERN = "walking-case1"

S3_BUCKET = "rnd-sagemaker"
S3_PREFIXES = [
    "Data/fm_downstream/CSI-Bench/Multitask/H5/ProximityRecognition",
    "Multimodal_CSI_Text/Dataset/csibench/multitask/proximity",
]


def fix_metadata_csv(
    csv_path: Path,
    pattern: str,
    correct_label: str = "0.5",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Fix label and distance for rows matching *pattern* in file_path."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    logger.info("Columns: %s", list(df.columns))

    # --- Before stats ---
    logger.info("\n=== BEFORE fix ===")
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())
    if "distance" in df.columns:
        logger.info(
            "Distance distribution:\n%s",
            df["distance"].value_counts().to_string(),
        )

    # --- Identify affected rows ---
    mask = df["file_path"].str.contains(pattern, na=False)
    n_affected = mask.sum()
    logger.info(
        "\nRows matching pattern '%s': %d / %d (%.1f%%)",
        pattern, n_affected, len(df), 100 * n_affected / len(df),
    )

    if n_affected == 0:
        logger.warning("No rows matched pattern '%s' — nothing to fix.", pattern)
        return df

    old_labels = df.loc[mask, "label"].value_counts()
    logger.info("Current labels of affected rows:\n%s", old_labels.to_string())

    if "distance" in df.columns:
        old_dist = df.loc[mask, "distance"].value_counts()
        logger.info(
            "Current distances of affected rows:\n%s", old_dist.to_string(),
        )

    # --- Apply fix ---
    df.loc[mask, "label"] = correct_label
    if "distance" in df.columns:
        df.loc[mask, "distance"] = correct_label

    # --- After stats ---
    logger.info("\n=== AFTER fix ===")
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())
    if "distance" in df.columns:
        logger.info(
            "Distance distribution:\n%s",
            df["distance"].value_counts().to_string(),
        )

    if not dry_run:
        backup = csv_path.with_suffix(".csv.bak")
        if not backup.exists():
            shutil.copy2(csv_path, backup)
            logger.info("Backed up original CSV to %s", backup)
        df.to_csv(csv_path, index=False)
        logger.info("Wrote fixed CSV to %s", csv_path)
    else:
        logger.info("[DRY RUN] Would write fixed CSV to %s", csv_path)

    return df


def fix_label_mapping(
    meta_dir: Path,
    dry_run: bool = False,
) -> None:
    """Write the correct 4-class label_mapping.json."""
    lm_path = meta_dir / "label_mapping.json"

    if lm_path.exists():
        with open(lm_path) as f:
            old = json.load(f)
        logger.info("Existing label_mapping.json: %s", json.dumps(old, indent=2))
        if not dry_run:
            backup = lm_path.with_suffix(".json.bak")
            if not backup.exists():
                shutil.copy2(lm_path, backup)
                logger.info("Backed up original label_mapping to %s", backup)

    if not dry_run:
        with open(lm_path, "w") as f:
            json.dump(CORRECT_LABEL_MAPPING, f, indent=2)
        logger.info("Wrote label_mapping.json: %s", json.dumps(CORRECT_LABEL_MAPPING))
    else:
        logger.info(
            "[DRY RUN] Would write label_mapping.json: %s",
            json.dumps(CORRECT_LABEL_MAPPING),
        )


def upload_to_s3(task_root: Path) -> None:
    """Upload fixed metadata files to all known S3 prefixes."""
    import boto3

    s3 = boto3.client("s3")

    files_to_upload = [
        "metadata/sample_metadata.csv",
        "metadata/label_mapping.json",
    ]

    for prefix in S3_PREFIXES:
        for rel in files_to_upload:
            local = task_root / rel
            if not local.is_file():
                logger.warning("Local file not found: %s — skipping", local)
                continue
            s3_key = f"{prefix}/{rel}"
            s3.upload_file(str(local), S3_BUCKET, s3_key)
            logger.info("Uploaded %s -> s3://%s/%s", local, S3_BUCKET, s3_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix proximity metadata: correct 0.5m labels",
    )
    parser.add_argument(
        "--task_root",
        type=str,
        required=True,
        help="Path to proximity task directory (contains metadata/ and splits/)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help=f"Substring in file_path that identifies 0.5m samples (default: {DEFAULT_PATTERN!r})",
    )
    parser.add_argument(
        "--correct_label",
        type=str,
        default="0.5",
        help="The correct label value for matched samples (default: '0.5')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report changes without writing any files",
    )
    parser.add_argument(
        "--upload_s3",
        action="store_true",
        help="Upload fixed files to S3 after writing locally",
    )
    args = parser.parse_args()

    task_root = Path(args.task_root)
    csv_path = task_root / "metadata" / "sample_metadata.csv"

    if not csv_path.is_file():
        logger.error("Metadata CSV not found: %s", csv_path)
        raise SystemExit(1)

    fix_metadata_csv(
        csv_path=csv_path,
        pattern=args.pattern,
        correct_label=args.correct_label,
        dry_run=args.dry_run,
    )

    fix_label_mapping(
        meta_dir=task_root / "metadata",
        dry_run=args.dry_run,
    )

    if args.upload_s3 and not args.dry_run:
        upload_to_s3(task_root)
    elif args.upload_s3:
        logger.info("[DRY RUN] Would upload to S3")

    logger.info("\n=== Done ===")
    if args.dry_run:
        logger.info("This was a dry run. Re-run without --dry_run to apply changes.")
    else:
        logger.info("Metadata fixed. Next steps:")
        logger.info("  1. Regenerate pair registry:")
        logger.info(
            "     python scripts/build_dataset_from_s3.py --datasets csibench --tasks proximity --skip_copy"
        )
        logger.info("  2. Retrain proximity models:")
        logger.info("     bash scripts/retrain_proximity_fixed.sh")


if __name__ == "__main__":
    main()
