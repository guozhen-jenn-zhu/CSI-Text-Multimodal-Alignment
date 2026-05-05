"""
Generate train_id.json, val_id.json, test_id.json under <signfi_root>/splits/.

Format matches XRF55 (each file is a JSON array of sample IDs as strings).

Split strategy per label:
- 70% of samples -> train
- 2 samples -> val
- Rest -> test

Uses sample_metadata.csv (id, label) and a fixed random seed for reproducibility.

The SignFi root defaults to ``<repo_root>/data/Dataset/signfi/sign_recognition``;
override with ``--root`` or ``$SIGNFI_ROOT`` (see ``_paths.py``).
"""

import argparse
import csv
import json
import os
import random

from _paths import metadata_csv_path, signfi_root, splits_dir

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_COUNT_PER_LABEL = 2


def load_ids_by_label(csv_path):
    """Load sample_metadata.csv and return {label: [id, id, ...]}."""
    by_label = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row.get("id", "").strip()
            try:
                lab = int(row.get("label", 0))
            except (ValueError, TypeError):
                continue
            by_label.setdefault(lab, []).append(sid)
    return by_label


def split_ids(by_label, seed, train_ratio, val_count):
    """
    Per label: 70% -> train, 2 -> val, rest -> test.
    Shuffle with fixed seed, then take train_count = int(0.7*N), val_count = min(2, rest), rest = test.
    """
    random.seed(seed)
    train_ids = []
    val_ids = []
    test_ids = []

    for label in sorted(by_label.keys()):
        ids = list(by_label[label])
        random.shuffle(ids)
        n = len(ids)
        n_train = int(train_ratio * n)
        rest = n - n_train
        n_val = min(val_count, rest)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    return train_ids, val_ids, test_ids


def main(root=None):
    print(f"SignFi root: {signfi_root(root)}")
    metadata_csv = metadata_csv_path(root)
    if not metadata_csv.is_file():
        raise SystemExit(f"Metadata not found: {metadata_csv}")

    by_label = load_ids_by_label(metadata_csv)
    if not by_label:
        raise SystemExit("No (id, label) rows found in metadata.")

    train_ids, val_ids, test_ids = split_ids(
        by_label, RANDOM_SEED, TRAIN_RATIO, VAL_COUNT_PER_LABEL
    )

    out_dir = splits_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train_id.json", train_ids), ("val_id.json", val_ids), ("test_id.json", test_ids)]:
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
        print(f"  {name}: {len(ids):,} ids")

    print(f"Written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SignFi train/val/test split JSONs from sample_metadata.csv.")
    parser.add_argument(
        "--root",
        default=None,
        help="SignFi root directory (overrides $SIGNFI_ROOT and the default <repo_root>/data/Dataset/signfi/sign_recognition)",
    )
    args = parser.parse_args()
    main(root=args.root)
