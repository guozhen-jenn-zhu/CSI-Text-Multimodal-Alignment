"""
Generate train_id.json, val_id.json, test_id.json in F:\\SignFi\\SignFi\\splits.

Format matches E:\\XRF55\\XRF55\\splits: each file is a JSON array of sample IDs (strings).

Split strategy per label:
- 70% of samples -> train
- 2 samples -> val
- Rest -> test

Uses sample_metadata.csv (id, label) and a fixed random seed for reproducibility.
"""

import csv
import json
import os
import random

SIGNFI_ROOT = r"F:\SignFi\SignFi"
METADATA_CSV = os.path.join(SIGNFI_ROOT, "metadata", "sample_metadata.csv")
SPLITS_DIR = os.path.join(SIGNFI_ROOT, "splits")
TRAIN_JSON = os.path.join(SPLITS_DIR, "train_id.json")
VAL_JSON = os.path.join(SPLITS_DIR, "val_id.json")
TEST_JSON = os.path.join(SPLITS_DIR, "test_id.json")

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
            if lab not in by_label:
                by_label[lab] = []
            by_label[lab].append(sid)
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
        n_test = rest - n_val
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    return train_ids, val_ids, test_ids


def main():
    if not os.path.isfile(METADATA_CSV):
        raise SystemExit(f"Metadata not found: {METADATA_CSV}")

    by_label = load_ids_by_label(METADATA_CSV)
    if not by_label:
        raise SystemExit("No (id, label) rows found in metadata.")

    train_ids, val_ids, test_ids = split_ids(
        by_label, RANDOM_SEED, TRAIN_RATIO, VAL_COUNT_PER_LABEL
    )

    os.makedirs(SPLITS_DIR, exist_ok=True)
    for path, ids in [(TRAIN_JSON, train_ids), (VAL_JSON, val_ids), (TEST_JSON, test_ids)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
        print(f"  {os.path.basename(path)}: {len(ids):,} ids")

    print(f"Written to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
