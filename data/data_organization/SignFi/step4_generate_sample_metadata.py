"""
Generate sample_metadata.csv for SignFi sub_Human data, in the same format as XRF55.

Columns: id, file_path, subject, activity, environment, device, label
- id: same as the .h5 filename (without extension), e.g. csid_dataset_home_276_1_277
- file_path: relative to the SignFi root, e.g. ./sub_Human/dataset_home_276/sign001/csid_dataset_home_276_1_277.h5
- subject: Human
- activity: sign label (from sign folder)
- environment: dataset name (e.g. dataset_home_276)
- device: WiFi
- label: from the sign folder name (sign001 -> 1, sign005 -> 5, sign276 -> 276)

The SignFi root defaults to ``<repo_root>/data/Dataset/signfi/sign_recognition``;
override with ``--root`` or ``$SIGNFI_ROOT`` (see ``_paths.py``).
"""

import argparse
import csv
import os

from _paths import (
    SUB_HUMAN_DIRNAME,
    metadata_csv_path,
    signfi_root,
    sub_human_dir,
)

COLUMNS = ["id", "file_path", "subject", "activity", "environment", "device", "label"]


def _label_from_sign_folder(sub_name):
    """From sign001, sign005, sign276 return 1, 5, 276. Return None if not signNNN."""
    if not sub_name.startswith("sign"):
        return None
    try:
        return int(sub_name[4:])
    except ValueError:
        return None


def collect_rows(root=None):
    sub_human = sub_human_dir(root)
    if not sub_human.is_dir():
        raise FileNotFoundError(f"sub_Human not found: {sub_human}")

    rows = []
    for dataset_name in sorted(os.listdir(sub_human)):
        dataset_path = sub_human / dataset_name
        if not dataset_path.is_dir() or dataset_name.startswith("sign"):
            continue
        for sub_name in sorted(os.listdir(dataset_path)):
            sub_path = dataset_path / sub_name
            if not sub_path.is_dir() or not sub_name.startswith("sign"):
                continue
            label = _label_from_sign_folder(sub_name)
            if label is None:
                continue
            for fname in sorted(os.listdir(sub_path)):
                if not fname.endswith(".h5"):
                    continue
                rel_dir = os.path.join(SUB_HUMAN_DIRNAME, dataset_name, sub_name)
                rel_path = os.path.join(rel_dir, fname).replace("\\", "/")
                file_path = "./" + rel_path
                sample_id = os.path.splitext(fname)[0]
                rows.append({
                    "id": sample_id,
                    "file_path": file_path,
                    "subject": "Human",
                    "activity": label,
                    "environment": dataset_name,
                    "device": "WiFi",
                    "label": label,
                })
    return rows


def main(root=None):
    print(f"SignFi root: {signfi_root(root)}")
    rows = collect_rows(root)
    output_csv = metadata_csv_path(root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Written {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SignFi sample_metadata.csv from the organized sub_Human layout.")
    parser.add_argument(
        "--root",
        default=None,
        help="SignFi root directory (overrides $SIGNFI_ROOT and the default <repo_root>/data/Dataset/signfi/sign_recognition)",
    )
    args = parser.parse_args()
    main(root=args.root)
