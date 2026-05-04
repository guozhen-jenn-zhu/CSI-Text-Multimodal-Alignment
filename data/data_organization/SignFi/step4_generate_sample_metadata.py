"""
Generate sample_metadata.csv for SignFi sub_Human data, in the same format as XRF55.

Columns: id, file_path, subject, activity, environment, device, label
- id: same as the .h5 filename (without extension), e.g. csid_dataset_home_276_1_277
- file_path: relative to SignFi root, e.g. ./sub_Human/dataset_home_276/sign001/csid_dataset_home_276_1_277.h5
- subject: Human
- activity: sign label (from sign folder)
- environment: dataset name (e.g. dataset_home_276)
- device: WiFi
- label: from the sign folder name (sign001 -> 1, sign005 -> 5, sign276 -> 276)
"""

import csv
import os

# Paths
SIGNFI_ROOT = r"F:\SignFi\SignFi"
SUB_HUMAN_DIR = "sub_Human"
OUTPUT_CSV = os.path.join(SIGNFI_ROOT, "metadata", "sample_metadata.csv")

COLUMNS = ["id", "file_path", "subject", "activity", "environment", "device", "label"]


def _label_from_sign_folder(sub_name):
    """From sign001, sign005, sign276 return 1, 5, 276. Return None if not signNNN."""
    if not sub_name.startswith("sign"):
        return None
    try:
        return int(sub_name[4:])
    except ValueError:
        return None


def collect_rows():
    sub_human = os.path.join(SIGNFI_ROOT, SUB_HUMAN_DIR)
    if not os.path.isdir(sub_human):
        raise FileNotFoundError(f"sub_Human not found: {sub_human}")

    rows = []
    for dataset_name in sorted(os.listdir(sub_human)):
        dataset_path = os.path.join(sub_human, dataset_name)
        if not os.path.isdir(dataset_path) or dataset_name.startswith("sign"):
            continue
        for sub_name in sorted(os.listdir(dataset_path)):
            sub_path = os.path.join(dataset_path, sub_name)
            if not os.path.isdir(sub_path) or not sub_name.startswith("sign"):
                continue
            label = _label_from_sign_folder(sub_name)
            if label is None:
                continue
            for fname in sorted(os.listdir(sub_path)):
                if not fname.endswith(".h5"):
                    continue
                rel_dir = os.path.join(SUB_HUMAN_DIR, dataset_name, sub_name)
                rel_path = os.path.join(rel_dir, fname).replace("\\", "/")
                file_path = "./" + rel_path
                sample_id = os.path.splitext(fname)[0]  # filename without .h5
                row = {
                    "id": sample_id,
                    "file_path": file_path,
                    "subject": "Human",
                    "activity": label,
                    "environment": dataset_name,
                    "device": "WiFi",
                    "label": label,
                }
                rows.append(row)
    return rows


def main():
    rows = collect_rows()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Written {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
