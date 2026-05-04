"""
Reformat SignFi .mat data to .h5 files.

1. Load CSI from .mat (default: dataset_home_276.mat)
2. Split by 4th dimension: [200, 30, 3, N] -> N samples of [200, 30, 3]
3. Reshape each sample to [200, 90]: cols 0-29 from [:,:,0], 30-59 from [:,:,1], 60-89 from [:,:,2]
4. Take amplitude (abs), transpose to [90, 200], add channel dim -> [90, 200, 1]
5. Save amplitude arrays and labels to .h5
6. Labels: use in-file labels when present, map indices to names via sign_labels.csv
"""

import os
import re
import numpy as np
import scipy.io
import h5py
import csv

# Default input .mat file (hard-coded)
DEFAULT_MAT_FILE = "dataset_lab_276_ul.mat"
LABELS_CSV = "sign_labels.csv"


def _csi_sort_key(name):
    """Sort csi1, csi2, csi3, ... by numeric suffix; others keep string order."""
    m = re.match(r"^(.+?)(\d+)$", name)
    if m is not None:
        return (m.group(1), int(m.group(2)))
    return (name, 0)


def find_all_csi_and_labels(mat_dict):
    """
    Find all CSI arrays (csi1, csi2, csid_lab, csiu, etc.) and the label array from loaded .mat dict.
    Returns: list of (prefix, csi_array) sorted by name.
    - Label: prefer "label_lab" when present (for dataset_lab_276_dl etc.), else first with "label" in name.
    - CSI: when using label_lab, only include "csid_lab" so lab datasets use csid_lab + label_lab.
    """
    csi_list = []
    label_arr = None
    label_key_used = None
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            if v.ndim == 4 and v.shape[0] == 200 and v.shape[1] == 30 and v.shape[2] == 3:
                csi_list.append((k, v))
            if v.ndim == 2 and "label" in k.lower():
                lab = v.flatten() if v.ndim == 2 else v
                if label_key_used is None or k == "label_lab":
                    label_arr = lab
                    label_key_used = k
    if not csi_list:
        raise KeyError("No 4D CSI array with shape (200, 30, 3, N) found in .mat")
    # For lab datasets: when using label_lab, keep only csid_lab and csiu_lab (dl vs ul)
    if label_key_used == "label_lab":
        lab_csi = [(k, v) for k, v in csi_list if k in ("csid_lab", "csiu_lab")]
        if lab_csi:
            csi_list = lab_csi
    csi_list.sort(key=lambda x: _csi_sort_key(x[0]))
    return csi_list, label_arr


def _sanitize_for_filename(s):
    """Replace characters unsuitable for filenames."""
    return s.replace("/", "-").replace("\\", "-").replace(" ", "_").strip()


def load_label_names(csv_path):
    """Load class names from sign_labels.csv (one label per line, index = row index)."""
    if not os.path.isfile(csv_path):
        return None
    names = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row:
                names.append(row[0].strip())
    return names


def sample_to_amplitude_channel(sample):
    """
    Convert one CSI sample [200, 30, 3] to amplitude image [90, 200, 1].
    - Reshape to [200, 90]: cols 0-29 = [:,:,0], 30-59 = [:,:,1], 60-89 = [:,:,2]
    - Take amplitude (abs)
    - Transpose to [90, 200], add channel -> [90, 200, 1]
    """
    # sample: (200, 30, 3)
    # Stack 2nd and 3rd dim: (200, 90)
    s = np.concatenate([sample[:, :, i] for i in range(sample.shape[2])], axis=1)
    # Amplitude: magnitude for complex CSI, otherwise as-is
    if np.iscomplexobj(s):
        s = np.abs(s)
    s = np.asarray(s, dtype=np.float64)
    # Transpose -> (90, 200), then (90, 200, 1)
    s = s.T[:, :, np.newaxis]
    return s.astype(np.float32)


def reformat_mat_to_h5(
    mat_path=None,
    out_dir=None,
    labels_csv_path=None,
):
    if mat_path is None:
        mat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_MAT_FILE)
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f".mat file not found: {mat_path}")

    base_dir = os.path.dirname(os.path.abspath(mat_path))
    original_basename = os.path.splitext(os.path.basename(mat_path))[0]
    # Default: create a folder named after the .mat (e.g. dataset_home_276/)
    if out_dir is None:
        out_dir = os.path.join(base_dir, original_basename)
    os.makedirs(out_dir, exist_ok=True)
    if labels_csv_path is None:
        labels_csv_path = os.path.join(base_dir, LABELS_CSV)

    print(f"Loading {mat_path} ...")
    mat = scipy.io.loadmat(mat_path)
    csi_list, label_arr = find_all_csi_and_labels(mat)

    n_csi = len(csi_list)
    n_per_csi = csi_list[0][1].shape[3]
    print(f"Found {n_csi} CSI array(s): {[p for p, _ in csi_list]}, {n_per_csi} samples each")

    # Labels: 1-based. Either shared (one label per sample, same for all CSI) or partitioned:
    #   partitioned: csi1 -> labels[0:n], csi2 -> labels[n:2n], csi3 -> labels[2n:3n], ...
    if label_arr is not None:
        label_arr = np.asarray(label_arr).flatten().astype(np.int32)
        total_labels = label_arr.size
        if total_labels == n_csi * n_per_csi:
            # Partitioned: csi_j uses labels[j*n_per_csi : (j+1)*n_per_csi]
            label_slices = [label_arr[j * n_per_csi : (j + 1) * n_per_csi] for j in range(n_csi)]
            print(f"Labels partitioned: {n_csi} blocks of {n_per_csi} (indices 1-{n_per_csi}, {n_per_csi+1}-{2*n_per_csi}, ...)")
        elif total_labels == n_per_csi:
            # Shared: same labels for all CSI arrays
            label_slices = [label_arr] * n_csi
            print("Labels shared across all CSI arrays")
        else:
            raise ValueError(f"Label length {total_labels} neither {n_per_csi} (shared) nor {n_csi * n_per_csi} (partitioned)")
    else:
        label_slices = [(np.arange(n_per_csi, dtype=np.int32) % 276) + 1] * n_csi  # fallback 1-based

    label_names = load_label_names(labels_csv_path)
    if label_names is None:
        label_names = [str(i) for i in range(276)]

    # Save one .h5 per sample per CSI array: {prefix}_{original_basename}_{label_index}_{sample_idx}.h5
    written = []
    for j, (prefix, csi) in enumerate(csi_list):
        labels_j = label_slices[j]
        for i in range(n_per_csi):
            sample = csi[:, :, :, i]  # (200, 30, 3)
            amp = sample_to_amplitude_channel(sample)  # (90, 200, 1)
            lab_idx = int(labels_j[i])
            h5_name = f"{prefix}_{original_basename}_{lab_idx}_{i + 1}.h5"
            h5_path = os.path.join(out_dir, h5_name)
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("amplitude", data=amp, compression="gzip")
                f.create_dataset("label_index", data=np.int32(lab_idx))
            written.append(h5_path)
    print(f"Saved {len(written)} files to {out_dir} (e.g. {os.path.basename(written[0])})")
    return written


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reformat SignFi .mat to .h5 (one file per sample: original_filename_label_index.h5)")
    parser.add_argument("--mat", default=None, help=f"Input .mat file (default: {DEFAULT_MAT_FILE})")
    parser.add_argument("--out-dir", default=None, help="Output directory for .h5 files (default: folder named after .mat, e.g. dataset_home_276/)")
    parser.add_argument("--labels-csv", default=None, help=f"Label names CSV (default: {LABELS_CSV})")
    args = parser.parse_args()
    reformat_mat_to_h5(mat_path=args.mat, out_dir=args.out_dir, labels_csv_path=args.labels_csv)
