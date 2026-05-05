# SignFi data organization

This folder contains the numbered scripts that turn the upstream **SignFi** `.mat` release into the per-sample H5 layout expected by every config/dataloader in this repository.

The released archive on Hugging Face Hub / Kaggle ships **only the metadata and splits** for SignFi (`metadata/sample_metadata.csv`, `splits/{train,val,test}_id.json`, `pair_registry.json`); you have to fetch the raw signal yourself and run the scripts here to materialize the per-sample `.h5` files.

> CSI-Bench and XRF55 are shipped fully processed and do not need this pipeline. See [`../XRF55/README.md`](../XRF55/README.md) if you ever want to rebuild the XRF55 layout from a fresh upstream pull.

## Default output layout

By default every step writes into

```text
<repo_root>/data/Dataset/signfi/sign_recognition/
├── sub_Human/
│   └── <dataset_name>/
│       └── sign<NNN>/<prefix>_<dataset>_<label>_<sample>.h5
├── metadata/
│   └── sample_metadata.csv
└── splits/
    ├── train_id.json
    ├── val_id.json
    └── test_id.json
```

This is the same path the trainer reads from (`configs/dataset/signfi.yaml`, the `signfi/sign_recognition` `task_root` entries in `configs/experiment/multitask_*.yaml`), so the released `metadata/sample_metadata.csv` and `pair_registry.json` line up with the per-sample `.h5` files the scripts produce.

Override the destination by passing `--root <path>` to any step or by exporting `SIGNFI_ROOT=<path>` once for the whole pipeline (precedence: `--root` > `$SIGNFI_ROOT` > the default above). The shared resolver lives in [`_paths.py`](./_paths.py).

## Prerequisites

1. Download the SignFi raw `.mat` archives from the [SignFi project page](https://github.com/yongsen/SignFi). Typical filenames:
   - `dataset_lab_276_dl.mat`
   - `dataset_lab_276_ul.mat`
   - `dataset_home_276.mat`
2. Place them next to `step1_mat_to_h5_reformat.py` (or pass `--mat <path>` to step 1 to read them from elsewhere).
3. Install the script-only dependencies (already covered by the project `requirements.txt`):
   - `numpy`, `scipy`, `h5py`

## Run the pipeline

```bash
cd data/data_organization/SignFi

python step1_mat_to_h5_reformat.py            # .mat -> per-sample .h5 (one file per sample)
python step2_organize_h5_by_label.py          # group .h5 files into sign001/, sign002/, ... folders
python step3_remove_label_index_from_h5.py    # strip the label_index dataset from each .h5
python step4_generate_sample_metadata.py      # build sample_metadata.csv
python step5_generate_splits.py               # build train/val/test split JSONs
```

Repeat **steps 1–3** for each `.mat` file you want to include (pass `--mat <path>` to step 1). Steps 4 and 5 then sweep the entire `sub_Human/` tree at once.

### Per-step reference

| Step | Script | Default I/O | Useful flags |
|---|---|---|---|
| 1 | `step1_mat_to_h5_reformat.py` | `.mat` → `<signfi_root>/sub_Human/<basename>/<prefix>_<basename>_<label>_<sample>.h5` | `--mat`, `--out-dir`, `--labels-csv`, `--root` |
| 2 | `step2_organize_h5_by_label.py` | Moves files in each `<basename>/` into `sign<NNN>/` subfolders by parsed label | `--root` |
| 3 | `step3_remove_label_index_from_h5.py` | Rewrites every `.h5` under `sub_Human/` without the `label_index` dataset (keeps `amplitude` etc.) | `--root` |
| 4 | `step4_generate_sample_metadata.py` | Walks `sub_Human/` → `<signfi_root>/metadata/sample_metadata.csv` | `--root` |
| 5 | `step5_generate_splits.py` | Reads `sample_metadata.csv` → `<signfi_root>/splits/{train,val,test}_id.json` (70 % train, 2 val, rest test, seed 42) | `--root` |

### Custom output location

```bash
# One-shot: write everything under /scratch/signfi/ instead of the repo default
export SIGNFI_ROOT=/scratch/signfi

python step1_mat_to_h5_reformat.py --mat /data/raw/dataset_home_276.mat
python step2_organize_h5_by_label.py
python step3_remove_label_index_from_h5.py
python step4_generate_sample_metadata.py
python step5_generate_splits.py
```

Or pass `--root /scratch/signfi` to each step individually if you don't want to touch the environment.

## Verifying the result

After step 5 the layout under `<signfi_root>/` should match what `configs/dataset/signfi.yaml` expects. A quick check:

```bash
python -c "import h5py, json, csv; \
p='data/Dataset/signfi/sign_recognition'; \
print('train ids:', len(json.load(open(f\"{p}/splits/train_id.json\")))); \
print('rows:', sum(1 for _ in csv.DictReader(open(f\"{p}/metadata/sample_metadata.csv\"))))"
```

If both numbers are non-zero and the per-sample `.h5` files referenced by `file_path` in the CSV exist, the dataloader and trainer will pick them up without further configuration.
