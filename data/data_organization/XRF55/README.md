# XRF55 data organization

This folder contains the numbered scripts that turn the upstream **XRF55** release into the per-sample H5 layout expected by every config/dataloader in this repository.

> **Most users do not need to run these scripts.** The released [Multimodal_CSI_Text dataset](https://www.kaggle.com/datasets/b43cfe263881c7804a6b840820452782b4a0db647abdc94adcc982fdc5a72aab) on Kaggle ships the XRF55 dataset already fully processed (per-sample `.h5` + `metadata/` + `splits/` + `pair_registry.json`) — just extract it under `data/Dataset/xrf55/` and you're done. The pipeline below is kept for transparency and so the released layout can be rebuilt from a fresh upstream pull.
>
> See [`../SignFi/README.md`](../SignFi/README.md) for the SignFi pipeline, which **does** need to be run because we are not allowed to redistribute the SignFi raw signal.

## Default output layout

By default every step writes into

```text
<repo_root>/data/Dataset/xrf55/action_recognition/
├── raw/                                      # upstream archive (you provide)
│   └── Scene<N>/WiFi/<user>_<act>_<repeat>.npy
├── organized/                                # step1 (.npy) + step2 (.h5)
│   └── user_<XX>/act_<XX>/env_<N>/...        # step3 renames to userXX_actXX_repeatXX_envN.<ext>
├── organized_npy/                            # step4 archives the .npy half here
│   └── user_<XX>/act_<XX>/env_<N>/<file>.npy
├── user_<XX>/act_<XX>/env_<N>/<file>.h5      # step5 (final canonical .h5 layout)
├── metadata/
│   └── sample_metadata.csv                   # step6
└── splits/                                   # step7 + step8
    ├── train_id.json
    ├── val_id.json
    └── test_id.json
```

This is the same path the trainer reads from (`configs/dataset/xrf55.yaml`, the `xrf55/action_recognition` `task_root` entries in `configs/experiment/multitask_*.yaml`), so the `.h5`, metadata and splits the scripts produce are immediately consumable.

Override the destination by passing `--root <path>` to any step or by exporting `XRF55_ROOT=<path>` once for the whole pipeline (precedence: `--root` > `$XRF55_ROOT` > the default above). The shared resolver lives in [`_paths.py`](./_paths.py).

## Prerequisites

1. Download the XRF55 raw archive from the [XRF55 project page](https://github.com/aiotgroup/XRF55-repo) and unpack it so that the upstream `Scene<N>/WiFi/*.npy` files live under `<xrf55_root>/raw/` (or any directory you'll pass via `--source` to step 1).
2. Install the script-only dependencies (already covered by the project `requirements.txt`):
   - `numpy`, `h5py`, `matplotlib` (only step 5 uses matplotlib for the optional first-file preview)

## Run the pipeline

```bash
cd data/data_organization/XRF55

python step1_organize_dataset.py              # raw Scene<N>/WiFi -> organized/user_XX/act_XX/env_N/*.npy
python step2_convert_npy_to_h5.py             # write a .h5 next to each .npy (same shape/dtype)
python step3_optional_rename_files_with_env.py  # rename to userXX_actXX_repeatXX_envN.{h5,npy}
python step4_separate_npy_h5.py               # move the .npy half into organized_npy/, leave .h5 in organized/
python step5_reshape_h5.py                    # add channel axis + reorder 270-dim into the canonical layout
python step6_generate_metadata.py             # build sample_metadata.csv
python step7_generate_splits.py               # build train/val/test split JSONs (repeat-based protocol)
python step8_shuffle_train_id.py              # shuffle train_id.json so the loader doesn't see deterministic order
```

Step 3 is technically optional — it just embeds the `env_<N>` tag in the filename so step 6 can parse user / activity / env / repeat directly from the filename instead of having to recover the env from the folder hierarchy. Skipping it is supported (step 6 falls back to the old `<user>_<act>_<repeat>.h5` filename + folder hint).

### Per-step reference

| Step | Script | Default I/O | Useful flags |
|---|---|---|---|
| 1 | `step1_organize_dataset.py` | `<xrf55_root>/raw/Scene<N>/WiFi/*.npy` → `<xrf55_root>/organized/user_<XX>/act_<XX>/env_<N>/*.npy` | `--source`, `--target`, `--root` |
| 2 | `step2_convert_npy_to_h5.py` | Each `.npy` under `<xrf55_root>/organized/` → sibling `.h5` (falls back to `raw/` if `organized/` is empty) | `--source`, `--target`, `--test`, `--root` |
| 3 | `step3_optional_rename_files_with_env.py` | Renames in place: `01_02_03.{h5,npy}` → `user01_act02_repeat03_env<N>.{h5,npy}` | `--source`, `--dry-run`, `--root` |
| 4 | `step4_separate_npy_h5.py` | Moves the `.npy` half from `organized/` into `organized_npy/`, leaves the `.h5` files in place | `--source`, `--target`, `--root` |
| 5 | `step5_reshape_h5.py` | Reads `.h5` from `organized/`, writes the reshaped `(270, 1000, 1)` files into `<xrf55_root>/` (mirrored relative paths) | `--source`, `--target`, `--in-place`, `--no-plot`, `--test`, `--root` |
| 6 | `step6_generate_metadata.py` | Walks `<xrf55_root>/` `.h5` tree → `<xrf55_root>/metadata/sample_metadata.csv` | `--source`, `--output`, `--root` |
| 7 | `step7_generate_splits.py` | Reads `sample_metadata.csv` → `<xrf55_root>/splits/{train,val,test}_id.json` using the official repeat-based protocol (train: 1–12, val: 13–14, test: 15–20) | `--metadata`, `--output`, `--root` |
| 8 | `step8_shuffle_train_id.py` | Shuffles `train_id.json` in place (default seed 42) | `--train-id`, `--output`, `--seed`, `--root` |

### Custom output location

```bash
# Stage the entire pipeline under /scratch/xrf55/ instead of the repo default
export XRF55_ROOT=/scratch/xrf55

# Optional: keep the raw archive somewhere else and point step 1 at it
python step1_organize_dataset.py --source /data/raw/XRF55
python step2_convert_npy_to_h5.py
python step3_optional_rename_files_with_env.py
python step4_separate_npy_h5.py
python step5_reshape_h5.py --no-plot
python step6_generate_metadata.py
python step7_generate_splits.py
python step8_shuffle_train_id.py
```

Or pass `--root /scratch/xrf55` to each step individually if you don't want to touch the environment.

## Verifying the result

After step 8 the layout under `<xrf55_root>/` should match what `configs/dataset/xrf55.yaml` expects. A quick check:

```bash
python -c "import json, csv; \
p='data/Dataset/xrf55/action_recognition'; \
print('train ids:', len(json.load(open(f\"{p}/splits/train_id.json\")))); \
print('rows:', sum(1 for _ in csv.DictReader(open(f\"{p}/metadata/sample_metadata.csv\"))))"
```

If both numbers are non-zero and the per-sample `.h5` files referenced by `file_path` in the CSV exist, the dataloader and trainer will pick them up without further configuration.
