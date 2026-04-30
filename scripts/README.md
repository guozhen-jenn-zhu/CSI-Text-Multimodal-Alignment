# `scripts/` — public entry points

This directory groups all command-line entry points for the CSI-Text
Multimodal Alignment benchmark. Every script here is intended for end users
(reproducing the paper, running the benchmark on your own data, training a
single model, etc.). For an end-to-end walkthrough see the top-level
[`README.md`](../README.md).

The layout is:

```text
scripts/
├── README.md                              # this file
├── launch_ddp.py                          # ── core utilities (flat, see §1)
├── run_benchmark.py
├── run_ablations.py
├── run_evaluate.py
├── generate_all_texts.py
├── generate_text_embeddings.py
├── preprocess_to_consolidated.py
├── pipelines/                             # ── §2: end-to-end orchestrators
│   ├── train_local.sh
│   ├── train_sagemaker.sh
│   ├── run_local_experiments.sh
│   └── run_neurips_experiments.sh
├── baselines/                             # ── §3: supervised baselines
│   ├── train_supervised.py
│   ├── train_supervised_baselines.sh
│   └── train_supervised_fall_motion.sh
└── paper_experiments/                     # ── §4: paper-reproducibility extras
    ├── run_held_out_classes.py
    ├── run_held_out_signfi.sh
    ├── run_open_vocab_eval.py
    ├── run_multi_seed.sh
    ├── generate_template_count_registries.py
    ├── run_template_count_ablation.sh
    └── aggregate_neurips_results.py
```

---

## 1. Core utilities (top level)

The handful of scripts the main `README.md` documents in detail. Use these
to build the dataset, train a single model, evaluate a checkpoint, or sweep
the full benchmark.

| Script | Purpose |
|--------|---------|
| [`generate_all_texts.py`](generate_all_texts.py) | Generate many-to-many CSI-text pair registries from metadata CSVs for every dataset/task. |
| [`generate_text_embeddings.py`](generate_text_embeddings.py) | Pre-compute text embeddings (HDF5, keyed by `text_id`) for a given pair registry and text encoder. |
| [`preprocess_to_consolidated.py`](preprocess_to_consolidated.py) | Consolidate per-sample H5 files into a single `train.h5` / `val.h5` / `test.h5` for fast training I/O. |
| [`launch_ddp.py`](launch_ddp.py) | Multi-GPU DDP launcher — auto-detects GPU count and invokes `torchrun` to run `training.train`. |
| [`run_benchmark.py`](run_benchmark.py) | Sweep all combinations of methods × encoders × tasks (the "full benchmark"). |
| [`run_ablations.py`](run_ablations.py) | Run any of the predefined ablation studies (`text_granularity`, `csi_encoder`, `embedding_dim`, …). |
| [`run_evaluate.py`](run_evaluate.py) | Run any subset of the five evaluation protocols (`zero_shot`, `ambient_qa`, `cross_task`, `cross_dataset`, `visualization`) against a trained checkpoint. |

---

## 2. `pipelines/` — end-to-end orchestrators

Shells that chain together text generation → embedding pre-compute →
training → evaluation. Use these to reproduce the full paper or to run on
a single machine.

| Script | Where to run | What it does |
|--------|-------------|--------------|
| [`pipelines/train_local.sh`](pipelines/train_local.sh) | Local Linux/macOS/Windows-Git-Bash | Single-task training pipeline — auto-detects GPUs, falls back to CPU. Driven by env vars (`DATASET`, `TASK`, `METHOD`, `ENCODER`, `BATCH_SIZE`, …). |
| [`pipelines/train_sagemaker.sh`](pipelines/train_sagemaker.sh) | AWS SageMaker g5/p4d | SageMaker variant of `train_local.sh`: handles multitask H5 symlinks, dependency install, and torchrun. |
| [`pipelines/run_local_experiments.sh`](pipelines/run_local_experiments.sh) | Local | Orchestrates the full priority-by-priority paper roadmap (encoder selection → main benchmark → cross-dataset/task transfer → ablations → ambient QA → analysis) on a single machine. |
| [`pipelines/run_neurips_experiments.sh`](pipelines/run_neurips_experiments.sh) | AWS SageMaker | SageMaker equivalent of `run_local_experiments.sh` with S3 sync of checkpoints and results. |

Both orchestrators support `PRIORITY=…`, `DRY_RUN=1`, and `SKIP_EXISTING=0/1`
flags — see the comment block at the top of each file.

---

## 3. `baselines/` — supervised baselines

Reference implementations of the supervised baselines reported in the
paper. These are *not* part of the alignment benchmark; they exist so
readers can reproduce the supervised numbers in Table 1.

| Script | Purpose |
|--------|---------|
| [`baselines/train_supervised.py`](baselines/train_supervised.py) | Train a single ResNet-18 1D supervised classifier on one task. |
| [`baselines/train_supervised_baselines.sh`](baselines/train_supervised_baselines.sh) | Train supervised baselines for all dataset/task combinations. |
| [`baselines/train_supervised_fall_motion.sh`](baselines/train_supervised_fall_motion.sh) | Convenience wrapper for `train_supervised_baselines.sh` restricted to CSI-Bench Fall-Detection and Motion-Source. |

---

## 4. `paper_experiments/` — paper-reproducibility extras

Scripts that drive specific paper figures/tables (extension experiments
and ablations). They are independent of the main benchmark, so feel free
to ignore them if you only want to use the dataset.

| Script | Paper experiment |
|--------|------------------|
| [`paper_experiments/run_held_out_classes.py`](paper_experiments/run_held_out_classes.py) | Zero-shot generalization to held-out SignFi classes (200 seen / 76 unseen). |
| [`paper_experiments/run_held_out_signfi.sh`](paper_experiments/run_held_out_signfi.sh) | Shell wrapper around `run_held_out_classes.py` (split → train → eval). |
| [`paper_experiments/run_open_vocab_eval.py`](paper_experiments/run_open_vocab_eval.py) | Open-vocabulary evaluation: zero-shot HAR using novel text prompts unseen during training. |
| [`paper_experiments/run_multi_seed.sh`](paper_experiments/run_multi_seed.sh) | Train every method × task across multiple random seeds for the error-bar / variance numbers. |
| [`paper_experiments/generate_template_count_registries.py`](paper_experiments/generate_template_count_registries.py) | Generate Level-1-only registries with 1/3/5/10 templates per class for the template-count ablation. |
| [`paper_experiments/run_template_count_ablation.sh`](paper_experiments/run_template_count_ablation.sh) | Train + evaluate the four template-count variants produced by the script above. |
| [`paper_experiments/aggregate_neurips_results.py`](paper_experiments/aggregate_neurips_results.py) | Aggregate per-experiment `metrics.json` files into the LaTeX tables and figures used in the paper. |

---

## Quickstart cheatsheet

```bash
# 1. Build the dataset (text registries + cached embeddings)
python scripts/generate_all_texts.py
python scripts/generate_text_embeddings.py \
    --registry_path data/generated_texts/csibench/har/pair_registry.json \
    --output_path   data/generated_texts/csibench/har/text_embeddings.h5

# 2. Train one model on one task
python scripts/launch_ddp.py --config configs/experiment/default.yaml \
    --override dataset=csibench task=har method=clip_infonce csi_encoder.name=resnet18_1d

# 3. Evaluate that checkpoint
python scripts/run_evaluate.py \
    --checkpoint runs/checkpoints/csibench-har-clip_infonce/best.pt \
    --config     configs/experiment/default.yaml \
    --output_dir runs/results/csibench/har/clip_infonce_resnet18_1d

# 4. Or reproduce the entire paper (local, single machine)
bash scripts/pipelines/run_local_experiments.sh
```

---

## Data / model paths

All scripts respect three environment variables for path layout:

| Variable | Default | What lives there |
|----------|---------|------------------|
| `REPO_ROOT` | git repo root | This codebase |
| `DATA_ROOT` | `${REPO_ROOT}/data/Dataset` (local) / `/home/sagemaker-user/Multimodal_CSI_Text/Dataset` (SageMaker) | Pair registries, metadata, splits, raw H5 |
| `MODEL_ROOT` | `${REPO_ROOT}/runs` (local) / `/home/sagemaker-user/Multimodal_CSI_Text/Model` (SageMaker) | Checkpoints, results, logs |

Override on the command line, e.g.:

```bash
DATA_ROOT=/mnt/csi/Dataset MODEL_ROOT=/mnt/csi/Model bash scripts/pipelines/train_local.sh
```
