# CSI-Text Multimodal Alignment

**The first WiFi CSI-text benchmark for many-to-many multimodal alignment and language-grounded ambient perception.**

---

## Motivating vision

Ambient intelligence aims to understand everyday environments in a way that is both **sensor-grounded** and **human-interpretable**. WiFi channel state information (CSI) is widely available and privacy-preserving compared to video, yet it has remained difficult to connect to open-ended language. This project releases **CSI-Text**, a unified CSI-text resource and codebase that aligns normalized CSI traces with natural-language descriptions so models can be **trained**, **compared**, and **evaluated** under shared protocols -- from zero-shot sensing to structured scene understanding.

---

## Key contributions

| Contribution | Description |
|--------------|-------------|
| **CSI-Text dataset** | Many-to-many pairings between CSI segments and text, aggregated from **CSI-Bench**, **XRF55**, and **SignFi**, with hierarchical text levels (semantic captions and composite captions). |
| **Benchmark suite** | Six alignment methods spanning contrastive, anchor-based, redundancy-reduction, and masked cross-modal objectives, with shared encoders and projection heads. |
| **Evaluation protocols** | Zero-shot classification, ambient QA, cross-task and cross-dataset transfer, and embedding analysis (visualization plus linear probing). |

Machine-readable dataset documentation: [`croissant.json`](croissant.json) (Croissant / Schema.org). Human-readable dataset documentation: [`datasheet.md`](datasheet.md) (Datasheet format).

---

## Dataset overview

The benchmark harmonizes **three public CSI corpora** into a common tensor format (`[C, K, T]` amplitude after preprocessing) and associates each segment with **one or more** text strings.

| Source dataset | Example tasks | Indicative class scale | Data format |
|----------------|---------------|------------------------|-------------|
| **CSI-Bench** | Fall detection, breath detection, motion source, localization, HAR, user ID, proximity | Binary to multi-class per task (7 task groups) | Per-sample H5, key `CSI_amps` |
| **XRF55** | Action recognition | **55** action classes | Per-sample H5, dynamic key |
| **SignFi** | Sign recognition | **276** sign glosses | Per-sample H5, key `amplitude` |

### Download

The processed CSI-Text dataset will be published on the Hugging Face Hub and mirrored on Kaggle. URLs will be finalized at publication time:

- **Hugging Face Hub** (primary): [link]
- **Kaggle** (mirror): [link]

Download the archive and extract it under `data/Dataset/` at the repository root (this is the default `DATA_ROOT` used by every config and pipeline script):

```bash
# From the repo root
mkdir -p data/Dataset

# Then download the release archive from one of the links above
# and extract its contents into data/Dataset/
```

To use a different location, set `DATA_ROOT` when invoking the pipeline (see [Quick start](#quick-start-run-the-full-experiment-pipeline)) or override `dataset_config.data_root=<path>` on the training command line.

#### Data layout and organization

The release archive bundles different artifacts per source dataset, depending on what we are allowed to redistribute:

| Source dataset | What the release ships | Action required |
|---|---|---|
| **CSI-Bench** | Full processed dataset: per-sample H5 + `metadata/` + `splits/` + `pair_registry.json` for every task | None — extract under `data/Dataset/csibench/` |
| **XRF55** | Full processed dataset: per-sample H5 + `metadata/` + `splits/` + `pair_registry.json` | None — extract under `data/Dataset/xrf55/` |
| **SignFi** | Metadata only: `metadata/sample_metadata.csv`, `splits/{train,val,test}_id.json`, `pair_registry.json` | Obtain the raw SignFi `.mat` files from the upstream SignFi release and run the reorganization pipeline in [`data/data_organization/SignFi/`](data/data_organization/SignFi/README.md) — see the folder README for step-by-step instructions |

The [`data/data_organization/`](data/data_organization/) folder contains the numbered scripts that produced the released layout for **SignFi** and **XRF55**, each with its own README:

- [`data/data_organization/SignFi/README.md`](data/data_organization/SignFi/README.md) — required for SignFi (raw `.mat` → per-sample `.h5` + metadata + splits).
- [`data/data_organization/XRF55/README.md`](data/data_organization/XRF55/README.md) — optional; XRF55 is shipped pre-organized and only needs these scripts when rebuilding from a fresh upstream pull.

### On-disk layout

After download, `data/Dataset/` should look like this:

```text
data/Dataset/
├── csibench/
│   ├── fall_detection/
│   │   ├── metadata/sample_metadata.csv         # columns: id, file_path, label, ...
│   │   ├── splits/{train,val,test}_id.json       # flat lists of sample IDs
│   │   ├── pair_registry.json                    # many-to-many CSI-text pairs
│   │   └── text_embeddings_<encoder>.h5          # generated on demand by the pipeline
│   ├── motion_source/
│   └── multitask/
│       ├── har/
│       ├── proximity/
│       └── user_id/
├── xrf55/
│   └── action_recognition/
└── signfi/
    └── sign_recognition/
```

Each task directory contains:

| File / folder | Description |
|---|---|
| `metadata/sample_metadata.csv` | Sample-level metadata: `id`, `file_path` (path to per-sample H5), `label`, plus task-specific attribute columns |
| `splits/{train,val,test}_id.json` | Flat JSON arrays of sample IDs assigned to each split |
| `pair_registry.json` | Many-to-many mapping between CSI sample IDs and natural-language captions (pre-bundled with the dataset; auto-regenerated by the orchestrator if missing) |
| `text_embeddings_<encoder>.h5` | Pre-computed text embeddings indexed by caption ID (auto-generated by the pipeline; e.g. `text_embeddings_clip_vit_b32.h5`, `text_embeddings_sentence_transformer.h5`) |

Each per-sample H5 file referenced by `file_path` in the CSV holds a single CSI amplitude tensor of shape `(F, T, 1)` where `F` is the number of subcarriers and `T` is the number of time steps. The dataset key inside the H5 file varies per source dataset (see the table above).

---

## Installation

### Requirements

- **Python 3.10 – 3.12**
- **PyTorch 2.0+** matching your CUDA version (or CPU-only)
- A POSIX shell (`bash`) for the pipeline scripts. On Windows use **Git Bash** or **WSL2**.

### Setup

Install PyTorch for your hardware, then install the rest:

```bash
# 1. Install PyTorch matching your CUDA version.
#    See https://pytorch.org/get-started/locally/ for the right command. Examples:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
pip install torch torchvision                                                      # CPU only

# 2. Install the rest of the dependencies
pip install -r requirements.txt
```

Key dependencies: `torch`, `h5py`, `pandas`, `scipy`, `transformers`, `sentence-transformers`, `open-clip-torch`, `omegaconf`, `boto3`, `wandb`.

### Verify install

```bash
python -c "import torch, transformers, h5py, omegaconf; \
    print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
```

---

## End-to-end workflow

The released dataset already ships with `pair_registry.json` files for every task, so the typical user only needs to download the dataset and run the orchestrator:

```text
Download dataset  -->  Train  -->  Evaluate
```

There are two ways to drive it:

- **Quick start** below — one command runs the full experiment matrix end-to-end.
- **Advanced usage** further below — per-script entry points for running a targeted benchmark sweep, ablation, or single-checkpoint evaluation outside the orchestrator.

---

## Quick start: run the full experiment pipeline

The entire training + evaluation suite can be driven by a single orchestrator that mirrors the SageMaker pipeline locally:

```bash
bash scripts/pipelines/run_local_experiments.sh
```

It auto-detects GPUs (falls back to CPU), runs **8 priority levels** end-to-end, and is **resumable** — every priority short-circuits if the expected `best.pt` / `metrics.json` already exists, so re-runs fast-forward through completed work and only resume incomplete stages.

### Priority levels

| Priority | What it does | Trains? |
|----------|--------------|---------|
| **0** | **Encoder preliminary** -- trains `clip_infonce` × `{resnet18_1d, patchtst, vit_1d}` on `csibench/har` and picks the best encoder for everything downstream | 3 runs |
| **1A** | **Main benchmark** -- 6 alignment methods × best encoder × 7 (dataset, task) pairs (5 CSI-Bench tasks + XRF55 + SignFi) | 42 runs |
| **1B** | **Multitask training** -- one shared model over the 5 CSI-Bench tasks, plus one over CSI-Bench + XRF55 + SignFi combined | 2 runs |
| **2A** | **Cross-dataset transfer** (eval-only) -- score a model trained on dataset X against dataset Y | 0 |
| **2B** | **Cross-task transfer** (eval-only) -- score single-task and multitask models on held-out tasks within CSI-Bench | 0 |
| **3** | **Text generation ablations** -- 3A: text granularity (`level1` vs `level1_level2`); 3B: many-to-many ratio (1 / 3 / 5 / 10 paraphrases × `clip_infonce` / `siglip`) | 10 runs |
| **4** | **Ambient scene QA** (eval-only) -- multitask model answers compositional questions ("which user is doing what at what proximity?") | 0 |
| **5** | **Architecture ablations** -- 5B: text encoder swap (CLIP vs sentence-transformers); 5C: projection dim ∈ {128, 256, 512} | 4 runs |
| **6** | **AM-FM pretrained encoder** -- 6A: frozen (projection-only); 6B: fine-tuned with `encoder_lr_scale=0.1`. Requires `omegaconf` to deserialize the checkpoint | 2 runs |
| **7** | **Analysis & visualization** (eval-only) -- final paper figures, t-SNE/UMAP embeddings, linear-probe accuracy; aggregates everything into `runs/results/_paper_tables/` | 0 |

`run_local_experiments.sh` is the local equivalent of `scripts/pipelines/sagemaker_scripts/run_sagemaker_experiments.sh` (which has the same priority structure but launches every job as a SageMaker training job).

### Common usage patterns

```bash
# Run everything from scratch with default hyperparameters
bash scripts/pipelines/run_local_experiments.sh

# Run only specific priorities (comma-separated, no spaces)
PRIORITY=0,1   bash scripts/pipelines/run_local_experiments.sh
PRIORITY=3,5   bash scripts/pipelines/run_local_experiments.sh   # ablations only
PRIORITY=7     bash scripts/pipelines/run_local_experiments.sh   # final figures only

# Dry run: print every command without executing
DRY_RUN=1 bash scripts/pipelines/run_local_experiments.sh

# Force re-run a priority by ignoring existing artifacts
SKIP_EXISTING=0 PRIORITY=1 bash scripts/pipelines/run_local_experiments.sh

# Quick smoke test: 2 epochs, smaller batch
MAX_EPOCHS=2 BATCH_SIZE=32 bash scripts/pipelines/run_local_experiments.sh

# Skip Priority 0 by pinning the encoder explicitly
BEST_ENCODER=resnet18_1d PRIORITY=1,2 bash scripts/pipelines/run_local_experiments.sh

# Point at custom data / output locations
DATA_ROOT=/d/Multimodal_CSI_Text/Dataset \
MODEL_ROOT=/d/Multimodal_CSI_Text/runs \
    bash scripts/pipelines/run_local_experiments.sh

# Use sentence-transformers as the text encoder for the entire pipeline
TEXT_ENCODER=sentence_transformer bash scripts/pipelines/run_local_experiments.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIORITY` | `all` | Which priorities to run (`all` or comma-separated, e.g. `0,1,3`) |
| `DATA_ROOT` | `<repo>/data/Dataset` | Root containing `csibench/`, `xrf55/`, `signfi/` task directories |
| `MODEL_ROOT` | `<repo>/runs` | Where checkpoints, results, and logs are written |
| `DRY_RUN` | `0` | `1` prints commands without running them |
| `SKIP_EXISTING` | `1` | `1` short-circuits on existing artifacts; `0` forces re-run |
| `NUM_GPUS` | `-1` (auto) | Force a specific GPU count, or `0` to run on CPU |
| `BATCH_SIZE` | `64` | Per-GPU batch size used for every training run |
| `MAX_EPOCHS` | `200` | Training epoch budget |
| `TEXT_ENCODER` | `clip_vit_b32` | `clip_vit_b32` or `sentence_transformer` |
| `BEST_ENCODER` | (auto) | Pin the encoder used by Priority 1+ instead of running Priority 0 |

### Output layout

```text
runs/
├── checkpoints/
│   ├── encoder_preliminary/<encoder>/csibench-har-clip_infonce/best.pt   # P0
│   ├── <dataset>-<task>-<method>/best.pt                                 # P1A
│   ├── csibench-multitask-clip_infonce/best.pt                           # P1B
│   ├── all-multitask-clip_infonce/best.pt                                # P1B
│   ├── ablation_text_level/<level>/...                                   # P3A
│   ├── ablation_m2m/<method>_<n>para/...                                 # P3B
│   ├── ablation_text_encoder/sentence_transformer/...                    # P5B
│   ├── ablation_embed_dim/dim_<n>/...                                    # P5C
│   └── amfm/{frozen,finetuned}/...                                       # P6
├── results/
│   ├── <dataset>/<task>/<method>_<encoder>/metrics.json                  # P0, P1A, P2*, P7
│   ├── multitask_csibench/...                                            # P2B
│   ├── ambient_qa/...                                                    # P4
│   ├── _encoder_preliminary_best.txt                                     # P0 winner
│   └── _paper_tables/                                                    # aggregated tables
└── logs/
    └── priority_<n>.log                                                  # tee'd stdout per priority
```

### Pre-flight: text registries and embeddings

The released dataset already includes a `pair_registry.json` for every task, so no manual pre-flight is needed for the standard workflow. The orchestrator generates per-task `text_embeddings_<encoder>.h5` files on demand whenever a multitask priority needs them.

If you want to **regenerate** registries with custom parameters (different paraphrase counts, LLM-based paraphrasing, new tasks), run `python scripts/generate_all_texts.py` — see the script's `--help` for available flags.

---

## Advanced usage

When you need finer-grained control than `run_local_experiments.sh` provides — e.g. running a custom benchmark sweep, a single ablation, or evaluating an arbitrary checkpoint — use the per-script entry points below.

---

## Run the full benchmark

Sweep all combinations of methods, encoders, and tasks:

```bash
python scripts/run_benchmark.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/experiment/default.yaml` | Base config |
| `--datasets` | `csibench xrf55 signfi` | Datasets to benchmark |
| `--methods` | all 6 methods | Alignment methods |
| `--encoders` | `resnet18_1d patchtst vit_1d` | CSI encoder architectures |
| `--num_gpus` | `0` (auto) | GPUs for each experiment |
| `--override` | (none) | Extra config overrides applied to every run |
| `--dry_run` | off | Print commands without executing |

**Examples:**

```bash
# Dry run to see all 162 experiment commands
python scripts/run_benchmark.py --dry_run

# Benchmark only CSI-Bench HAR with all methods and encoders
python scripts/run_benchmark.py --datasets csibench --override task=har

# Benchmark only contrastive methods on 4 GPUs
python scripts/run_benchmark.py --methods clip_infonce siglip --num_gpus 4
```

The full benchmark runs: **6 methods x 3 encoders x 9 tasks = 162 experiments**
(7 CSI-Bench tasks + 1 XRF55 task + 1 SignFi task).

---

## Run ablation studies

```bash
python scripts/run_ablations.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/experiment/default.yaml` | Base config |
| `--ablation` | all ablations | Which ablation(s) to run |
| `--num_gpus` | `0` (auto) | GPUs per experiment |
| `--override` | (none) | Extra config overrides |
| `--dry_run` | off | Print without executing |

Available ablations:

| Ablation name | Description | Sweep values |
|---------------|-------------|-------------|
| `text_granularity` | Effect of text levels | `level1`, `level1_level2` |
| `many_to_many_ratio` | Paraphrase count | 1, 3, 5, 10 |
| `csi_encoder` | Encoder architecture | `resnet18_1d`, `patchtst`, `vit_1d` |
| `text_encoder` | Text backbone | `clip_vit_b32`, `sentence_transformer` |
| `embedding_dim` | Projection dimension | 128, 256, 512 |
| `batch_size` | Contrastive batch size | 64, 128, 256, 512 |

**Examples:**

```bash
# Run only the encoder ablation
python scripts/run_ablations.py --ablation csi_encoder

# Run embedding dim + batch size ablations on 8 GPUs
python scripts/run_ablations.py --ablation embedding_dim batch_size --num_gpus 8

# Dry run all ablations
python scripts/run_ablations.py --dry_run
```

---

## Evaluate a trained checkpoint

```bash
python scripts/run_evaluate.py \
    --checkpoint checkpoints/csibench/clip_infonce/best.pt \
    --config configs/experiment/default.yaml \
    --output_dir results/csibench_clip_infonce
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to trained `.pt` checkpoint |
| `--config` | (required) | Experiment YAML config |
| `--override` | (none) | Config overrides |
| `--eval_protocols` | `zero_shot ambient_qa visualization` | Which protocols to run |
| `--output_dir` | `results` | Output directory for metrics and figures |

Available evaluation protocols: `zero_shot`, `ambient_qa`, `cross_task`, `cross_dataset`, `visualization`. The default set runs `zero_shot`, `ambient_qa`, and `visualization`; `cross_task` and `cross_dataset` are still supported but must be requested explicitly (the orchestrator drives them via the dedicated Priority 2A/2B stages).

**Examples:**

```bash
# Run only zero-shot evaluation
python scripts/run_evaluate.py \
    --checkpoint best.pt --config configs/experiment/default.yaml \
    --eval_protocols zero_shot

# Run all protocols for a SigLIP model
python scripts/run_evaluate.py \
    --checkpoint checkpoints/csibench/siglip/best.pt \
    --config configs/experiment/default.yaml \
    --override method=siglip \
    --output_dir results/siglip_eval
```

Output files:
- `metrics.json` -- all evaluation metrics in a single JSON
- `tables.tex` -- LaTeX table for the paper
- `tsne_embeddings.png` -- t-SNE visualization (if `visualization` protocol is run)

---

## Project structure

```text
CSI-Text-Multimodal-Alignment/
├── configs/
│   ├── dataset/                       # csibench.yaml, xrf55.yaml, signfi.yaml
│   ├── method/                        # clip_infonce, siglip, language_bind, centrobind, barlow_twins, masked_predict
│   └── experiment/                    # default, amfm_alignment, multitask_*; SageMaker variants live under experiment/sagemaker_configs/
├── data/
│   ├── metadata_loader.py             # Unified CSV + split JSON + per-sample H5 loader
│   ├── dataset.py                     # Single-task PyTorch Dataset (lazy/preload, DistributedSampler)
│   ├── multitask_dataset.py           # Multi-task Dataset wrapper for joint training
│   ├── pair_registry.py               # Many-to-many CSI-text registry
│   ├── preprocessing/                 # Per-dataset loaders + unified CSI normalization
│   ├── text_generation/               # Templates, LLM paraphrase, pair registry builder
│   └── data_organization/             # Reorganization scripts that produced the released layout (SignFi/, XRF55/); SignFi steps are required to reformat raw .mat files before training
├── models/
│   ├── encoders/                      # CSI encoders (ResNet-1D, PatchTST, ViT-1D, AM-FM), text encoders, projections
│   └── alignment/                     # 6 alignment method implementations
├── training/
│   ├── train.py                       # Single-task DDP + AMP training loop
│   ├── multitask_train.py             # Multi-task joint training loop
│   ├── losses.py                      # Loss functions
│   └── scheduler.py                   # Warmup cosine / step / plateau schedulers
├── evaluation/
│   ├── zero_shot.py                   # Zero-shot text-prompt classification
│   ├── few_shot.py                    # Few-shot probing
│   ├── retrieval.py                   # CSI <-> text retrieval (R@K, mean rank)
│   ├── ambient_qa.py                  # Ambient question-answering protocol
│   ├── cross_task_transfer.py         # Cross-task / cross-dataset transfer
│   └── visualization.py               # UMAP / t-SNE embedding plots
├── utils/
│   └── s3_sync.py                     # S3 upload/download utilities
├── scripts/
│   ├── launch_ddp.py                  # Multi-GPU DDP launcher
│   ├── run_benchmark.py               # Full benchmark sweep
│   ├── run_ablations.py               # Ablation studies
│   ├── run_evaluate.py                # Evaluation protocol runner
│   ├── generate_all_texts.py          # Text pair registry generation (all datasets)
│   ├── generate_text_embeddings.py    # Encoder-specific text embedding pre-compute
│   ├── preprocess_to_consolidated.py  # H5 consolidation for fast I/O
│   ├── pipelines/                     # End-to-end orchestrators (run_local_experiments.sh, train_local.sh; SageMaker variants under pipelines/sagemaker_scripts/)
│   ├── baselines/                     # Supervised baselines (no text alignment)
│   └── paper_experiments/             # Paper-reproducibility extras (multi-seed, held-out, template-count, open-vocab)
├── datasheet.md                       # Datasheet for Datasets (Gebru et al.)
├── croissant.json                     # Croissant metadata (NeurIPS artifact)
├── requirements.txt
└── LICENSE
```

---

## Benchmark methods

All methods are selected via `configs/method/<name>.yaml` and `method=` in the experiment config.

| Method | Config file | Family | Description |
|--------|-------------|--------|-------------|
| CLIP-style InfoNCE | `clip_infonce.yaml` | Contrastive | Symmetric InfoNCE between CSI and text embeddings |
| SigLIP-style | `siglip.yaml` | Contrastive | Sigmoid pairwise contrastive loss (no global softmax) |
| LanguageBind-style | `language_bind.yaml` | Anchor-based | Asymmetric InfoNCE with frozen text anchor |
| CentroBind-style | `centrobind.yaml` | Anchor-based | Centroid-based contrastive learning with momentum |
| Barlow Twins | `barlow_twins.yaml` | Non-contrastive | Cross-modal redundancy reduction |
| Masked prediction | `masked_predict.yaml` | Generative/hybrid | Masked cross-modal reconstruction with alignment |

---

## Evaluation protocols

| Protocol | Module | What it measures |
|----------|--------|------------------|
| **Zero-shot classification** | `evaluation/zero_shot.py` | CSI-to-text matching via text class prototypes |
| **Ambient QA** | `evaluation/ambient_qa.py` | Multiple-choice scene questions (activity, proximity, identity) |
| **Cross-task transfer** | `evaluation/cross_task_transfer.py` | Zero-shot on a held-out task using target-task text prototypes |
| **Cross-dataset transfer** | `evaluation/cross_task_transfer.py` | Zero-shot on a different dataset entirely |
| **Embedding analysis** | `evaluation/visualization.py` | t-SNE/UMAP visualization + linear probe accuracy |

---

## License

- **Code** (this repository): **MIT** -- see [`LICENSE`](LICENSE).
- **CSI-Text curated text, manifests, and processed splits**: **CC-BY-SA-4.0**, unless noted otherwise.
- **Original CSI measurements** remain under their respective source licenses (CSI-Bench, XRF55, SignFi).

---

## Contact

Open an issue on the GitHub repository for bugs, clarifications, or collaboration regarding the benchmark and dataset release.
