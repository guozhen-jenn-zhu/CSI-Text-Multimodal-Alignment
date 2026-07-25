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

### Data layout on SageMaker

All datasets are stored under `/data/fm_downstream/` with a uniform structure:

```text
/data/fm_downstream/
  FallDetection/
    metadata/sample_metadata.csv       # columns: id, file_path, label, ...
    splits/
      train_id.json                    # flat list of sample IDs
      val_id.json
      test_id.json
    sub_Human/.../*.h5                 # per-sample H5 files, shape (F, T, 1)
  BreathingDetection/
  Localization/
  MotionSourceRecognition/
  HumanActivityRecognition/
  ProximityRecognition/
  HumanIdentification/
  XRF55/
  SignFi/
```

Each H5 file contains a single CSI amplitude tensor of shape `(F, T, 1)` where F is the number of subcarriers and T is the number of time steps. The dataset key inside the H5 file varies by dataset (see table above).

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `h5py`, `pandas`, `scipy`, `transformers`, `sentence-transformers`, `open-clip-torch`, `boto3`, `wandb`.

---

## End-to-end workflow

The pipeline follows these steps in order:

```text
1. Generate texts  -->  2. (Optional) Consolidate H5  -->  3. Train  -->  4. Evaluate  -->  5. S3 backup
```

---

## Step 1: Generate text pair registries

Generate many-to-many CSI-text pair registries from metadata CSVs for all datasets and tasks.

### Generate for all datasets at once

```bash
python scripts/generate_all_texts.py
```

This reads every `configs/dataset/*.yaml`, discovers all tasks, loads the metadata CSV for each, and writes pair registries to `data/generated_texts/`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | `csibench xrf55 signfi` | Which datasets to process |
| `--output_dir` | `data/generated_texts` | Output root directory |
| `--use_llm` | off | Enable LLM paraphrasing (requires `OPENAI_API_KEY`) |
| `--llm_model` | `gpt-4o-mini` | OpenAI model for paraphrasing |
| `--num_paraphrases` | `5` | Paraphrases per template |
| `--s3_sync` | off | Upload output to S3 when done |

**Examples:**

```bash
# Generate for CSI-Bench only, with LLM paraphrasing
python scripts/generate_all_texts.py --datasets csibench --use_llm

# Generate for all datasets and upload to S3
python scripts/generate_all_texts.py --s3_sync
```

### Generate for a single task (standalone)

```bash
python -m data.text_generation.generate_texts \
    --dataset csibench \
    --task_root /data/fm_downstream/HumanActivityRecognition \
    --h5_key CSI_amps \
    --output data/generated_texts/csibench/har/pair_registry.json
```

| Flag | Required | Description |
|------|----------|-------------|
| `--dataset` | yes | `csibench`, `xrf55`, or `signfi` |
| `--task_root` | yes | Path to the task directory containing `metadata/` and `splits/` |
| `--h5_key` | no | H5 dataset key (`CSI_amps`, `amplitude`, or `auto` for XRF55) |
| `--output` | yes | Output path for the pair registry JSON |
| `--use_llm` | no | Enable LLM paraphrasing |
| `--num_paraphrases` | no | Number of paraphrases per template (default: 5) |

---

## Step 2 (optional): Consolidate per-sample H5 files

For faster training I/O, consolidate thousands of per-sample H5 files into a single HDF5 per split. This is a one-time preprocessing step.

```bash
python scripts/preprocess_to_consolidated.py \
    --task_root /data/fm_downstream/HumanActivityRecognition \
    --h5_key CSI_amps \
    --output_dir data/processed/csibench/har
```

This produces `train.h5`, `val.h5`, `test.h5` under the output directory, each containing:
- `csi_data`: shape `[N, C, K_target, T_target]`, float32, gzip-compressed
- `labels`: shape `[N]`
- `sample_ids`: variable-length strings

| Flag | Default | Description |
|------|---------|-------------|
| `--task_root` | (required) | Path to the task directory |
| `--h5_key` | `CSI_amps` | H5 dataset key |
| `--output_dir` | (required) | Where to write consolidated H5 files |
| `--splits` | `train val test` | Which splits to consolidate |
| `--target_subcarriers` | `64` | Interpolate subcarrier axis to this size |
| `--target_time_steps` | `500` | Interpolate time axis to this size |
| `--s3_sync` | off | Upload output to S3 |

**Example -- consolidate XRF55:**

```bash
python scripts/preprocess_to_consolidated.py \
    --task_root /data/fm_downstream/XRF55 \
    --h5_key auto \
    --output_dir data/processed/xrf55/action_recognition \
    --s3_sync
```

---

## Step 3: Train

### Single-GPU training

```bash
python -m training.train --config configs/experiment/default.yaml
```

Override any config value with `--override key.subkey=value`:

```bash
python -m training.train \
    --config configs/experiment/default.yaml \
    --override dataset=csibench task=har method=clip_infonce csi_encoder.name=resnet18_1d
```

### Multi-GPU training (DDP)

Use the launcher script which auto-detects GPU count and invokes `torchrun`:

```bash
python scripts/launch_ddp.py --config configs/experiment/default.yaml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/experiment/default.yaml` | Experiment config path |
| `--num_gpus` | `0` (auto-detect) | Number of GPUs to use |
| `--override` | (none) | Config overrides as `key=value` |
| `--resume` | (none) | Checkpoint path to resume from |

**Examples:**

```bash
# Explicitly use 4 GPUs
python scripts/launch_ddp.py --num_gpus 4 --config configs/experiment/default.yaml

# Resume training from a checkpoint
python scripts/launch_ddp.py --resume checkpoints/csibench/clip_infonce/best.pt

# Override method and encoder
python scripts/launch_ddp.py --override method=siglip csi_encoder.name=patchtst
```

### Key training config options

Edit `configs/experiment/default.yaml` or pass as overrides:

| Config key | Default | Description |
|-----------|---------|-------------|
| `training.batch_size` | `128` | Per-GPU batch size |
| `training.learning_rate` | `1e-3` | AdamW learning rate |
| `training.max_epochs` | `100` | Maximum training epochs |
| `training.amp` | `true` | Enable mixed precision |
| `training.amp_dtype` | `bfloat16` | `bfloat16` for A100, `float16` for A10G |
| `training.accumulation_steps` | `1` | Gradient accumulation steps |
| `training.preload` | `false` | Load all samples into memory at init |
| `training.num_workers` | `4` | DataLoader workers per GPU |
| `training.early_stopping_patience` | `15` | Epochs without improvement before stopping |
| `s3.enabled` | `true` | Auto-sync best checkpoint to S3 |

### Instance recommendations

| Instance | GPUs | Memory | Recommended `amp_dtype` | Suggested `batch_size` |
|----------|------|--------|-------------------------|------------------------|
| p4d.24xlarge | 8x A100 80GB | 1.1 TB | `bfloat16` | 256-512 |
| g5.24xlarge | 4x A10G 24GB | 384 GB | `float16` | 128-256 |

---

## Step 4: Run the full benchmark

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

## Step 5: Run ablation studies

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

## Step 6: Evaluate a trained checkpoint

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
| `--eval_protocols` | all 5 protocols | Which protocols to run |
| `--output_dir` | `results` | Output directory for metrics and figures |

Available evaluation protocols: `zero_shot`, `ambient_qa`, `cross_task`, `cross_dataset`, `cross_domain`, `visualization`, `retrieval`, `few_shot`.

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

## Step 7: Cross-domain evaluation (cross-device / cross-environment)

CSI-Bench tasks ship additional split files that hold out entire environments
(`splits/cross_env.json`) and entire device types (`splits/cross_device.json`).
The `cross_domain` protocol scores an **already-trained** checkpoint on those
splits, so no retraining is needed:

```bash
python -m scripts.run_evaluate \
    --checkpoint checkpoints/csibench-har-clip_infonce/best.pt \
    --config configs/experiment/sagemaker_g5.yaml \
    --eval_protocols zero_shot cross_domain \
    --output_dir results/csibench_har_cross_domain \
    --no_timestamp
```

Including `zero_shot` in the same run is recommended: the IID test accuracy is
then reported alongside each domain split so `summary.md` and `tables.tex` can
show the degradation (`Delta vs IID`).

| Flag | Default | Description |
|------|---------|-------------|
| `--cross_domain_splits` | `cross_env cross_device` | Split names to evaluate; each resolves to `<task_root>/splits/<name>.json` |

Splits that do not exist for a task are recorded as skipped rather than failing
the run.

To sweep all five CSI-Bench tasks:

```bash
# Uses the SageMaker checkpoint/results layout by default
WITH_IID=1 bash scripts/run_cross_domain_eval.sh

# A subset of tasks, one domain axis, and a local layout
TASK=har,proximity SPLITS=cross_env \
REPO_ROOT=$PWD DATA_ROOT=/data/Dataset MODEL_ROOT=/data/Model \
    bash scripts/run_cross_domain_eval.sh
```

Additional output files:
- `cross_domain/<split>.json` -- per-split metrics
- `cross_domain/summary.json` -- all splits plus success/failure counts

---

## S3 backup

All scripts support automatic backup to `s3://rnd-sagemaker/Multimodal_CSI_Text/`. The S3 bucket is organized as:

| S3 prefix | Contents |
|-----------|----------|
| `Model/` | Trained checkpoints |
| `Dataset/` | Generated pair registries, text embeddings, consolidated H5 |
| `Evaluation/` | Metrics JSON, LaTeX tables, figures |

S3 sync is configured in `configs/experiment/default.yaml`:

```yaml
s3:
  enabled: true
  bucket: rnd-sagemaker
  prefix: Multimodal_CSI_Text
  sync_checkpoints: true
  sync_evaluation: true
  sync_dataset: true
```

- **Training**: the best checkpoint is automatically uploaded after each improvement when `s3.enabled` is true.
- **Text generation / preprocessing**: pass `--s3_sync` to upload output.
- **Evaluation**: use the `S3Syncer` API in your evaluation script or manually sync:

```python
from utils.s3_sync import S3Syncer
syncer = S3Syncer(bucket="rnd-sagemaker", prefix="Multimodal_CSI_Text")
syncer.upload_evaluation("results/")
syncer.upload_checkpoint("checkpoints/csibench/clip_infonce/best.pt")
```

---

## Project structure

```text
CSI-Text-Multimodal-Alignment/
├── configs/
│   ├── dataset/             # csibench.yaml, xrf55.yaml, signfi.yaml
│   ├── method/              # clip_infonce, siglip, language_bind, centrobind, barlow_twins, masked_predict
│   └── experiment/          # default.yaml (training hyperparameters, S3, AMP)
├── data/
│   ├── metadata_loader.py   # Unified CSV + split JSON + per-sample H5 loader
│   ├── dataset.py           # PyTorch Dataset with lazy/preload modes, DistributedSampler
│   ├── pair_registry.py     # Many-to-many CSI-text registry
│   ├── preprocessing/       # Legacy loaders + unified CSI normalization
│   └── text_generation/     # Templates, LLM paraphrase, pair registry builder
├── models/
│   ├── encoders/            # CSI encoders (ResNet-1D, PatchTST, ViT-1D), text encoders, projections
│   └── alignment/           # 6 alignment method implementations
├── training/
│   ├── train.py             # DDP + AMP training loop with S3 backup
│   ├── losses.py            # Loss functions
│   └── scheduler.py         # Warmup cosine / step / plateau schedulers
├── evaluation/              # Zero-shot, ambient QA, cross-task/dataset transfer, visualization
├── utils/
│   └── s3_sync.py           # S3 upload/download utilities
├── scripts/
│   ├── launch_ddp.py        # Multi-GPU DDP launcher
│   ├── run_benchmark.py     # Full benchmark sweep
│   ├── run_ablations.py     # Ablation studies
│   ├── generate_all_texts.py  # Text pair registry generation
│   ├── preprocess_to_consolidated.py  # H5 consolidation for fast I/O
│   ├── run_evaluate.py      # Evaluation protocol runner
│   └── run_cross_domain_eval.sh  # Cross-device / cross-environment sweep
├── datasheet.md             # Datasheet for Datasets (Gebru et al.)
├── croissant.json           # Croissant metadata (NeurIPS artifact)
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
| **Cross-domain generalization** | `scripts/run_evaluate.py` (`cross_domain`) | Zero-shot on held-out environments / devices via the `cross_env` and `cross_device` splits |
| **Embedding analysis** | `evaluation/visualization.py` | t-SNE/UMAP visualization + linear probe accuracy |

---

## Citation

```bibtex
@misc{zhu2026csitext,
  title        = {CSI-Text: A Multimodal Benchmark for Language-Grounded WiFi Sensing},
  author       = {Zhu, Guozhen and others},
  year         = {2026},
  note         = {NeurIPS 2026 E\&D Track},
  howpublished = {\url{https://github.com/guozhen-jenn-zhu/CSI-Text-Multimodal-Alignment}}
}
```

---

## License

- **Code** (this repository): **MIT** -- see [`LICENSE`](LICENSE).
- **CSI-Text curated text, manifests, and processed splits**: **CC-BY-SA-4.0**, unless noted otherwise.
- **Original CSI measurements** remain under their respective source licenses (CSI-Bench, XRF55, SignFi).

---

## Contact

Open an issue on the GitHub repository for bugs, clarifications, or collaboration regarding the benchmark and dataset release.
