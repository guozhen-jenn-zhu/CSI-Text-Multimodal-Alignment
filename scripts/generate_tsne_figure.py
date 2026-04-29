#!/usr/bin/env python3
"""Generate a publication-quality t-SNE figure from a trained checkpoint.

Supports two modes:
  1. **Local** (SageMaker): data and checkpoint already on disk.
  2. **S3**: downloads checkpoint + a random subset of test data from S3.

Local example (SageMaker)
-------------------------
python scripts/generate_tsne_figure.py \
    --checkpoint /home/sagemaker-user/Multimodal_CSI_Text/Model/checkpoints/xrf55-action_recognition-clip_infonce/best.pt \
    --config configs/experiment/default.yaml \
    --save_path paper_results/figures/xrf55_action_recognition_clip_infonce.pdf

S3 example (laptop)
-------------------
python scripts/generate_tsne_figure.py \
    --s3_checkpoint s3://rnd-sagemaker/Multimodal_CSI_Text/Model/checkpoints/xrf55-action_recognition-clip_infonce/best.pt \
    --s3_data_root  s3://rnd-sagemaker/Multimodal_CSI_Text/Dataset/xrf55/action_recognition \
    --config paper_results/raw/xrf55/action_recognition/clip_infonce_resnet18_1d/config.yaml \
    --samples_per_class 20 \
    --save_path paper_results/figures/xrf55_action_recognition_clip_infonce.pdf
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import CSITextDataset
from evaluation.visualization import EmbeddingVisualizer
from evaluation.zero_shot import ZeroShotClassifier
from scripts.run_evaluate import build_class_texts, build_model_from_config

logger = logging.getLogger(__name__)


# ── S3 helpers (only used in S3 mode) ────────────────────────────────

def _s3_cp(src: str, dst: str) -> None:
    subprocess.check_call(["aws", "s3", "cp", src, dst, "--quiet"])


def _download_metadata(s3_data_root: str, local_root: Path) -> None:
    for rel in [
        "pair_registry.json",
        "splits/test_id.json",
        "splits/train_id.json",
        "metadata/label_mapping.json",
        "metadata/sample_metadata.csv",
    ]:
        dest = local_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _s3_cp(f"{s3_data_root.rstrip('/')}/{rel}", str(dest))


def _select_subset(local_root: Path, samples_per_class: int, seed: int = 42) -> list[str]:
    import pandas as pd

    with open(local_root / "splits" / "test_id.json") as f:
        test_ids = set(str(i) for i in json.load(f))
    df = pd.read_csv(local_root / "metadata" / "sample_metadata.csv")
    df["id"] = df["id"].astype(str)
    test_df = df[df["id"].isin(test_ids)]

    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for _label, group in test_df.groupby("label"):
        ids = group["id"].tolist()
        chosen = rng.choice(ids, size=min(samples_per_class, len(ids)), replace=False)
        selected.extend(chosen.tolist())
    logger.info("Selected %d test samples (%d/class, %d classes)",
                len(selected), samples_per_class, test_df["label"].nunique())
    return selected


def _download_h5_files(s3_data_root: str, local_root: Path, sample_ids: list[str]) -> None:
    import pandas as pd

    df = pd.read_csv(local_root / "metadata" / "sample_metadata.csv")
    df["id"] = df["id"].astype(str)
    id_to_path = dict(zip(df["id"], df["file_path"]))

    total = len(sample_ids)
    for i, sid in enumerate(sample_ids):
        rel = id_to_path[sid].lstrip("./")
        dest = local_root / rel
        if dest.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        _s3_cp(f"{s3_data_root.rstrip('/')}/{rel}", str(dest))
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info("  downloaded %d / %d H5 files", i + 1, total)


def _prepare_s3_data(args) -> tuple[Path, Path]:
    """Download checkpoint + subset of data from S3. Returns (ckpt_path, data_root)."""
    tmp_base = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="tsne_gen_"))
    local_data = tmp_base / "data"
    local_data.mkdir(parents=True, exist_ok=True)

    ckpt_local = tmp_base / "best.pt"
    if not ckpt_local.is_file():
        logger.info("Downloading checkpoint from S3...")
        _s3_cp(args.s3_checkpoint, str(ckpt_local))

    logger.info("Downloading metadata from S3...")
    _download_metadata(args.s3_data_root, local_data)

    selected = _select_subset(local_data, args.samples_per_class)
    logger.info("Downloading %d H5 files from S3...", len(selected))
    _download_h5_files(args.s3_data_root, local_data, selected)

    return ckpt_local, local_data


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate publication-quality t-SNE figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    local = parser.add_argument_group("local mode (SageMaker)")
    local.add_argument("--checkpoint", default=None,
                       help="Path to trained checkpoint (.pt) on local disk")

    s3 = parser.add_argument_group("S3 mode (laptop)")
    s3.add_argument("--s3_checkpoint", default=None)
    s3.add_argument("--s3_data_root", default=None)
    s3.add_argument("--samples_per_class", type=int, default=20,
                    help="Samples per class to download in S3 mode (ignored in local mode)")
    s3.add_argument("--tmp_dir", default=None)

    parser.add_argument("--config", required=True,
                        help="Experiment YAML (used for model architecture + data paths)")
    parser.add_argument("--save_path", default="paper_results/figures/xrf55_action_recognition_clip_infonce.pdf")
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 6])
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # ── Resolve mode ──
    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        data_root = config["task_root"]
        registry_path = config["registry_path"]
    elif args.s3_checkpoint:
        ckpt_path, data_root_path = _prepare_s3_data(args)
        data_root = str(data_root_path)
        registry_path = str(data_root_path / "pair_registry.json")
        config["task_root"] = data_root
        config["registry_path"] = registry_path
        config["global_data_root"] = str(data_root_path.parent)
        config.pop("text_embeddings_path", None)
    else:
        parser.error("Provide either --checkpoint (local) or --s3_checkpoint (S3)")

    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Data root: %s", data_root)

    # ── Build model + load checkpoint ──
    model, text_encoder, text_projection, csi_projection = build_model_from_config(config, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("Model loaded")

    # ── Build text prototypes ──
    dataset = config.get("dataset", "xrf55")
    task = config.get("task", "action_recognition")
    class_texts = build_class_texts(dataset, task, registry_path=registry_path)
    logger.info("Class texts: %d classes", len(class_texts))

    classifier = ZeroShotClassifier(
        class_texts, text_encoder, text_projection, device=str(device),
    )
    prototypes = classifier.prototypes.cpu().numpy()

    # ── Encode CSI test samples ──
    preproc = config.get("dataset_config", {}).get("preprocessing", {})
    ds = CSITextDataset(
        task_root=data_root,
        split="test",
        h5_key=config.get("dataset_config", {}).get("h5_key", "auto"),
        registry_path=registry_path,
        target_subcarriers=preproc.get("target_subcarriers", 64),
        target_time_steps=preproc.get("target_time_steps", 500),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False,
        num_workers=4, collate_fn=CSITextDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    csi_encoder = model.csi_encoder
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            csi = batch["csi"].to(device)
            features = csi_encoder(csi)
            if csi_projection is not None:
                features = csi_projection(features)
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            if (i + 1) % 20 == 0:
                logger.info("  encoded %d / %d batches", i + 1, len(loader))

    csi_embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels)
    logger.info("CSI embeddings: %s  (%d unique classes)", csi_embeddings.shape, len(np.unique(labels)))

    # ── Generate t-SNE figure ──
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    viz = EmbeddingVisualizer()
    viz.plot_tsne(
        csi_embeddings, prototypes, labels,
        perplexity=args.perplexity,
        save_path=str(save_path),
        figsize=tuple(args.figsize),
        compact_legend=True,
    )
    logger.info("Figure saved to %s", save_path)

    npz_path = save_path.with_suffix(".npz")
    np.savez(npz_path, csi_embeddings=csi_embeddings, text_prototypes=prototypes, labels=labels)
    logger.info("Embeddings saved to %s", npz_path)


if __name__ == "__main__":
    main()
