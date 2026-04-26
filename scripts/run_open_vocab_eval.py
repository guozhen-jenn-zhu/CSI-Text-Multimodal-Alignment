"""Open-vocabulary generalization evaluation on HAR.

Loads a trained CSI-Text alignment checkpoint and evaluates zero-shot
classification with novel text prompts that were **never** seen during
training.  Compares against the original training templates as a baseline.

Results are saved to ``paper_results/open_vocabulary/`` — no existing
files, registries, or checkpoints are modified.

Usage (on SageMaker — data already at default paths)
----------------------------------------------------
python scripts/run_open_vocab_eval.py \\
    --checkpoint /home/sagemaker-user/.../csibench-har-clip_infonce/best.pt \\
    --config paper_results/raw/csibench/har/clip_infonce_resnet18_1d/config.yaml

Usage (local — override data paths)
------------------------------------
python scripts/run_open_vocab_eval.py \\
    --checkpoint /local/best.pt \\
    --config paper_results/raw/csibench/har/clip_infonce_resnet18_1d/config.yaml \\
    --data_root /local/Dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from data import build_dataloaders
from data.text_generation.templates.open_vocab_har_templates import (
    FAMILY_DESCRIPTIONS,
    OPEN_VOCAB_HAR_FAMILIES,
)
from data.text_generation.templates.task_templates import get_task_templates
from evaluation.zero_shot import ZeroShotClassifier
from models.alignment import build_alignment_model
from models.encoders import ProjectionHead, build_csi_encoder, build_text_encoder
from training.train import _apply_overrides, _merge_configs

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "jumping", 1: "running", 2: "seated_breathing", 3: "walking", 4: "waving"}


# ---------------------------------------------------------------------------
# Model / data helpers (self-contained to avoid ``scripts.*`` package imports)
# ---------------------------------------------------------------------------

def _extract_model_kwargs(method_name: str, loss_cfg: dict, embed_dim: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    temp = loss_cfg.get("temperature")
    if temp is not None:
        kwargs["temperature"] = temp
    if method_name == "siglip":
        kwargs["bias"] = loss_cfg.get("bias", -10.0)
    elif method_name == "centrobind":
        kwargs["embed_dim"] = embed_dim
        kwargs["num_centroids"] = loss_cfg.get("num_centroids", 16)
        kwargs["momentum"] = loss_cfg.get("centroid_momentum", 0.999)
    elif method_name == "barlow_twins":
        kwargs["lambda_coeff"] = loss_cfg.get("lambda_coeff", 0.005)
    elif method_name == "masked_predict":
        kwargs["embed_dim"] = embed_dim
        kwargs["mask_ratio"] = loss_cfg.get("mask_ratio", 0.75)
        kwargs["reconstruction_weight"] = loss_cfg.get("reconstruction_weight", 1.0)
        kwargs["alignment_weight"] = loss_cfg.get("alignment_weight", 0.5)
    return kwargs


def build_model_from_config(
    config: dict, device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, Optional[torch.nn.Module], torch.nn.Module]:
    ds_cfg = config.get("dataset_config", {})
    preproc = ds_cfg.get("preprocessing", {})
    proj_cfg = config.get("projection", {})
    method_cfg = config.get("method_config", {})
    loss_cfg = method_cfg.get("loss", {})
    embed_dim = proj_cfg.get("dim", 512)

    input_channels = preproc.get("input_channels", 1)
    input_subcarriers = preproc.get("target_subcarriers", 64)
    input_timesteps = preproc.get("target_time_steps", 500)

    enc_cfg = config.get("csi_encoder", {})
    enc_kwargs: dict = {"dropout": float(enc_cfg.get("dropout", 0.0))}
    if enc_cfg.get("name", "resnet18_1d").lower() == "amfm":
        for k in ("emb_dim", "encoder_heads", "encoder_layers",
                   "encoder_ff_dim", "encoder_dropout", "c_out",
                   "freq_out", "max_len", "in_channels"):
            if k in enc_cfg:
                enc_kwargs[k] = enc_cfg[k]

    csi_encoder = build_csi_encoder(
        name=enc_cfg.get("name", "resnet18_1d"),
        input_channels=input_channels,
        input_subcarriers=input_subcarriers,
        input_timesteps=input_timesteps,
        embed_dim=embed_dim,
        **enc_kwargs,
    )

    txt_cfg = config.get("text_encoder", {})
    text_encoder = build_text_encoder(name=txt_cfg.get("name", "clip_vit_b32"))
    text_embed_dim = text_encoder.embed_dim

    proj_dropout = float(proj_cfg.get("dropout", 0.0))
    csi_projection = ProjectionHead(
        input_dim=embed_dim,
        output_dim=proj_cfg.get("dim", 512),
        hidden_dim=proj_cfg.get("hidden_dim", 1024),
        num_layers=proj_cfg.get("num_layers", 2),
        dropout=proj_dropout,
    )

    method_name = config["method"]
    if method_name == "language_bind":
        text_projection = None
    else:
        text_projection = ProjectionHead(
            input_dim=text_embed_dim,
            output_dim=proj_cfg.get("dim", 512),
            hidden_dim=proj_cfg.get("hidden_dim", 1024),
            num_layers=proj_cfg.get("num_layers", 2),
            dropout=proj_dropout,
        )

    model_kwargs = _extract_model_kwargs(method_name, loss_cfg, embed_dim)
    model = build_alignment_model(
        method_name=method_name,
        csi_encoder=csi_encoder,
        csi_projection=csi_projection,
        text_projection=text_projection,
        **model_kwargs,
    )
    model.to(device)
    text_encoder.to(device)
    if text_projection is not None:
        text_projection.to(device)
    csi_projection.to(device)
    return model, text_encoder, text_projection, csi_projection


def _dataset_yaml_defaults(dataset_name: str) -> dict[str, Any]:
    p = Path("configs/dataset") / f"{dataset_name}.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _resolve_task_root(
    config: dict, task: Optional[str] = None, dataset_name: Optional[str] = None,
) -> str:
    dn = dataset_name if dataset_name is not None else config["dataset"]
    t = task if task is not None else config.get("task", "har")

    if dn == config["dataset"] and t == config.get("task") and config.get("task_root"):
        return config["task_root"]

    mt_cfg_tasks = config.get("multitask", {}).get("tasks", [])
    if isinstance(mt_cfg_tasks, list):
        for entry in mt_cfg_tasks:
            if isinstance(entry, dict) and entry.get("task_name") == t and entry.get("task_root"):
                return entry["task_root"]

    ds_cfg = config.get("dataset_config", {}) if dn == config["dataset"] else _dataset_yaml_defaults(dn)
    default_data_root = f"data/processed/{dn}"
    data_root = ds_cfg.get("data_root", default_data_root)

    global_data_root = config.get("global_data_root")
    if global_data_root and data_root == default_data_root:
        data_root = str(Path(global_data_root) / dn)

    has_custom_data_root = data_root != default_data_root

    mt_tasks = ds_cfg.get("multitask", {}).get("tasks", {})
    if isinstance(mt_tasks, dict) and t in mt_tasks:
        return str(Path(data_root) / "multitask" / t)

    if not has_custom_data_root:
        tasks = ds_cfg.get("tasks", {})
        if isinstance(tasks, dict) and t in tasks:
            folder = tasks[t].get("folder", t)
            return str(Path(data_root) / folder)

    return str(Path(data_root) / t)


def _loader_config_from_experiment(
    config: dict,
    task: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> dict[str, Any]:
    dn = dataset_name if dataset_name is not None else config["dataset"]
    ds_cfg = config.get("dataset_config", {}) if dn == config["dataset"] else _dataset_yaml_defaults(dn)
    preproc = ds_cfg.get("preprocessing", {})
    training_cfg = config["training"]
    t = task if task is not None else config.get("task", "har")
    task_root = _resolve_task_root(config, task=t, dataset_name=dn)
    reg_default = f"data/processed/{dn}/pair_registry.json"
    return {
        "task_root": task_root,
        "h5_key": ds_cfg.get("h5_key", "CSI_amps"),
        "registry_path": config.get("registry_path", reg_default) if dn == config["dataset"] else reg_default,
        "text_embeddings_path": config.get("text_embeddings_path") if dn == config["dataset"] else None,
        "target_subcarriers": preproc.get("target_subcarriers", 64),
        "target_time_steps": preproc.get("target_time_steps", 500),
        "batch_size": int(training_cfg.get("batch_size", 128)),
        "num_workers": int(training_cfg.get("num_workers", 4)),
        "prefetch_factor": int(training_cfg.get("prefetch_factor", 2)),
        "persistent_workers": training_cfg.get("persistent_workers", True),
    }


def _original_class_texts() -> dict[int, list[str]]:
    """Build class_texts from the original HAR training templates."""
    entries = get_task_templates("csibench", "har")
    return {e["class_id"]: e["templates"] for e in sorted(entries, key=lambda x: x["class_id"])}


@torch.no_grad()
def _encode_csi_test_set(
    model: torch.nn.Module,
    dataloader,
    csi_projection: Optional[torch.nn.Module],
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Encode all test CSI samples once (reused across prompt families)."""
    model.eval()
    csi_encoder = model.csi_encoder if hasattr(model, "csi_encoder") else model
    if csi_projection is None and hasattr(model, "csi_projection"):
        csi_projection = model.csi_projection

    all_embeddings: list[torch.Tensor] = []
    all_labels: list[np.ndarray] = []
    for batch in dataloader:
        csi = batch["csi"].to(device)
        features = csi_encoder(csi)
        if csi_projection is not None:
            features = csi_projection(features)
        all_embeddings.append(features.cpu())
        all_labels.append(batch["labels"].numpy())

    return torch.cat(all_embeddings, dim=0), np.concatenate(all_labels)


def _classify_with_texts(
    csi_embeddings: torch.Tensor,
    labels: np.ndarray,
    class_texts: dict[int, list[str]],
    text_encoder: torch.nn.Module,
    text_projection: Optional[torch.nn.Module],
    device: torch.device,
) -> dict[str, Any]:
    """Build text prototypes and classify pre-encoded CSI embeddings."""
    classifier = ZeroShotClassifier(
        class_texts, text_encoder, text_projection, device=str(device),
    )
    raw = classifier.evaluate(csi_embeddings, labels)

    per_class_f1 = raw.get("per_class_f1", {})
    far = raw.get("false_alarm_rate", {})
    return {
        "accuracy": raw["accuracy"],
        "weighted_f1": raw["weighted_f1"],
        "per_class_f1": {CLASS_NAMES.get(k, k): v for k, v in per_class_f1.items()},
        "false_alarm_rate_macro": far.get("macro", 0.0) if isinstance(far, dict) else 0.0,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Open-vocabulary HAR evaluation with novel text prompts",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="Experiment YAML or config embedded in checkpoint")
    parser.add_argument(
        "--data_root", default=None,
        help="Override data root (base dir containing csibench/). "
             "Clears the checkpoint's task_root so the path is re-resolved.",
    )
    parser.add_argument(
        "--output_dir",
        default="paper_results/open_vocabulary",
        help="Directory for results (default: paper_results/open_vocabulary)",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides key.subkey=value",
    )
    args = parser.parse_args()

    # --- Load config -------------------------------------------------------
    with open(args.config) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    config = _merge_configs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "config" in ckpt:
        file_ds_cfg = config.get("dataset_config", {})
        config = _merge_configs(ckpt["config"])
        merged_ds = dict(file_ds_cfg)
        merged_ds.update(config.get("dataset_config", {}))
        config["dataset_config"] = merged_ds

    if args.override:
        _apply_overrides(config, args.override)

    if args.data_root:
        config["global_data_root"] = args.data_root
        config.pop("task_root", None)
        config.pop("registry_path", None)
        config.pop("text_embeddings_path", None)
        config.get("dataset_config", {}).pop("data_root", None)

    # --- Build model -------------------------------------------------------
    model, text_encoder, text_projection, csi_projection = build_model_from_config(config, device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # --- Build test dataloader (HAR) ----------------------------------------
    loader_cfg = _loader_config_from_experiment(config, task="har", dataset_name="csibench")
    loaders = build_dataloaders(loader_cfg)
    test_loader = loaders["test"]
    logger.info("Test set: %d samples", len(test_loader.dataset))

    # --- Encode CSI test set once (reused across all prompt families) --------
    logger.info("Encoding CSI test samples ...")
    csi_embeddings, labels = _encode_csi_test_set(
        model, test_loader, csi_projection, device,
    )
    logger.info("Encoded %d samples, %d unique labels", len(labels), len(np.unique(labels)))

    # --- Evaluate each prompt family ----------------------------------------
    prompt_sets: dict[str, dict[int, list[str]]] = {
        "original_templates": _original_class_texts(),
    }
    for family_name, family_texts in OPEN_VOCAB_HAR_FAMILIES.items():
        prompt_sets[family_name] = family_texts

    all_results: dict[str, Any] = {}
    for name, class_texts in prompt_sets.items():
        logger.info("--- Evaluating prompt family: %s ---", name)
        for cid, texts in sorted(class_texts.items()):
            logger.info("  class %d (%s): %s", cid, CLASS_NAMES.get(cid, "?"), texts)

        metrics = _classify_with_texts(
            csi_embeddings=csi_embeddings,
            labels=labels,
            class_texts=class_texts,
            text_encoder=text_encoder,
            text_projection=text_projection,
            device=device,
        )
        metrics["texts"] = {CLASS_NAMES.get(k, k): v for k, v in class_texts.items()}
        metrics["texts_per_class"] = len(next(iter(class_texts.values())))

        if name in FAMILY_DESCRIPTIONS:
            metrics["description"] = FAMILY_DESCRIPTIONS[name]

        all_results[name] = metrics
        logger.info(
            "  -> accuracy=%.4f  weighted_f1=%.4f",
            metrics["accuracy"],
            metrics["weighted_f1"],
        )

    # --- Print summary table ------------------------------------------------
    print("\n" + "=" * 75)
    print("Open-Vocabulary HAR Evaluation — Summary")
    print("=" * 75)
    print(f"{'Prompt Family':<25s} {'Style':<30s} {'#T/C':>4s} {'Acc':>8s} {'wF1':>8s}")
    print("-" * 75)
    for name, m in all_results.items():
        style = m.get("description", "Training templates (baseline)")
        tc = m.get("texts_per_class", "?")
        print(f"{name:<25s} {style:<30s} {tc:>4} {m['accuracy']:>7.1%} {m['weighted_f1']:>7.1%}")
    print("=" * 75 + "\n")

    # --- Save results (back up any previous file first) ---------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "har_novel_prompts.json"

    if out_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out_path.with_name(f"har_novel_prompts_{ts}.bak.json")
        shutil.copy2(out_path, backup)
        logger.warning("Previous results backed up to %s", backup)

    output = {
        "experiment": "open_vocabulary_har",
        "description": (
            "Zero-shot HAR evaluation with novel text prompts unseen during "
            "training. Tests whether CSI-text alignment generalises to open "
            "vocabulary (arbitrary text descriptions) rather than memorising "
            "specific training template sentences."
        ),
        "checkpoint": str(args.checkpoint),
        "config_file": str(args.config),
        "timestamp": datetime.now().isoformat(),
        "model": {
            "method": config.get("method", "unknown"),
            "csi_encoder": config.get("csi_encoder", {}).get("name", "unknown"),
            "text_encoder": config.get("text_encoder", {}).get("name", "unknown"),
        },
        "num_test_samples": len(test_loader.dataset),
        "results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
