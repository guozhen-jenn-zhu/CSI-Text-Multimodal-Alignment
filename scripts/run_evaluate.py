"""Run evaluation protocols on a trained model checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is on sys.path so `data.*` / `evaluation.*` /
# `models.*` / `training.*` imports work when this script is invoked as
# `python scripts/run_evaluate.py` (which only puts scripts/ on sys.path).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import build_dataloaders
from data.multitask_dataset import resolve_task_roots
from data.pair_registry import PairRegistry
from data.text_generation.templates.task_templates import get_task_templates
from evaluation import (
    AmbientQA,
    CrossDatasetTransfer,
    CrossTaskTransfer,
    EmbeddingVisualizer,
    FewShotLinearProbe,
    LinearProbe,
    run_retrieval_evaluation,
    run_zero_shot_evaluation,
)
from evaluation.cross_task_transfer import TASK_TO_DATASET
from models.alignment import build_alignment_model
from models.encoders import ProjectionHead, build_csi_encoder, build_text_encoder
from training.train import _apply_overrides, _merge_configs

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Model / config helpers (unchanged)
# ------------------------------------------------------------------

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


def build_model_from_config(config: dict, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, Optional[torch.nn.Module], torch.nn.Module]:
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
        # Don't load pretrained weights during eval -- the full model
        # state_dict from the alignment checkpoint already has them.

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


# ------------------------------------------------------------------
# Text / data helpers (unchanged)
# ------------------------------------------------------------------

def _local_class_texts(entries: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Build class_texts from raw template entries (fallback when no registry)."""
    ordered = sorted(entries, key=lambda e: e["class_id"])
    return {e["class_id"]: e["templates"] for e in ordered}


def build_class_texts(
    dataset: str,
    task: str,
    registry_path: Optional[str] = None,
) -> dict[int, list[str]]:
    """Build class_texts mapping, preferring the pair registry when available.

    If *registry_path* points to a valid v2 registry, class texts are loaded
    from it (including any LLM paraphrases). Otherwise, falls back to raw
    templates.
    """
    if registry_path and Path(registry_path).is_file():
        try:
            reg = PairRegistry.load(registry_path)
            ct = reg.get_class_texts()
            if ct:
                logger.info("Loaded class_texts from registry: %s (%d classes)", registry_path, len(ct))
                return ct
        except Exception as e:
            logger.warning("Failed to load class_texts from registry %s: %s — falling back to templates", registry_path, e)

    entries = get_task_templates(dataset, task)
    return _local_class_texts(entries)


def _dataset_yaml_defaults(dataset_name: str) -> dict[str, Any]:
    p = Path("configs/dataset") / f"{dataset_name}.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _resolve_task_root(config: dict, task: Optional[str] = None, dataset_name: Optional[str] = None) -> str:
    """Resolve the task_root path, checking config overrides first.

    Resolution order:
    1. Return ``config["task_root"]`` if it matches the requested dataset/task.
    2. Check the top-level ``multitask.tasks`` list for an explicit per-task
       ``task_root`` (set by multitask experiment configs).
    3. Use ``data_root`` from ``dataset_config``, checking the dataset YAML's
       ``multitask.tasks`` dict and ``tasks`` dict for folder structure.
       When ``config["global_data_root"]`` is set (via ``--data_root``), it
       overrides the default ``data/processed/<dataset>`` base path with
       ``<global_data_root>/<dataset>`` for cross-dataset targets.
    4. Fall back to ``data_root/<task>``.

    When ``data_root`` is explicitly set (e.g. SageMaker paths), the folder
    mappings from the dataset YAML (like ``FallDetection``) are skipped and
    the task name is used directly as the folder name.
    """
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
    multi_task: bool = False,
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


def cross_dataset_source_name(config: dict) -> str:
    d, t = config["dataset"], config.get("task", "har")
    if d == "csibench" and t == "har":
        return "csibench_har"
    if d == "xrf55":
        return "xrf55"
    if d == "signfi":
        return "signfi"
    return f"{d}_{t}"


# ------------------------------------------------------------------
# Ambient QA helpers (unchanged)
# ------------------------------------------------------------------

def _register_ambient_answers(
    aq: AmbientQA, config: dict, registry_dir: Optional[str] = None,
) -> None:
    """Register answer texts for ambient QA attributes.

    Tries to load from per-task registries first (uses the first template text
    per class). Falls back to raw templates.
    """
    task_attr_map = {
        "har": "activity",
        "proximity": "proximity",
        "user_id": "user_identity",
    }

    for task_name, attr in task_attr_map.items():
        if attr not in aq.QUESTION_TEMPLATES:
            continue

        reg_path = None
        if registry_dir:
            candidate = Path(registry_dir) / task_name / "pair_registry.json"
            if candidate.is_file():
                reg_path = str(candidate)

        ct = build_class_texts("csibench", task_name, registry_path=reg_path)
        if ct:
            answers = {cid: texts[0] for cid, texts in sorted(ct.items()) if texts}
            aq.register_answers(attr, answers)


def _normalize_csi_single(
    csi: np.ndarray,
    target_subcarriers: int = 64,
    target_time_steps: int = 500,
) -> np.ndarray:
    """Interpolate and z-normalise a single CSI sample.

    Parameters
    ----------
    csi : np.ndarray, shape ``[C, K, T]``

    Returns
    -------
    np.ndarray, shape ``[C, target_subcarriers, target_time_steps]``
    """
    from scipy.interpolate import interp1d

    C, K, T = csi.shape

    if K != target_subcarriers:
        x_old = np.linspace(0, 1, K)
        x_new = np.linspace(0, 1, target_subcarriers)
        out = np.empty((C, target_subcarriers, T), dtype=csi.dtype)
        for c in range(C):
            f = interp1d(x_old, csi[c], axis=0, kind="linear")
            out[c] = f(x_new)
        csi = out
        K = target_subcarriers

    if T != target_time_steps:
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_time_steps)
        out = np.empty((C, K, target_time_steps), dtype=csi.dtype)
        for c in range(C):
            f = interp1d(x_old, csi[c], axis=1, kind="linear")
            out[c] = f(x_new)
        csi = out

    mean = csi.mean()
    std = csi.std()
    if std > 1e-8:
        csi = (csi - mean) / std
    else:
        csi = csi - mean

    return csi.astype(np.float32)


def _normalize_raw_label(task_name: str, raw_label: Any) -> str:
    """Normalise a raw metadata label to the composite-template key format."""
    s = str(raw_label)
    if task_name == "user_id":
        if s.startswith("U") and s[1:].isdigit():
            return f"user_{int(s[1:])}"
        return s
    if task_name == "har":
        return {"seated-breathing": "seated_breathing", "wavinghand": "waving"}.get(s, s)
    if task_name == "proximity":
        try:
            return f"{float(s)}m"
        except ValueError:
            return s
    return s


def _run_ambient_qa_multitask(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    text_projection: Optional[torch.nn.Module],
    csi_projection: torch.nn.Module,
    config: dict,
    device: torch.device,
    registry_dir: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    from data.metadata_loader import MetadataLoader

    attr_map = {"har": "activity", "proximity": "proximity", "user_id": "user_identity"}
    mt_entries = [
        e for e in config.get("multitask", {}).get("tasks", [])
        if isinstance(e, dict) and e.get("task_name") in attr_map
    ]
    if not mt_entries:
        return {"error": "no_multitask_tasks_in_config",
                "message": "Config has no multitask.tasks entries matching har/proximity/user_id."}

    task_loaders: dict[str, MetadataLoader] = {}
    for entry in mt_entries:
        task_loaders[entry["task_name"]] = MetadataLoader(
            entry["task_root"], h5_key=entry.get("h5_key", "CSI_amps"),
        )

    test_id_sets = {
        t: set(ld.filter_valid_ids(ld.get_split_ids("test")))
        for t, ld in task_loaders.items()
    }
    shared_ids = sorted(set.intersection(*test_id_sets.values()))
    if not shared_ids:
        return {"error": "no_shared_test_samples",
                "message": "No shared test sample IDs across multitask tasks."}

    ref_loader = next(iter(task_loaders.values()))
    preproc = config.get("dataset_config", {}).get("preprocessing", {})
    tgt_sc = preproc.get("target_subcarriers", 64)
    tgt_ts = preproc.get("target_time_steps", 500)

    normed_samples: list[np.ndarray] = []
    for sid in shared_ids:
        raw = ref_loader.load_sample(sid)  # [C, K, T]
        normed = _normalize_csi_single(raw, tgt_sc, tgt_ts)
        normed_samples.append(normed)
    csi_np = np.stack(normed_samples, axis=0)  # [N, C, tgt_sc, tgt_ts]
    csi_tensor = torch.from_numpy(csi_np).to(device)

    model.eval()
    emb_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in csi_tensor.split(256):
            feat = model.csi_encoder(chunk)
            if csi_projection is not None:
                feat = csi_projection(feat)
            emb_chunks.append(feat.cpu())
    csi_emb = torch.cat(emb_chunks, dim=0)

    # Collect both integer labels (Level 1) and raw string labels (Level 2)
    tasks_and_labels: dict[str, np.ndarray] = {}
    raw_labels_per_task: dict[str, list[str]] = {}

    for task_name, loader in task_loaders.items():
        attr = attr_map[task_name]
        raw_labels = [loader.get_label(sid) for sid in shared_ids]
        raw_labels_per_task[task_name] = [
            _normalize_raw_label(task_name, lbl) for lbl in raw_labels
        ]

        lmap_path = Path(loader.task_root) / "metadata" / "label_mapping.json"
        if lmap_path.is_file():
            with open(lmap_path) as _lf:
                label_to_idx = json.load(_lf)["label_to_idx"]
            int_labels = [label_to_idx[str(lbl)] for lbl in raw_labels]
        else:
            unique = sorted(set(raw_labels))
            lbl2int = {lbl: i for i, lbl in enumerate(unique)}
            int_labels = [lbl2int[lbl] for lbl in raw_labels]
        tasks_and_labels[attr] = np.array(int_labels, dtype=np.int64)

    aq = AmbientQA(text_encoder, text_projection, device=str(device))
    _register_ambient_answers(aq, config, registry_dir=registry_dir)

    # Level 1: per-attribute QA
    level1 = aq.evaluate_multi_attribute(
        csi_emb, tasks_and_labels, list(tasks_and_labels.keys()),
    )

    result: dict[str, Any] = {"level1": level1}

    # Level 2: composite (holistic scene) QA
    has_all_tasks = all(t in raw_labels_per_task for t in ("user_id", "har", "proximity"))
    if has_all_tasks:
        gt_tuples = [
            (user, act, prox)
            for user, act, prox in zip(
                raw_labels_per_task["user_id"],
                raw_labels_per_task["har"],
                raw_labels_per_task["proximity"],
            )
        ]
        try:
            level2 = aq.evaluate_composite_qa(csi_emb, gt_tuples)
            result["level2"] = level2

            if output_dir is not None:
                viz_dir = Path(output_dir)
                viz_dir.mkdir(parents=True, exist_ok=True)

                AmbientQA.generate_composite_confusion(
                    pred_indices=level2["predictions"],
                    gt_indices=level2["gt_indices"],
                    index_to_tuple=level2["index_to_tuple"],
                    save_path=str(viz_dir / "composite_confusion_matrix.png"),
                )
                AmbientQA.plot_similarity_heatmap(
                    sim_matrix=level2["sim_matrix"],
                    gt_indices=level2["gt_indices"],
                    index_to_tuple=level2["index_to_tuple"],
                    labels_dict=tasks_and_labels,
                    save_path=str(viz_dir / "composite_similarity_heatmap.png"),
                    samples_per_activity=2,
                )
        except Exception as e:
            logger.warning("Level 2 composite QA failed: %s", e)
            result["level2"] = {"error": str(e)}
    else:
        logger.warning(
            "Level 2 composite QA skipped: missing tasks. Have: %s",
            list(raw_labels_per_task.keys()),
        )

    return result


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------


def _strip_heavy_ambient_qa(aq: dict[str, Any]) -> dict[str, Any]:
    """Strip large arrays from the ambient QA result dict.

    Handles the nested Level 1 / Level 2 structure produced by
    ``_run_ambient_qa_multitask``.
    """
    drop_keys = {"predictions", "gt_indices", "sim_matrix", "index_to_tuple"}
    out: dict[str, Any] = {}

    for level_key in ("level1", "level2"):
        level = aq.get(level_key)
        if level is None:
            continue
        stripped: dict[str, Any] = {}
        for k, v in level.items():
            if k in drop_keys:
                continue
            if k == "per_attribute":
                stripped[k] = {
                    a: {x: y for x, y in d.items() if x != "predictions"}
                    for a, d in v.items()
                }
            else:
                stripped[k] = v
        out[level_key] = stripped

    for k, v in aq.items():
        if k not in ("level1", "level2"):
            out[k] = v

    return _numpy_json(out)


def _numpy_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_json(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_numpy_json(x) for x in obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def _strip_heavy(m: dict[str, Any]) -> dict[str, Any]:
    """Strip large arrays from metrics to keep JSON small and readable."""
    drop = {"csi_embeddings", "text_embeddings", "prototypes", "failure_details", "labels"}
    out = {k: v for k, v in m.items() if k not in drop}
    if "per_attribute" in out:
        out["per_attribute"] = {
            a: {x: y for x, y in d.items() if x != "predictions"}
            for a, d in out["per_attribute"].items()
        }
    if "predictions" in out:
        preds = out["predictions"]
        if hasattr(preds, "tolist"):
            preds = preds[:200].tolist()
        elif isinstance(preds, list):
            preds = preds[:200]
        out["predictions"] = preds
    return _numpy_json(out)


def _deep_strip_heavy(obj: Any) -> Any:
    """Recursively strip heavy arrays from nested dicts at any depth."""
    _heavy = {"labels", "csi_embeddings", "text_embeddings", "prototypes", "failure_details"}
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _heavy:
                continue
            if k == "predictions":
                if hasattr(v, "tolist"):
                    out[k] = v[:200].tolist()
                elif isinstance(v, list):
                    out[k] = v[:200]
                else:
                    out[k] = v
            elif k == "per_attribute" and isinstance(v, dict):
                out[k] = {
                    a: {x: y for x, y in d.items() if x != "predictions"}
                    for a, d in v.items() if isinstance(d, dict)
                }
            else:
                out[k] = _deep_strip_heavy(v)
        return out
    if isinstance(obj, list):
        return [_deep_strip_heavy(x) for x in obj]
    return obj


def _save_json(data: Any, path: Path, *, merge: bool = False) -> None:
    """Save JSON with numpy conversion, creating parent dirs as needed.

    If *merge* is True and *path* already exists, top-level keys from *data*
    are merged into the existing file (new keys overwrite old ones).  This
    prevents partial reruns from erasing results of earlier protocols.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = _numpy_json(data)
    if merge and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(converted, dict):
                existing.update(converted)
                converted = existing
        except (json.JSONDecodeError, OSError):
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)


# ------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------

def _build_output_dir(
    args_output_dir: str,
    config: dict,
    task: str,
    no_timestamp: bool,
    is_multitask: bool = False,
) -> Path:
    """Build a structured output directory path.

    When using the default ``results``, auto-generates::

        results/<dataset>/<task>/<method>_<encoder>/            (single-task)
        results/multitask_<dataset>/<method>_<encoder>/          (multi-task)

    A timestamp subdirectory is appended unless *no_timestamp* is set.
    """
    dataset = config.get("dataset", "csibench")
    method = config.get("method", "unknown")
    encoder = config.get("csi_encoder", {}).get("name", "unknown")

    if args_output_dir == "results":
        if is_multitask:
            base = Path("results") / f"multitask_{dataset}" / f"{method}_{encoder}"
        else:
            base = Path("results") / dataset / task / f"{method}_{encoder}"
    else:
        base = Path(args_output_dir)

    if no_timestamp:
        out = base
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = base / stamp

    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------------------------------------------
# Summary / LaTeX generation
# ------------------------------------------------------------------

def generate_summary_md(
    results: dict[str, Any],
    checkpoint_path: str,
    config_path: str,
    dataset: str,
    task: str,
) -> str:
    """Generate a human-readable Markdown summary of evaluation results."""
    lines = [
        "# Evaluation Summary",
        "",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Config**: `{config_path}`",
        f"- **Dataset**: {dataset}",
        f"- **Task**: {task}",
        f"- **Method**: {results.get('config', {}).get('method', 'N/A')}",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    zs = results.get("zero_shot")
    if zs and "accuracy" in zs:
        lines.extend([
            "## Zero-Shot Classification",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {zs['accuracy']:.4f} |",
            f"| Weighted F1 | {zs.get('weighted_f1', 0):.4f} |",
        ])
        far = zs.get("false_alarm_rate", {}).get("macro")
        if far is not None:
            lines.append(f"| FAR (macro) | {far:.4f} |")
        lines.append("")

        pcf1 = zs.get("per_class_f1", {})
        if pcf1:
            lines.extend(["### Per-Class F1", "", "| Class | F1 |", "|-------|-----|"])
            for cid, f1 in sorted(pcf1.items(), key=lambda x: str(x[0])):
                lines.append(f"| {cid} | {f1:.4f} |")
            lines.append("")

    ret = results.get("retrieval")
    if ret and isinstance(ret, dict) and "csi_to_text" in ret:
        c2t = ret["csi_to_text"]
        t2c = ret["text_to_csi"]
        lines.extend([
            "## Retrieval",
            "",
            "| Direction | R@1 | R@5 | R@10 | Median Rank |",
            "|-----------|-----|-----|------|-------------|",
        ])
        c2t_cells = [
            f"{c2t.get('recall@1', 0):.4f}",
            f"{c2t.get('recall@5', 0):.4f}",
            f"{c2t.get('recall@10', 0):.4f}",
            f"{c2t.get('median_rank', 0):.1f}",
        ]
        t2c_cells = [
            f"{t2c.get('recall@1', 0):.4f}",
            f"{t2c.get('recall@5', 0):.4f}",
            f"{t2c.get('recall@10', 0):.4f}",
            f"{t2c.get('median_rank', 0):.1f}",
        ]
        lines.append(f"| CSI -> Text | " + " | ".join(c2t_cells) + " |")
        lines.append(f"| Text -> CSI | " + " | ".join(t2c_cells) + " |")
        lines.append("")

    fs = results.get("few_shot")
    if fs and isinstance(fs, dict):
        lines.extend([
            "## Few-Shot Linear Probe",
            "",
            "| K-Shot | Accuracy (mean +/- std) | F1 (mean +/- std) |",
            "|--------|-------------------------|-------------------|",
        ])
        for key in sorted(fs.keys()):
            v = fs[key]
            if isinstance(v, dict) and "accuracy_mean" in v:
                lines.append(
                    f"| {v.get('k', key)} | "
                    f"{v['accuracy_mean']:.4f} +/- {v.get('accuracy_std', 0):.4f} | "
                    f"{v.get('f1_mean', 0):.4f} +/- {v.get('f1_std', 0):.4f} |"
                )
        lines.append("")

    aq = results.get("ambient_qa")
    if aq and "joint_accuracy" in aq:
        lines.extend([
            "## Ambient QA",
            "",
            f"- **Joint Accuracy**: {aq['joint_accuracy']:.4f}",
            "",
        ])

    ct = results.get("cross_task")
    if ct and isinstance(ct, dict):
        valid_pairs = {k: v for k, v in ct.items() if isinstance(v, dict) and "accuracy" in v}
        if valid_pairs:
            lines.extend([
                "## Cross-Task Transfer",
                "",
                "| Direction | Accuracy | Weighted F1 |",
                "|-----------|----------|-------------|",
            ])
            for pair, res in sorted(valid_pairs.items()):
                lines.append(f"| {pair} | {res['accuracy']:.4f} | {res.get('weighted_f1', 0):.4f} |")
            lines.append("")

    cd = results.get("cross_dataset")
    if cd and isinstance(cd, dict):
        valid_pairs = {k: v for k, v in cd.items() if isinstance(v, dict) and "accuracy" in v}
        if valid_pairs:
            lines.extend([
                "## Cross-Dataset Transfer",
                "",
                "| Direction | Accuracy | Weighted F1 |",
                "|-----------|----------|-------------|",
            ])
            for pair, res in sorted(valid_pairs.items()):
                lines.append(f"| {pair} | {res['accuracy']:.4f} | {res.get('weighted_f1', 0):.4f} |")
            lines.append("")

    viz = results.get("visualization")
    if viz and isinstance(viz, dict):
        lp = viz.get("linear_probe", {})
        if lp and "accuracy" in lp:
            lines.extend([
                "## Linear Probe",
                "",
                f"- **Accuracy**: {lp['accuracy']:.4f}",
                "",
            ])

    return "\n".join(lines)


def generate_multitask_summary_md(
    all_results: dict[str, dict[str, Any]],
    checkpoint_path: str,
    config_path: str,
    dataset: str,
) -> str:
    """Generate a combined summary for multitask evaluation across all tasks."""
    lines = [
        "# Multitask Evaluation Summary",
        "",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Config**: `{config_path}`",
        f"- **Dataset**: {dataset}",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Per-Task Zero-Shot Results",
        "",
        "| Task | Accuracy | Weighted F1 | FAR (macro) |",
        "|------|----------|-------------|-------------|",
    ]

    for task_name, task_res in sorted(all_results.items()):
        zs = task_res.get("zero_shot", task_res)
        if "error" in task_res and "accuracy" not in zs:
            lines.append(f"| {task_name} | ERROR | ERROR | ERROR |")
            continue
        acc = zs.get("accuracy", 0)
        f1 = zs.get("weighted_f1", 0)
        far_obj = zs.get("false_alarm_rate", {})
        far = far_obj.get("macro", 0) if isinstance(far_obj, dict) else 0
        lines.append(f"| {task_name} | {acc:.4f} | {f1:.4f} | {far:.4f} |")

    lines.append("")

    has_cross_task = False
    for task_name, task_res in sorted(all_results.items()):
        ct = task_res.get("cross_task", {})
        valid = {k: v for k, v in ct.items() if isinstance(v, dict) and "accuracy" in v}
        if valid:
            if not has_cross_task:
                lines.extend(["## Cross-Task Transfer", ""])
                has_cross_task = True
            lines.extend([
                f"### Source: {task_name}",
                "",
                "| Target | Accuracy | Weighted F1 |",
                "|--------|----------|-------------|",
            ])
            for pair, res in sorted(valid.items()):
                lines.append(f"| {pair} | {res['accuracy']:.4f} | {res.get('weighted_f1', 0):.4f} |")
            lines.append("")

    has_lp = False
    for task_name, task_res in sorted(all_results.items()):
        lp = task_res.get("visualization", {}).get("linear_probe", {})
        if lp and "accuracy" in lp:
            if not has_lp:
                lines.extend(["## Linear Probe", "", "| Task | Accuracy |", "|------|----------|"])
                has_lp = True
            lines.append(f"| {task_name} | {lp['accuracy']:.4f} |")
    if has_lp:
        lines.append("")

    return "\n".join(lines)


def write_latex_tables(results: dict[str, Any], path: Path) -> None:
    """Write comprehensive LaTeX tables for all evaluation protocols."""
    lines = ["% Auto-generated by run_evaluate.py", ""]

    zs = results.get("zero_shot")
    if zs and "accuracy" in zs:
        far = zs.get("false_alarm_rate", {}).get("macro", 0)
        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Protocol & Accuracy & Weighted F1 & FAR \\\\",
            "\\hline",
            f"Zero-shot & {zs['accuracy']:.4f} & {zs.get('weighted_f1', 0):.4f} & {far:.4f} \\\\",
        ])
        aq = results.get("ambient_qa")
        if aq and "joint_accuracy" in aq:
            lines.append(
                f"Ambient QA (joint) & {aq['joint_accuracy']:.4f} & --- & --- \\\\"
            )
        lp = results.get("visualization", {}).get("linear_probe", {})
        if lp and "accuracy" in lp:
            lines.append(
                f"Linear Probe & {lp['accuracy']:.4f} & --- & --- \\\\"
            )
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Evaluation summary.}",
            "\\end{table}",
            "",
        ])

    ct = results.get("cross_task", {})
    valid_ct = {k: v for k, v in ct.items() if isinstance(v, dict) and "accuracy" in v}
    if valid_ct:
        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lcc}",
            "\\hline",
            "Direction & Accuracy & Weighted F1 \\\\",
            "\\hline",
        ])
        for pair, res in sorted(valid_ct.items()):
            safe_pair = pair.replace("_", "\\_").replace("->", " $\\rightarrow$ ")
            lines.append(f"{safe_pair} & {res['accuracy']:.4f} & {res.get('weighted_f1', 0):.4f} \\\\")
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Cross-task transfer results.}",
            "\\end{table}",
            "",
        ])

    cd = results.get("cross_dataset", {})
    valid_cd = {k: v for k, v in cd.items() if isinstance(v, dict) and "accuracy" in v}
    if valid_cd:
        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lcc}",
            "\\hline",
            "Direction & Accuracy & Weighted F1 \\\\",
            "\\hline",
        ])
        for pair, res in sorted(valid_cd.items()):
            safe_pair = pair.replace("_", "\\_").replace("->", " $\\rightarrow$ ")
            lines.append(f"{safe_pair} & {res['accuracy']:.4f} & {res.get('weighted_f1', 0):.4f} \\\\")
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Cross-dataset transfer results.}",
            "\\end{table}",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_multitask_latex_tables(all_results: dict[str, dict], path: Path) -> None:
    """Write LaTeX tables summarising multitask per-task results."""
    lines = ["% Auto-generated by run_evaluate.py (multitask)", ""]

    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Task & Accuracy & Weighted F1 & FAR \\\\",
        "\\hline",
    ])
    for task_name, task_res in sorted(all_results.items()):
        zs = task_res.get("zero_shot", task_res)
        if "accuracy" not in zs:
            continue
        far_obj = zs.get("false_alarm_rate", {})
        far = far_obj.get("macro", 0) if isinstance(far_obj, dict) else 0
        safe_name = task_name.replace("_", "\\_")
        lines.append(f"{safe_name} & {zs['accuracy']:.4f} & {zs.get('weighted_f1', 0):.4f} & {far:.4f} \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Multitask per-task zero-shot results.}",
        "\\end{table}",
        "",
    ])

    has_ct = False
    for task_name, task_res in sorted(all_results.items()):
        ct = task_res.get("cross_task", {})
        valid_ct = {k: v for k, v in ct.items() if isinstance(v, dict) and "accuracy" in v}
        if valid_ct:
            if not has_ct:
                lines.extend([
                    "\\begin{table}[t]",
                    "\\centering",
                    "\\begin{tabular}{lcc}",
                    "\\hline",
                    "Direction & Accuracy & Weighted F1 \\\\",
                    "\\hline",
                ])
                has_ct = True
            for pair, res in sorted(valid_ct.items()):
                safe_pair = pair.replace("_", "\\_").replace("->", " $\\rightarrow$ ")
                lines.append(f"{safe_pair} & {res['accuracy']:.4f} & {res.get('weighted_f1', 0):.4f} \\\\")
    if has_ct:
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Cross-task transfer results.}",
            "\\end{table}",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------
# Core evaluation for a single task
# ------------------------------------------------------------------

def _resolve_registry_path(
    task: str,
    registry_dir: Optional[str],
    config: dict,
    dataset_name: Optional[str] = None,
) -> Optional[str]:
    """Find the pair registry for *task*, checking registry_dir then task_root."""
    if registry_dir:
        candidate = Path(registry_dir) / task / "pair_registry.json"
        if candidate.is_file():
            return str(candidate)
    dn = dataset_name or TASK_TO_DATASET.get(task, config.get("dataset", "csibench"))
    task_root = _resolve_task_root(config, task=task, dataset_name=dn)
    fallback = Path(task_root) / "pair_registry.json"
    if fallback.is_file():
        return str(fallback)
    return None


def _evaluate_single_task(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    text_projection: Optional[torch.nn.Module],
    csi_projection: torch.nn.Module,
    config: dict[str, Any],
    dataset: str,
    task: str,
    out_dir: Path,
    eval_protocols: list[str],
    registry_dir: Optional[str],
    device: torch.device,
    cross_task_dataset: Optional[str] = None,
) -> dict[str, Any]:
    """Run all requested protocols for a single task and save results.

    Args:
        cross_task_dataset: Dataset key for ``CrossTaskTransfer.DATASET_TASKS``
            lookup.  Defaults to ``"all"`` so that cross-task evaluation
            covers every task across all datasets regardless of training
            dataset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "config": {"dataset": dataset, "task": task, "method": config.get("method")},
    }

    needs_class_texts = any(p in eval_protocols for p in ("zero_shot", "visualization", "cross_task", "cross_dataset", "retrieval", "few_shot"))
    class_texts: dict = {}
    loaders: dict = {}
    if needs_class_texts:
        reg_path = _resolve_registry_path(task, registry_dir, config, dataset_name=dataset)
        class_texts = build_class_texts(dataset, task, registry_path=reg_path)
        loader_cfg = _loader_config_from_experiment(config, task=task, dataset_name=dataset)
        loaders = build_dataloaders(loader_cfg)

        # Remove phantom classes (text prototypes with zero data samples)
        if loaders.get("test") and hasattr(loaders["test"].dataset, "int_labels"):
            present = set(int(x) for x in np.unique(loaders["test"].dataset.int_labels))
            phantom = set(class_texts.keys()) - present
            if phantom:
                logger.warning(
                    "Removing %d phantom class IDs from class_texts "
                    "(no test data): %s", len(phantom), phantom,
                )
                class_texts = {k: v for k, v in class_texts.items() if k in present}

    # --- Zero-shot ---
    zs_full: Optional[dict[str, Any]] = None
    if any(p in eval_protocols for p in ("zero_shot", "visualization", "few_shot")):
        zs_full = run_zero_shot_evaluation(
            model=model,
            dataloader=loaders["test"],
            class_texts=class_texts,
            text_encoder=text_encoder,
            text_projection=text_projection,
            csi_projection=csi_projection,
            device=str(device),
        )
    if "zero_shot" in eval_protocols and zs_full is not None:
        results["zero_shot"] = _strip_heavy(zs_full)

    # --- Ambient QA ---
    if "ambient_qa" in eval_protocols and dataset == "csibench":
        try:
            aq_dir = out_dir / "ambient_qa"
            aq_res = _run_ambient_qa_multitask(
                model, text_encoder, text_projection, csi_projection, config, device,
                registry_dir=registry_dir,
                output_dir=aq_dir,
            )
            results["ambient_qa"] = _strip_heavy_ambient_qa(aq_res)
        except Exception as e:
            logger.warning("Ambient QA skipped: %s", e)
            results["ambient_qa"] = {"error": str(e)}

    # --- Cross-task transfer ---
    ct_dataset = cross_task_dataset or "all"
    if "cross_task" in eval_protocols and ct_dataset in CrossTaskTransfer.DATASET_TASKS:
        ct_obj = CrossTaskTransfer(text_encoder, text_projection, device=str(device))
        all_tasks = CrossTaskTransfer.DATASET_TASKS[ct_dataset]
        other_tasks = [t for t in all_tasks if t != task]

        if other_tasks:
            ct_dir = out_dir / "cross_task"
            ct_dir.mkdir(parents=True, exist_ok=True)

            target_loaders: dict[str, DataLoader] = {}
            target_texts: dict[str, dict[int, list[str]]] = {}
            ct_errors: dict[str, str] = {}

            for tgt in other_tasks:
                try:
                    tgt_dataset = TASK_TO_DATASET.get(tgt, dataset)
                    cfg_t = _loader_config_from_experiment(
                        config, task=tgt, multi_task=False, dataset_name=tgt_dataset,
                    )
                    ld = build_dataloaders(cfg_t)
                    target_loaders[tgt] = ld["test"]
                    tgt_reg = _resolve_registry_path(tgt, registry_dir, config, tgt_dataset)
                    target_texts[tgt] = build_class_texts(tgt_dataset, tgt, registry_path=tgt_reg)
                except Exception as e:
                    logger.warning("Cross-task target %s failed: %s", tgt, e)
                    ct_errors[tgt] = str(e)

            raw_ct = ct_obj.evaluate_all_pairs(
                ct_dataset,
                model,
                target_loaders,
                target_texts,
                source_task=task,
                csi_projection=csi_projection,
            )

            ct_results: dict[str, Any] = {}
            for key, val in raw_ct.items():
                stripped = _strip_heavy(val) if isinstance(val, dict) else val
                ct_results[key] = stripped
                safe_key = key.replace("->", "_to_")
                _save_json(stripped, ct_dir / f"{safe_key}.json")

            for tgt, err_msg in ct_errors.items():
                err_key = f"{task}->{tgt}"
                ct_results[err_key] = {"error": err_msg}

            results["cross_task"] = ct_results

            _save_json(
                {"source_task": task, "num_targets": len(other_tasks),
                 "num_successful": len(raw_ct), "num_failed": len(ct_errors),
                 "pairs": ct_results},
                ct_dir / "summary.json",
            )
            logger.info("Cross-task: %d/%d pairs evaluated", len(raw_ct), len(other_tasks))
        else:
            logger.info("Cross-task: no other tasks for dataset '%s'", ct_dataset)

    # --- Cross-dataset transfer ---
    if "cross_dataset" in eval_protocols:
        src = cross_dataset_source_name(config)
        cdt = CrossDatasetTransfer(text_encoder, text_projection, device=str(device))
        target_configs: dict[str, dict[str, Any]] = {}
        cd_errors: dict[str, str] = {}
        for _s, tgt in CrossDatasetTransfer.TRANSFER_DIRECTIONS:
            if _s != src:
                continue
            try:
                if tgt == "xrf55":
                    cfg_t = _loader_config_from_experiment(
                        config, task="action_recognition", dataset_name="xrf55",
                    )
                    ld = build_dataloaders(cfg_t)["test"]
                    target_configs[tgt] = {
                        "dataloader": ld,
                        "class_texts": build_class_texts("xrf55", "action_recognition"),
                    }
                elif tgt == "csibench_har":
                    cfg_t = _loader_config_from_experiment(config, task="har", dataset_name="csibench")
                    ld = build_dataloaders(cfg_t)["test"]
                    har_reg = _resolve_registry_path("har", registry_dir, config, "csibench")
                    target_configs[tgt] = {
                        "dataloader": ld,
                        "class_texts": build_class_texts("csibench", "har", registry_path=har_reg),
                    }
            except Exception as e:
                logger.warning("Cross-dataset target %s: %s", tgt, e)
                cd_errors[tgt] = str(e)

        raw_cd = cdt.evaluate_all_directions(model, src, target_configs, csi_projection)

        cd_results: dict[str, Any] = {}
        if raw_cd or cd_errors:
            cd_dir = out_dir / "cross_dataset"
            cd_dir.mkdir(parents=True, exist_ok=True)

            for key, val in raw_cd.items():
                stripped = _strip_heavy(val) if isinstance(val, dict) else _numpy_json(val)
                cd_results[key] = stripped
                safe_key = key.replace("->", "_to_")
                _save_json(stripped, cd_dir / f"{safe_key}.json")

            for tgt, err_msg in cd_errors.items():
                err_key = f"{src}->{tgt}"
                cd_results[err_key] = {"error": err_msg}

            _save_json(
                {"source": src, "num_successful": len(raw_cd),
                 "num_failed": len(cd_errors), "pairs": cd_results},
                cd_dir / "summary.json",
            )

        results["cross_dataset"] = cd_results

    # --- Retrieval metrics ---
    if "retrieval" in eval_protocols and needs_class_texts:
        try:
            retrieval_res = run_retrieval_evaluation(
                model=model,
                dataloader=loaders["test"],
                class_texts=class_texts,
                text_encoder=text_encoder,
                text_projection=text_projection,
                csi_projection=csi_projection,
                device=str(device),
            )
            results["retrieval"] = _numpy_json(retrieval_res)
        except Exception as e:
            logger.warning("Retrieval evaluation failed: %s", e)
            results["retrieval"] = {"error": str(e)}

    # --- Visualization + Linear Probe ---
    train_x: Optional[np.ndarray] = None
    train_y: Optional[np.ndarray] = None
    if "visualization" in eval_protocols and zs_full is not None:
        proto = zs_full["prototypes"].numpy()
        csi_np = zs_full["csi_embeddings"].numpy()
        labels = zs_full["labels"]
        viz = EmbeddingVisualizer()
        fig_path = out_dir / "tsne_embeddings.png"
        viz.plot_tsne(csi_np, proto, labels, save_path=str(fig_path))
        train_emb_chunks: list[torch.Tensor] = []
        train_label_chunks: list[np.ndarray] = []
        model.eval()
        ce = model.csi_encoder
        with torch.no_grad():
            for batch in loaders["train"]:
                csi = batch["csi"].to(device)
                f = ce(csi)
                if csi_projection is not None:
                    f = csi_projection(f)
                train_emb_chunks.append(f.cpu().numpy())
                train_label_chunks.append(batch["labels"].numpy())
        train_x = np.concatenate(train_emb_chunks, axis=0)
        train_y = np.concatenate(train_label_chunks, axis=0)
        probe = LinearProbe()
        probe_res = probe.evaluate(train_x, train_y, csi_np, labels)
        results["visualization"] = {
            "tsne_path": str(fig_path),
            "linear_probe": _numpy_json(
                {k: v for k, v in probe_res.items() if k != "predictions"},
            ),
        }

    # --- Few-shot linear probe ---
    if "few_shot" in eval_protocols and zs_full is not None:
        csi_np_fs = zs_full["csi_embeddings"].numpy()
        labels_fs = zs_full["labels"]
        if train_x is None:
            train_emb_chunks_fs: list[torch.Tensor] = []
            train_label_chunks_fs: list[np.ndarray] = []
            model.eval()
            ce_fs = model.csi_encoder
            with torch.no_grad():
                for batch in loaders["train"]:
                    csi = batch["csi"].to(device)
                    f = ce_fs(csi)
                    if csi_projection is not None:
                        f = csi_projection(f)
                    train_emb_chunks_fs.append(f.cpu().numpy())
                    train_label_chunks_fs.append(batch["labels"].numpy())
            train_x = np.concatenate(train_emb_chunks_fs, axis=0)
            train_y = np.concatenate(train_label_chunks_fs, axis=0)
        try:
            fs_probe = FewShotLinearProbe(
                shots=(1, 5, 10, 50), n_trials=10,
            )
            fs_results = fs_probe.evaluate(train_x, train_y, csi_np_fs, labels_fs)
            results["few_shot"] = _numpy_json(fs_results)
        except Exception as e:
            logger.warning("Few-shot evaluation failed: %s", e)
            results["few_shot"] = {"error": str(e)}

    # Save combined metrics.json (merge with existing to preserve prior protocols)
    clean_results = _deep_strip_heavy(_numpy_json(results))
    _save_json(clean_results, out_dir / "metrics.json", merge=True)
    logger.info("Saved %s", out_dir / "metrics.json")

    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate a trained CSI-Text alignment checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="Experiment YAML used for training")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional config overrides key.subkey=value",
    )
    parser.add_argument(
        "--eval_protocols",
        nargs="+",
        default=["zero_shot", "ambient_qa", "visualization"],
        help="Protocols to run by default: zero_shot, ambient_qa, visualization. "
             "Pass any of these explicitly to opt in: cross_task, cross_dataset, "
             "retrieval, few_shot.",
    )
    parser.add_argument("--output_dir", default="results", help="Directory for metrics and tables")
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Do not append a timestamp subdirectory to the output path. "
             "Useful when the caller already constructs a unique directory.",
    )
    parser.add_argument(
        "--registry_dir",
        default=None,
        help="Directory containing per-task pair registries (e.g. data/generated_texts/csibench). "
             "If provided, class_texts are loaded from registries instead of raw templates.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Base directory containing all dataset folders (e.g. /data/Dataset). "
             "Used to resolve task_root for cross-task targets from other datasets.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Evaluate a multitask checkpoint on each listed task separately. "
             "Produces per-task metrics under output_dir/<task>/.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    config = _merge_configs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    resolve_task_roots(config)

    model, text_encoder, text_projection, csi_projection = build_model_from_config(config, device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # ------------------------------------------------------------------
    # Multi-task evaluation: evaluate each task separately, then combine
    # ------------------------------------------------------------------
    if args.tasks:
        multitask_dataset = config.get("dataset", "csibench")
        base_out = _build_output_dir(
            args.output_dir, config, "multitask",
            no_timestamp=args.no_timestamp, is_multitask=True,
        )
        config.pop("task_root", None)

        all_results: dict[str, dict[str, Any]] = {}
        for eval_task in args.tasks:
            logger.info("=== Evaluating task: %s ===", eval_task)
            task_out = base_out / eval_task

            eval_dataset = TASK_TO_DATASET.get(eval_task, multitask_dataset)
            _apply_overrides(config, [f"task={eval_task}", f"dataset={eval_dataset}"])

            try:
                task_results = _evaluate_single_task(
                    model=model,
                    text_encoder=text_encoder,
                    text_projection=text_projection,
                    csi_projection=csi_projection,
                    config=config,
                    dataset=eval_dataset,
                    task=eval_task,
                    out_dir=task_out,
                    eval_protocols=args.eval_protocols,
                    registry_dir=args.registry_dir,
                    device=device,
                    cross_task_dataset="all",
                )
                all_results[eval_task] = _deep_strip_heavy(_numpy_json(task_results))
            except Exception as e:
                logger.warning("Task %s failed: %s", eval_task, e)
                all_results[eval_task] = {"error": str(e)}

            logger.info("  -> %s", task_out)

        # Save config snapshot
        with open(base_out / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Combined summary (merge to preserve prior protocol results)
        _save_json(all_results, base_out / "summary.json", merge=True)

        summary_md = generate_multitask_summary_md(
            all_results, str(ckpt_path), args.config, multitask_dataset,
        )
        (base_out / "summary.md").write_text(summary_md, encoding="utf-8")

        _write_multitask_latex_tables(all_results, base_out / "tables.tex")

        logger.info("Multitask results: %s", base_out)
        logger.info("  Files: summary.json, summary.md, tables.tex, config.yaml, <task>/metrics.json")
        return

    # ------------------------------------------------------------------
    # Standard single-task evaluation
    # ------------------------------------------------------------------
    dataset = config["dataset"]
    task = config.get("task", "har")
    out_dir = _build_output_dir(
        args.output_dir, config, task,
        no_timestamp=args.no_timestamp,
    )

    results = _evaluate_single_task(
        model=model,
        text_encoder=text_encoder,
        text_projection=text_projection,
        csi_projection=csi_projection,
        config=config,
        dataset=dataset,
        task=task,
        out_dir=out_dir,
        eval_protocols=args.eval_protocols,
        registry_dir=args.registry_dir,
        device=device,
    )

    # Save config snapshot
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Generate summary.md
    clean_results = _deep_strip_heavy(_numpy_json(results))
    summary_md = generate_summary_md(
        clean_results, str(ckpt_path), args.config, dataset, task,
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    # LaTeX tables
    write_latex_tables(clean_results, out_dir / "tables.tex")

    logger.info("Results saved to: %s", out_dir)
    logger.info("  Files: metrics.json, summary.md, tables.tex, config.yaml")


if __name__ == "__main__":
    main()
