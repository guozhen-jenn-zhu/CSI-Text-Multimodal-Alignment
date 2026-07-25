"""Run evaluation protocols on a trained model checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ``transformers`` / ``datasets`` (pulled in by the text encoders) probe for
# TensorFlow / Flax / JAX at import time; a broken TF install in the
# environment makes that probe raise instead of degrading gracefully. This
# project is PyTorch-only, so opt out before any such import runs. ``setdefault``
# keeps an explicit caller override intact.
for _backend_flag in ("USE_TF", "USE_FLAX", "USE_JAX"):
    os.environ.setdefault(_backend_flag, "0")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import build_dataloaders
from data.pair_registry import PairRegistry
from data.text_generation.templates.task_templates import get_task_templates
from evaluation import (
    AmbientQA,
    CrossDatasetTransfer,
    CrossTaskTransfer,
    EmbeddingVisualizer,
    FewShotFinetune,
    FewShotLinearProbe,
    LinearProbe,
    run_retrieval_evaluation,
    run_zero_shot_evaluation,
)
from evaluation.composition import SCENE_TASK_ATTRIBUTES, normalize_attribute_value
from evaluation.cross_task_transfer import TASK_TO_DATASET
from models.alignment import build_alignment_model
from models.encoders import ProjectionHead, build_csi_encoder, build_text_encoder
from training.train import _apply_overrides, _merge_configs

logger = logging.getLogger(__name__)

# Split files defining held-out domains, relative to each task's ``splits/``
# directory (resolved as ``<name>.json`` by MetadataLoader.get_split_ids).
DEFAULT_CROSS_DOMAIN_SPLITS = ("cross_env", "cross_device")


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
# Ambient QA / scene helpers
# ------------------------------------------------------------------

def _register_ambient_answers(
    aq: AmbientQA, config: dict, registry_dir: Optional[str] = None,
) -> None:
    """Register answer texts for ambient QA attributes.

    Tries to load from per-task registries first (uses the first template text
    per class). Falls back to raw templates.
    """
    for task_name, attr in SCENE_TASK_ATTRIBUTES.items():
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
    return normalize_attribute_value(task_name, raw_label)


class SceneBatchError(RuntimeError):
    """Raised when the co-labeled scene subset cannot be assembled."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_result(self) -> dict[str, str]:
        return {"error": self.code, "message": self.message}


@dataclass
class MultitaskSceneBatch:
    """Encoded samples co-labeled with activity, proximity and user identity.

    Shared by ambient QA (:func:`_run_ambient_qa_multitask`) and the
    compositional-generalization analysis, so both always evaluate the same
    population with the same preprocessing.
    """

    sample_ids: list[str]
    csi_embeddings: torch.Tensor
    int_labels: dict[str, np.ndarray] = field(default_factory=dict)
    triples: Optional[list[tuple[str, str, str]]] = None


def collect_multitask_scene_embeddings(
    model: torch.nn.Module,
    csi_projection: Optional[torch.nn.Module],
    config: dict,
    device: torch.device,
    split: str = "test",
) -> MultitaskSceneBatch:
    """Encode the samples shared by the har / proximity / user_id splits.

    Raises
    ------
    SceneBatchError
        When the config has no matching multitask tasks, or the tasks share no
        samples in *split*.
    """
    from data.metadata_loader import MetadataLoader

    mt_entries = [
        e for e in config.get("multitask", {}).get("tasks", [])
        if isinstance(e, dict) and e.get("task_name") in SCENE_TASK_ATTRIBUTES
    ]
    if not mt_entries:
        raise SceneBatchError(
            "no_multitask_tasks_in_config",
            "Config has no multitask.tasks entries matching har/proximity/user_id.",
        )

    task_loaders: dict[str, MetadataLoader] = {}
    for entry in mt_entries:
        task_loaders[entry["task_name"]] = MetadataLoader(
            entry["task_root"], h5_key=entry.get("h5_key", "CSI_amps"),
        )

    id_sets = {
        t: set(ld.filter_valid_ids(ld.get_split_ids(split)))
        for t, ld in task_loaders.items()
    }
    shared_ids = sorted(set.intersection(*id_sets.values()))
    if not shared_ids:
        raise SceneBatchError(
            f"no_shared_{split}_samples",
            f"No shared {split} sample IDs across multitask tasks.",
        )

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
    int_labels: dict[str, np.ndarray] = {}
    normalized_per_task: dict[str, list[str]] = {}

    for task_name, loader in task_loaders.items():
        attr = SCENE_TASK_ATTRIBUTES[task_name]
        raw_labels = [loader.get_label(sid) for sid in shared_ids]
        normalized_per_task[task_name] = [
            _normalize_raw_label(task_name, lbl) for lbl in raw_labels
        ]

        lmap_path = Path(loader.task_root) / "metadata" / "label_mapping.json"
        if lmap_path.is_file():
            with open(lmap_path) as _lf:
                label_to_idx = json.load(_lf)["label_to_idx"]
            ints = [label_to_idx[str(lbl)] for lbl in raw_labels]
        else:
            unique = sorted(set(raw_labels))
            lbl2int = {lbl: i for i, lbl in enumerate(unique)}
            ints = [lbl2int[lbl] for lbl in raw_labels]
        int_labels[attr] = np.array(ints, dtype=np.int64)

    triples: Optional[list[tuple[str, str, str]]] = None
    if all(t in normalized_per_task for t in ("user_id", "har", "proximity")):
        triples = list(
            zip(
                normalized_per_task["user_id"],
                normalized_per_task["har"],
                normalized_per_task["proximity"],
            )
        )
    else:
        logger.warning(
            "Composite triples unavailable: have tasks %s",
            list(normalized_per_task.keys()),
        )

    return MultitaskSceneBatch(
        sample_ids=shared_ids,
        csi_embeddings=csi_emb,
        int_labels=int_labels,
        triples=triples,
    )


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
    try:
        batch = collect_multitask_scene_embeddings(
            model, csi_projection, config, device, split="test",
        )
    except SceneBatchError as exc:
        return exc.as_result()

    csi_emb = batch.csi_embeddings
    tasks_and_labels = batch.int_labels

    aq = AmbientQA(text_encoder, text_projection, device=str(device))
    _register_ambient_answers(aq, config, registry_dir=registry_dir)

    # Level 1: per-attribute QA
    level1 = aq.evaluate_multi_attribute(
        csi_emb, tasks_and_labels, list(tasks_and_labels.keys()),
    )

    result: dict[str, Any] = {"level1": level1}

    # Level 2: composite (holistic scene) QA
    if batch.triples is not None:
        gt_tuples = batch.triples
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
            "Level 2 composite QA skipped: config lacks one of har/proximity/user_id.",
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

def _cross_domain_rows(
    cross_domain: dict[str, Any],
    iid_accuracy: Optional[float] = None,
) -> tuple[list[str], list[str]]:
    """Format cross-domain results as Markdown table rows.

    Returns ``(scored_rows, unscored_notes)`` where the notes describe splits
    that were skipped or errored.
    """
    rows: list[str] = []
    notes: list[str] = []
    for split_name, res in sorted(cross_domain.items()):
        if not isinstance(res, dict):
            continue
        if "accuracy" not in res:
            reason = res.get("skipped") or res.get("error") or "unavailable"
            notes.append(f"- `{split_name}`: {reason}")
            continue
        delta = (
            f"{res['accuracy'] - iid_accuracy:+.4f}"
            if iid_accuracy is not None
            else "n/a"
        )
        rows.append(
            f"| {split_name} | {res['accuracy']:.4f} | "
            f"{res.get('weighted_f1', 0):.4f} | {res.get('num_samples', 'n/a')} | "
            f"{res.get('num_classes', 'n/a')} | {delta} |"
        )
    return rows, notes


FT_SCOPE_LABELS = {
    "projection": "Projection-only",
    "encoder": "Full encoder",
}


def _ft_scope_label(scope: str) -> str:
    return FT_SCOPE_LABELS.get(scope, scope)


def _cross_domain_few_shot_view(
    split_res: dict[str, Any],
) -> Optional[tuple[list[str], list[int], Optional[float], dict[str, dict[int, dict]]]]:
    """Normalise one split's few-shot result into ``(scopes, ks, baseline, table)``.

    ``table[scope][k]`` holds the metric dict.  Returns ``None`` when the
    split produced no fine-tuning numbers.
    """
    if not isinstance(split_res, dict):
        return None
    ft = split_res.get("finetune")
    if not isinstance(ft, dict):
        return None

    table: dict[str, dict[int, dict]] = {}
    ks: set[int] = set()
    for scope, per_k in ft.items():
        if not isinstance(per_k, dict):
            continue
        by_k: dict[int, dict] = {}
        for m in per_k.values():
            if isinstance(m, dict) and "accuracy_mean" in m:
                by_k[int(m["k"])] = m
                ks.add(int(m["k"]))
        if by_k:
            table[scope] = by_k

    if not table:
        return None

    ordered = [s for s in FT_SCOPE_LABELS if s in table]
    ordered += [s for s in sorted(table) if s not in ordered]
    baseline = (split_res.get("zero_shot_baseline") or {}).get("accuracy")
    return ordered, sorted(ks), baseline, table


def _cross_domain_few_shot_md(cd_fs: dict[str, Any], heading: str) -> list[str]:
    """Render the cross-domain few-shot section as Markdown lines."""
    body: list[str] = []
    notes: list[str] = []

    for split_name, split_res in sorted(cd_fs.items()):
        view = _cross_domain_few_shot_view(split_res)
        if view is None:
            if isinstance(split_res, dict):
                reason = split_res.get("skipped") or split_res.get("error")
                if reason:
                    notes.append(f"- `{split_name}`: {reason}")
            continue
        scopes, ks, baseline, table = view

        body.extend([
            f"### Split: {split_name}",
            "",
            "| K | " + " | ".join(_ft_scope_label(s) for s in scopes) + " |",
            "|---|" + "|".join(["---"] * len(scopes)) + "|",
        ])
        if baseline is not None:
            body.append(
                "| 0 (zero-shot) | "
                + " | ".join(f"{baseline:.4f}" for _ in scopes)
                + " |"
            )
        for k in ks:
            cells = []
            for scope in scopes:
                m = table[scope].get(k)
                if m is None:
                    cells.append("---")
                    continue
                cell = f"{m['accuracy_mean']:.4f} +/- {m.get('accuracy_std', 0):.4f}"
                if baseline is not None:
                    cell += f" ({m['accuracy_mean'] - baseline:+.4f})"
                cells.append(cell)
            body.append(f"| {k} | " + " | ".join(cells) + " |")
        body.append("")

    if not body and not notes:
        return []

    lines = [heading, ""]
    if body:
        lines.extend([
            "Accuracy on the remaining samples of the held-out split "
            "(mean +/- std over trials; delta is versus the 0-shot baseline).",
            "",
            *body,
        ])
    if notes:
        lines.extend([*notes, ""])
    return lines


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

    cdom = results.get("cross_domain")
    if cdom and isinstance(cdom, dict):
        iid_acc = zs.get("accuracy") if isinstance(zs, dict) else None
        rows, notes = _cross_domain_rows(cdom, iid_acc)
        if rows or notes:
            lines.extend(["## Cross-Domain Generalization", ""])
            if iid_acc is not None:
                lines.extend([f"IID test accuracy: {iid_acc:.4f}", ""])
            if rows:
                lines.extend([
                    "| Split | Accuracy | Weighted F1 | Samples | Classes | Delta vs IID |",
                    "|-------|----------|-------------|---------|---------|--------------|",
                    *rows,
                    "",
                ])
            if notes:
                lines.extend([*notes, ""])

    cd_fs = results.get("cross_domain_few_shot")
    if cd_fs and isinstance(cd_fs, dict):
        lines.extend(
            _cross_domain_few_shot_md(cd_fs, "## Cross-Domain Few-Shot Fine-Tuning")
        )

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

    has_cross_domain = False
    for task_name, task_res in sorted(all_results.items()):
        cdom = task_res.get("cross_domain", {})
        if not isinstance(cdom, dict):
            continue
        iid_acc = task_res.get("zero_shot", {}).get("accuracy")
        rows, notes = _cross_domain_rows(cdom, iid_acc)
        if not rows and not notes:
            continue
        if not has_cross_domain:
            lines.extend(["## Cross-Domain Generalization", ""])
            has_cross_domain = True
        lines.extend([f"### Task: {task_name}", ""])
        if rows:
            lines.extend([
                "| Split | Accuracy | Weighted F1 | Samples | Classes | Delta vs IID |",
                "|-------|----------|-------------|---------|---------|--------------|",
                *rows,
                "",
            ])
        if notes:
            lines.extend([*notes, ""])

    has_cd_fs = False
    for task_name, task_res in sorted(all_results.items()):
        cd_fs = task_res.get("cross_domain_few_shot", {})
        if not isinstance(cd_fs, dict):
            continue
        section = _cross_domain_few_shot_md(cd_fs, f"### Task: {task_name}")
        if not section:
            continue
        if not has_cd_fs:
            lines.extend(["## Cross-Domain Few-Shot Fine-Tuning", ""])
            has_cd_fs = True
        # Split tables are one level deeper under the per-task heading.
        lines.extend(s.replace("### Split:", "#### Split:", 1) for s in section)

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

    cdom = results.get("cross_domain", {})
    valid_cdom = {k: v for k, v in cdom.items() if isinstance(v, dict) and "accuracy" in v}
    if valid_cdom:
        iid_acc = results.get("zero_shot", {}).get("accuracy")
        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Split & Accuracy & Weighted F1 & $N$ & $\\Delta$ vs IID \\\\",
            "\\hline",
        ])
        if iid_acc is not None:
            lines.append(
                f"IID test & {iid_acc:.4f} & "
                f"{results.get('zero_shot', {}).get('weighted_f1', 0):.4f} & --- & --- \\\\"
            )
        for split_name, res in sorted(valid_cdom.items()):
            safe_name = split_name.replace("_", "\\_")
            delta = f"{res['accuracy'] - iid_acc:+.4f}" if iid_acc is not None else "---"
            lines.append(
                f"{safe_name} & {res['accuracy']:.4f} & "
                f"{res.get('weighted_f1', 0):.4f} & {res.get('num_samples', '---')} & {delta} \\\\"
            )
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Cross-domain (cross-device / cross-environment) generalization.}",
            "\\end{table}",
            "",
        ])

    lines.extend(
        _cross_domain_few_shot_latex(results.get("cross_domain_few_shot", {}))
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _cross_domain_few_shot_latex_rows(
    cd_fs: dict[str, Any],
    scope_order: list[str],
    task_name: Optional[str] = None,
) -> list[str]:
    """LaTeX body rows for the cross-domain few-shot table.

    ``scope_order`` fixes the column layout so rows from several tasks can
    share one table.
    """
    rows: list[str] = []
    prefix = f"{_latex_escape(task_name)} & " if task_name else ""

    for split_name, split_res in sorted(cd_fs.items()):
        view = _cross_domain_few_shot_view(split_res)
        if view is None:
            continue
        _, ks, baseline, table = view
        safe_split = _latex_escape(split_name)

        if baseline is not None:
            rows.append(
                f"{prefix}{safe_split} & 0 & "
                + " & ".join(f"{baseline:.4f}" for _ in scope_order)
                + " \\\\"
            )
        for k in ks:
            cells = []
            for scope in scope_order:
                m = table.get(scope, {}).get(k)
                cells.append(
                    "---" if m is None
                    else f"{m['accuracy_mean']:.4f} $\\pm$ {m.get('accuracy_std', 0):.4f}"
                )
            rows.append(f"{prefix}{safe_split} & {k} & " + " & ".join(cells) + " \\\\")

    return rows


def _cross_domain_few_shot_scopes(cd_fs: dict[str, Any]) -> list[str]:
    """Union of fine-tuning scopes present, in a stable display order."""
    found: set[str] = set()
    for split_res in cd_fs.values():
        view = _cross_domain_few_shot_view(split_res)
        if view is not None:
            found.update(view[0])
    ordered = [s for s in FT_SCOPE_LABELS if s in found]
    return ordered + sorted(found - set(ordered))


def _cross_domain_few_shot_table(
    rows: list[str], scope_order: list[str], with_task_column: bool,
) -> list[str]:
    if not rows:
        return []
    col_spec = ("lll" if with_task_column else "ll") + "c" * len(scope_order)
    header_prefix = "Task & Split & $K$" if with_task_column else "Split & $K$"
    header = (
        header_prefix + " & "
        + " & ".join(_ft_scope_label(s) for s in scope_order) + " \\\\"
    )
    return [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header,
        "\\hline",
        *rows,
        "\\hline",
        "\\end{tabular}",
        "\\caption{Cross-domain few-shot fine-tuning accuracy "
        "(mean $\\pm$ std over trials; $K=0$ is the zero-shot baseline).}",
        "\\end{table}",
        "",
    ]


def _cross_domain_few_shot_latex(cd_fs: dict[str, Any]) -> list[str]:
    if not isinstance(cd_fs, dict) or not cd_fs:
        return []
    scope_order = _cross_domain_few_shot_scopes(cd_fs)
    rows = _cross_domain_few_shot_latex_rows(cd_fs, scope_order)
    return _cross_domain_few_shot_table(rows, scope_order, with_task_column=False)


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

    has_cdom = False
    for task_name, task_res in sorted(all_results.items()):
        cdom = task_res.get("cross_domain", {})
        if not isinstance(cdom, dict):
            continue
        valid_cdom = {k: v for k, v in cdom.items() if isinstance(v, dict) and "accuracy" in v}
        if not valid_cdom:
            continue
        if not has_cdom:
            lines.extend([
                "\\begin{table}[t]",
                "\\centering",
                "\\begin{tabular}{llccc}",
                "\\hline",
                "Task & Split & Accuracy & Weighted F1 & $\\Delta$ vs IID \\\\",
                "\\hline",
            ])
            has_cdom = True
        iid_acc = task_res.get("zero_shot", {}).get("accuracy")
        safe_task = task_name.replace("_", "\\_")
        for split_name, res in sorted(valid_cdom.items()):
            safe_split = split_name.replace("_", "\\_")
            delta = f"{res['accuracy'] - iid_acc:+.4f}" if iid_acc is not None else "---"
            lines.append(
                f"{safe_task} & {safe_split} & {res['accuracy']:.4f} & "
                f"{res.get('weighted_f1', 0):.4f} & {delta} \\\\"
            )
    if has_cdom:
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Cross-domain (cross-device / cross-environment) generalization.}",
            "\\end{table}",
            "",
        ])

    all_cd_fs = {
        t: r.get("cross_domain_few_shot", {})
        for t, r in all_results.items()
        if isinstance(r.get("cross_domain_few_shot"), dict)
    }
    if all_cd_fs:
        scope_order = _cross_domain_few_shot_scopes(
            {f"{t}/{s}": v for t, cd in all_cd_fs.items() for s, v in cd.items()}
        )
        fs_rows: list[str] = []
        for task_name, cd_fs in sorted(all_cd_fs.items()):
            fs_rows.extend(
                _cross_domain_few_shot_latex_rows(cd_fs, scope_order, task_name)
            )
        lines.extend(
            _cross_domain_few_shot_table(fs_rows, scope_order, with_task_column=True)
        )

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


def _prune_phantom_classes(
    class_texts: dict[int, list[str]],
    loader: Optional[DataLoader],
) -> dict[int, list[str]]:
    """Drop text prototypes for classes that have no samples in *loader*.

    Keeping prototypes for absent classes would let the classifier predict
    labels that cannot occur, depressing accuracy for reasons unrelated to
    the model.
    """
    ds = getattr(loader, "dataset", None) if loader is not None else None
    if ds is None or not hasattr(ds, "int_labels"):
        return dict(class_texts)

    present = {int(x) for x in np.unique(ds.int_labels)}
    phantom = set(class_texts) - present
    if phantom:
        logger.warning(
            "Removing %d phantom class IDs from class_texts (no data): %s",
            len(phantom), sorted(phantom),
        )
    return {k: v for k, v in class_texts.items() if k in present}


def _run_cross_domain_eval(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    text_projection: Optional[torch.nn.Module],
    csi_projection: Optional[torch.nn.Module],
    config: dict[str, Any],
    dataset: str,
    task: str,
    class_texts: dict[int, list[str]],
    out_dir: Path,
    device: torch.device,
    split_names: list[str],
) -> dict[str, Any]:
    """Zero-shot evaluation on held-out domain splits.

    Each split file (e.g. ``splits/cross_env.json``) defines a test set drawn
    from environments or devices unseen during training.  Comparing accuracy
    here against the IID ``test`` split quantifies cross-domain degradation.
    """
    cd_dir = out_dir / "cross_domain"
    cd_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _loader_config_from_experiment(config, task=task, dataset_name=dataset)
    results: dict[str, Any] = {}

    for split_name in split_names:
        try:
            loader_cfg = {
                **base_cfg,
                "test_split": split_name,
                "build_splits": ["test"],
            }
            loader = build_dataloaders(loader_cfg)["test"]

            split_texts = _prune_phantom_classes(class_texts, loader)
            if not split_texts:
                raise ValueError(
                    f"No class texts overlap the samples in split '{split_name}'"
                )

            metrics = _strip_heavy(
                run_zero_shot_evaluation(
                    model=model,
                    dataloader=loader,
                    class_texts=split_texts,
                    text_encoder=text_encoder,
                    text_projection=text_projection,
                    csi_projection=csi_projection,
                    device=str(device),
                )
            )
            metrics["split"] = split_name
            metrics["num_samples"] = len(loader.dataset)
            metrics["num_classes"] = len(split_texts)
            results[split_name] = metrics

            logger.info(
                "Cross-domain [%s / %s]: accuracy=%.4f  weighted_f1=%.4f  "
                "(%d samples, %d classes)",
                task, split_name, metrics.get("accuracy", float("nan")),
                metrics.get("weighted_f1", float("nan")),
                metrics["num_samples"], metrics["num_classes"],
            )
        except FileNotFoundError as e:
            logger.warning(
                "Cross-domain split '%s' not available for task %s: %s",
                split_name, task, e,
            )
            results[split_name] = {"skipped": "split file not found", "detail": str(e)}
        except Exception as e:
            logger.warning(
                "Cross-domain split '%s' failed for task %s: %s", split_name, task, e,
            )
            results[split_name] = {"error": str(e)}

        _save_json(results[split_name], cd_dir / f"{split_name}.json")

    num_ok = sum(1 for v in results.values() if "accuracy" in v)
    _save_json(
        {
            "dataset": dataset,
            "task": task,
            "num_successful": num_ok,
            "num_failed": len(results) - num_ok,
            "splits": results,
        },
        cd_dir / "summary.json",
    )
    return results


def _gather_split_tensors(loader: DataLoader) -> tuple[torch.Tensor, np.ndarray]:
    """Materialise a loader's CSI samples and integer labels on the CPU."""
    csi_chunks: list[torch.Tensor] = []
    label_chunks: list[np.ndarray] = []
    for batch in loader:
        csi_chunks.append(batch["csi"].cpu())
        label_chunks.append(batch["labels"].cpu().numpy())
    if not csi_chunks:
        raise ValueError(
            "Split contains no usable samples (check that the H5 files "
            "referenced by the split file exist)"
        )
    csi = torch.cat(csi_chunks, dim=0)
    labels = np.concatenate(label_chunks, axis=0)

    n_bytes = csi.element_size() * csi.nelement()
    if n_bytes > 8 * 1024 ** 3:
        logger.warning(
            "Held-out split occupies %.1f GiB of host RAM; consider reducing "
            "target_time_steps or evaluating fewer splits at once.",
            n_bytes / 1024 ** 3,
        )
    return csi, labels


def _tensor_batches(
    csi: torch.Tensor, labels: np.ndarray, batch_size: int,
) -> list[dict[str, Any]]:
    """Wrap in-memory tensors as loader-style batches.

    Lets the zero-shot baseline reuse the samples already gathered for
    fine-tuning instead of reading every H5 file a second time.  The slices
    are views, so this costs no extra memory.
    """
    return [
        {
            "csi": csi[start : start + batch_size],
            "labels": torch.from_numpy(labels[start : start + batch_size]),
        }
        for start in range(0, len(labels), batch_size)
    ]


def _run_cross_domain_few_shot(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    text_projection: Optional[torch.nn.Module],
    csi_projection: Optional[torch.nn.Module],
    config: dict[str, Any],
    dataset: str,
    task: str,
    class_texts: dict[int, list[str]],
    out_dir: Path,
    device: torch.device,
    split_names: list[str],
    ft_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Few-shot fine-tuning on each held-out domain split.

    For every split, K labelled samples per class are used to adapt the CSI
    branch (with a fresh linear head) and the remaining samples are scored.
    The split's zero-shot accuracy is recorded alongside as the K=0 baseline.
    """
    fs_dir = out_dir / "cross_domain_few_shot"
    fs_dir.mkdir(parents=True, exist_ok=True)

    if csi_projection is None:
        csi_projection = getattr(model, "csi_projection", None)

    base_cfg = _loader_config_from_experiment(config, task=task, dataset_name=dataset)
    results: dict[str, Any] = {}

    for split_name in split_names:
        try:
            loader_cfg = {
                **base_cfg,
                "test_split": split_name,
                "build_splits": ["test"],
            }
            loader = build_dataloaders(loader_cfg)["test"]
            split_texts = _prune_phantom_classes(class_texts, loader)
            batch_size = int(base_cfg.get("batch_size", 128))

            # Read the split once; the baseline and the fine-tuning sweep
            # both work off these tensors.
            csi, labels = _gather_split_tensors(loader)
            num_classes = int(len(np.unique(labels)))

            if num_classes < 2:
                logger.warning(
                    "Split '%s' of task %s has a single class; few-shot "
                    "classification needs at least two.", split_name, task,
                )
                results[split_name] = {
                    "skipped": "fewer than 2 classes present in the split",
                    "split": split_name,
                    "num_samples": int(len(labels)),
                    "num_classes": num_classes,
                }
                _save_json(_numpy_json(results[split_name]), fs_dir / f"{split_name}.json")
                continue

            # The baseline needs the text branch; a failure there must not
            # discard the fine-tuning numbers, which use no text at all.
            baseline: dict[str, Any] = {}
            if split_texts:
                try:
                    zs = _strip_heavy(
                        run_zero_shot_evaluation(
                            model=model,
                            dataloader=_tensor_batches(csi, labels, batch_size),
                            class_texts=split_texts,
                            text_encoder=text_encoder,
                            text_projection=text_projection,
                            csi_projection=csi_projection,
                            device=str(device),
                        )
                    )
                    baseline = {
                        "accuracy": zs.get("accuracy"),
                        "weighted_f1": zs.get("weighted_f1"),
                    }
                except Exception as e:
                    logger.warning(
                        "Zero-shot baseline failed for split '%s': %s", split_name, e,
                    )
                    baseline = {"error": str(e)}
            else:
                logger.warning(
                    "No class texts overlap split '%s'; reporting fine-tuning "
                    "without a 0-shot baseline.", split_name,
                )

            finetuner = FewShotFinetune(
                shots=ft_cfg.get("shots", (1, 5, 10, 20)),
                scopes=ft_cfg.get("scopes", ("projection", "encoder")),
                n_trials=ft_cfg.get("n_trials", 5),
                epochs=ft_cfg.get("epochs", 100),
                lr=ft_cfg.get("lr", 1e-3),
                weight_decay=ft_cfg.get("weight_decay", 1e-4),
                batch_size=batch_size,
                seed=ft_cfg.get("seed", 42),
            )
            ft_metrics = finetuner.evaluate(
                csi=csi,
                labels=labels,
                encoder=model.csi_encoder,
                projection=csi_projection,
                device=device,
            )

            results[split_name] = {
                "split": split_name,
                "num_samples": int(len(labels)),
                "num_classes": num_classes,
                "zero_shot_baseline": baseline,
                "finetune": ft_metrics,
            }

            for scope, per_k in ft_metrics.items():
                for k_label, m in sorted(per_k.items(), key=lambda kv: kv[1]["k"]):
                    logger.info(
                        "Cross-domain few-shot [%s / %s / %s] %s: "
                        "accuracy=%.4f +/- %.4f",
                        task, split_name, scope, k_label,
                        m["accuracy_mean"], m["accuracy_std"],
                    )
        except FileNotFoundError as e:
            logger.warning(
                "Cross-domain few-shot split '%s' not available for task %s: %s",
                split_name, task, e,
            )
            results[split_name] = {"skipped": "split file not found", "detail": str(e)}
        except Exception as e:
            logger.warning(
                "Cross-domain few-shot split '%s' failed for task %s: %s",
                split_name, task, e,
            )
            results[split_name] = {"error": str(e)}

        _save_json(_numpy_json(results[split_name]), fs_dir / f"{split_name}.json")

    num_ok = sum(1 for v in results.values() if "finetune" in v)
    _save_json(
        {
            "dataset": dataset,
            "task": task,
            "shots": list(ft_cfg.get("shots", (1, 5, 10, 20))),
            "scopes": list(ft_cfg.get("scopes", ("projection", "encoder"))),
            "n_trials": ft_cfg.get("n_trials", 5),
            "epochs": ft_cfg.get("epochs", 100),
            "num_successful": num_ok,
            "num_failed": len(results) - num_ok,
            "splits": _numpy_json(results),
        },
        fs_dir / "summary.json",
    )
    return results


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
    cross_domain_splits: Optional[list[str]] = None,
    cross_domain_ft_cfg: Optional[dict[str, Any]] = None,
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

    iid_protocols = ("zero_shot", "visualization", "cross_task", "cross_dataset", "retrieval", "few_shot")
    needs_iid_loaders = any(p in eval_protocols for p in iid_protocols)
    needs_class_texts = needs_iid_loaders or any(
        p in eval_protocols for p in ("cross_domain", "cross_domain_few_shot")
    )
    class_texts: dict = {}
    class_texts_full: dict = {}
    loaders: dict = {}
    if needs_class_texts:
        reg_path = _resolve_registry_path(task, registry_dir, config, dataset_name=dataset)
        class_texts_full = build_class_texts(dataset, task, registry_path=reg_path)
        class_texts = dict(class_texts_full)

    if needs_iid_loaders:
        loader_cfg = _loader_config_from_experiment(config, task=task, dataset_name=dataset)
        loaders = build_dataloaders(loader_cfg)

        # Remove phantom classes (text prototypes with zero data samples)
        class_texts = _prune_phantom_classes(class_texts, loaders.get("test"))

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

    # --- Cross-domain (cross-device / cross-environment) ---
    if "cross_domain" in eval_protocols:
        results["cross_domain"] = _run_cross_domain_eval(
            model=model,
            text_encoder=text_encoder,
            text_projection=text_projection,
            csi_projection=csi_projection,
            config=config,
            dataset=dataset,
            task=task,
            class_texts=class_texts_full,
            out_dir=out_dir,
            device=device,
            split_names=list(cross_domain_splits or DEFAULT_CROSS_DOMAIN_SPLITS),
        )

    # --- Cross-domain few-shot fine-tuning ---
    if "cross_domain_few_shot" in eval_protocols:
        results["cross_domain_few_shot"] = _run_cross_domain_few_shot(
            model=model,
            text_encoder=text_encoder,
            text_projection=text_projection,
            csi_projection=csi_projection,
            config=config,
            dataset=dataset,
            task=task,
            class_texts=class_texts_full,
            out_dir=out_dir,
            device=device,
            split_names=list(cross_domain_splits or DEFAULT_CROSS_DOMAIN_SPLITS),
            ft_cfg=cross_domain_ft_cfg or {},
        )

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
        emb_path = out_dir / "tsne_data.npz"
        np.savez(emb_path, csi_embeddings=csi_np, text_prototypes=proto, labels=labels)
        logger.info("Saved embedding data to %s", emb_path)
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
        default=["zero_shot", "ambient_qa", "cross_task", "cross_dataset", "visualization"],
        help="Protocols to run. Options: zero_shot, ambient_qa, cross_task, "
             "cross_dataset, cross_domain, cross_domain_few_shot, "
             "visualization, retrieval, few_shot",
    )
    parser.add_argument(
        "--cross_domain_splits",
        nargs="+",
        default=list(DEFAULT_CROSS_DOMAIN_SPLITS),
        help="Split names evaluated by the cross_domain and "
             "cross_domain_few_shot protocols. Each must resolve to "
             "<task_root>/splits/<name>.json.",
    )
    parser.add_argument(
        "--cross_domain_shots",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="Labelled samples per class used to adapt on each held-out "
             "domain split (cross_domain_few_shot).",
    )
    parser.add_argument(
        "--cross_domain_ft_scopes",
        nargs="+",
        default=["projection", "encoder"],
        help="Fine-tuning scopes: 'projection' trains csi_projection + head "
             "with a frozen encoder, 'encoder' trains the whole CSI branch.",
    )
    parser.add_argument(
        "--cross_domain_ft_epochs",
        type=int,
        default=100,
        help="Full-batch gradient steps on the support set per fine-tune run.",
    )
    parser.add_argument("--cross_domain_ft_lr", type=float, default=1e-3)
    parser.add_argument("--cross_domain_ft_weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--cross_domain_ft_trials",
        type=int,
        default=5,
        help="Random support draws per shot count; results are mean +/- std.",
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

    cross_domain_ft_cfg = {
        "shots": tuple(args.cross_domain_shots),
        "scopes": tuple(args.cross_domain_ft_scopes),
        "n_trials": args.cross_domain_ft_trials,
        "epochs": args.cross_domain_ft_epochs,
        "lr": args.cross_domain_ft_lr,
        "weight_decay": args.cross_domain_ft_weight_decay,
    }

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

    model, text_encoder, text_projection, csi_projection = build_model_from_config(config, device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # Checkpoint directories are sometimes shared by several training runs, so
    # record which model this evaluation actually scored.
    val_loss = ckpt.get("val_loss") if isinstance(ckpt, dict) else None
    logger.info(
        "Loaded %s | encoder=%s text_encoder=%s method=%s task=%s epoch=%s val_loss=%s",
        ckpt_path,
        config.get("csi_encoder", {}).get("name", "resnet18_1d"),
        config.get("text_encoder", {}).get("name", "clip_vit_b32"),
        config.get("method"),
        config.get("task") or config.get("tasks"),
        ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss,
    )

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
                    cross_domain_splits=args.cross_domain_splits,
                    cross_domain_ft_cfg=cross_domain_ft_cfg,
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
        cross_domain_splits=args.cross_domain_splits,
        cross_domain_ft_cfg=cross_domain_ft_cfg,
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
