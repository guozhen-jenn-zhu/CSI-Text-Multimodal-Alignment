"""Run evaluation protocols on a trained model checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

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
    LinearProbe,
    run_zero_shot_evaluation,
)
from models.alignment import build_alignment_model
from models.encoders import ProjectionHead, build_csi_encoder, build_text_encoder
from training.train import _apply_overrides, _merge_configs

logger = logging.getLogger(__name__)


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
    csi_encoder = build_csi_encoder(
        name=enc_cfg.get("name", "resnet18_1d"),
        input_channels=input_channels,
        input_subcarriers=input_subcarriers,
        input_timesteps=input_timesteps,
        embed_dim=embed_dim,
    )

    txt_cfg = config.get("text_encoder", {})
    text_encoder = build_text_encoder(name=txt_cfg.get("name", "clip_vit_b32"))
    text_embed_dim = text_encoder.embed_dim

    csi_projection = ProjectionHead(
        input_dim=embed_dim,
        output_dim=proj_cfg.get("dim", 512),
        hidden_dim=proj_cfg.get("hidden_dim", 1024),
        num_layers=proj_cfg.get("num_layers", 2),
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
    """Resolve the task_root path, checking config overrides first."""
    dn = dataset_name if dataset_name is not None else config["dataset"]
    t = task if task is not None else config.get("task", "har")

    if dn == config["dataset"] and t == config.get("task") and config.get("task_root"):
        return config["task_root"]

    ds_cfg = config.get("dataset_config", {}) if dn == config["dataset"] else _dataset_yaml_defaults(dn)
    data_root = ds_cfg.get("data_root", f"data/processed/{dn}")

    tasks = ds_cfg.get("tasks", {})
    if isinstance(tasks, dict) and t in tasks:
        folder = tasks[t].get("folder", t)
    else:
        folder = t

    from pathlib import Path
    return str(Path(data_root) / folder)


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


def _register_ambient_answers(
    aq: AmbientQA, config: dict, registry_dir: Optional[str] = None,
) -> None:
    """Register answer texts for ambient QA attributes.

    Tries to load from per-task registries first (uses the first template text
    per class). Falls back to raw templates.
    """
    task_attr_map = {
        "har": "activity",
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


def _run_ambient_qa_multitask(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    text_projection: Optional[torch.nn.Module],
    csi_projection: torch.nn.Module,
    config: dict,
    device: torch.device,
    registry_dir: Optional[str] = None,
) -> dict[str, Any]:
    cfg = _loader_config_from_experiment(config, task="multi_task", multi_task=True)
    loaders = build_dataloaders(cfg)
    test_loader = loaders["test"]
    aq = AmbientQA(text_encoder, text_projection, device=str(device))
    _register_ambient_answers(aq, config, registry_dir=registry_dir)
    all_task_labels: dict[str, list[np.ndarray]] = {}
    model.eval()
    csi_encoder = model.csi_encoder
    emb_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            csi = batch["csi"].to(device)
            feat = csi_encoder(csi)
            if csi_projection is not None:
                feat = csi_projection(feat)
            emb_chunks.append(feat.cpu())
            tl = batch.get("task_labels")
            if tl is None:
                continue
            for k, v in tl.items():
                all_task_labels.setdefault(k, []).append(v.numpy())
    if not emb_chunks:
        return {"error": "no_batches", "message": "Ambient QA requires multi-task CSI-Bench test data."}
    csi_emb = torch.cat(emb_chunks, dim=0)
    tasks_and_labels: dict[str, np.ndarray] = {}
    key_map = {"har": "activity", "user_id": "user_identity", "proximity": "proximity"}
    for raw_k, arr_list in all_task_labels.items():
        attr = key_map.get(raw_k, raw_k)
        tasks_and_labels[attr] = np.concatenate(arr_list, axis=0)
    return aq.evaluate_multi_attribute(csi_emb, tasks_and_labels, list(tasks_and_labels.keys()))


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
    drop = {"csi_embeddings", "text_embeddings", "prototypes", "failure_details"}
    out = {k: v for k, v in m.items() if k not in drop}
    if "per_attribute" in out:
        out["per_attribute"] = {
            a: {x: y for x, y in d.items() if x != "predictions"}
            for a, d in out["per_attribute"].items()
        }
    if "predictions" in out:
        preds = out["predictions"]
        out["predictions"] = preds[:200].tolist() if hasattr(preds, "tolist") else preds
    return _numpy_json(out)


def write_latex_tables(results: dict[str, Any], path: Path) -> None:
    lines = [
        "% Auto-generated by run_evaluate.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Protocol & Accuracy & Weighted F1 \\\\",
        "\\hline",
    ]
    zs = results.get("zero_shot")
    if zs and "accuracy" in zs:
        lines.append(
            f"Zero-shot & {zs['accuracy']:.4f} & {zs.get('weighted_f1', 0):.4f} \\\\"
        )
    aq = results.get("ambient_qa")
    if aq and "joint_accuracy" in aq:
        lines.append(
            f"Ambient QA (joint) & {aq['joint_accuracy']:.4f} & --- \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\caption{Evaluation summary.}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


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
        help="Protocols to run",
    )
    parser.add_argument("--output_dir", default="results", help="Directory for metrics and tables")
    parser.add_argument(
        "--registry_dir",
        default=None,
        help="Directory containing per-task pair registries (e.g. data/generated_texts/csibench). "
             "If provided, class_texts are loaded from registries instead of raw templates.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    config = _merge_configs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = _merge_configs(ckpt["config"])
    if args.override:
        _apply_overrides(config, args.override)

    model, text_encoder, text_projection, csi_projection = build_model_from_config(config, device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    dataset = config["dataset"]
    task = config.get("task", "har")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"config": {"dataset": dataset, "task": task, "method": config.get("method")}}

    reg_path = None
    if args.registry_dir:
        candidate = Path(args.registry_dir) / task / "pair_registry.json"
        if candidate.is_file():
            reg_path = str(candidate)
    class_texts = build_class_texts(dataset, task, registry_path=reg_path)
    loader_cfg = _loader_config_from_experiment(config)
    loaders = build_dataloaders(loader_cfg)

    zs_full: Optional[dict[str, Any]] = None
    if "zero_shot" in args.eval_protocols or "visualization" in args.eval_protocols:
        zs_full = run_zero_shot_evaluation(
            model=model,
            dataloader=loaders["test"],
            class_texts=class_texts,
            text_encoder=text_encoder,
            text_projection=text_projection,
            csi_projection=csi_projection,
            device=str(device),
        )
    if "zero_shot" in args.eval_protocols and zs_full is not None:
        results["zero_shot"] = _strip_heavy(zs_full)

    if "ambient_qa" in args.eval_protocols and dataset == "csibench":
        try:
            aq_res = _run_ambient_qa_multitask(
                model, text_encoder, text_projection, csi_projection, config, device,
                registry_dir=args.registry_dir,
            )
            results["ambient_qa"] = _strip_heavy(aq_res)
        except Exception as e:
            logger.warning("Ambient QA skipped: %s", e)
            results["ambient_qa"] = {"error": str(e)}

    if "cross_task" in args.eval_protocols and dataset == "csibench":
        ct = CrossTaskTransfer(text_encoder, text_projection, device=str(device))
        target_loaders: dict[str, DataLoader] = {}
        target_texts: dict[str, dict[int, list[str]]] = {}
        pairs = CrossTaskTransfer.TRANSFER_PAIRS.get("csibench", [])
        for _, tgt in pairs:
            if tgt not in target_loaders:
                cfg_t = _loader_config_from_experiment(config, task=tgt, multi_task=False)
                ld = build_dataloaders(cfg_t)
                target_loaders[tgt] = ld["test"]
                tgt_reg = None
                if args.registry_dir:
                    tgt_cand = Path(args.registry_dir) / tgt / "pair_registry.json"
                    if tgt_cand.is_file():
                        tgt_reg = str(tgt_cand)
                target_texts[tgt] = build_class_texts("csibench", tgt, registry_path=tgt_reg)
        results["cross_task"] = _numpy_json(
            ct.evaluate_all_pairs(
                "csibench",
                model,
                target_loaders,
                target_texts,
                csi_projection,
            )
        )

    if "cross_dataset" in args.eval_protocols:
        src = cross_dataset_source_name(config)
        cdt = CrossDatasetTransfer(text_encoder, text_projection, device=str(device))
        target_configs: dict[str, dict[str, Any]] = {}
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
                    har_reg = None
                    if args.registry_dir:
                        har_cand = Path(args.registry_dir) / "har" / "pair_registry.json"
                        if har_cand.is_file():
                            har_reg = str(har_cand)
                    target_configs[tgt] = {
                        "dataloader": ld,
                        "class_texts": build_class_texts("csibench", "har", registry_path=har_reg),
                    }
                elif tgt == "xrf55_gesture_subset":
                    cfg_t = _loader_config_from_experiment(
                        config, task="sign_recognition", dataset_name="signfi",
                    )
                    ld = build_dataloaders(cfg_t)["test"]
                    target_configs[tgt] = {
                        "dataloader": ld,
                        "class_texts": build_class_texts("signfi", "sign_recognition"),
                    }
            except Exception as e:
                logger.warning("Cross-dataset target %s: %s", tgt, e)
        results["cross_dataset"] = _numpy_json(
            cdt.evaluate_all_directions(model, src, target_configs, csi_projection)
        )

    if "visualization" in args.eval_protocols and zs_full is not None:
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

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_latex_tables(results, out_dir / "tables.tex")
    logger.info("Wrote %s and %s", metrics_path, out_dir / "tables.tex")


if __name__ == "__main__":
    main()
