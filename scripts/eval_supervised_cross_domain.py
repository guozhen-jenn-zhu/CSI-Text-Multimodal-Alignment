#!/usr/bin/env python3
"""Score a trained supervised CSI classifier on the held-out domain splits.

``scripts/train_supervised.py`` trains a ResNet-18 1D + linear head on CSI
alone and reports the IID ``test`` split only.  This script reuses that
checkpoint to also measure ``cross_env`` and ``cross_device`` accuracy, giving
the CSI-only counterpart to the ``cross_domain`` protocol that
``scripts/run_evaluate.py`` runs for the multimodal-aligned models.  With both
sides scored on the same splits, the accuracy drop from IID to a held-out
environment or device is directly comparable.

Architecture hyper-parameters are recovered from the checkpoint tensors (and
cross-checked against ``supervised_results.json`` when it sits next to the
weights), so no flags need to be repeated from the training run.

Usage::

    python -m scripts.eval_supervised_cross_domain \\
        --checkpoint_dir "$MODEL_ROOT/results/supervised_baselines/csibench_har" \\
        --task_root "$DATA_ROOT/csibench/multitask/har"

Results mirror the aligned layout: ``cross_domain/<split>.json`` plus a
``metrics.json`` holding the IID entry and every domain split.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from scripts.train_supervised import (
    SupervisedCSIClassifier,
    build_dataloader,
    evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IID_SPLIT = "test"


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Read the plain state_dict that train_supervised.py saves."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError, ValueError):
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state_dict")
    return state


def _infer_architecture(
    state: Dict[str, torch.Tensor], input_channels: int,
) -> Dict[str, int]:
    """Recover num_classes / embed_dim / base_width / subcarriers from weights."""
    try:
        num_classes, embed_dim = state["classifier.weight"].shape
        base_width, in_chan = state["encoder.conv1.weight"].shape[:2]
    except KeyError as e:
        raise ValueError(
            f"Checkpoint is missing {e}; expected a SupervisedCSIClassifier "
            "state_dict from scripts/train_supervised.py"
        ) from e

    if in_chan % input_channels != 0:
        raise ValueError(
            f"conv1 expects {in_chan} input channels, which is not divisible by "
            f"--input_channels {input_channels}"
        )

    return {
        "num_classes": int(num_classes),
        "embed_dim": int(embed_dim),
        "base_width": int(base_width),
        "target_subcarriers": in_chan // input_channels,
    }


def _check_label_mapping(task_root: Path, allow_missing: bool) -> Optional[dict]:
    """Class indices must not be re-derived per split.

    ``CSITextDataset`` only keeps indices stable across splits when
    ``metadata/label_mapping.json`` exists; without it each split falls back to
    sorted-unique assignment, so a domain split that omits a class would shift
    every later index and score the classifier against the wrong columns.
    """
    lm_path = task_root / "metadata" / "label_mapping.json"
    if lm_path.is_file():
        with open(lm_path) as f:
            return json.load(f)

    msg = (
        f"No label mapping at {lm_path}. Class indices would be rebuilt from "
        "each split's own labels, so a split missing a class would silently "
        "misalign the classifier outputs."
    )
    if not allow_missing:
        raise FileNotFoundError(
            msg + " Pass --allow_missing_label_map only if every split is known "
            "to contain all classes."
        )
    logger.warning("%s Continuing because --allow_missing_label_map was set.", msg)
    return None


def _score_split(
    model: nn.Module,
    task_root: str,
    split: str,
    h5_key: str,
    loader_kwargs: Dict[str, Any],
    num_classes: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run the classifier over one split and collect comparable metrics."""
    loader = build_dataloader(
        task_root=task_root, split=split, h5_key=h5_key, **loader_kwargs
    )
    dataset = loader.dataset
    ds_classes = getattr(dataset, "num_classes", None)
    if ds_classes is not None and int(ds_classes) != num_classes:
        raise ValueError(
            f"split '{split}' maps labels into {ds_classes} classes but the "
            f"classifier has {num_classes} outputs"
        )

    # CSITextDataset drops IDs whose H5 file is absent, so a partially synced
    # dataset would otherwise be scored on a silent subset of the split.
    n_declared = len(dataset.loader.get_split_ids(split))
    n_loaded = len(dataset.sample_ids)
    n_missing = n_declared - n_loaded
    if n_missing > 0:
        logger.warning(
            "Split '%s': %d of %d samples have no H5 file on disk and were "
            "dropped; the accuracy below covers only the %d loadable samples",
            split, n_missing, n_declared, n_loaded,
        )
    if n_loaded == 0:
        raise ValueError(
            f"split '{split}' has no loadable samples "
            f"({n_declared} IDs declared, all missing their H5 files)"
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss, accuracy, preds, labels = evaluate(model, loader, criterion, device)

    present = sorted({int(x) for x in labels.tolist()})
    per_class = f1_score(labels, preds, labels=present, average=None, zero_division=0)

    return {
        "split": split,
        "accuracy": float(accuracy),
        "weighted_f1": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, preds, labels=present, average="macro", zero_division=0)
        ),
        "per_class_f1": {int(c): float(v) for c, v in zip(present, per_class)},
        "loss": float(loss),
        "num_samples": int(len(labels)),
        "num_split_ids": n_declared,
        "num_missing_files": n_missing,
        "num_classes": len(present),
        "num_model_classes": num_classes,
    }


def _check_iid_reproduction(
    iid_result: Optional[Dict[str, Any]],
    train_results: Dict[str, Any],
    tol: float,
) -> Optional[Dict[str, Any]]:
    """Re-scoring the IID split must reproduce the accuracy training reported.

    This is the end-to-end evidence that the weights, the H5 files and the label
    map all loaded the way they did during training: a wrong checkpoint, a
    shifted label map or mis-shaped CSI all move this number.
    """
    recorded = train_results.get("test_accuracy")
    if iid_result is None or not isinstance(recorded, (int, float)):
        return None
    measured = iid_result.get("accuracy")
    if not isinstance(measured, (int, float)):
        return None

    delta = float(measured) - float(recorded)
    consistent = abs(delta) <= tol
    if consistent:
        logger.info(
            "IID check: accuracy %.4f matches the %.4f recorded at training time "
            "(delta %+.4f)", measured, recorded, delta,
        )
    else:
        logger.warning(
            "IID check FAILED: accuracy %.4f but training recorded %.4f "
            "(delta %+.4f, tolerance %.4f). The cross-domain numbers below are "
            "suspect: verify the checkpoint belongs to this task, that "
            "metadata/label_mapping.json is unchanged, and that h5_key / "
            "target_subcarriers / target_time_steps match the training config.",
            measured, recorded, delta, tol,
        )

    return {
        "measured": float(measured),
        "recorded_at_training": float(recorded),
        "delta": delta,
        "tolerance": tol,
        "consistent": consistent,
    }


def _summary_md(
    results: Dict[str, Any],
    checkpoint: Path,
    task_root: str,
    num_classes: int,
    iid_check: Optional[Dict[str, Any]] = None,
) -> str:
    iid = results.get(IID_SPLIT, {})
    iid_acc = iid.get("accuracy")

    lines = [
        "# Supervised CSI-Only Cross-Domain Evaluation",
        "",
        f"- **Checkpoint**: `{checkpoint}`",
        f"- **Task root**: `{task_root}`",
        "- **Encoder**: ResNet-18 1D + linear head (no text supervision)",
        f"- **Classes**: {num_classes}",
        "",
        "| Split | Accuracy | Weighted F1 | Samples | Classes | Delta vs IID |",
        "|-------|----------|-------------|---------|---------|--------------|",
    ]

    notes: List[str] = []
    ordered = [IID_SPLIT] + sorted(k for k in results if k != IID_SPLIT)
    for split in ordered:
        res = results.get(split)
        if res is None:
            continue
        if "accuracy" not in res:
            notes.append(f"- `{split}`: {res.get('skipped') or res.get('error')}")
            continue
        if split == IID_SPLIT or iid_acc is None:
            delta = "---"
        else:
            delta = f"{res['accuracy'] - iid_acc:+.4f}"
        label = "IID test" if split == IID_SPLIT else split
        lines.append(
            f"| {label} | {res['accuracy']:.4f} | {res['weighted_f1']:.4f} | "
            f"{res['num_samples']} | {res['num_classes']} | {delta} |"
        )

    if notes:
        lines.extend(["", "Unavailable splits:", ""] + notes)

    if iid_check:
        verdict = "matches" if iid_check["consistent"] else "DOES NOT MATCH"
        lines.extend([
            "",
            f"IID accuracy {verdict} the {iid_check['recorded_at_training']:.4f} "
            f"recorded at training time (delta {iid_check['delta']:+.4f}, "
            f"tolerance {iid_check['tolerance']:.4f}).",
        ])
        if not iid_check["consistent"]:
            lines.append(
                "Treat the cross-domain rows as unverified until this is resolved."
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint_dir",
        help="Supervised results directory holding best.pt (and "
             "supervised_results.json, used to recover the training config)",
    )
    src.add_argument("--checkpoint", help="Path to a supervised best.pt")
    parser.add_argument(
        "--task_root",
        default=None,
        help="Task directory with metadata/ and splits/ "
             "(default: config.task_root from supervised_results.json)",
    )
    parser.add_argument("--h5_key", default=None, help="Default: from training config")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[IID_SPLIT, "cross_env", "cross_device"],
        help=f"Splits to score; '{IID_SPLIT}' is recorded as the IID reference",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <checkpoint_dir>/cross_domain_eval",
    )
    parser.add_argument(
        "--iid_tol",
        type=float,
        default=0.005,
        help="Accepted drift between the re-scored IID accuracy and the one "
             "recorded in supervised_results.json",
    )
    parser.add_argument("--input_channels", type=int, default=None)
    parser.add_argument("--target_time_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--allow_missing_label_map",
        action="store_true",
        help="Proceed even without metadata/label_mapping.json (unsafe: class "
             "indices are then rebuilt per split)",
    )
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        ckpt_dir = ckpt_path.parent
    else:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_path = ckpt_dir / "best.pt"

    if not ckpt_path.is_file():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 1

    train_cfg: Dict[str, Any] = {}
    train_results: Dict[str, Any] = {}
    train_results_path = ckpt_dir / "supervised_results.json"
    if train_results_path.is_file():
        with open(train_results_path) as f:
            train_results = json.load(f)
        train_cfg = train_results.get("config", {}) or {}
        logger.info("Recovered training config from %s", train_results_path)

    task_root = args.task_root or train_cfg.get("task_root")
    if not task_root:
        logger.error(
            "No --task_root given and none recorded in %s", train_results_path
        )
        return 1
    if not Path(task_root).is_dir():
        logger.error("task_root does not exist: %s", task_root)
        return 1

    # The recorded path is absolute on the training host, so only the task name
    # is meaningful when comparing against a local checkout.
    trained_on = train_cfg.get("task_root")
    if trained_on and Path(trained_on).name != Path(task_root).name:
        logger.warning(
            "Checkpoint was trained on task '%s' (%s) but is being scored on "
            "'%s'; confirm this is the pairing you want",
            Path(trained_on).name, trained_on, task_root,
        )

    h5_key = args.h5_key or train_cfg.get("h5_key", "auto")
    input_channels = args.input_channels or int(train_cfg.get("input_channels", 1))
    target_time_steps = args.target_time_steps or int(
        train_cfg.get("target_time_steps", 500)
    )

    state = _load_state_dict(ckpt_path)
    arch = _infer_architecture(state, input_channels)
    logger.info(
        "Checkpoint architecture: num_classes=%d embed_dim=%d base_width=%d "
        "subcarriers=%d",
        arch["num_classes"], arch["embed_dim"], arch["base_width"],
        arch["target_subcarriers"],
    )

    for key, inferred in (
        ("num_classes", arch["num_classes"]),
        ("embed_dim", arch["embed_dim"]),
        ("base_width", arch["base_width"]),
        ("target_subcarriers", arch["target_subcarriers"]),
    ):
        recorded = train_cfg.get(key)
        if recorded is not None and int(recorded) != inferred:
            logger.warning(
                "Training config says %s=%s but the weights say %s; using the weights",
                key, recorded, inferred,
            )

    try:
        _check_label_mapping(Path(task_root), args.allow_missing_label_map)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedCSIClassifier(
        input_channels=input_channels,
        input_subcarriers=arch["target_subcarriers"],
        input_timesteps=target_time_steps,
        num_classes=arch["num_classes"],
        embed_dim=arch["embed_dim"],
        base_width=arch["base_width"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("Loaded %s on %s", ckpt_path, device)

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_dir / "cross_domain_eval"
    cd_dir = out_dir / "cross_domain"
    cd_dir.mkdir(parents=True, exist_ok=True)

    loader_kwargs = {
        "target_subcarriers": arch["target_subcarriers"],
        "target_time_steps": target_time_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    results: Dict[str, Any] = {}
    for split in args.splits:
        try:
            results[split] = _score_split(
                model=model,
                task_root=str(task_root),
                split=split,
                h5_key=h5_key,
                loader_kwargs=loader_kwargs,
                num_classes=arch["num_classes"],
                device=device,
            )
            res = results[split]
            logger.info(
                "Supervised [%s]: accuracy=%.4f  weighted_f1=%.4f "
                "(%d samples, %d classes present)",
                split, res["accuracy"], res["weighted_f1"],
                res["num_samples"], res["num_classes"],
            )
        except FileNotFoundError as e:
            logger.warning("Split '%s' not available: %s", split, e)
            results[split] = {"skipped": "split file not found", "detail": str(e)}
        except Exception as e:
            logger.warning("Split '%s' failed: %s", split, e)
            results[split] = {"error": str(e)}

        if split != IID_SPLIT:
            with open(cd_dir / f"{split}.json", "w") as f:
                json.dump(results[split], f, indent=2)

    iid_check = _check_iid_reproduction(
        results.get(IID_SPLIT), train_results, args.iid_tol
    )

    cross_domain = {k: v for k, v in results.items() if k != IID_SPLIT}
    num_ok = sum(1 for v in cross_domain.values() if "accuracy" in v)
    with open(cd_dir / "summary.json", "w") as f:
        json.dump(
            {
                "task_root": str(task_root),
                "num_successful": num_ok,
                "num_failed": len(cross_domain) - num_ok,
                "splits": cross_domain,
            },
            f,
            indent=2,
        )

    metrics = {
        "model": "supervised_resnet18_1d",
        "checkpoint": str(ckpt_path),
        "task_root": str(task_root),
        "num_classes": arch["num_classes"],
        "iid": results.get(IID_SPLIT, {}),
        "cross_domain": cross_domain,
    }
    if iid_check:
        metrics["iid_reproduction"] = iid_check
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    (out_dir / "summary.md").write_text(
        _summary_md(
            results, ckpt_path, str(task_root), arch["num_classes"], iid_check,
        ),
        encoding="utf-8",
    )

    logger.info("Results written to %s", out_dir)
    logger.info("  Files: metrics.json, summary.md, cross_domain/<split>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
