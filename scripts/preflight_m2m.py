#!/usr/bin/env python3
"""Verify every checkpoint, registry and split before the m2m retrieval sweep.

The sweep touches 42 checkpoints and takes a while, so this checks the things
that would otherwise fail halfway through, and checks the one thing a
successful-looking run can still get wrong: whether the sample ids coming out
of the dataloader actually match the keys in the pair registry.  If they do not
overlap, every query is silently dropped and the resulting mAP is meaningless.

Per task/method cell it verifies:

1. the checkpoint file exists, loads, and carries a config;
2. the model builds from that config and accepts the state dict with
   ``strict=True``;
3. the pair registry exists and loads;
4. the test split loads and its sample ids overlap the registry;
5. one real batch flows through the CSI encoder and projection, and the batch
   carries ``sample_ids``;
6. the output directory is writable.

Usage::

    python scripts/preflight_m2m.py
    python scripts/preflight_m2m.py --quick          # skip the batch forward pass
    python scripts/preflight_m2m.py --only_tasks csibench:har --only_methods clip_infonce
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

for _backend_flag in ("USE_TF", "USE_FLAX", "USE_JAX"):
    os.environ.setdefault(_backend_flag, "0")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import CSITextDataset  # noqa: E402
from data.pair_registry import PairRegistry  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_TASKS = [
    "csibench:har",
    "csibench:fall_detection",
    "csibench:motion_source",
    "csibench:proximity",
    "csibench:user_id",
    "xrf55:action_recognition",
    "signfi:sign_recognition",
]
METHODS = [
    "clip_infonce", "siglip", "language_bind", "centrobind",
    "barlow_twins", "masked_predict",
]

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


def resolve_task_root(data_root: Path, dataset: str, task: str) -> Path:
    if dataset == "csibench" and task in ("har", "proximity", "user_id"):
        return data_root / "csibench" / "multitask" / task
    if dataset == "csibench":
        return data_root / "csibench" / task
    return data_root / dataset / task


def resolve_registry_parent(data_root: Path, dataset: str, task: str) -> Path:
    if dataset == "csibench" and task in ("har", "proximity", "user_id"):
        return data_root / "csibench" / "multitask"
    if dataset == "csibench":
        return data_root / "csibench"
    return data_root / dataset


class CellReport:
    """Accumulates per-check outcomes for one task/method cell."""

    def __init__(self, dataset: str, task: str, method: str) -> None:
        self.cell = f"{dataset}/{task}/{method}"
        self.checks: dict[str, tuple[str, str]] = {}
        self.details: dict[str, Any] = {}

    def record(self, name: str, status: str, message: str = "") -> None:
        self.checks[name] = (status, message)

    @property
    def status(self) -> str:
        statuses = {s for s, _ in self.checks.values()}
        if FAIL in statuses:
            return FAIL
        if WARN in statuses:
            return WARN
        if statuses == {SKIP}:
            return SKIP
        return PASS

    def failures(self) -> list[str]:
        return [
            f"{name}: {msg}"
            for name, (status, msg) in self.checks.items()
            if status in (FAIL, WARN)
        ]


def check_checkpoint(report: CellReport, ckpt_path: Path, device: str) -> Optional[dict]:
    if not ckpt_path.is_file():
        report.record("checkpoint", FAIL, f"not found: {ckpt_path}")
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        report.record("checkpoint", FAIL, f"torch.load failed: {e}")
        return None

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        report.record("checkpoint", WARN, "no 'model_state_dict' key; treating file as a bare state dict")
        return {"model_state_dict": ckpt, "config": None}

    size_mb = ckpt_path.stat().st_size / 1e6
    report.details["checkpoint_epoch"] = ckpt.get("epoch")
    report.details["checkpoint_val_loss"] = ckpt.get("val_loss")
    report.record(
        "checkpoint", PASS,
        f"{size_mb:.0f} MB, epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')}",
    )
    return ckpt


def check_model(
    report: CellReport, ckpt: dict, overrides: list[str], device: str,
) -> Optional[tuple]:
    from scripts.run_evaluate import (
        _apply_overrides,
        _merge_configs,
        build_model_from_config,
    )

    if ckpt.get("config") is None:
        report.record("model", FAIL, "checkpoint carries no config; cannot rebuild the model")
        return None

    try:
        config = _merge_configs(ckpt["config"])
        _apply_overrides(config, overrides)
        model, text_encoder, text_projection, csi_projection = build_model_from_config(
            config, torch.device(device),
        )
    except Exception as e:
        report.record("model", FAIL, f"build failed: {e}")
        return None

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as e:
        report.record("model", FAIL, f"load_state_dict(strict=True) failed: {e}")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    report.details["encoder"] = config.get("csi_encoder", {}).get("name")
    report.details["text_encoder"] = config.get("text_encoder", {}).get("name")
    report.record(
        "model", PASS,
        f"{n_params/1e6:.1f}M params, encoder={report.details['encoder']}, "
        f"text={report.details['text_encoder']}",
    )
    return config, model, text_encoder, text_projection, csi_projection


def check_registry(report: CellReport, registry_path: Path) -> Optional[PairRegistry]:
    if not registry_path.is_file():
        report.record("registry", FAIL, f"not found: {registry_path}")
        return None
    try:
        registry = PairRegistry.load(str(registry_path))
    except Exception as e:
        report.record("registry", FAIL, f"load failed: {e}")
        return None

    stats = registry.stats()
    report.details["registry_stats"] = stats
    if stats["num_texts"] < 2:
        report.record("registry", FAIL, f"gallery of {stats['num_texts']} caption(s) cannot be ranked")
        return registry

    report.record(
        "registry", PASS,
        f"{stats['num_texts']} captions, {stats['num_samples']} samples, "
        f"{stats['avg_texts_per_sample']:.1f} captions/sample",
    )
    return registry


def check_split_overlap(
    report: CellReport, task_root: Path, config: dict, registry: PairRegistry,
) -> Optional[CSITextDataset]:
    """The check that matters: do dataloader sample ids exist in the registry?"""
    ds_cfg = config.get("dataset_config", {})
    preproc = ds_cfg.get("preprocessing", {})
    try:
        dataset = CSITextDataset(
            task_root=str(task_root),
            split="test",
            h5_key=ds_cfg.get("h5_key", "CSI_amps"),
            registry_path=None,  # the m2m protocol supplies its own registry
            target_subcarriers=preproc.get("target_subcarriers", 64),
            target_time_steps=preproc.get("target_time_steps", 500),
            preload=False,
        )
    except Exception as e:
        report.record("test_split", FAIL, f"could not build the test dataset: {e}")
        return None

    if len(dataset) == 0:
        report.record("test_split", FAIL, "test split is empty")
        return None
    report.record("test_split", PASS, f"{len(dataset)} test samples")

    known = sum(1 for sid in dataset.sample_ids if sid in registry.sample_to_texts)
    frac = known / len(dataset.sample_ids)
    report.details["split_registry_overlap"] = frac
    example = dataset.sample_ids[0]
    if frac == 0:
        report.record(
            "id_overlap", FAIL,
            f"NO test sample id is in the registry (e.g. {example!r}); "
            "the split files and the registry describe different sample spaces",
        )
    elif frac < 0.99:
        report.record(
            "id_overlap", WARN,
            f"only {frac*100:.1f}% of test ids are in the registry; "
            f"{len(dataset.sample_ids) - known} queries would be dropped",
        )
    else:
        report.record("id_overlap", PASS, f"{frac*100:.1f}% of test ids found in the registry")
    return dataset


def check_batch(
    report: CellReport,
    dataset: CSITextDataset,
    model,
    csi_projection,
    device: str,
    batch_size: int,
) -> None:
    """Pull one real batch through the encoder: proves H5 reads and the forward pass."""
    from torch.utils.data import DataLoader

    try:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=CSITextDataset.collate_fn,
        )
        batch = next(iter(loader))
    except Exception as e:
        report.record("batch", FAIL, f"could not read a batch: {e}")
        return

    if "sample_ids" not in batch:
        report.record("batch", FAIL, "batch has no 'sample_ids'; the m2m protocol will raise KeyError")
        return

    try:
        model.eval()
        csi_encoder = model.csi_encoder if hasattr(model, "csi_encoder") else model
        proj = csi_projection
        if proj is None and hasattr(model, "csi_projection"):
            proj = model.csi_projection
        with torch.no_grad():
            features = csi_encoder(batch["csi"].to(device))
            if proj is not None:
                features = proj(features)
    except Exception as e:
        report.record("batch", FAIL, f"forward pass failed: {e}")
        return

    if features.shape[0] != len(batch["sample_ids"]):
        report.record(
            "batch", FAIL,
            f"embedding count {features.shape[0]} != sample id count {len(batch['sample_ids'])}",
        )
        return

    report.details["embed_dim"] = int(features.shape[1])
    report.record(
        "batch", PASS,
        f"csi{tuple(batch['csi'].shape)} -> embeddings{tuple(features.shape)}",
    )


def check_writable(report: CellReport, out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        report.record("output", FAIL, f"{out_dir} is not writable: {e}")
        return
    report.record("output", PASS, str(out_dir))


def run_cell(
    dataset_name: str, task: str, method: str, args: argparse.Namespace,
) -> CellReport:
    report = CellReport(dataset_name, task, method)
    data_root = Path(args.data_root).expanduser()
    model_root = Path(args.model_root).expanduser()

    ckpt_path = model_root / "checkpoints" / f"{dataset_name}-{task}-{method}" / "best.pt"
    task_root = resolve_task_root(data_root, dataset_name, task)
    registry_path = resolve_registry_parent(data_root, dataset_name, task) / task / "pair_registry.json"
    out_dir = (
        model_root / "results" / "m2m_retrieval" / dataset_name / task
        / f"{method}_{args.encoder}"
    )

    ckpt = check_checkpoint(report, ckpt_path, args.device)
    registry = check_registry(report, registry_path)
    check_writable(report, out_dir)

    built = None
    if ckpt is not None:
        built = check_model(
            report, ckpt,
            [
                f"dataset={dataset_name}", f"task={task}", f"method={method}",
                f"csi_encoder.name={args.encoder}", f"task_root={task_root}",
            ],
            args.device,
        )

    if built is None or registry is None:
        report.record("test_split", SKIP, "blocked by an earlier failure")
        report.record("id_overlap", SKIP, "blocked by an earlier failure")
        report.record("batch", SKIP, "blocked by an earlier failure")
        return report

    config, model, _text_encoder, _text_projection, csi_projection = built
    config["global_data_root"] = str(data_root)

    ds = check_split_overlap(report, task_root, config, registry)
    if ds is None:
        report.record("batch", SKIP, "blocked by an earlier failure")
        return report

    if args.quick:
        report.record("batch", SKIP, "--quick")
    else:
        check_batch(report, ds, model, csi_projection, args.device, args.batch_size)

    return report


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/home/sagemaker-user/Multimodal_CSI_Text/Dataset")
    parser.add_argument("--model_root", default="/home/sagemaker-user/Multimodal_CSI_Text/Model")
    parser.add_argument("--encoder", default="resnet18_1d")
    parser.add_argument("--only_tasks", nargs="*", default=None, metavar="DATASET:TASK")
    parser.add_argument("--only_methods", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip the batch forward pass (no H5 reads, no model inference)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json_out", default=None, help="Optional path for a machine-readable report")
    args = parser.parse_args()

    pairs = args.only_tasks or DATASET_TASKS
    methods = args.only_methods or METHODS

    print("=" * 78)
    print(f"Preflight for the many-to-many retrieval sweep  (device={args.device})")
    print(f"  data_root : {args.data_root}")
    print(f"  model_root: {args.model_root}")
    print("=" * 78)

    reports: list[CellReport] = []
    for pair in pairs:
        if ":" not in pair:
            parser.error(f"--only_tasks expects DATASET:TASK, got {pair!r}")
        dataset_name, task = pair.split(":", 1)
        for method in methods:
            try:
                report = run_cell(dataset_name, task, method, args)
            except Exception as e:
                report = CellReport(dataset_name, task, method)
                report.record("unexpected", FAIL, f"{e}\n{traceback.format_exc()}")
            reports.append(report)

            marker = {PASS: "ok  ", FAIL: "FAIL", WARN: "warn", SKIP: "skip"}[report.status]
            print(f"[{marker}] {report.cell}")
            for name, (status, msg) in report.checks.items():
                if status == PASS:
                    print(f"         {name}: {msg}")
                elif status != SKIP:
                    print(f"    >>>  {name}: {status} -- {msg}")

    n_pass = sum(1 for r in reports if r.status == PASS)
    n_warn = sum(1 for r in reports if r.status == WARN)
    n_fail = sum(1 for r in reports if r.status == FAIL)

    print("=" * 78)
    print(f"{n_pass} ready, {n_warn} with warnings, {n_fail} blocked, {len(reports)} total")
    if n_fail:
        print("\nBlocked cells:")
        for r in reports:
            if r.status == FAIL:
                print(f"  {r.cell}")
                for f in r.failures():
                    print(f"      {f}")
    print("=" * 78)

    if args.json_out:
        payload = [
            {
                "cell": r.cell,
                "status": r.status,
                "checks": {k: {"status": s, "message": m} for k, (s, m) in r.checks.items()},
                "details": r.details,
            }
            for r in reports
        ]
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.json_out}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
