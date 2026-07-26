#!/usr/bin/env python3
"""Tabulate multimodal-aligned against CSI-only cross-domain performance.

Reads the two sets of metrics produced on the same held-out domain splits --
``scripts/run_cross_domain_eval.sh`` for the aligned models (zero-shot over text
prototypes) and ``scripts/run_supervised_cross_domain.sh`` for the supervised
ResNet-18 1D classifiers (argmax over a linear head) -- and reports IID
accuracy, per-split accuracy, each model's drop from IID, and the gap between
the two models.

The IID column comes from the aligned run's ``zero_shot`` entry, so make sure
that run scored the intended checkpoint; ``scripts/resolve_checkpoint.py``
rejects directories where a later training run replaced ``best.pt`` (HAR
clip_infonce being the known case).

Usage::

    python -m scripts.compare_aligned_vs_supervised \\
        --model_root /home/sagemaker-user/Multimodal_CSI_Text/Model \\
        --method clip_infonce

Writes ``comparison_aligned_vs_supervised.md`` and ``.tex`` to --output_dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TASKS = ("har", "proximity", "user_id", "fall_detection", "motion_source")
DEFAULT_SPLITS = ("cross_env", "cross_device")

TASK_LABELS = {
    "har": "HAR",
    "proximity": "Proximity",
    "user_id": "User ID",
    "fall_detection": "Fall Det.",
    "motion_source": "Motion Src.",
}

# Data-variant retrains carry their own tag in both checkpoint and results paths.
MODEL_TAGS = {"proximity": "proximity_fixed"}


def _model_tag(task: str) -> str:
    return MODEL_TAGS.get(task, task)


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        logger.warning("Missing metrics file: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def _accuracy(entry: Any) -> Optional[float]:
    if isinstance(entry, dict) and isinstance(entry.get("accuracy"), (int, float)):
        return float(entry["accuracy"])
    return None


def _reason(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("skipped") or entry.get("error") or "no accuracy")
    return "not evaluated"


def collect(
    aligned_root: Path,
    supervised_root: Path,
    tasks: Tuple[str, ...],
    splits: Tuple[str, ...],
    dataset: str,
    method: str,
    encoder: str,
) -> Dict[str, Dict[str, Any]]:
    """Pair up the aligned and supervised metrics for each task."""
    rows: Dict[str, Dict[str, Any]] = {}

    for task in tasks:
        tag = _model_tag(task)
        aligned = _read_json(
            aligned_root / dataset / tag / f"{method}_{encoder}" / "metrics.json"
        )
        supervised = _read_json(
            supervised_root / f"{dataset}_{tag}" / "cross_domain_eval" / "metrics.json"
        )
        if aligned is None and supervised is None:
            continue

        entry: Dict[str, Any] = {
            "model_tag": tag,
            "aligned_iid": (aligned or {}).get("zero_shot"),
            "supervised_iid": (supervised or {}).get("iid"),
            "supervised_iid_check": (supervised or {}).get("iid_reproduction"),
            "splits": {},
        }
        check = entry["supervised_iid_check"]
        if isinstance(check, dict) and not check.get("consistent", True):
            logger.warning(
                "Task '%s': the CSI-only IID accuracy (%.4f) did not reproduce the "
                "%.4f recorded at training time; its rows are unverified",
                task, check.get("measured", float("nan")),
                check.get("recorded_at_training", float("nan")),
            )
        for split in splits:
            entry["splits"][split] = {
                "aligned": ((aligned or {}).get("cross_domain") or {}).get(split),
                "supervised": ((supervised or {}).get("cross_domain") or {}).get(split),
            }
        rows[task] = entry

    return rows


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}" if value is not None else "--"


def _fmt_delta(value: Optional[float]) -> str:
    return f"{value * 100:+.1f}" if value is not None else "--"


def _drop(iid: Optional[float], split_acc: Optional[float]) -> Optional[float]:
    if iid is None or split_acc is None:
        return None
    return split_acc - iid


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _split_view(
    rows: Dict[str, Dict[str, Any]], split: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Per-task numbers, paired means, and reasons for missing entries.

    Means cover only the tasks scored by both models; averaging over different
    task sets would invite a comparison between two unrelated subsets.
    """
    per_task: List[Dict[str, Any]] = []
    notes: List[str] = []
    aligned_drops: List[float] = []
    supervised_drops: List[float] = []
    gaps: List[float] = []

    for task, entry in rows.items():
        pair = entry["splits"].get(split, {})
        a_iid = _accuracy(entry["aligned_iid"])
        s_iid = _accuracy(entry["supervised_iid"])
        a_acc = _accuracy(pair.get("aligned"))
        s_acc = _accuracy(pair.get("supervised"))
        a_drop = _drop(a_iid, a_acc)
        s_drop = _drop(s_iid, s_acc)
        gap = a_acc - s_acc if (a_acc is not None and s_acc is not None) else None

        if a_drop is not None and s_drop is not None:
            aligned_drops.append(a_drop)
            supervised_drops.append(s_drop)
        if gap is not None:
            gaps.append(gap)

        per_task.append({
            "task": task,
            "label": TASK_LABELS.get(task, task),
            "aligned_iid": a_iid,
            "aligned_acc": a_acc,
            "aligned_drop": a_drop,
            "supervised_iid": s_iid,
            "supervised_acc": s_acc,
            "supervised_drop": s_drop,
            "gap": gap,
        })

        if a_acc is None:
            notes.append(f"- `{task}` aligned: {_reason(pair.get('aligned'))}")
        if s_acc is None:
            notes.append(f"- `{task}` CSI-only: {_reason(pair.get('supervised'))}")

    means = {
        "n_paired": len(aligned_drops),
        "aligned_drop": _mean(aligned_drops),
        "supervised_drop": _mean(supervised_drops),
        "gap": _mean(gaps),
    }
    return per_task, means, notes


def build_markdown(
    rows: Dict[str, Dict[str, Any]],
    splits: Tuple[str, ...],
    method: str,
    encoder: str,
) -> str:
    lines = [
        "# Multimodal-Aligned vs CSI-Only: Cross-Domain Generalization",
        "",
        f"- **Aligned model**: {method} + {encoder}, zero-shot over text prototypes",
        "- **CSI-only model**: ResNet-18 1D + linear head, supervised cross-entropy",
        "- **Splits**: IID `test` versus the held-out `cross_env` / `cross_device` splits",
        "",
        "Accuracy in %. `Drop` is the change from that model's own IID accuracy, so it "
        "measures domain robustness independently of absolute accuracy. `Gap` is "
        "aligned minus CSI-only on the same split.",
        "",
        "Caveats to check before quoting these numbers:",
        "",
        "- Aligned IID is the `zero_shot` entry of the aligned run, so it is only as "
        "trustworthy as the checkpoint that run scored. For HAR make sure it came "
        "from the CLIP checkpoint (`epoch_150_p0_clip.pt`) rather than a `best.pt` "
        "left behind by the sentence-transformer ablation.",
        "- The CSI-only side records an `iid_reproduction` check against the accuracy "
        "its training run reported; a failed check means that row's data or weights "
        "did not load as they did during training.",
        "",
    ]

    for split in splits:
        lines.extend([
            f"## {split}",
            "",
            "| Task | Aligned IID | Aligned split | Aligned drop | "
            "CSI-only IID | CSI-only split | CSI-only drop | Gap |",
            "|------|-------------|---------------|--------------|"
            "--------------|----------------|---------------|-----|",
        ])

        per_task, means, notes = _split_view(rows, split)

        for row in per_task:
            lines.append(
                f"| {row['label']} | {_fmt_pct(row['aligned_iid'])} | "
                f"{_fmt_pct(row['aligned_acc'])} | {_fmt_delta(row['aligned_drop'])} | "
                f"{_fmt_pct(row['supervised_iid'])} | "
                f"{_fmt_pct(row['supervised_acc'])} | "
                f"{_fmt_delta(row['supervised_drop'])} | {_fmt_delta(row['gap'])} |"
            )

        lines.append(
            f"| **Mean over {means['n_paired']} paired task(s)** | -- | -- | "
            f"{_fmt_delta(means['aligned_drop'])} | -- | -- | "
            f"{_fmt_delta(means['supervised_drop'])} | {_fmt_delta(means['gap'])} |"
        )

        if notes:
            lines.extend(["", "Unavailable entries:", ""] + notes)
        lines.append("")

    flagged = [
        (task, entry["supervised_iid_check"])
        for task, entry in rows.items()
        if isinstance(entry.get("supervised_iid_check"), dict)
        and not entry["supervised_iid_check"].get("consistent", True)
    ]
    if flagged:
        lines.extend(["## Failed IID reproduction checks", ""])
        for task, check in flagged:
            lines.append(
                f"- `{task}` CSI-only: measured {check['measured'] * 100:.1f} vs "
                f"{check['recorded_at_training'] * 100:.1f} recorded at training "
                "time. Its rows above are unverified."
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def build_latex(
    rows: Dict[str, Dict[str, Any]],
    splits: Tuple[str, ...],
    method: str,
    encoder: str,
) -> str:
    safe_method = method.replace("_", r"\_")
    safe_encoder = encoder.replace("_", r"\_")
    lines: List[str] = []

    for split in splits:
        safe_split = split.replace("_", r"\_")
        per_task, means, _ = _split_view(rows, split)

        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Cross-domain generalization on the {safe_split} split: "
            f"{safe_method} alignment versus a supervised CSI-only "
            f"{safe_encoder} classifier. Accuracy in \\%; $\\Delta$ is the change "
            "from each model's own IID accuracy.}",
            f"\\label{{tab:cross-domain-{split.replace('_', '-')}-aligned-vs-csi}}",
            "\\begin{tabular}{lccccccc}",
            "\\toprule",
            "& \\multicolumn{3}{c}{Aligned (zero-shot)} & "
            "\\multicolumn{3}{c}{CSI-only (supervised)} & \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "Task & IID & Split & $\\Delta$ & IID & Split & $\\Delta$ & Gap \\\\",
            "\\midrule",
        ])

        for row in per_task:
            lines.append(
                f"{row['label']} & {_fmt_pct(row['aligned_iid'])} & "
                f"{_fmt_pct(row['aligned_acc'])} & {_fmt_delta(row['aligned_drop'])} & "
                f"{_fmt_pct(row['supervised_iid'])} & "
                f"{_fmt_pct(row['supervised_acc'])} & "
                f"{_fmt_delta(row['supervised_drop'])} & {_fmt_delta(row['gap'])} \\\\"
            )

        lines.extend([
            "\\midrule",
            f"Mean ({means['n_paired']} paired) & --- & --- & "
            f"{_fmt_delta(means['aligned_drop'])} & --- & --- & "
            f"{_fmt_delta(means['supervised_drop'])} & {_fmt_delta(means['gap'])} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model_root",
        default="/home/sagemaker-user/Multimodal_CSI_Text/Model",
        help="Root holding results/ (default: the SageMaker layout)",
    )
    parser.add_argument(
        "--aligned_root",
        default=None,
        help="Default: <model_root>/results",
    )
    parser.add_argument(
        "--supervised_root",
        default=None,
        help="Default: <model_root>/results/supervised_baselines",
    )
    parser.add_argument("--dataset", default="csibench")
    parser.add_argument("--method", default="clip_infonce")
    parser.add_argument("--encoder", default="resnet18_1d")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <model_root>/results/analysis",
    )
    args = parser.parse_args()

    model_root = Path(args.model_root)
    aligned_root = Path(args.aligned_root) if args.aligned_root else model_root / "results"
    supervised_root = (
        Path(args.supervised_root)
        if args.supervised_root
        else model_root / "results" / "supervised_baselines"
    )
    out_dir = Path(args.output_dir) if args.output_dir else model_root / "results" / "analysis"

    rows = collect(
        aligned_root=aligned_root,
        supervised_root=supervised_root,
        tasks=tuple(args.tasks),
        splits=tuple(args.splits),
        dataset=args.dataset,
        method=args.method,
        encoder=args.encoder,
    )

    if not rows:
        logger.error(
            "No metrics found. Looked for aligned results under %s and supervised "
            "results under %s", aligned_root, supervised_root,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "comparison_aligned_vs_supervised.md"
    tex_path = out_dir / "comparison_aligned_vs_supervised.tex"
    json_path = out_dir / "comparison_aligned_vs_supervised.json"

    md_path.write_text(
        build_markdown(rows, tuple(args.splits), args.method, args.encoder),
        encoding="utf-8",
    )
    tex_path.write_text(
        build_latex(rows, tuple(args.splits), args.method, args.encoder),
        encoding="utf-8",
    )
    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "method": args.method,
                "encoder": args.encoder,
                "aligned_root": str(aligned_root),
                "supervised_root": str(supervised_root),
                "tasks": rows,
            },
            f,
            indent=2,
        )

    print(md_path.read_text(encoding="utf-8"))
    logger.info("Wrote %s", md_path)
    logger.info("Wrote %s", tex_path)
    logger.info("Wrote %s", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
