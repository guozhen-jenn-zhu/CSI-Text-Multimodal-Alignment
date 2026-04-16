"""Aggregate experiment results into paper-ready tables and figures.

Usage:
    python scripts/aggregate_neurips_results.py \
        --results_root /path/to/results \
        --output_dir /path/to/paper_tables \
        --stage all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METHOD_DISPLAY = {
    "clip_infonce": "CLIP InfoNCE",
    "siglip": "SigLIP",
    "language_bind": "LanguageBind",
    "centrobind": "CentroBind",
    "barlow_twins": "Barlow Twins",
    "masked_predict": "Masked Pred.",
}

ENCODER_DISPLAY = {
    "resnet18_1d": "ResNet-18",
    "patchtst": "PatchTST",
    "vit_1d": "ViT-1D",
    "amfm": "AM-FM",
}

TASK_DISPLAY = {
    "har": "HAR",
    "fall_detection": "Fall Det.",
    "motion_source": "Motion Src.",
    "proximity": "Proximity",
    "user_id": "User ID",
    "action_recognition": "XRF55 (55)",
    "sign_recognition": "SignFi (276)",
}

SUPERVISED_BASELINES = {
    "har": 0.8806,
    "fall_detection": 0.9493,
    "motion_source": 0.9956,
    "proximity": 0.8641,
    "user_id": 0.9955,
}


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


KNOWN_ENCODERS = ["resnet18_1d", "patchtst", "vit_1d", "amfm"]


def _split_method_encoder(name: str) -> tuple[str, str] | None:
    """Split a directory name like 'clip_infonce_resnet18_1d' into (method, encoder).

    Both method and encoder names can contain underscores, so a simple
    split on ``_`` is ambiguous.  We match against known encoder suffixes
    instead.
    """
    for enc in KNOWN_ENCODERS:
        suffix = f"_{enc}"
        if name.endswith(suffix):
            return name[: -len(suffix)], enc
    return None


def _find_results(
    root: Path,
    dataset: str | None = None,
    task: str | None = None,
    method: str | None = None,
    encoder: str | None = None,
) -> list[tuple[str, str, str, str, dict]]:
    """Discover all metrics.json files under root and return parsed entries."""
    entries = []
    for mf in sorted(root.rglob("metrics.json")):
        parts = mf.relative_to(root).parts
        if len(parts) < 3:
            continue
        d, t = parts[0], parts[1]
        method_enc = parts[2]

        parsed = _split_method_encoder(method_enc)
        if parsed is None:
            continue
        m, e = parsed

        if dataset and d != dataset:
            continue
        if task and t != task:
            continue
        if method and m != method:
            continue
        if encoder and e != encoder:
            continue

        data = _load_metrics(mf)
        if data:
            entries.append((d, t, m, e, data))
    return entries


def generate_main_benchmark_table(root: Path, output_dir: Path) -> None:
    """Table 1: Zero-shot accuracy across methods and tasks."""
    entries = _find_results(root)
    if not entries:
        logger.warning("No results found for main benchmark table")
        return

    tasks_ordered = ["har", "fall_detection", "motion_source",
                     "proximity", "user_id",
                     "action_recognition", "sign_recognition"]
    methods_ordered = list(METHOD_DISPLAY.keys())

    task_cols = [TASK_DISPLAY.get(t, t) for t in tasks_ordered]
    lines_md = [
        "# Table 1: Zero-Shot Classification Benchmark",
        "",
        "| Method | " + " | ".join(task_cols) + " |",
        "|--------" + "|----------" * len(task_cols) + "|",
    ]
    lines_tex = [
        "% Table 1: Zero-shot classification benchmark",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Zero-shot classification accuracy (\\%) across tasks and datasets. "
        "Best per column in \\textbf{bold}.}",
        "\\begin{tabular}{l" + "c" * len(tasks_ordered) + "}",
        "\\toprule",
        "Method & " + " & ".join(task_cols) + " \\\\",
        "\\midrule",
    ]

    results_grid: dict[str, dict[str, float]] = {}
    for d, t, m, e, data in entries:
        zs = data.get("zero_shot", {})
        acc = zs.get("accuracy")
        if acc is None:
            continue
        results_grid.setdefault(m, {})[t] = max(
            results_grid.get(m, {}).get(t, -1), acc
        )

    best_per_task: dict[str, float] = {}
    for t in tasks_ordered:
        vals = [results_grid.get(m, {}).get(t, -1) for m in methods_ordered]
        positive = [v for v in vals if v >= 0]
        best_per_task[t] = max(positive) if positive else -1

    for method in methods_ordered:
        row = results_grid.get(method, {})
        cells_md = []
        cells_tex = []
        for t in tasks_ordered:
            v = row.get(t)
            if v is None:
                cells_md.append("—")
                cells_tex.append("—")
            else:
                pct = v * 100
                is_best = abs(v - best_per_task.get(t, -1)) < 1e-6
                cells_md.append(f"**{pct:.1f}**" if is_best else f"{pct:.1f}")
                cells_tex.append(
                    f"\\textbf{{{pct:.1f}}}" if is_best else f"{pct:.1f}"
                )
        mname = METHOD_DISPLAY.get(method, method)
        lines_md.append(f"| {mname} | " + " | ".join(cells_md) + " |")
        lines_tex.append(f"{mname} & " + " & ".join(cells_tex) + " \\\\")

    sup_cells = []
    for t in tasks_ordered:
        bl = SUPERVISED_BASELINES.get(t)
        sup_cells.append(f"{bl*100:.1f}" if bl else "—")
    lines_md.append(f"| *Supervised (CSI-Bench)* | " + " | ".join(sup_cells) + " |")
    lines_tex.append("\\midrule")
    lines_tex.append(f"Supervised & " + " & ".join(sup_cells) + " \\\\")

    lines_md.append("")
    lines_tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table1_main_benchmark.md").write_text("\n".join(lines_md))
    (output_dir / "table1_main_benchmark.tex").write_text("\n".join(lines_tex))
    logger.info("Wrote Table 1 to %s", output_dir)


def generate_cross_dataset_table(root: Path, output_dir: Path) -> None:
    """Table 2: Cross-dataset transfer accuracy matrix."""
    entries = _find_results(root)
    lines_md = [
        "# Table 2: Cross-Dataset Transfer",
        "",
        "| Method | CSI-Bench HAR -> XRF55 | XRF55 -> CSI-Bench HAR | SignFi -> XRF55 Gesture |",
        "|--------|------------------------|------------------------|------------------------|",
    ]
    lines_tex = [
        "% Table 2: Cross-dataset transfer",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-dataset zero-shot transfer accuracy (\\%).}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & HAR $\\rightarrow$ XRF55 & XRF55 $\\rightarrow$ HAR & SignFi $\\rightarrow$ XRF55 \\\\",
        "\\midrule",
    ]

    directions = [
        ("csibench_har->xrf55", "CSI-Bench HAR -> XRF55"),
        ("xrf55->csibench_har", "XRF55 -> CSI-Bench HAR"),
        ("signfi->xrf55_gesture_subset", "SignFi -> XRF55 Gesture"),
    ]

    method_cd: dict[str, dict[str, float]] = {}
    for d, t, m, e, data in entries:
        cd = data.get("cross_dataset", {})
        for dir_key, _ in directions:
            pair = cd.get(dir_key, {})
            if isinstance(pair, dict) and "accuracy" in pair:
                method_cd.setdefault(m, {})[dir_key] = max(
                    method_cd.get(m, {}).get(dir_key, -1), pair["accuracy"]
                )

    for method in METHOD_DISPLAY:
        row = method_cd.get(method, {})
        cells = []
        for dir_key, _ in directions:
            v = row.get(dir_key)
            cells.append(f"{v*100:.1f}" if v is not None and v >= 0 else "—")
        mname = METHOD_DISPLAY.get(method, method)
        lines_md.append(f"| {mname} | " + " | ".join(cells) + " |")
        lines_tex.append(f"{mname} & " + " & ".join(cells) + " \\\\")

    lines_md.append("")
    lines_tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table2_cross_dataset.md").write_text("\n".join(lines_md))
    (output_dir / "table2_cross_dataset.tex").write_text("\n".join(lines_tex))
    logger.info("Wrote Table 2 to %s", output_dir)


def generate_cross_task_table(root: Path, output_dir: Path) -> None:
    """Table 4: Cross-task transfer matrix."""
    entries = _find_results(root)

    source_tasks = ["har", "fall_detection", "motion_source"]
    target_tasks = ["har", "fall_detection", "motion_source", "proximity", "user_id"]

    lines_md = [
        "# Table 4: Cross-Task Transfer",
        "",
        "| Source -> Target | " + " | ".join(TASK_DISPLAY.get(t, t) for t in target_tasks) + " |",
        "|------------------" + "|----------" * len(target_tasks) + "|",
    ]

    for d, t, m, e, data in entries:
        if d != "csibench":
            continue
        ct = data.get("cross_task", {})
        if not ct:
            continue
        cells = []
        for tgt in target_tasks:
            if tgt == t:
                cells.append("—")
                continue
            key = f"{t}->{tgt}"
            pair = ct.get(key, {})
            if isinstance(pair, dict) and "accuracy" in pair:
                cells.append(f"{pair['accuracy']*100:.1f}")
            else:
                cells.append("—")
        src_label = f"{TASK_DISPLAY.get(t, t)} ({METHOD_DISPLAY.get(m, m)[:4]})"
        lines_md.append(f"| {src_label} | " + " | ".join(cells) + " |")

    lines_md.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table4_cross_task.md").write_text("\n".join(lines_md))
    logger.info("Wrote Table 4 to %s", output_dir)


def generate_encoder_ablation_table(root: Path, output_dir: Path) -> None:
    """Table 3 (partial): Encoder ablation results from Priority 0."""
    entries = _find_results(root, dataset="csibench", task="har", method="clip_infonce")

    lines_md = [
        "# Encoder Ablation (clip_infonce on HAR)",
        "",
        "| Encoder | Zero-Shot Acc | Weighted F1 | Linear Probe |",
        "|---------|---------------|-------------|--------------|",
    ]

    for d, t, m, e, data in entries:
        zs = data.get("zero_shot", {})
        acc = zs.get("accuracy")
        f1 = zs.get("weighted_f1")
        lp = data.get("visualization", {}).get("linear_probe", {}).get("accuracy")
        enc_name = ENCODER_DISPLAY.get(e, e)
        cells = [
            f"{acc*100:.1f}" if acc is not None else "—",
            f"{f1*100:.1f}" if f1 is not None else "—",
            f"{lp*100:.1f}" if lp is not None else "—",
        ]
        lines_md.append(f"| {enc_name} | " + " | ".join(cells) + " |")

    lines_md.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table3_encoder_ablation.md").write_text("\n".join(lines_md))
    logger.info("Wrote encoder ablation table to %s", output_dir)


def generate_full_summary(root: Path, output_dir: Path) -> None:
    """Comprehensive summary of all results."""
    entries = _find_results(root)
    summary: dict[str, Any] = {
        "total_experiments": len(entries),
        "experiments": [],
    }

    for d, t, m, e, data in entries:
        zs = data.get("zero_shot", {})
        entry = {
            "dataset": d,
            "task": t,
            "method": m,
            "encoder": e,
            "zero_shot_accuracy": zs.get("accuracy"),
            "zero_shot_f1": zs.get("weighted_f1"),
        }
        lp = data.get("visualization", {}).get("linear_probe", {})
        if lp:
            entry["linear_probe_accuracy"] = lp.get("accuracy")
        ct = data.get("cross_task", {})
        if ct:
            ct_accs = {k: v["accuracy"] for k, v in ct.items()
                       if isinstance(v, dict) and "accuracy" in v}
            if ct_accs:
                entry["cross_task"] = ct_accs
        cd = data.get("cross_dataset", {})
        if cd:
            cd_accs = {k: v["accuracy"] for k, v in cd.items()
                       if isinstance(v, dict) and "accuracy" in v}
            if cd_accs:
                entry["cross_dataset"] = cd_accs
        summary["experiments"].append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote full summary (%d experiments) to %s", len(entries), output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Aggregate NeurIPS experiment results")
    parser.add_argument("--results_root", required=True, help="Root directory containing results")
    parser.add_argument("--output_dir", required=True, help="Output directory for tables")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["main_benchmark", "cross_dataset", "cross_task", "encoder_ablation", "all"],
        help="Which tables to generate",
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)

    if args.stage in ("main_benchmark", "all"):
        generate_main_benchmark_table(root, out)
    if args.stage in ("cross_dataset", "all"):
        generate_cross_dataset_table(root, out)
    if args.stage in ("cross_task", "all"):
        generate_cross_task_table(root, out)
    if args.stage in ("encoder_ablation", "all"):
        generate_encoder_ablation_table(root, out)
    if args.stage == "all":
        generate_full_summary(root, out)


if __name__ == "__main__":
    main()
