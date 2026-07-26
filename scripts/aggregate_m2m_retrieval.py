#!/usr/bin/env python3
"""Aggregate the many-to-many retrieval sweep into paper tables.

Reads every ``metrics.json`` written by ``run_m2m_retrieval_evals.sh`` and
builds a task x method view of the ranking metrics, so the benchmark can be
reported with mAP / nDCG / R-Precision / MRR instead of Rank-1 alone.

Usage::

    python scripts/aggregate_m2m_retrieval.py \\
        --results_root /home/sagemaker-user/Multimodal_CSI_Text/Model/results/m2m_retrieval \\
        --output_dir paper_results/m2m_retrieval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

KNOWN_ENCODERS = ["resnet18_1d", "patchtst", "vit_1d", "amfm"]

TASK_ORDER = [
    "har", "fall_detection", "motion_source", "proximity", "user_id",
    "action_recognition", "sign_recognition",
]
METHOD_ORDER = [
    "clip_infonce", "siglip", "language_bind", "centrobind",
    "barlow_twins", "masked_predict",
]

TASK_LABELS = {
    "har": "HAR",
    "fall_detection": "Fall detection",
    "motion_source": "Motion source",
    "proximity": "Proximity",
    "user_id": "User ID",
    "action_recognition": "XRF55 action",
    "sign_recognition": "SignFi sign",
}
METHOD_LABELS = {
    "clip_infonce": "CLIP InfoNCE",
    "siglip": "SigLIP",
    "language_bind": "LanguageBind",
    "centrobind": "CentroBind",
    "barlow_twins": "Barlow Twins",
    "masked_predict": "Masked Predict",
}


def split_method_encoder(name: str) -> Optional[tuple[str, str]]:
    for enc in KNOWN_ENCODERS:
        suffix = f"_{enc}"
        if name.endswith(suffix):
            return name[: -len(suffix)], enc
    return None


def scan_runs(root: Path) -> list[dict[str, Any]]:
    """Collect every run holding a ``retrieval_m2m`` block."""
    runs: list[dict[str, Any]] = []
    for mf in sorted(root.rglob("metrics.json")):
        parts = mf.relative_to(root).parts
        if len(parts) < 3:
            continue
        dataset, task, method_enc = parts[0], parts[1], parts[2]
        parsed = split_method_encoder(method_enc)
        if parsed is None:
            continue
        method, encoder = parsed

        try:
            with open(mf, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", mf, e)
            continue

        m2m = data.get("retrieval_m2m")
        if not isinstance(m2m, dict) or "csi_to_text" not in m2m:
            continue
        if "error" in m2m:
            logger.warning("%s/%s/%s reported an error: %s", dataset, task, method, m2m["error"])
            continue

        runs.append({
            "dataset": dataset,
            "task": task,
            "method": method,
            "encoder": encoder,
            "path": str(mf),
            "m2m": m2m,
        })
    return runs


def _order_key(values: list[str], order: list[str]) -> list[str]:
    """Preferred order first, then anything unexpected, alphabetically."""
    known = [v for v in order if v in values]
    extra = sorted(v for v in values if v not in order)
    return known + extra


def _get(run: dict, direction: str, level: str, metric: str) -> Optional[float]:
    block = run["m2m"].get(direction, {})
    if not isinstance(block, dict):
        return None
    level_block = block.get(level, {})
    if not isinstance(level_block, dict):
        return None
    value = level_block.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "--"


def build_matrix(
    runs: list[dict], tasks: list[str], methods: list[str],
    direction: str, level: str, metric: str,
) -> dict[tuple[str, str], Optional[float]]:
    lookup = {(r["task"], r["method"]): r for r in runs}
    return {
        (t, m): (_get(lookup[(t, m)], direction, level, metric) if (t, m) in lookup else None)
        for t in tasks for m in methods
    }


def markdown_matrix(
    matrix: dict, tasks: list[str], methods: list[str], title: str, note: str,
) -> list[str]:
    lines = [f"### {title}", "", note, "", "| Task | " + " | ".join(
        METHOD_LABELS.get(m, m) for m in methods
    ) + " | best |", "|" + "|".join("---" for _ in range(len(methods) + 2)) + "|"]
    for t in tasks:
        row = [matrix.get((t, m)) for m in methods]
        present = [v for v in row if v is not None]
        best = max(present) if present else None
        cells = [
            (f"**{_fmt(v)}**" if best is not None and v == best else _fmt(v))
            for v in row
        ]
        lines.append(
            f"| {TASK_LABELS.get(t, t)} | " + " | ".join(cells) + f" | {_fmt(best)} |"
        )
    lines.append("")
    return lines


def build_markdown(runs: list[dict], tasks: list[str], methods: list[str]) -> str:
    lines = [
        "# Many-to-Many Retrieval Results",
        "",
        f"{len(runs)} runs across {len(tasks)} tasks and {len(methods)} methods. "
        "Every query is ranked against the full caption gallery and scored "
        "against all of its valid matches.",
        "",
        "Two relevance rules are reported from the same ranking:",
        "",
        "- **strict** -- only the captions on that sample's own registry edge "
        "list count, so a model must separate the sample's own Level-2 "
        "composite from composites describing the same activity by a different "
        "person at a different distance.",
        "- **class-level** -- any caption of the correct class counts. This is "
        "the class-level ranking view a reviewer would expect alongside mAP.",
        "",
        "## Protocol scale",
        "",
        "| Task | Gallery captions | Test CSI samples | Strict pos/query | Class-level pos/query |",
        "|------|------------------|------------------|------------------|-----------------------|",
    ]

    seen: set[str] = set()
    for r in sorted(runs, key=lambda r: TASK_ORDER.index(r["task"]) if r["task"] in TASK_ORDER else 99):
        if r["task"] in seen:
            continue
        seen.add(r["task"])
        gallery = r["m2m"].get("gallery", {})
        lines.append(
            f"| {TASK_LABELS.get(r['task'], r['task'])} "
            f"| {gallery.get('num_texts', '--')} "
            f"| {gallery.get('num_test_samples', '--')} "
            f"| {_fmt(_get(r, 'csi_to_text', 'strict', 'avg_positives_per_query'), 1)} "
            f"| {_fmt(_get(r, 'csi_to_text', 'class_level', 'avg_positives_per_query'), 1)} |"
        )

    lines.extend(["", "## CSI -> Text", ""])
    for level, level_label in (("strict", "strict"), ("class_level", "class-level")):
        lines.extend(markdown_matrix(
            build_matrix(runs, tasks, methods, "csi_to_text", level, "mAP"),
            tasks, methods, f"mAP ({level_label} relevance)",
            "Mean average precision over the full caption gallery.",
        ))
    lines.extend(markdown_matrix(
        build_matrix(runs, tasks, methods, "csi_to_text", "strict", "normalized_mAP"),
        tasks, methods, "Normalized mAP (strict)",
        "mAP rescaled onto [chance -> 1], so tasks with different gallery "
        "sizes and positive counts are comparable.",
    ))
    lines.extend(markdown_matrix(
        build_matrix(runs, tasks, methods, "csi_to_text", "strict", "mAP_macro"),
        tasks, methods, "Macro mAP (strict)",
        "Averaged within each class first, so a frequent class cannot carry the score.",
    ))
    lines.extend(markdown_matrix(
        build_matrix(runs, tasks, methods, "csi_to_text", "strict", "mAP_worst_class"),
        tasks, methods, "Worst-class mAP (strict)",
        "The weakest class in each run. This is where the remaining headroom is.",
    ))
    lines.extend(markdown_matrix(
        build_matrix(runs, tasks, methods, "csi_to_text", "strict", "ndcg@10"),
        tasks, methods, "nDCG@10 (strict)",
        "Rank-sensitive quality of the top 10 captions.",
    ))

    lines.extend(["## Text -> CSI", ""])
    for level, level_label in (("strict", "strict"), ("class_level", "class-level")):
        lines.extend(markdown_matrix(
            build_matrix(runs, tasks, methods, "text_to_csi", level, "mAP"),
            tasks, methods, f"mAP ({level_label} relevance)",
            "Each caption ranks the whole test split; every CSI sample it is "
            "paired with counts as a positive.",
        ))

    lines.extend(_rank1_contrast_lines(runs, tasks, methods))
    lines.extend(_dedup_lines(runs, tasks, methods))
    lines.extend(_chance_lines(runs, tasks))
    return "\n".join(lines)


def _rank1_contrast_lines(runs: list[dict], tasks: list[str], methods: list[str]) -> list[str]:
    """Side-by-side of the Rank-1 style metric and mAP, per task."""
    lines = [
        "## Rank-1 versus mAP",
        "",
        "`hit@1` is the many-to-many analogue of Rank-1: did any valid caption "
        "land first? Getting one of many positives to the top is far easier "
        "than ranking all of them well, so hit@1 saturates while mAP does not. "
        "The last column is how much of the apparent performance is an artefact "
        "of only asking for one correct caption.",
        "",
        "| Task | Method | hit@1 (strict) | mAP (strict) | hit@1 - mAP |",
        "|------|--------|----------------|--------------|-------------|",
    ]
    lookup = {(r["task"], r["method"]): r for r in runs}
    for t in tasks:
        for m in methods:
            run = lookup.get((t, m))
            if run is None:
                continue
            hit = _get(run, "csi_to_text", "strict", "hit_rate@1")
            m_ap = _get(run, "csi_to_text", "strict", "mAP")
            if hit is None or m_ap is None:
                continue
            lines.append(
                f"| {TASK_LABELS.get(t, t)} | {METHOD_LABELS.get(m, m)} "
                f"| {hit:.4f} | {m_ap:.4f} | {hit - m_ap:+.4f} |"
            )
    lines.append("")
    return lines


def _dedup_lines(runs: list[dict], tasks: list[str], methods: list[str]) -> list[str]:
    """Effect of collapsing captions the frozen text encoder cannot separate."""
    rows: list[str] = []
    lookup = {(r["task"], r["method"]): r for r in runs}
    for t in tasks:
        for m in methods:
            run = lookup.get((t, m))
            if run is None:
                continue
            info = run["m2m"].get("near_duplicate_control", {})
            dedup = run["m2m"].get("csi_to_text_dedup")
            if not isinstance(info, dict) or "num_groups" not in info:
                continue
            strict = run["m2m"].get("csi_to_text", {}).get("strict", {})
            dedup_strict = dedup.get("strict", {}) if isinstance(dedup, dict) else {}
            rows.append(
                f"| {TASK_LABELS.get(t, t)} | {METHOD_LABELS.get(m, m)} "
                f"| {info['num_captions']} -> {info['num_groups']} "
                f"| {_fmt(strict.get('mAP'))} "
                f"| {_fmt(strict.get('chance', {}).get('mAP'))} "
                f"| {_fmt(strict.get('normalized_mAP'))} "
                f"| {_fmt(dedup_strict.get('mAP'))} "
                f"| {_fmt(dedup_strict.get('chance', {}).get('mAP'))} "
                f"| {_fmt(dedup_strict.get('normalized_mAP'))} |"
            )
    if not rows:
        return []
    return [
        "## Near-duplicate caption control",
        "",
        "Captions the frozen text encoder cannot tell apart are merged into one "
        "gallery entry before re-scoring, so a model is not credited repeatedly "
        "for the same caption. A large gallery collapse means the nominal "
        "gallery size overstates how many distinct things a model must separate.",
        "",
        "Merging shrinks the gallery *and* the number of positives per query, so "
        "chance mAP moves too and the raw mAP columns are not directly "
        "comparable across the collapse. Read the normalized columns, which "
        "rescale each side onto its own [chance -> 1] range.",
        "",
        "| Task | Method | Gallery | mAP | chance | norm mAP | mAP (dedup) | chance (dedup) | norm mAP (dedup) |",
        "|------|--------|---------|-----|--------|----------|-------------|----------------|------------------|",
        *rows,
        "",
    ]


def _chance_lines(runs: list[dict], tasks: list[str]) -> list[str]:
    lines = [
        "## Chance baselines",
        "",
        "Exact expected scores of a uniformly random ranking with the same "
        "number of positives, so the tables above can be read against a floor.",
        "",
        "| Task | Gallery | Strict chance mAP | Class-level chance mAP | Strict chance hit@1 |",
        "|------|---------|-------------------|------------------------|---------------------|",
    ]
    seen: set[str] = set()
    for t in tasks:
        run = next((r for r in runs if r["task"] == t), None)
        if run is None or t in seen:
            continue
        seen.add(t)
        c2t = run["m2m"].get("csi_to_text", {})
        strict_chance = c2t.get("strict", {}).get("chance", {})
        class_chance = c2t.get("class_level", {}).get("chance", {})
        lines.append(
            f"| {TASK_LABELS.get(t, t)} "
            f"| {run['m2m'].get('gallery', {}).get('num_texts', '--')} "
            f"| {_fmt(strict_chance.get('mAP'))} "
            f"| {_fmt(class_chance.get('mAP'))} "
            f"| {_fmt(strict_chance.get('hit_rate@1'))} |"
        )
    lines.append("")
    return lines


def build_latex(runs: list[dict], tasks: list[str], methods: list[str]) -> str:
    matrix = build_matrix(runs, tasks, methods, "csi_to_text", "strict", "mAP")
    chance = {
        t: (next((r for r in runs if r["task"] == t), {}).get("m2m", {})
            .get("csi_to_text", {}).get("strict", {}).get("chance", {}).get("mAP"))
        for t in tasks
    }

    rows = []
    for t in tasks:
        values = [matrix.get((t, m)) for m in methods]
        present = [v for v in values if v is not None]
        best = max(present) if present else None
        cells = [
            ("\\textbf{" + _fmt(v, 3) + "}" if best is not None and v == best else _fmt(v, 3))
            for v in values
        ]
        rows.append(
            f"{TASK_LABELS.get(t, t)} & {_fmt(chance.get(t), 3)} & "
            + " & ".join(cells) + " \\\\"
        )

    col_spec = "l" + "c" * (len(methods) + 1)
    header = "Task & Chance & " + " & ".join(
        METHOD_LABELS.get(m, m) for m in methods
    ) + " \\\\"

    return "\n".join([
        "% Auto-generated by scripts/aggregate_m2m_retrieval.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header,
        "\\hline",
        *rows,
        "\\hline",
        "\\end{tabular}",
        "\\caption{CSI $\\rightarrow$ Text mean average precision under strict "
        "many-to-many relevance: every query is ranked against the full caption "
        "gallery and scored against all of its valid matches. Chance is the "
        "exact expected mAP of a random ranking with the same number of "
        "positives. Unlike Rank-1 accuracy, mAP does not saturate on the coarse "
        "tasks, leaving clear headroom.}",
        "\\label{tab:m2m-retrieval}",
        "\\end{table}",
        "",
    ])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_root",
        default="/home/sagemaker-user/Multimodal_CSI_Text/Model/results/m2m_retrieval",
        help="Root written by run_m2m_retrieval_evals.sh",
    )
    parser.add_argument("--output_dir", default="paper_results/m2m_retrieval")
    parser.add_argument(
        "--encoder",
        default="resnet18_1d",
        help="Only aggregate runs with this CSI encoder (use 'all' to keep every encoder)",
    )
    args = parser.parse_args()

    root = Path(args.results_root).expanduser()
    if not root.is_dir():
        logger.error("Results root does not exist: %s", root)
        return 1

    runs = scan_runs(root)
    if args.encoder != "all":
        runs = [r for r in runs if r["encoder"] == args.encoder]
    if not runs:
        logger.error("No runs with a retrieval_m2m block found under %s", root)
        return 1

    tasks = _order_key(sorted({r["task"] for r in runs}), TASK_ORDER)
    methods = _order_key(sorted({r["method"] for r in runs}), METHOD_ORDER)
    logger.info("Aggregating %d runs: %d tasks x %d methods", len(runs), len(tasks), len(methods))

    missing = [
        f"{t}/{m}" for t in tasks for m in methods
        if not any(r["task"] == t and r["method"] == m for r in runs)
    ]
    if missing:
        logger.warning("Missing %d task/method cells: %s", len(missing), ", ".join(missing))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.md").write_text(
        build_markdown(runs, tasks, methods), encoding="utf-8",
    )
    (output_dir / "tables.tex").write_text(
        build_latex(runs, tasks, methods), encoding="utf-8",
    )
    with open(output_dir / "m2m_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results_root": str(root),
                "encoder": args.encoder,
                "tasks": tasks,
                "methods": methods,
                "missing_cells": missing,
                "runs": runs,
            },
            f, indent=2,
        )

    logger.info("Wrote summary.md, tables.tex and m2m_retrieval.json to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
