#!/usr/bin/env python3
"""Assess how much headroom the benchmark still has.

Most tasks sit above 90% accuracy, which invites the question of whether the
benchmark is saturated.  This script assembles the evidence that it is not:

1. **Saturation table** -- per task, accuracy against its supervised ceiling
   next to many-to-many mAP against its chance floor.  Accuracy is close to
   saturated on the coarse tasks; ranking quality is not.
2. **Hardest slices** -- protocols already in the repository where performance
   is far from ceiling (unseen classes, open-vocabulary specialist registers,
   joint scene QA, the largest label spaces).
3. **Separability scatter** -- per-task text-side separability against mAP,
   linking the template-overlap audit to the retrieval results.

Inputs are the artefacts written by ``aggregate_m2m_retrieval.py`` and
``analyze_template_overlap.py``; both are optional and the corresponding
sections are skipped when absent.

Usage::

    python scripts/analyze_benchmark_difficulty.py --output_dir paper_results/m2m_retrieval
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

# Reference numbers transcribed from paper_results/RESULTS_ORGANIZED.md
# (Table 1a zero-shot, best method per task, and the supervised baseline row).
# Keep in sync if those tables are regenerated.
TASK_REFERENCE: dict[str, dict[str, Any]] = {
    "har": {
        "label": "HAR", "num_classes": 5,
        "best_zero_shot": 0.987, "best_method": "CLIP InfoNCE", "supervised": 0.979,
    },
    "fall_detection": {
        "label": "Fall detection", "num_classes": 2,
        "best_zero_shot": 0.959, "best_method": "SigLIP", "supervised": 0.955,
    },
    "motion_source": {
        "label": "Motion source", "num_classes": 4,
        "best_zero_shot": 0.996, "best_method": "CLIP InfoNCE", "supervised": 0.998,
    },
    "proximity": {
        "label": "Proximity", "num_classes": 4,
        "best_zero_shot": 0.955, "best_method": "Masked Pred.", "supervised": 0.963,
    },
    "user_id": {
        "label": "User ID", "num_classes": 7,
        "best_zero_shot": 1.000, "best_method": "SigLIP", "supervised": 1.000,
    },
    "action_recognition": {
        "label": "XRF55 action", "num_classes": 55,
        "best_zero_shot": 0.781, "best_method": "CLIP InfoNCE", "supervised": 0.829,
    },
    "sign_recognition": {
        "label": "SignFi sign", "num_classes": 276,
        "best_zero_shot": 0.958, "best_method": "CLIP InfoNCE", "supervised": 0.994,
    },
}

TASK_ORDER = list(TASK_REFERENCE)

# Protocols already in the repository where performance is far from ceiling.
# (metric, value, chance, source) -- see paper_results/RESULTS_ORGANIZED.md.
HARD_SLICES: list[dict[str, Any]] = [
    {
        "setting": "SignFi held-out classes, zero-shot",
        "metric": "Accuracy", "value": 0.113, "chance": 1 / 76,
        "source": "Table 2a (76 unseen classes)",
        "note": "Unseen sign vocabulary is essentially unsolved.",
    },
    {
        "setting": "SignFi held-out classes, 10-shot",
        "metric": "Accuracy", "value": 0.746, "chance": 1 / 76,
        "source": "Table 2a",
        "note": "Ten labelled examples recover most, but not all, of the gap.",
    },
    {
        "setting": "Open-vocabulary HAR, sensor-grounded prompts",
        "metric": "Accuracy", "value": 0.354, "chance": 1 / 5,
        "source": "Table 5",
        "note": "WiFi-physics vocabulary is barely covered by the text encoder.",
    },
    {
        "setting": "Open-vocabulary HAR, domain-expert prompts",
        "metric": "Accuracy", "value": 0.591, "chance": 1 / 5,
        "source": "Table 5",
        "note": "Clinical vocabulary, unanchored.",
    },
    {
        "setting": "Open-vocabulary HAR, sensor-grounded anchored",
        "metric": "Accuracy", "value": 0.771, "chance": 1 / 5,
        "source": "Table 5",
        "note": "Prompt anchoring helps but does not close the gap.",
    },
    {
        "setting": "Ambient QA Level 1, joint (all 3 attributes)",
        "metric": "Accuracy", "value": 0.704, "chance": 1 / (5 * 4 * 7),
        "source": "Table 3b",
        "note": "Per-attribute accuracy is high; getting all three right is not.",
    },
    {
        "setting": "Ambient QA Level 2, 140-way composite",
        "metric": "Top-1", "value": 0.869, "chance": 1 / 140,
        "source": "Table 3c",
        "note": "Composite scene retrieval, the hardest QA setting.",
    },
    {
        "setting": "XRF55 action recognition (55 classes)",
        "metric": "Accuracy", "value": 0.781, "chance": 1 / 55,
        "source": "Table 1a",
        "note": "Below its own supervised ceiling of 82.9%.",
    },
    {
        "setting": "Multitask-all model, proximity",
        "metric": "Accuracy", "value": 0.845, "chance": 1 / 4,
        "source": "Table 3a",
        "note": "Joint training costs 10 points versus the single-task model.",
    },
    {
        "setting": "Barlow Twins, motion source",
        "metric": "Accuracy", "value": 0.612, "chance": 1 / 4,
        "source": "Table 1a",
        "note": "Method choice still matters enormously on some tasks.",
    },
]


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "--"


def _pct(value: Optional[float]) -> str:
    return f"{value*100:.1f}%" if value is not None else "--"


def load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        logger.warning("Not found, skipping: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def best_m2m_per_task(m2m: Optional[dict]) -> dict[str, dict[str, Any]]:
    """Best strict CSI->Text run per task, keyed by task name."""
    if not m2m:
        return {}
    best: dict[str, dict[str, Any]] = {}
    for run in m2m.get("runs", []):
        strict = run.get("m2m", {}).get("csi_to_text", {}).get("strict", {})
        m_ap = strict.get("mAP")
        if not isinstance(m_ap, (int, float)):
            continue
        task = run["task"]
        if task not in best or m_ap > best[task]["mAP"]:
            best[task] = {
                "method": run["method"],
                "mAP": float(m_ap),
                "chance_mAP": strict.get("chance", {}).get("mAP"),
                "normalized_mAP": strict.get("normalized_mAP"),
                "mAP_macro": strict.get("mAP_macro"),
                "mAP_worst_class": strict.get("mAP_worst_class"),
                "hit_rate@1": strict.get("hit_rate@1"),
                "ndcg@10": strict.get("ndcg@10"),
                "gallery": run.get("m2m", {}).get("gallery", {}).get("num_texts"),
            }
    return best


def separability_per_task(overlap: Optional[dict]) -> dict[str, dict[str, Any]]:
    if not overlap:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for audit in overlap.get("audits", []):
        em = audit.get("embedding", {})
        if "skipped" in em:
            continue
        out[audit["task"]] = {
            "nn1_same_class_rate": em.get("nn1_same_class_rate"),
            "text_only_probe_accuracy": em.get("text_only_probe_accuracy"),
            "jaccard_between": audit.get("lexical", {}).get("mean_jaccard_between_class"),
            "collision_rate": audit.get("lexical", {}).get("between_class_collision_rate"),
            "fraction_near_duplicate": audit.get("near_duplicates", {}).get("fraction_removed"),
        }
    return out


def saturation_lines(best: dict[str, dict]) -> list[str]:
    lines = [
        "## Saturation and headroom",
        "",
        "Accuracy is measured against a supervised ceiling; many-to-many mAP is "
        "measured against an exact chance floor. `Gap to sup.` is how far "
        "zero-shot accuracy still is from the supervised baseline (negative "
        "means the aligned model already beats it), and `mAP headroom` is how "
        "much of the range between chance and a perfect ranking remains. A task "
        "can be simultaneously saturated on accuracy and wide open on mAP -- "
        "that is the case for most of this benchmark.",
        "",
        "| Task | Classes | Chance acc | Best zero-shot | Supervised | Gap to sup. "
        "| Best mAP | Chance mAP | Norm. mAP | mAP headroom |",
        "|------|---------|------------|----------------|------------|-------------"
        "|----------|------------|-----------|--------------|",
    ]
    for task in TASK_ORDER:
        ref = TASK_REFERENCE[task]
        b = best.get(task, {})
        chance_acc = 1.0 / ref["num_classes"]
        acc_gap = ref["supervised"] - ref["best_zero_shot"]
        norm = b.get("normalized_mAP")
        map_headroom = (1.0 - norm) if isinstance(norm, (int, float)) else None
        lines.append(
            f"| {ref['label']} | {ref['num_classes']} | {_pct(chance_acc)} "
            f"| {_pct(ref['best_zero_shot'])} ({ref['best_method']}) "
            f"| {_pct(ref['supervised'])} | {acc_gap*100:+.1f} pp "
            f"| {_fmt(b.get('mAP'))} | {_fmt(b.get('chance_mAP'))} "
            f"| {_fmt(norm)} | {_pct(map_headroom)} |"
        )
    lines.extend([
        "",
        "| Task | Best mAP | Macro mAP | Worst-class mAP | hit@1 | hit@1 - mAP |",
        "|------|----------|-----------|-----------------|-------|-------------|",
    ])
    for task in TASK_ORDER:
        b = best.get(task)
        if not b:
            continue
        hit, m_ap = b.get("hit_rate@1"), b.get("mAP")
        gap = (hit - m_ap) if (hit is not None and m_ap is not None) else None
        lines.append(
            f"| {TASK_REFERENCE[task]['label']} | {_fmt(m_ap)} "
            f"| {_fmt(b.get('mAP_macro'))} | {_fmt(b.get('mAP_worst_class'))} "
            f"| {_fmt(hit)} | {_fmt(gap)} |"
        )
    lines.append("")
    return lines


def hard_slices_lines() -> list[str]:
    lines = [
        "## Hardest existing slices",
        "",
        "Protocols already reported in this repository where the benchmark is "
        "clearly unsolved. The headline 90%-plus numbers come from in-distribution "
        "zero-shot classification on coarse label sets; every axis that moves away "
        "from that drops sharply.",
        "",
        "| Setting | Metric | Value | Chance | Source |",
        "|---------|--------|-------|--------|--------|",
    ]
    for slice_ in HARD_SLICES:
        lines.append(
            f"| {slice_['setting']} | {slice_['metric']} | {_pct(slice_['value'])} "
            f"| {_pct(slice_['chance'])} | {slice_['source']} |"
        )
    lines.extend(["", "Notes:", ""])
    for slice_ in HARD_SLICES:
        lines.append(f"- **{slice_['setting']}** -- {slice_['note']}")
    lines.append("")
    return lines


def separability_lines(
    best: dict[str, dict], separability: dict[str, dict],
) -> list[str]:
    if not separability:
        return []
    lines = [
        "## Text-side separability versus retrieval quality",
        "",
        "If the captions alone already determine the class, then the text half of "
        "the retrieval problem contributes no difficulty and the score reflects "
        "the CSI encoder plus the caption vocabulary. Comparing the two columns "
        "shows how much of the retrieval score survives once the model has to "
        "rank *all* valid captions rather than pick a class.",
        "",
        "| Task | 1-NN same class | Text-only probe | Between-class Jaccard "
        "| Near-dup. removed | Best strict mAP |",
        "|------|-----------------|-----------------|-----------------------"
        "|-------------------|-----------------|",
    ]
    for task in TASK_ORDER:
        sep = separability.get(task)
        if not sep:
            continue
        lines.append(
            f"| {TASK_REFERENCE[task]['label']} "
            f"| {_fmt(sep.get('nn1_same_class_rate'))} "
            f"| {_fmt(sep.get('text_only_probe_accuracy'))} "
            f"| {_fmt(sep.get('jaccard_between'))} "
            f"| {_pct(sep.get('fraction_near_duplicate'))} "
            f"| {_fmt(best.get(task, {}).get('mAP'))} |"
        )
    lines.append("")
    return lines


def write_scatter(
    best: dict[str, dict], separability: dict[str, dict], path: Path,
) -> bool:
    points = [
        (
            separability[t].get("text_only_probe_accuracy")
            or separability[t].get("nn1_same_class_rate"),
            best[t]["mAP"],
            TASK_REFERENCE[t]["label"],
        )
        for t in TASK_ORDER
        if t in separability and t in best
    ]
    points = [p for p in points if p[0] is not None and p[1] is not None]
    if len(points) < 2:
        logger.warning("Not enough paired points for the separability scatter")
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping the scatter figure")
        return False

    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=70)
    for x, y, label in points:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Text-side separability (caption alone predicts the class)")
    ax.set_ylabel("Best CSI $\\rightarrow$ Text mAP (strict relevance)")
    ax.set_title("Caption separability does not translate into ranking quality")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return True


def build_latex(best: dict[str, dict]) -> str:
    rows = []
    for task in TASK_ORDER:
        ref = TASK_REFERENCE[task]
        b = best.get(task, {})
        rows.append(
            f"{ref['label']} & {ref['num_classes']} & "
            f"{ref['best_zero_shot']*100:.1f} & {ref['supervised']*100:.1f} & "
            f"{_fmt(b.get('chance_mAP'), 3)} & {_fmt(b.get('mAP'), 3)} & "
            f"{_fmt(b.get('mAP_macro'), 3)} & {_fmt(b.get('mAP_worst_class'), 3)} & "
            f"{_fmt(b.get('normalized_mAP'), 3)} \\\\"
        )
    return "\n".join([
        "% Auto-generated by scripts/analyze_benchmark_difficulty.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "Task & $C$ & Zero-shot & Supervised & Chance mAP & mAP & Macro mAP & "
        "Worst class & Norm. mAP \\\\",
        "\\hline",
        *rows,
        "\\hline",
        "\\end{tabular}",
        "\\caption{Accuracy is near its supervised ceiling on most tasks, but "
        "many-to-many mAP is far from 1 and the worst-class mAP is lower still. "
        "Zero-shot and supervised accuracies are percentages for the best method "
        "per task; mAP columns are the best method's CSI $\\rightarrow$ Text "
        "score under strict relevance, and normalised mAP rescales onto "
        "[chance $\\rightarrow$ 1].}",
        "\\label{tab:benchmark-headroom}",
        "\\end{table}",
        "",
    ])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="paper_results/m2m_retrieval")
    parser.add_argument(
        "--m2m_json",
        default=None,
        help="Defaults to <output_dir>/m2m_retrieval.json from aggregate_m2m_retrieval.py",
    )
    parser.add_argument(
        "--overlap_json",
        default="paper_results/template_overlap/overlap_stats.json",
        help="Written by analyze_template_overlap.py",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    m2m_path = Path(args.m2m_json) if args.m2m_json else output_dir / "m2m_retrieval.json"

    m2m = load_json(m2m_path)
    overlap = load_json(Path(args.overlap_json))

    best = best_m2m_per_task(m2m)
    separability = separability_per_task(overlap)

    if not best:
        logger.warning(
            "No many-to-many results found at %s; the mAP columns will be empty. "
            "Run scripts/run_m2m_retrieval_evals.sh then aggregate_m2m_retrieval.py first.",
            m2m_path,
        )

    lines = [
        "# Benchmark Difficulty and Headroom",
        "",
        "Six of seven tasks exceed 90% zero-shot accuracy, which raises the "
        "question of whether the benchmark is saturated. This report separates "
        "two things that the accuracy number conflates: picking the right class "
        "from a handful of candidates, which is largely solved, and ranking a "
        "sample against every caption it is validly paired with, which is not.",
        "",
    ]
    lines.extend(saturation_lines(best))
    lines.extend(hard_slices_lines())
    lines.extend(separability_lines(best, separability))

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "difficulty_vs_separability.png"
    if write_scatter(best, separability, figure_path):
        lines.extend([
            "## Figure",
            "",
            f"![Caption separability versus mAP]({figure_path.name})",
            "",
        ])

    (output_dir / "benchmark_difficulty.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "benchmark_difficulty.tex").write_text(build_latex(best), encoding="utf-8")
    with open(output_dir / "benchmark_difficulty.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_reference": TASK_REFERENCE,
                "best_m2m_per_task": best,
                "text_separability": separability,
                "hard_slices": HARD_SLICES,
            },
            f, indent=2,
        )

    logger.info("Wrote benchmark_difficulty.{md,tex,json} to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
