"""Collect the compositional-generalization results into paper-ready tables.

Reads the ``composition_analysis.json`` written by
``scripts/run_composition_analysis.py`` for each model (baseline, HOC-strict,
HOC-text) plus the audit and split manifest, and emits a comparison table in
Markdown and LaTeX.

Models that have not been trained yet are reported as missing rather than
silently dropped, so the table always shows what is still outstanding.

Usage
-----
PYTHONPATH=. python scripts/summarize_composition_results.py \
    --results_dir paper_results/composition
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "baseline": "Baseline (all combinations trained)",
    "hoc_strict": "HOC-strict (samples withheld)",
    "hoc_text": "HOC-text (composite captions withheld)",
}

NA = "--"


def _fmt(value: Optional[float], scale: float = 100.0, digits: int = 1) -> str:
    if value is None:
        return NA
    return f"{scale * value:.{digits}f}"


def _get(d: dict[str, Any], *path: str) -> Any:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _load(results_dir: Path, name: str) -> Optional[dict[str, Any]]:
    path = results_dir / name / "composition_analysis.json"
    if not path.is_file():
        logger.warning("Missing %s -- has %s been trained and analysed?", path, name)
        return None
    with open(path) as f:
        return json.load(f)


def _markdown(
    audit: Optional[dict[str, Any]],
    manifest: Optional[dict[str, Any]],
    models: dict[str, Optional[dict[str, Any]]],
) -> str:
    lines: list[str] = ["# Unseen-Composition Results", ""]

    if audit:
        lines.append(
            f"The co-labeled subset realises **{audit['realized_triples']} of "
            f"{audit['total_combinations']}** attribute combinations "
            f"({100 * audit['coverage']:.1f}%), so the effective Level 2 "
            f"candidate set is {audit['effective_combinations']:.0f}, not "
            f"{audit['total_combinations']}."
        )
        lines.append("")

    if manifest:
        sel = manifest["selection"]
        enforced = ", ".join(
            p.replace("__", " x ").replace("user_identity", "user")
            for p in sel.get("pair_coverage_enforced", [])
        )
        lines.append(
            f"**{sel['num_held_out']} of {sel['num_train_combinations']}** "
            f"training combinations are withheld "
            f"({100 * sel['holdout_fraction_achieved']:.0f}%), covering "
            f"{sel['unseen_test_samples']} unseen and "
            f"{sel['seen_test_samples']} seen test samples. Every attribute "
            f"value and every {enforced or 'enforceable'} cell still occurs in "
            f"training, so each unseen combination is built from parts the "
            f"model has seen."
        )
        lines.append("")

    lines.append("## Level 2 composite retrieval, seen vs unseen combinations")
    lines.append("")
    lines.append(
        "| Model | Group | Setting | Samples | Top-1 (%) | Top-3 (%) | MRR "
        "| Activity (%) | Proximity (%) | User (%) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for name, label in MODEL_LABELS.items():
        res = models.get(name)
        if res is None:
            lines.append(f"| {label} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} |")
            continue
        breakdown = res.get("seen_unseen", {})
        for setting in ("generalized", "closed"):
            for group in ("seen", "unseen"):
                m = _get(breakdown, setting, group)
                if not m or not m.get("num_samples"):
                    continue
                dec = m.get("decomposed_per_attribute", {})
                lines.append(
                    f"| {label} | {group} | {setting} | {m['num_samples']} "
                    f"| {_fmt(m.get('top1_accuracy'))} "
                    f"| {_fmt(m.get('top3_accuracy'))} "
                    f"| {m.get('mrr', float('nan')):.3f} "
                    f"| {_fmt(dec.get('activity'))} "
                    f"| {_fmt(dec.get('proximity'))} "
                    f"| {_fmt(dec.get('user_identity'))} |"
                )
    lines.append("")

    lines.append("## Reference points on the same test samples")
    lines.append("")
    lines.append(
        "| Model | Composite Top-1 (%) | Realized-only Top-1 (%) "
        "| Factorized Top-1 (%) | Level 1 joint (%) |"
    )
    lines.append("|---|---|---|---|---|")
    for name, label in MODEL_LABELS.items():
        res = models.get(name)
        if res is None:
            lines.append(f"| {label} | {NA} | {NA} | {NA} | {NA} |")
            continue
        lines.append(
            f"| {label} "
            f"| {_fmt(_get(res, 'candidate_set_control', 'all_combinations', 'top1_accuracy'))} "
            f"| {_fmt(_get(res, 'candidate_set_control', 'realized_only', 'top1_accuracy'))} "
            f"| {_fmt(_get(res, 'factorized_reference', 'top1_accuracy'))} "
            f"| {_fmt(_get(res, 'factorized_reference', 'independent_argmax_joint_accuracy'))} |"
        )
    lines.append("")

    baseline = models.get("baseline")
    if baseline:
        lines.append("## Zero-training controls (baseline model)")
        lines.append("")
        lines.append("| Slot | Options | Accuracy (%) | Chance (%) |")
        lines.append("|---|---|---|---|")
        for attr, stats in baseline.get("slot_counterfactual", {}).items():
            lines.append(
                f"| {attr} | {stats['num_options']} "
                f"| {_fmt(stats['accuracy'])} | {_fmt(stats['chance'])} |"
            )
        lines.append("")
        lines.append("| Composition pattern | Top-1 (%) | MRR |")
        lines.append("|---|---|---|")
        for fam, m in baseline.get("novel_templates", {}).get("families", {}).items():
            lines.append(
                f"| {fam} | {_fmt(m.get('top1_accuracy'))} "
                f"| {m.get('mrr', float('nan')):.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _latex(
    audit: Optional[dict[str, Any]],
    manifest: Optional[dict[str, Any]],
    models: dict[str, Optional[dict[str, Any]]],
) -> str:
    short = {
        "baseline": "Baseline (all trained)",
        "hoc_strict": "HOC-strict",
        "hoc_text": "HOC-text",
    }

    rows: list[str] = []
    for position, (name, label) in enumerate(short.items()):
        if position:
            rows.append("\\midrule")
        res = models.get(name)
        if res is None:
            rows.append(
                f"{label} & \\multicolumn{{8}}{{c}}{{\\textit{{not yet trained}}}} \\\\"
            )
            continue
        breakdown = res.get("seen_unseen", {})
        fac = _fmt(_get(res, "factorized_reference", "top1_accuracy"))
        lvl1 = _fmt(_get(res, "factorized_reference", "independent_argmax_joint_accuracy"))
        first = True
        for group in ("seen", "unseen"):
            gen = _get(breakdown, "generalized", group)
            if not gen or not gen.get("num_samples"):
                continue
            closed = _get(breakdown, "closed", group) or {}
            dec = gen.get("decomposed_per_attribute", {})
            prefix = label if first else ""
            first = False
            rows.append(
                f"{prefix} & {group} & {gen['num_samples']} "
                f"& {_fmt(gen.get('top1_accuracy'))} "
                f"& {_fmt(gen.get('top3_accuracy'))} "
                f"& {gen.get('mrr', float('nan')):.3f} "
                f"& {_fmt(closed.get('top1_accuracy'))} "
                f"& {fac if group == 'unseen' else NA} "
                f"& {lvl1 if group == 'unseen' else NA} \\\\"
            )

    caption_bits = []
    if audit:
        caption_bits.append(
            f"Only {audit['realized_triples']} of {audit['total_combinations']} "
            f"attribute combinations occur in the co-labeled subset"
        )
    if manifest:
        sel = manifest["selection"]
        caption_bits.append(
            f"{sel['num_held_out']} of {sel['num_train_combinations']} are "
            f"withheld from training, with every attribute value and every "
            f"activity--proximity cell still present"
        )
    caption = ". ".join(caption_bits)

    body = "\n".join(rows) if rows else "\\multicolumn{9}{c}{no results yet} \\\\"

    return (
        "% Generated by scripts/summarize_composition_results.py -- do not edit by hand.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Level 2 composite scene retrieval on seen versus unseen "
        "attribute combinations. " + caption + ". All rows are measured on the "
        "same test split; for the baseline, which was trained on every "
        "combination, the unseen rows act as a control showing that those test "
        "samples are not intrinsically harder. Generalized scores compete "
        "against all 140 candidates; closed scores against the held-out "
        "combinations only. The factorized reference scores each combination as "
        "the sum of the three independent Level 1 similarities, and Level 1 "
        "joint is the accuracy of three separate per-attribute argmaxes.}\n"
        "\\label{tab:unseen-composition}\n"
        "\\small\n"
        "\\begin{tabular}{llrrrrrrr}\n"
        "\\toprule\n"
        " & & & \\multicolumn{3}{c}{Generalized (140-way)} & Closed & "
        "\\multicolumn{2}{c}{Composition-free} \\\\\n"
        "\\cmidrule(lr){4-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-9}\n"
        "Model & Combinations & $N$ & Top-1 & Top-3 & MRR & Top-1 & "
        "Factorized & Level 1 joint \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


PRETTY_ATTR = {
    "user_identity": "User",
    "activity": "Activity",
    "proximity": "Proximity",
}


def _audit_table_tex(audit: Optional[dict[str, Any]]) -> str:
    """Attribute-pair coverage and dependence, for the appendix."""
    if not audit:
        return "% No composition_audit.json found.\n"

    # LaTeX math mode keeps the thousands separator from picking up extra space.
    num_samples = f"{audit['num_samples']:,}".replace(",", "{,}")

    rows = []
    for pair, stats in audit["pairwise"].items():
        left, right = (PRETTY_ATTR.get(p, p) for p in pair.split("__"))
        rows.append(
            f"{left} $\\times$ {right} & {stats['realized_cells']} / "
            f"{stats['total_cells']} & {100 * stats['coverage']:.1f} & "
            f"{stats['mutual_information_bits']:.3f} & "
            f"{stats['normalized_mi']:.3f} \\\\"
        )

    return (
        "% Generated by scripts/summarize_composition_results.py "
        "-- do not edit by hand.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Attribute-combination structure of the co-labeled CSI-Bench "
        f"subset ({num_samples} samples). Only "
        f"{audit['realized_triples']} of {audit['total_combinations']} "
        "combinations occur, so the effective Level 2 candidate set is "
        f"{audit['effective_combinations']:.0f}. The user axis is fully crossed "
        "with the recorded activity--proximity conditions; the sparsity is "
        "concentrated in activity $\\times$ proximity, because activities other "
        "than walking were recorded only at 1.5\\,m.}\n"
        "\\label{tab:composition-audit}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Attribute pair & Realized cells & Coverage (\\%) & MI (bits) & "
        "Normalized MI \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _heldout_table_tex(manifest: Optional[dict[str, Any]]) -> str:
    """The withheld combinations and their per-split sample counts."""
    counts = (manifest or {}).get("held_out_sample_counts")
    if not counts:
        return "% No composition_split.json found.\n"

    rows = []
    for triple, per_split in counts.items():
        user, activity, prox = triple.split("|")
        user = user.replace("user_", "User~")
        activity = activity.replace("_", " ")
        rows.append(
            f"{user} & {activity} & {prox} & {per_split.get('train', 0)} & "
            f"{per_split.get('val', 0)} & {per_split.get('test', 0)} \\\\"
        )

    sel = manifest["selection"]
    return (
        "% Generated by scripts/summarize_composition_results.py "
        "-- do not edit by hand.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{The {sel['num_held_out']} attribute combinations withheld "
        f"from training, out of {sel['num_train_combinations']} that occur. "
        "Train and val samples carrying these combinations are removed "
        "(HOC-strict) or keep only their Level 1 captions (HOC-text); the test "
        "split is untouched in both cases. Every attribute value and every "
        "activity--proximity cell still occurs in training, so each withheld "
        "combination is built from parts the model has seen.}\n"
        "\\label{tab:composition-heldout}\n"
        "\\small\n"
        "\\begin{tabular}{lllrrr}\n"
        "\\toprule\n"
        "User & Activity & Proximity & Train & Val & Test \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _controls_table_tex(models: dict[str, Optional[dict[str, Any]]]) -> str:
    """Zero-training controls on the baseline scene model."""
    res = models.get("baseline") or {}

    slot_rows = [
        f"{PRETTY_ATTR.get(attr, attr)} & {stats['num_options']} & "
        f"{_fmt(stats['accuracy'])} & {_fmt(stats['chance'])} \\\\"
        for attr, stats in res.get("slot_counterfactual", {}).items()
    ] or [f"\\multicolumn{{4}}{{c}}{{{NA}}} \\\\"]

    families = res.get("novel_templates", {}).get("families", {})
    pattern_rows = [
        f"{name.replace('_', ' ')} & {_fmt(m.get('top1_accuracy'))} & "
        f"{m.get('mrr', float('nan')):.3f} \\\\"
        for name, m in families.items()
    ] or [f"\\multicolumn{{3}}{{c}}{{{NA}}} \\\\"]

    return (
        "% Generated by scripts/summarize_composition_results.py "
        "-- do not edit by hand.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Zero-training controls on the baseline scene model. "
        "\\textbf{Left}: single-slot counterfactual probe, varying one "
        "attribute while the other two are fixed to ground truth. "
        "\\textbf{Right}: Level 2 retrieval under the four composition patterns "
        "seen in training and three patterns that reorder the slots and were "
        "never used to build a caption.}\n"
        "\\label{tab:composition-controls}\n"
        "\\small\n"
        "\\begin{minipage}{0.46\\linewidth}\n"
        "\\centering\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Slot & Options & Acc.\\ (\\%) & Chance (\\%) \\\\\n"
        "\\midrule\n" + "\n".join(slot_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{minipage}\\hfill\n"
        "\\begin{minipage}{0.50\\linewidth}\n"
        "\\centering\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Composition pattern & Top-1 (\\%) & MRR \\\\\n"
        "\\midrule\n" + "\n".join(pattern_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )


def _numbers_tex(
    audit: Optional[dict[str, Any]],
    manifest: Optional[dict[str, Any]],
    models: dict[str, Optional[dict[str, Any]]],
) -> str:
    """Emit the scalars the prose cites, so the text never needs hand-editing."""
    macros: dict[str, str] = {}

    if audit:
        macros["RealizedCombinations"] = str(audit["realized_triples"])
        macros["TotalCombinations"] = str(audit["total_combinations"])
        macros["CombinationCoverage"] = f"{100 * audit['coverage']:.1f}"
        macros["EffectiveCombinations"] = f"{audit['effective_combinations']:.0f}"
        macros["ChanceRealized"] = _fmt(
            _get(audit, "prior_baselines", "chance_realized_only"), digits=2,
        )
        macros["MajorityCombination"] = _fmt(
            _get(audit, "prior_baselines", "majority_triple_accuracy"), digits=2,
        )
        ap = _get(audit, "pairwise", "activity__proximity") or {}
        macros["ActProxCells"] = str(ap.get("realized_cells", ""))
        macros["ActProxTotalCells"] = str(ap.get("total_cells", ""))
        macros["ActProxNMI"] = f"{ap.get('normalized_mi', float('nan')):.2f}"

    if manifest:
        sel = manifest["selection"]
        macros["HeldOutCombinations"] = str(sel["num_held_out"])
        macros["TrainCombinations"] = str(sel["num_train_combinations"])
        macros["HoldoutPercent"] = f"{100 * sel['holdout_fraction_achieved']:.0f}"
        macros["UnseenTestSamples"] = str(sel["unseen_test_samples"])
        macros["SeenTestSamples"] = str(sel["seen_test_samples"])

    for name in MODEL_LABELS:
        res = models.get(name)
        stem = "".join(part.capitalize() for part in name.split("_"))
        macros[f"{stem}Seen"] = _fmt(
            _get(res or {}, "seen_unseen", "generalized", "seen", "top1_accuracy")
        )
        macros[f"{stem}Unseen"] = _fmt(
            _get(res or {}, "seen_unseen", "generalized", "unseen", "top1_accuracy")
        )
        macros[f"{stem}UnseenClosed"] = _fmt(
            _get(res or {}, "seen_unseen", "closed", "unseen", "top1_accuracy")
        )
        macros[f"{stem}Factorized"] = _fmt(
            _get(res or {}, "factorized_reference", "top1_accuracy")
        )
        macros[f"{stem}LevelOneJoint"] = _fmt(
            _get(res or {}, "factorized_reference",
                 "independent_argmax_joint_accuracy")
        )

    lines = [
        "% Generated by scripts/summarize_composition_results.py "
        "-- do not edit by hand.",
        "% Values shown as -- are still waiting on a training run.",
    ]
    lines += [f"\\newcommand{{\\comp{k}}}{{{v}}}" for k, v in macros.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the unseen-composition experiment",
    )
    parser.add_argument(
        "--results_dir", type=str, default="paper_results/composition",
    )
    parser.add_argument(
        "--paper_dir", type=str, default="paper/latex/generated",
        help="Where to write the table and macros the paper inputs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results_dir = Path(args.results_dir)

    audit = None
    audit_path = results_dir / "composition_audit.json"
    if audit_path.is_file():
        with open(audit_path) as f:
            audit = json.load(f)

    manifest = None
    manifest_path = results_dir / "composition_split.json"
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)

    models = {name: _load(results_dir, name) for name in MODEL_LABELS}

    md_path = results_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(_markdown(audit, manifest, models))

    table_tex = _latex(audit, manifest, models)
    numbers_tex = _numbers_tex(audit, manifest, models)

    tex_path = results_dir / "tables.tex"
    with open(tex_path, "w") as f:
        f.write(table_tex)

    written = [md_path, tex_path]
    if args.paper_dir:
        paper_dir = Path(args.paper_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("composition_table.tex", table_tex),
            ("composition_numbers.tex", numbers_tex),
            ("composition_audit_table.tex", _audit_table_tex(audit)),
            ("composition_heldout_table.tex", _heldout_table_tex(manifest)),
            ("composition_controls_table.tex", _controls_table_tex(models)),
        ):
            with open(paper_dir / name, "w") as f:
                f.write(content)
            written.append(paper_dir / name)

    for path in written:
        print(f"Wrote {path}")

    missing = [n for n, r in models.items() if r is None]
    if missing:
        print(f"Still missing analyses for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
