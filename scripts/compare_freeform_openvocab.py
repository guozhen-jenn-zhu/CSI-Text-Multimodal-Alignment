"""Compare open-vocabulary HAR results: template-trained vs freeform-trained.

Reads the ``har_novel_prompts.json`` file(s) produced by
``scripts/run_open_vocab_eval.py`` for each model and writes a markdown report
with **two separate tables**:

1. **Original open-vocabulary families** — ``original_templates`` plus the eight
   pre-existing novel families. The template-model numbers here can be *reused*
   from a previously cached results file (no re-run needed).
2. **New held-out freeform families** — ``freeform_colloquial`` and
   ``freeform_colloquial_ensemble`` (added later). Because the cached template
   results predate these families, score the template checkpoint on just these
   families with ``run_open_vocab_eval.py --families ...`` and pass that file in
   addition to the cached one; ``--template`` accepts multiple files and merges
   their ``results`` (later files win on key conflicts).

Usage
-----
# Reuse cached template results for the original families, add a small file
# with the template model scored on only the two new families:
PYTHONPATH=. python scripts/compare_freeform_openvocab.py \
    --template paper_results/open_vocabulary/original_template_vit1d/har_novel_prompts.json \
               paper_results/open_vocabulary/template_new_families/har_novel_prompts.json \
    --freeform paper_results/open_vocabulary/freeform_trained/har_novel_prompts.json \
    --out      paper_results/open_vocabulary/freeform_vs_template.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

# Family groupings ----------------------------------------------------------
ORIGINAL_FAMILIES = [
    "original_templates",
    "domain_expert",
    "layperson",
    "sensor_grounded",
    "domain_expert_anchored",
    "sensor_grounded_anchored",
    "domain_expert_ensemble",
    "layperson_ensemble",
    "sensor_grounded_ensemble",
]
NEW_FAMILIES = [
    "freeform_colloquial",
    "freeform_colloquial_ensemble",
]


def _load_and_merge(paths: list[str]) -> dict[str, Any]:
    """Load one or more result files and merge their ``results`` dicts.

    Later files override earlier ones on family-key conflicts. Model metadata
    and num_test_samples are taken from the first file that provides them.
    """
    merged: dict[str, Any] = {"model": {}, "num_test_samples": None, "results": {}}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not merged["model"] and data.get("model"):
            merged["model"] = data["model"]
        if merged["num_test_samples"] is None and data.get("num_test_samples") is not None:
            merged["num_test_samples"] = data.get("num_test_samples")
        merged["results"].update(data.get("results", {}))
        merged.setdefault("_source_files", []).append(str(p))
    return merged


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else "N/A"


def _fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    """b - a, in percentage points, with sign."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return f"{(b - a) * 100.0:+.1f}"
    return "N/A"


def _model_desc(meta: dict[str, Any]) -> str:
    m = meta.get("model", {})
    return (
        f"method={m.get('method', '?')}, "
        f"csi={m.get('csi_encoder', '?')}, "
        f"text={m.get('text_encoder', '?')}"
    )


def _render_table(
    families: list[str],
    tmpl_res: dict[str, Any],
    free_res: dict[str, Any],
) -> list[str]:
    lines = [
        "| Prompt Family | Description | #T/C | Tmpl Acc | Free Acc | ΔAcc (pp) | "
        "Tmpl wF1 | Free wF1 | ΔwF1 (pp) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    any_row = False
    for fam in families:
        t = tmpl_res.get(fam, {})
        f = free_res.get(fam, {})
        if not t and not f:
            continue  # neither model has this family -> skip
        any_row = True
        desc = t.get("description") or f.get("description") or (
            "Training templates (baseline)" if fam == "original_templates" else ""
        )
        tc = t.get("texts_per_class", f.get("texts_per_class", "?"))
        t_acc, f_acc = t.get("accuracy"), f.get("accuracy")
        t_f1, f_f1 = t.get("weighted_f1"), f.get("weighted_f1")
        lines.append(
            f"| {fam} | {desc} | {tc} | "
            f"{_fmt_pct(t_acc)} | {_fmt_pct(f_acc)} | {_fmt_delta(t_acc, f_acc)} | "
            f"{_fmt_pct(t_f1)} | {_fmt_pct(f_f1)} | {_fmt_delta(t_f1, f_f1)} |"
        )
    if not any_row:
        lines.append("| _(no data)_ | | | | | | | | |")
    return lines


def _mean_acc(res: dict[str, Any], fams: list[str]) -> Optional[float]:
    vals = [res[x]["accuracy"] for x in fams if x in res and "accuracy" in res[x]]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare template-trained vs freeform-trained open-vocab HAR results",
    )
    parser.add_argument(
        "--template", nargs="+", required=True,
        help="One or more har_novel_prompts.json files for the template-trained "
             "model (merged; e.g. cached original families + a new-families file).",
    )
    parser.add_argument(
        "--freeform", nargs="+", required=True,
        help="One or more har_novel_prompts.json files for the freeform-trained model.",
    )
    parser.add_argument(
        "--out", default="paper_results/open_vocabulary/freeform_vs_template.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    tmpl = _load_and_merge(args.template)
    free = _load_and_merge(args.freeform)
    tmpl_res: dict[str, Any] = tmpl["results"]
    free_res: dict[str, Any] = free["results"]

    lines: list[str] = []
    lines.append("# Open-Vocabulary HAR: Freeform-trained vs Template-trained")
    lines.append("")
    lines.append(f"- Template-trained model: {_model_desc(tmpl)}")
    lines.append(f"- Freeform-trained model: {_model_desc(free)}")
    lines.append(f"- Template test samples: {tmpl.get('num_test_samples', '?')}")
    lines.append(f"- Freeform test samples: {free.get('num_test_samples', '?')}")

    # Fair-comparison guardrail: warn on encoder / sample-count mismatch.
    t_enc = tmpl.get("model", {}).get("csi_encoder")
    f_enc = free.get("model", {}).get("csi_encoder")
    t_txt = tmpl.get("model", {}).get("text_encoder")
    f_txt = free.get("model", {}).get("text_encoder")
    warnings: list[str] = []
    if t_enc and f_enc and t_enc != f_enc:
        warnings.append(
            f"CSI encoders differ (template={t_enc}, freeform={f_enc}); "
            "the comparison is confounded by architecture, not just text source."
        )
    if t_txt and f_txt and t_txt != f_txt:
        warnings.append(
            f"Text encoders differ (template={t_txt}, freeform={f_txt}); "
            "prompt embeddings are not comparable across a different text encoder."
        )
    if (tmpl.get("num_test_samples") and free.get("num_test_samples")
            and tmpl["num_test_samples"] != free["num_test_samples"]):
        warnings.append(
            f"Test-sample counts differ (template={tmpl['num_test_samples']}, "
            f"freeform={free['num_test_samples']}); ensure the same test split."
        )
    if warnings:
        lines.append("")
        for w in warnings:
            lines.append(f"> WARNING: {w}")
    lines.append("")
    lines.append("Delta columns are (freeform - template) in percentage points.")
    lines.append("")

    # Table 1 — original open-vocabulary families (template numbers reusable) ---
    present_original = [
        f for f in ORIGINAL_FAMILIES if f in tmpl_res or f in free_res
    ]
    lines.append("## Table 1 — Original open-vocabulary families")
    lines.append("")
    lines.append("_Template-model numbers reused from the previously cached evaluation._")
    lines.append("")
    lines.extend(_render_table(present_original, tmpl_res, free_res))
    lines.append("")

    # Table 2 — new held-out freeform families --------------------------------
    lines.append("## Table 2 — New held-out freeform families")
    lines.append("")
    lines.extend(_render_table(NEW_FAMILIES, tmpl_res, free_res))
    lines.append("")

    t_mean = _mean_acc(tmpl_res, NEW_FAMILIES)
    f_mean = _mean_acc(free_res, NEW_FAMILIES)
    if t_mean is not None or f_mean is not None:
        lines.append("### Mean accuracy over new held-out freeform families")
        lines.append("")
        lines.append("| Model | Mean Acc |")
        lines.append("|---|---|")
        lines.append(f"| Template-trained | {_fmt_pct(t_mean)} |")
        lines.append(f"| Freeform-trained | {_fmt_pct(f_mean)} |")
        lines.append(f"| Delta (pp) | {_fmt_delta(t_mean, f_mean)} |")
        lines.append("")

    # Provenance
    lines.append("---")
    lines.append("")
    lines.append("Source files:")
    for p in tmpl.get("_source_files", []):
        lines.append(f"- template: `{p}`")
    for p in free.get("_source_files", []):
        lines.append(f"- freeform: `{p}`")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nComparison written to {out_path}")


if __name__ == "__main__":
    main()
