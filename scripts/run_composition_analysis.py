"""Run compositional-generalization controls on a trained scene model.

Level 2 ambient QA reports 86.9% top-1 over 140 composite scene descriptions.
This script asks whether that reflects composition or memorised co-occurrence,
using four controls that need no retraining:

1. Candidate-set control -- score over all 140 combinations vs only the
   combinations that occur in the data.  A flat comparison shows that pruning
   impossible combinations is not what produces the headline number.
2. Factorized reference -- score every combination as the sum of the three
   independent Level 1 attribute similarities.  The Level 2 gain over Level 1
   joint accuracy only demonstrates dependency modelling if it also beats this.
3. Slot counterfactual probe -- vary one attribute at a time with the other two
   fixed to ground truth, testing whether the sentence is read slot-wise.
4. Novel composition templates -- re-score with sentence patterns disjoint from
   the four used to build training captions.

With ``--held_out`` it additionally reports the seen vs unseen composition
breakdown for a model trained on a held-out-composition split.

Usage
-----
PYTHONPATH=. python scripts/run_composition_analysis.py \
    --checkpoint /path/to/best_model.pt \
    --config configs/experiment/sagemaker_multitask_scene.yaml \
    --registry_dir data/generated_texts/csibench \
    --output_dir paper_results/composition/multitask_scene
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

for _backend_flag in ("USE_TF", "USE_FLAX", "USE_JAX"):
    os.environ.setdefault(_backend_flag, "0")

import torch
import yaml

from evaluation.ambient_qa import AmbientQA
from evaluation.composition import (
    NOVEL_COMPOSITE_PATTERNS,
    TRIPLE_ATTRIBUTES,
    CompositionAnalyzer,
    audit_composition,
    load_scene_attribute_table,
)
from scripts.run_evaluate import (
    SceneBatchError,
    _register_ambient_answers,
    build_model_from_config,
    collect_multitask_scene_embeddings,
)
from training.train import _apply_overrides, _merge_configs

logger = logging.getLogger(__name__)


def _parse_triple(text: str) -> tuple[str, str, str]:
    parts = text.split("|")
    if len(parts) != 3:
        raise ValueError(
            f"Expected 'user|activity|proximity', got {text!r}"
        )
    return parts[0], parts[1], parts[2]


def _load_held_out(path: str) -> list[tuple[str, str, str]]:
    """Read held-out combinations from a split manifest or a bare list."""
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("held_out_triples", payload.get("held_out", []))
    return [_parse_triple(t) for t in payload]


def _build_attribute_answers(
    aq: AmbientQA,
    class_keys: dict[str, dict[int, str]],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Level 1 answer embeddings plus the attribute value each answer denotes."""
    embeddings: dict[str, torch.Tensor] = {}
    keys: dict[str, list[str]] = {}

    for attr in TRIPLE_ATTRIBUTES:
        answers = aq.QUESTION_TEMPLATES.get(attr, {}).get("answers", {})
        if not answers:
            raise ValueError(
                f"No Level 1 answers registered for '{attr}'; "
                "pass --registry_dir pointing at the pair registries."
            )
        mapping = class_keys.get(attr)
        if mapping is None:
            raise ValueError(
                f"No label_mapping.json available for attribute '{attr}'."
            )
        sorted_ids = sorted(answers.keys())
        unknown = [aid for aid in sorted_ids if aid not in mapping]
        if unknown:
            raise ValueError(
                f"Answer ids {unknown} for '{attr}' are absent from "
                "label_mapping.json; registry and metadata disagree."
            )
        embeddings[attr] = aq.build_answer_embeddings(attr)
        keys[attr] = [mapping[aid] for aid in sorted_ids]

    return embeddings, keys


def _format_report(results: dict[str, Any]) -> str:
    lines: list[str] = ["# Compositional Generalization Analysis", ""]

    meta = results["meta"]
    lines.append(
        f"Checkpoint: `{meta['checkpoint']}`  \n"
        f"Split: `{meta['split']}`  \n"
        f"Samples: {meta['num_samples']}"
    )
    lines.append("")

    audit = results.get("audit")
    if audit:
        lines.append(
            f"The co-labeled subset realises **{audit['realized_triples']} of "
            f"{audit['total_combinations']}** attribute combinations "
            f"({100 * audit['coverage']:.1f}%); effective count "
            f"(2^H) = {audit['effective_combinations']:.1f}."
        )
        lines.append("")

    def _row(label: str, m: dict[str, Any]) -> str:
        if not m.get("num_samples"):
            return f"| {label} | - | - | - | - | - |"
        return (
            f"| {label} | {m['num_candidates']} "
            f"| {100 * m['top1_accuracy']:.1f} "
            f"| {100 * m['top3_accuracy']:.1f} "
            f"| {100 * m['top5_accuracy']:.1f} "
            f"| {m['mrr']:.3f} |"
        )

    header = (
        "| Setting | Candidates | Top-1 (%) | Top-3 (%) | Top-5 (%) | MRR |\n"
        "|---|---|---|---|---|---|"
    )

    csc = results["candidate_set_control"]
    lines += ["## Candidate-set control", "", header]
    lines.append(_row("All combinations", csc["all_combinations"]))
    lines.append(_row("Realized combinations only", csc["realized_only"]))
    lines.append("")

    fac = results["factorized_reference"]
    lines += ["## Composition-free reference", ""]
    lines.append(
        f"Independent per-slot argmax (Level 1 joint accuracy): "
        f"**{100 * fac['independent_argmax_joint_accuracy']:.1f}%**."
    )
    lines.append("")
    lines.append(header)
    lines.append(_row("Sum of independent Level 1 scores", fac))
    lines.append(_row("Composite text embedding", csc["all_combinations"]))
    lines.append("")

    lines += ["## Single-slot counterfactual probe", ""]
    lines.append("| Slot | Options | Accuracy (%) | Chance (%) |")
    lines.append("|---|---|---|---|")
    for attr, stats in results["slot_counterfactual"].items():
        lines.append(
            f"| {attr} | {stats['num_options']} "
            f"| {100 * stats['accuracy']:.1f} | {100 * stats['chance']:.1f} |"
        )
    lines.append("")

    nov = results["novel_templates"]
    lines += ["## Composition templates", ""]
    lines.append(
        "Disjoint from training captions: "
        f"**{nov['disjointness']['all_disjoint']}**."
    )
    lines.append("")
    lines.append(header)
    for name, m in nov["families"].items():
        lines.append(_row(name, m))
    lines.append("")

    breakdown = results.get("seen_unseen")
    if breakdown:
        lines += ["## Seen vs unseen compositions", ""]
        lines.append(
            f"Held-out combinations: {breakdown['num_held_out_combinations']}."
        )
        lines.append("")
        lines.append(
            "| Group | Setting | Samples | Candidates | Top-1 (%) | Top-3 (%) | MRR |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for setting in ("generalized", "closed"):
            for group, m in breakdown.get(setting, {}).items():
                if not m.get("num_samples"):
                    continue
                lines.append(
                    f"| {group} | {setting} | {m['num_samples']} "
                    f"| {m['num_candidates']} "
                    f"| {100 * m['top1_accuracy']:.1f} "
                    f"| {100 * m['top3_accuracy']:.1f} "
                    f"| {m['mrr']:.3f} |"
                )
        if "harmonic_mean_top1" in breakdown:
            lines.append("")
            lines.append(
                f"Harmonic mean of seen / unseen top-1: "
                f"**{100 * breakdown['harmonic_mean_top1']:.1f}%**."
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compositional-generalization controls for the scene model",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config", type=str,
        default="configs/experiment/sagemaker_multitask_scene.yaml",
    )
    parser.add_argument(
        "--registry_dir", type=str, default="data/generated_texts/csibench",
        help="Directory of per-task pair registries, used for Level 1 answers.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="paper_results/composition/analysis",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--template_idx", type=int, default=0,
        help="Training composition pattern used for the main Level 2 scoring.",
    )
    parser.add_argument(
        "--held_out", type=str, default=None,
        help="Split manifest listing held-out combinations "
             "(enables the seen vs unseen breakdown).",
    )
    parser.add_argument(
        "--check_files", action="store_true",
        help="Verify H5 files exist when determining realized combinations "
             "(slower; the default trusts the metadata CSV).",
    )
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    config = _merge_configs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        file_ds_cfg = config.get("dataset_config", {})
        config = _merge_configs(ckpt["config"])
        merged_ds = dict(file_ds_cfg)
        merged_ds.update(config.get("dataset_config", {}))
        config["dataset_config"] = merged_ds
    if args.override:
        _apply_overrides(config, args.override)

    model, text_encoder, text_projection, csi_projection = build_model_from_config(
        config, device,
    )
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state, strict=True)

    task_entries = config.get("multitask", {}).get("tasks", [])
    table = load_scene_attribute_table(
        task_entries, require_valid_files=args.check_files,
    )
    audit = audit_composition(table)

    try:
        batch = collect_multitask_scene_embeddings(
            model, csi_projection, config, device, split=args.split,
        )
    except SceneBatchError as exc:
        raise SystemExit(f"{exc.code}: {exc.message}")

    if batch.triples is None:
        raise SystemExit(
            "Config lacks one of har/proximity/user_id, so composite "
            "combinations cannot be formed."
        )

    aq = AmbientQA(text_encoder, text_projection, device=str(device))
    _register_ambient_answers(aq, config, registry_dir=args.registry_dir)
    answer_embeddings, answer_keys = _build_attribute_answers(aq, table.class_keys)

    analyzer = CompositionAnalyzer(text_encoder, text_projection, device=str(device))
    gt_triples = batch.triples
    csi_emb = batch.csi_embeddings

    results: dict[str, Any] = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "split": args.split,
            "template_idx": args.template_idx,
            "num_samples": len(gt_triples),
        },
        "audit": audit,
    }

    logger.info("Candidate-set control ...")
    results["candidate_set_control"] = analyzer.candidate_set_control(
        csi_emb, gt_triples, table.realized_triples(),
        template_idx=args.template_idx,
    )

    logger.info("Factorized composition-free reference ...")
    attribute_sims = analyzer.attribute_similarities(
        csi_emb, answer_embeddings, answer_keys,
    )
    results["factorized_reference"] = CompositionAnalyzer.factorized_reference(
        attribute_sims, gt_triples,
    )

    logger.info("Single-slot counterfactual probe ...")
    results["slot_counterfactual"] = analyzer.slot_counterfactual_probe(
        csi_emb, gt_triples, template_idx=args.template_idx,
    )

    logger.info("Novel composition templates ...")
    results["novel_templates"] = analyzer.novel_template_control(
        csi_emb, gt_triples, NOVEL_COMPOSITE_PATTERNS,
    )

    held_out: Optional[list[tuple[str, str, str]]] = None
    if args.held_out:
        held_out = _load_held_out(args.held_out)
        logger.info("Seen vs unseen breakdown over %d held-out combinations ...",
                    len(held_out))
        results["seen_unseen"] = analyzer.seen_unseen_breakdown(
            csi_emb, gt_triples, held_out, template_idx=args.template_idx,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "composition_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    report = _format_report(results)
    with open(out_dir / "composition_analysis.md", "w") as f:
        f.write(report)

    print(report)
    print(f"\nWrote {out_dir / 'composition_analysis.json'}")
    print(f"Wrote {out_dir / 'composition_analysis.md'}")


if __name__ == "__main__":
    main()
