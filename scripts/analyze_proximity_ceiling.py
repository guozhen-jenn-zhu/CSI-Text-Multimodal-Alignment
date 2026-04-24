"""Analyze the proximity zero-shot ceiling effect.

Investigates why all alignment methods cluster at ~73% on proximity despite
very different loss functions, while supervised reaches 97.7%.

Key analyses:
1. Text embedding distances between proximity class prototypes
2. Comparison with other tasks' inter-class text distances
3. Impact of the phantom 0.5m class (no data samples)
4. Confusion matrix analysis across methods

Usage:
    python scripts/analyze_proximity_ceiling.py [--output_dir paper_results/analysis]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TASKS_TO_ANALYZE = ["proximity", "har", "motion_source", "fall_detection", "user_id"]


def compute_text_similarities(task_templates: list[dict]) -> dict:
    """Compute pairwise cosine similarities between class text prototypes.

    Uses simple word-level analysis (no model needed) and also computes
    character-level Jaccard overlap as a proxy for text distinguishability.
    """
    class_names = [t["class_name"] for t in task_templates]
    class_texts = {t["class_id"]: t["templates"] for t in task_templates}
    n_classes = len(class_names)

    all_texts = []
    for cid in sorted(class_texts.keys()):
        joined = " ".join(class_texts[cid]).lower()
        all_texts.append(joined)

    word_sets = [set(text.split()) for text in all_texts]
    jaccard_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0

    off_diag = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            off_diag.append(jaccard_matrix[i, j])

    template_lengths = [
        len(class_texts[cid]) for cid in sorted(class_texts.keys())
    ]

    diff_words_per_pair: dict[str, list[str]] = {}
    sorted_cids = sorted(class_texts.keys())
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            unique_to_i = word_sets[i] - word_sets[j]
            unique_to_j = word_sets[j] - word_sets[i]
            pair_key = f"{class_names[i]}_vs_{class_names[j]}"
            diff_words_per_pair[pair_key] = {
                "unique_to_first": sorted(unique_to_i)[:10],
                "unique_to_second": sorted(unique_to_j)[:10],
                "jaccard": float(jaccard_matrix[i, j]),
            }

    return {
        "class_names": class_names,
        "n_classes": n_classes,
        "jaccard_matrix": jaccard_matrix.tolist(),
        "mean_pairwise_jaccard": float(np.mean(off_diag)) if off_diag else 0.0,
        "min_pairwise_jaccard": float(np.min(off_diag)) if off_diag else 0.0,
        "max_pairwise_jaccard": float(np.max(off_diag)) if off_diag else 0.0,
        "template_counts": template_lengths,
        "diff_words_per_pair": diff_words_per_pair,
    }


def analyze_confusion_matrices(results_root: Path) -> dict:
    """Load and analyze proximity confusion matrices across methods."""
    methods = ["clip_infonce", "siglip", "language_bind", "centrobind",
               "barlow_twins", "masked_predict"]
    analysis = {}

    for method in methods:
        mf = results_root / "csibench" / "proximity" / f"{method}_resnet18_1d" / "metrics.json"
        if not mf.exists():
            continue
        with open(mf) as f:
            data = json.load(f)

        zs = data.get("zero_shot", {})
        cm = zs.get("confusion_matrix")
        per_f1 = zs.get("per_class_f1", {})
        acc = zs.get("accuracy", 0)

        if cm is not None:
            cm_arr = np.array(cm)
            per_class_acc = np.diag(cm_arr) / cm_arr.sum(axis=1).clip(1)

            has_phantom = cm_arr.shape[0] == 4
            phantom_predictions = int(cm_arr[:, 0].sum()) if has_phantom else 0
            total_samples = int(cm_arr.sum())

            analysis[method] = {
                "accuracy": acc,
                "confusion_matrix": cm,
                "per_class_accuracy": per_class_acc.tolist(),
                "per_class_f1": per_f1,
                "phantom_0_5m_predictions": phantom_predictions,
                "total_samples": total_samples,
                "phantom_prediction_rate": phantom_predictions / total_samples if total_samples > 0 else 0,
            }

    return analysis


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="paper_results/analysis")
    args = parser.parse_args()

    from data.text_generation.templates.task_templates import (
        CSIBENCH_TEMPLATES,
        get_task_templates,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Proximity Ceiling Analysis",
        "",
        "## 1. Text Similarity Between Classes",
        "",
        "Higher Jaccard similarity = harder to distinguish via text alone.",
        "",
    ]

    all_similarities = {}
    for task in TASKS_TO_ANALYZE:
        if task in CSIBENCH_TEMPLATES:
            templates = get_task_templates("csibench", task)
            sims = compute_text_similarities(templates)
            all_similarities[task] = sims

            report_lines.append(f"### {task.replace('_', ' ').title()}")
            report_lines.append(f"- Classes: {sims['n_classes']} ({', '.join(sims['class_names'])})")
            report_lines.append(f"- Mean pairwise Jaccard: **{sims['mean_pairwise_jaccard']:.4f}**")
            report_lines.append(f"- Min pairwise Jaccard: {sims['min_pairwise_jaccard']:.4f}")
            report_lines.append(f"- Max pairwise Jaccard: {sims['max_pairwise_jaccard']:.4f}")
            report_lines.append("")

            for pair_key, pair_info in sims["diff_words_per_pair"].items():
                report_lines.append(
                    f"  - {pair_key}: Jaccard={pair_info['jaccard']:.4f}"
                )
            report_lines.append("")

    report_lines.extend([
        "## 2. Cross-Task Comparison",
        "",
        "| Task | Classes | Mean Jaccard | Zero-Shot Best | Supervised | Gap |",
        "|------|---------|-------------|----------------|------------|-----|",
    ])

    task_performance = {
        "har": (0.987, 0.979),
        "fall_detection": (0.959, 0.955),
        "motion_source": (0.996, 0.998),
        "proximity": (0.736, 0.977),
        "user_id": (1.000, 1.000),
    }

    for task in TASKS_TO_ANALYZE:
        if task in all_similarities:
            s = all_similarities[task]
            zs_best, sup = task_performance.get(task, (0, 0))
            gap = (sup - zs_best) * 100
            report_lines.append(
                f"| {task} | {s['n_classes']} | {s['mean_pairwise_jaccard']:.4f} "
                f"| {zs_best*100:.1f}% | {sup*100:.1f}% | {gap:+.1f}% |"
            )
    report_lines.append("")

    report_lines.extend([
        "## 3. Proximity-Specific Issues",
        "",
        "### Phantom 0.5m Class",
        "",
        "The proximity task has 4 class labels (0.5m, 1.5m, 2.5m, 3.5m) but the",
        "0.5m class has **zero data samples** in the dataset. However, the zero-shot",
        "classifier constructs a text prototype for 0.5m, meaning some test samples",
        "may be incorrectly assigned to this phantom class.",
        "",
        "### Numerical vs Semantic Distinction",
        "",
        "The proximity classes differ only by a number ('0.5', '1.5', '2.5', '3.5').",
        "In text embedding space, these numbers are likely close together because:",
        "- CLIP was trained on image-caption pairs, not numerical distance estimation",
        "- '1.5 meters' and '2.5 meters' share most context words",
        "- The semantic difference ('close' vs 'moderate' vs 'far') is subtle in text",
        "",
        "This contrasts sharply with HAR where classes like 'jumping' vs 'seated' are",
        "semantically distinct concepts with very different word choices.",
        "",
    ])

    cm_analysis = {}
    results_root = Path("paper_results/raw")
    if results_root.exists():
        cm_analysis = analyze_confusion_matrices(results_root)
        if cm_analysis:
            report_lines.extend([
                "## 4. Confusion Matrix Analysis",
                "",
                "| Method | Accuracy | Phantom Predictions | Per-class Accuracy |",
                "|--------|----------|--------------------|--------------------|",
            ])
            for method, info in sorted(cm_analysis.items()):
                pca = info.get("per_class_accuracy", [])
                pca_str = ", ".join(f"{a:.2f}" for a in pca)
                phantom = info.get("phantom_predictions", 0)
                phantom_rate = info.get("phantom_prediction_rate", 0)
                report_lines.append(
                    f"| {method} | {info['accuracy']*100:.1f}% | "
                    f"{phantom} ({phantom_rate*100:.1f}%) | "
                    f"[{pca_str}] |"
                )
            report_lines.append("")

            report_lines.extend([
                "### Key Observation",
                "",
                "If phantom prediction rate is high across all methods, this confirms",
                "that the 0.5m text prototype is 'stealing' predictions from real classes,",
                "creating a systematic ceiling effect.",
                "",
            ])

    report_lines.extend([
        "## 5. Recommendations",
        "",
        "1. **Remove phantom 0.5m class** from zero-shot evaluation (use only 3 classes with actual data)",
        "2. **Re-evaluate proximity** with 3-class setup to see if accuracy improves",
        "3. **Report both 3-class and 4-class** results for completeness",
        "4. **Frame as a finding**: text-based supervision struggles with fine-grained",
        "   continuous attributes where linguistic distinction is small",
        "",
    ])

    report_path = output_dir / "proximity_ceiling_analysis.md"
    report_path.write_text("\n".join(report_lines))
    logger.info("Wrote proximity analysis to %s", report_path)

    summary = {
        "text_similarities": {
            task: {
                k: v for k, v in sims.items()
                if k not in ("diff_words_per_pair",)
            }
            for task, sims in all_similarities.items()
        },
        "confusion_analysis": {
            m: {k: v for k, v in info.items() if k != "confusion_matrix"}
            for m, info in cm_analysis.items()
        } if results_root.exists() else {},
    }
    with open(output_dir / "proximity_ceiling_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
