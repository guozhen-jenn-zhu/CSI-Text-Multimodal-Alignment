#!/usr/bin/env python3
"""Audit how distinguishable the benchmark's captions are from each other.

A reviewer asked whether high retrieval scores partly reflect the caption
vocabulary rather than the CSI-text alignment.  This script answers that from
the text side alone: no checkpoint, no CSI data, CPU only.

Per task it reports

* registry scale -- unique captions, captions per sample, samples per caption;
* lexical overlap -- word-level Jaccard within a class versus between classes,
  and the rate at which captions from different classes collide;
* embedding separability -- 1-nearest-neighbour same-class rate and a
  text-only linear probe on the class label, both in frozen CLIP space.  If a
  caption alone predicts its class perfectly, the text side contributes no
  difficulty and the retrieval score is carried entirely by the CSI encoder;
* near-duplicates -- how much the gallery shrinks when captions the text
  encoder cannot separate are collapsed, matching the control that
  ``retrieval_m2m`` applies to mAP.

Usage::

    python scripts/analyze_template_overlap.py \\
        --data_root /home/sagemaker-user/Multimodal_CSI_Text/Dataset \\
        --output_dir paper_results/template_overlap

    # Audit a single registry file directly (e.g. a freeform variant)
    python scripts/analyze_template_overlap.py \\
        --registry har=/path/to/pair_registry_freeform.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pair_registry import PairRegistry  # noqa: E402
from evaluation.retrieval_m2m import (  # noqa: E402
    DEFAULT_DEDUP_COSINE,
    MAX_DEDUP_GALLERY,
    duplicate_groups,
)

logger = logging.getLogger(__name__)

# (dataset, task) pairs and their location under --data_root, matching the
# layout used by scripts/run_m2m_retrieval_evals.sh.
DATASET_TASKS: list[tuple[str, str]] = [
    ("csibench", "har"),
    ("csibench", "fall_detection"),
    ("csibench", "motion_source"),
    ("csibench", "proximity"),
    ("csibench", "user_id"),
    ("xrf55", "action_recognition"),
    ("signfi", "sign_recognition"),
]

JACCARD_COLLISION_THRESHOLD = 0.5


def resolve_registry_path(data_root: Path, dataset: str, task: str) -> Path:
    if dataset == "csibench" and task in ("har", "proximity", "user_id"):
        return data_root / "csibench" / "multitask" / task / "pair_registry.json"
    if dataset == "csibench":
        return data_root / "csibench" / task / "pair_registry.json"
    return data_root / dataset / task / "pair_registry.json"


# ---------------------------------------------------------------------------
# Lexical overlap
# ---------------------------------------------------------------------------

def lexical_overlap(
    texts: list[str], text_classes: list[str], max_pairs: int = 400_000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Word-level Jaccard within a class versus between classes.

    Pairs are subsampled when the gallery is large enough that the full
    quadratic comparison would dominate the runtime.
    """
    word_sets = [set(t.lower().split()) for t in texts]
    n = len(texts)
    total_pairs = n * (n - 1) // 2

    rng = rng or np.random.default_rng(0)
    if total_pairs <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        i = rng.integers(0, n, size=max_pairs * 2)
        j = rng.integers(0, n, size=max_pairs * 2)
        pairs = [(int(a), int(b)) for a, b in zip(i, j) if a < b][:max_pairs]

    within: list[float] = []
    between: list[float] = []
    collisions = 0
    worst: list[tuple[float, str, str]] = []

    for a, b in pairs:
        union = word_sets[a] | word_sets[b]
        if not union:
            continue
        jac = len(word_sets[a] & word_sets[b]) / len(union)
        if text_classes[a] == text_classes[b]:
            within.append(jac)
        else:
            between.append(jac)
            if jac > JACCARD_COLLISION_THRESHOLD:
                collisions += 1
                if len(worst) < 2000:
                    worst.append((jac, texts[a], texts[b]))

    worst.sort(key=lambda x: -x[0])
    return {
        "num_pairs_compared": len(pairs),
        "pairs_subsampled": total_pairs > max_pairs,
        "mean_jaccard_within_class": float(np.mean(within)) if within else 0.0,
        "mean_jaccard_between_class": float(np.mean(between)) if between else 0.0,
        "max_jaccard_between_class": float(np.max(between)) if between else 0.0,
        "between_class_collision_rate": (
            float(collisions / len(between)) if between else 0.0
        ),
        "between_class_collision_threshold": JACCARD_COLLISION_THRESHOLD,
        "worst_between_class_pairs": [
            {"jaccard": round(j, 4), "text_a": a, "text_b": b} for j, a, b in worst[:5]
        ],
    }


# ---------------------------------------------------------------------------
# Embedding separability
# ---------------------------------------------------------------------------

def encode_texts(texts: list[str], encoder_name: str, batch_size: int) -> np.ndarray:
    """Encode captions with a frozen text encoder and L2-normalise."""
    import torch

    from models.encoders import build_text_encoder

    encoder = build_text_encoder(encoder_name)
    encoder.eval()

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            emb = encoder.encode(texts[start : start + batch_size])
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            chunks.append(emb.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def embedding_separability(
    embeddings: np.ndarray, text_classes: list[str], seed: int = 0,
) -> dict:
    """1-NN same-class rate and a text-only linear probe on the class label."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    classes = np.array(text_classes)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = sims.argmax(axis=1)
    nn_same_class = float((classes[nn_idx] == classes).mean())

    unique, counts = np.unique(classes, return_counts=True)
    result = {
        "nn1_same_class_rate": nn_same_class,
        "mean_cosine_within_class": None,
        "mean_cosine_between_class": None,
        "text_only_probe_accuracy": None,
        "text_only_probe_note": "",
        "num_classes": int(len(unique)),
    }

    same = classes[:, None] == classes[None, :]
    off_diag = ~np.eye(len(classes), dtype=bool)
    finite = np.where(np.isfinite(sims), sims, 0.0)
    within_mask = same & off_diag
    between_mask = (~same) & off_diag
    if within_mask.any():
        result["mean_cosine_within_class"] = float(finite[within_mask].mean())
    if between_mask.any():
        result["mean_cosine_between_class"] = float(finite[between_mask].mean())

    min_count = int(counts.min())
    if len(unique) < 2:
        result["text_only_probe_note"] = "only one class in the gallery"
    elif min_count < 2:
        result["text_only_probe_note"] = (
            f"smallest class has {min_count} caption(s); cross-validation needs 2+"
        )
    else:
        folds = min(5, min_count)
        probe = LogisticRegression(max_iter=2000, random_state=seed)
        scores = cross_val_score(probe, embeddings, classes, cv=folds)
        result["text_only_probe_accuracy"] = float(scores.mean())
        result["text_only_probe_note"] = f"{folds}-fold cross-validation"

    return result


def _census_one(
    embeddings: np.ndarray,
    texts: list[str],
    text_classes: list[str],
    threshold: float,
    labels: Optional[list[str]],
) -> dict:
    groups = duplicate_groups(embeddings, threshold, labels=labels)
    n = len(texts)
    collapsed = [g for g in groups if len(g) > 1]
    cross_class = [g for g in collapsed if len({text_classes[i] for i in g}) > 1]
    return {
        "cosine_threshold": threshold,
        "class_constrained": labels is not None,
        "num_captions": n,
        "num_groups": len(groups),
        "num_captions_removed": n - len(groups),
        "fraction_removed": float((n - len(groups)) / n) if n else 0.0,
        "num_collapsed_groups": len(collapsed),
        "num_cross_class_groups": len(cross_class),
        "largest_group": max((len(g) for g in groups), default=0),
        "examples": [
            {
                "classes": sorted({text_classes[i] for i in g}),
                "texts": [texts[i] for i in g[:3]],
            }
            for g in sorted(collapsed, key=lambda g: -len(g))[:3]
        ],
        "cross_class_examples": [
            {
                "classes": sorted({text_classes[i] for i in g}),
                "texts": [texts[i] for i in g[:3]],
            }
            for g in sorted(cross_class, key=lambda g: -len(g))[:3]
        ],
    }


def near_duplicate_census(
    embeddings: np.ndarray,
    texts: list[str],
    text_classes: list[str],
    threshold: float,
    sweep: Sequence[float] = (0.95, 0.98, 0.99),
) -> dict:
    """How much the gallery shrinks when indistinguishable captions are merged.

    Reports two groupings.  ``unconstrained`` is the diagnostic: captions are
    merged on cosine alone, so any group spanning several classes is a pair the
    frozen encoder cannot separate *despite* them meaning different things --
    exactly the template-overlap failure worth reporting.  ``class_constrained``
    matches the control that ``retrieval_m2m`` applies to mAP, where merging is
    confined to captions of the same class.

    A threshold sweep is included because the right cut-off depends on how
    templated the captions are: shared scaffolding puts a floor under the
    cosine between any two captions of the same register.
    """
    n = len(texts)
    if n > MAX_DEDUP_GALLERY:
        return {"skipped": f"gallery of {n} exceeds {MAX_DEDUP_GALLERY}"}

    return {
        "unconstrained": _census_one(embeddings, texts, text_classes, threshold, None),
        "class_constrained": _census_one(
            embeddings, texts, text_classes, threshold, text_classes,
        ),
        "threshold_sweep": [
            {
                "cosine_threshold": t,
                "num_groups": len(duplicate_groups(embeddings, t)),
                "num_cross_class_groups": sum(
                    1 for g in duplicate_groups(embeddings, t)
                    if len({text_classes[i] for i in g}) > 1
                ),
            }
            for t in sweep
        ],
    }


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------

def audit_registry(
    registry_path: Path,
    dataset: str,
    task: str,
    encoder_name: str,
    batch_size: int,
    dedup_cosine: float,
    skip_embeddings: bool,
) -> dict:
    registry = PairRegistry.load(str(registry_path))
    text_ids = registry.get_all_text_ids()
    texts = [registry.texts[tid] for tid in text_ids]

    text_to_class: dict[str, str] = {}
    for class_key, tids in registry.class_to_texts.items():
        for tid in tids:
            text_to_class.setdefault(tid, str(class_key))
    text_classes = [text_to_class.get(tid, "") for tid in text_ids]

    logger.info(
        "%s/%s: %d captions, %d samples", dataset, task, len(texts),
        len(registry.sample_to_texts),
    )

    audit: dict = {
        "dataset": dataset,
        "task": task,
        "registry_path": str(registry_path),
        "registry_stats": registry.stats(),
        "num_unlabelled_captions": int(sum(1 for c in text_classes if not c)),
        "lexical": lexical_overlap(texts, text_classes),
    }

    if skip_embeddings:
        audit["embedding"] = {"skipped": "--skip_embeddings"}
        audit["near_duplicates"] = {"skipped": "--skip_embeddings"}
        return audit

    embeddings = encode_texts(texts, encoder_name, batch_size)
    audit["text_encoder"] = encoder_name
    audit["embedding"] = embedding_separability(embeddings, text_classes)
    audit["near_duplicates"] = near_duplicate_census(
        embeddings, texts, text_classes, dedup_cosine,
    )
    return audit


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_markdown(audits: list[dict]) -> str:
    lines = [
        "# Template Overlap Audit",
        "",
        "How distinguishable are the benchmark's captions from one another, "
        "independent of any trained model? Computed from the pair registries "
        "and a frozen text encoder only.",
        "",
        "## Scale",
        "",
        "| Task | Captions | Samples | Captions/sample | Samples/caption | Classes |",
        "|------|----------|---------|-----------------|-----------------|---------|",
    ]
    for a in audits:
        s = a["registry_stats"]
        lines.append(
            f"| {a['dataset']}/{a['task']} | {s['num_texts']} | {s['num_samples']} "
            f"| {s['avg_texts_per_sample']:.1f} | {s['avg_samples_per_text']:.1f} "
            f"| {s['num_classes']} |"
        )

    lines.extend([
        "",
        "## Lexical overlap (word-level Jaccard)",
        "",
        "A low between-class Jaccard means captions from different classes share "
        "few words, so telling them apart needs little linguistic subtlety.",
        "",
        "| Task | Within class | Between class | Max between | Collisions (J>0.5) |",
        "|------|--------------|---------------|-------------|--------------------|",
    ])
    for a in audits:
        lx = a["lexical"]
        lines.append(
            f"| {a['dataset']}/{a['task']} "
            f"| {lx['mean_jaccard_within_class']:.4f} "
            f"| {lx['mean_jaccard_between_class']:.4f} "
            f"| {lx['max_jaccard_between_class']:.4f} "
            f"| {lx['between_class_collision_rate']*100:.2f}% |"
        )

    has_embeddings = any("skipped" not in a.get("embedding", {}) for a in audits)
    if has_embeddings:
        lines.extend([
            "",
            "## Embedding separability (frozen text encoder)",
            "",
            "`1-NN same class` is the fraction of captions whose nearest neighbour "
            "in text space shares its class. `Text-only probe` is a logistic "
            "regression predicting the class label from the caption embedding "
            "alone. Values near 1.0 mean the caption already determines the class, "
            "so the text side of retrieval contributes no difficulty.",
            "",
            "| Task | 1-NN same class | Text-only probe | Cos within | Cos between |",
            "|------|-----------------|-----------------|------------|-------------|",
        ])
        for a in audits:
            em = a.get("embedding", {})
            if "skipped" in em:
                continue
            cells = [
                f"{a['dataset']}/{a['task']}",
                _fmt(em.get("nn1_same_class_rate")),
                _fmt(em.get("text_only_probe_accuracy")),
                _fmt(em.get("mean_cosine_within_class")),
                _fmt(em.get("mean_cosine_between_class")),
            ]
            lines.append("| " + " | ".join(cells) + " |")

        lines.extend([
            "",
            "## Near-duplicate captions",
            "",
            "Captions the frozen text encoder cannot separate (cosine above the "
            "threshold) are merged into one gallery entry. A large collapse means "
            "the nominal gallery size overstates how many distinct things a model "
            "must tell apart.",
            "",
            "Two groupings are reported. *Unconstrained* merges on cosine alone, so "
            "any group spanning several classes is a set of captions the encoder "
            "cannot separate even though they describe different activities -- the "
            "template-overlap failure itself. *Class-constrained* confines merging "
            "to captions of one class and is the control `retrieval_m2m` applies "
            "to mAP, so it never credits a model for confusing two activities.",
            "",
            "| Task | Grouping | Threshold | Captions | Groups | Removed | Cross-class groups | Largest |",
            "|------|----------|-----------|----------|--------|---------|--------------------|---------|",
        ])
        for a in audits:
            nd = a.get("near_duplicates", {})
            if "skipped" in nd:
                lines.append(
                    f"| {a['dataset']}/{a['task']} | - | - | - | - "
                    f"| skipped: {nd['skipped']} | - | - |"
                )
                continue
            for label, key in (("unconstrained", "unconstrained"),
                               ("class-constrained", "class_constrained")):
                c = nd.get(key)
                if not c:
                    continue
                lines.append(
                    f"| {a['dataset']}/{a['task']} | {label} "
                    f"| {c['cosine_threshold']:.2f} "
                    f"| {c['num_captions']} | {c['num_groups']} "
                    f"| {c['num_captions_removed']} ({c['fraction_removed']*100:.1f}%) "
                    f"| {c['num_cross_class_groups']} | {c['largest_group']} |"
                )

        lines.extend([
            "",
            "### Threshold sensitivity (unconstrained grouping)",
            "",
            "Cross-class groups at a given cut-off measure how much of the cosine "
            "is carried by shared template scaffolding rather than by meaning.",
            "",
            "| Task | " + " | ".join(
                f"cos>{t['cosine_threshold']:.2f}"
                for t in (audits[0].get("near_duplicates", {}).get("threshold_sweep") or [])
            ) + " |",
        ])
        sweep0 = audits[0].get("near_duplicates", {}).get("threshold_sweep") or []
        lines.append("|---|" + "---|" * len(sweep0))
        for a in audits:
            sweep = a.get("near_duplicates", {}).get("threshold_sweep")
            if not sweep:
                continue
            cells = " | ".join(
                f"{t['num_groups']} grp, {t['num_cross_class_groups']} x-class"
                for t in sweep
            )
            lines.append(f"| {a['dataset']}/{a['task']} | {cells} |")

        for a in audits:
            xc = (a.get("near_duplicates", {}).get("unconstrained", {})
                  .get("cross_class_examples") or [])
            if not xc:
                continue
            lines.extend([
                "",
                f"#### Cross-class collisions, {a['dataset']}/{a['task']}",
                "",
            ])
            for ex in xc:
                lines.append(f"- classes {', '.join(ex['classes'])}:")
                for t in ex["texts"]:
                    lines.append(f"  - \"{t}\"")

    lines.extend([
        "",
        "## Worst between-class lexical collisions",
        "",
    ])
    for a in audits:
        worst = a["lexical"].get("worst_between_class_pairs", [])
        if not worst:
            continue
        lines.extend([f"### {a['dataset']}/{a['task']}", ""])
        for pair in worst[:3]:
            lines.append(
                f"- J={pair['jaccard']:.3f} -- \"{pair['text_a']}\" vs "
                f"\"{pair['text_b']}\""
            )
        lines.append("")

    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def build_latex(audits: list[dict]) -> str:
    rows = []
    for a in audits:
        lx = a["lexical"]
        em = a.get("embedding", {})
        nd = a.get("near_duplicates", {}).get("class_constrained", {})
        probe = em.get("text_only_probe_accuracy")
        rows.append(
            f"{a['task'].replace('_', ' ')} & "
            f"{a['registry_stats']['num_texts']} & "
            f"{lx['mean_jaccard_between_class']:.3f} & "
            f"{lx['between_class_collision_rate']*100:.1f}\\% & "
            f"{_fmt(em.get('nn1_same_class_rate'))} & "
            f"{_fmt(probe)} & "
            f"{nd.get('fraction_removed', 0)*100:.1f}\\% \\\\"
        )
    return "\n".join([
        "% Auto-generated by scripts/analyze_template_overlap.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Task & Captions & Jaccard (between) & Collisions & 1-NN same class & "
        "Text-only probe & Near-dup. removed \\\\",
        "\\hline",
        *rows,
        "\\hline",
        "\\end{tabular}",
        "\\caption{Caption distinguishability, measured from the pair registries "
        "and a frozen text encoder alone. A high 1-NN same-class rate and a "
        "text-only probe accuracy near 1 mean the caption vocabulary already "
        "separates the classes, so cross-modal retrieval scores are carried by "
        "the CSI encoder rather than by text ambiguity.}",
        "\\label{tab:template-overlap}",
        "\\end{table}",
        "",
    ])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="/home/sagemaker-user/Multimodal_CSI_Text/Dataset",
        help="Dataset root holding the per-task pair registries",
    )
    parser.add_argument(
        "--registry",
        nargs="*",
        default=[],
        metavar="TASK=PATH",
        help="Audit specific registry files instead of discovering them under --data_root",
    )
    parser.add_argument("--output_dir", default="paper_results/template_overlap")
    parser.add_argument("--text_encoder", default="clip_vit_b32")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dedup_cosine", type=float, default=DEFAULT_DEDUP_COSINE)
    parser.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Lexical statistics only; no text encoder is loaded",
    )
    args = parser.parse_args()

    targets: list[tuple[str, str, Path]] = []
    if args.registry:
        for spec in args.registry:
            if "=" not in spec:
                parser.error(f"--registry expects TASK=PATH, got {spec!r}")
            task, path = spec.split("=", 1)
            targets.append(("custom", task, Path(path).expanduser()))
    else:
        data_root = Path(args.data_root).expanduser()
        for dataset, task in DATASET_TASKS:
            targets.append((dataset, task, resolve_registry_path(data_root, dataset, task)))

    audits: list[dict] = []
    for dataset, task, path in targets:
        if not path.is_file():
            logger.warning("Registry not found, skipping %s/%s: %s", dataset, task, path)
            continue
        try:
            audits.append(audit_registry(
                path, dataset, task, args.text_encoder, args.batch_size,
                args.dedup_cosine, args.skip_embeddings,
            ))
        except Exception as e:
            logger.error("Audit failed for %s/%s: %s", dataset, task, e)

    if not audits:
        logger.error("No registries could be audited. Check --data_root / --registry.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.md").write_text(build_markdown(audits), encoding="utf-8")
    (output_dir / "tables.tex").write_text(build_latex(audits), encoding="utf-8")
    with open(output_dir / "overlap_stats.json", "w", encoding="utf-8") as f:
        json.dump({"audits": audits}, f, indent=2)

    logger.info("Wrote %d task audits to %s", len(audits), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
