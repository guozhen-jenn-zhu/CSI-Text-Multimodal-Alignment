"""Compositional generalization analysis for ambient scene understanding.

Level 2 ambient QA (see :mod:`evaluation.ambient_qa`) matches a CSI embedding
against 140 composite scene descriptions spanning 5 activities x 4 proximities
x 7 users.  A high score there only demonstrates composition if the model is
not simply exploiting the subset of attribute combinations that actually occurs
in CSI-Bench.  This module provides:

* the data-side audit of that combination structure
  (:func:`load_scene_attribute_table`, :func:`audit_composition`), and
* the embedding-space controls that separate composition from co-occurrence
  (:class:`CompositionAnalyzer`).

Both halves share one definition of the ``(user, activity, proximity)`` triple
so that the audit, the split builder, and the evaluator cannot drift apart.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from data.text_generation.templates.task_templates import (
    _ACTIVITY_DESCRIPTIONS,
    _PROXIMITY_DESCRIPTIONS,
    _USER_DESCRIPTIONS,
    CSIBENCH_COMPOSITE_TEMPLATES,
    generate_composite_caption,
)

logger = logging.getLogger(__name__)

# Multitask task name -> ambient-QA attribute name.  Matches the mapping used
# by ``_run_ambient_qa_multitask`` in scripts/run_evaluate.py.
SCENE_TASK_ATTRIBUTES: dict[str, str] = {
    "har": "activity",
    "proximity": "proximity",
    "user_id": "user_identity",
}

# Position of each attribute inside the composite triple.  The ordering matches
# ``AmbientQA.build_composite_embeddings``, which yields ``(user, activity,
# proximity)`` tuples.
TRIPLE_ATTRIBUTES: tuple[str, str, str] = ("user_identity", "activity", "proximity")
TRIPLE_TASKS: tuple[str, str, str] = ("user_id", "har", "proximity")

ACTIVITY_KEYS: tuple[str, ...] = tuple(_ACTIVITY_DESCRIPTIONS.keys())
PROXIMITY_KEYS: tuple[str, ...] = tuple(_PROXIMITY_DESCRIPTIONS.keys())
USER_KEYS: tuple[str, ...] = tuple(_USER_DESCRIPTIONS.keys())

TRIPLE_VALUE_SPACE: dict[str, tuple[str, ...]] = {
    "user_identity": USER_KEYS,
    "activity": ACTIVITY_KEYS,
    "proximity": PROXIMITY_KEYS,
}

# Composition patterns disjoint from the four used to build training captions
# (``CSIBENCH_COMPOSITE_TEMPLATES["template_patterns"]``).  They reorder the
# slots and change the surrounding wording, so matching them requires composing
# attributes rather than recalling a memorised sentence.
NOVEL_COMPOSITE_PATTERNS: dict[str, str] = {
    "novel_distance_first": (
        "About {distance} from the receiver, {user_desc} can be seen {activity}."
    ),
    "novel_two_clause": (
        "{user_desc} is {activity}; the person stands roughly {distance} "
        "from the antenna."
    ),
    "novel_report_style": (
        "Scene report: {activity}, subject {user_desc}, range {distance}."
    ),
}

Triple = tuple[str, str, str]


# ======================================================================
# Data side: attribute table and combination audit
# ======================================================================


def normalize_attribute_value(task_name: str, raw_label: Any) -> str:
    """Normalise a raw metadata label to the composite-template key format.

    ``U03 -> user_3``, ``seated-breathing -> seated_breathing``,
    ``1.5 -> 1.5m``.  Values that are already normalised pass through.
    """
    s = str(raw_label)
    if task_name == "user_id":
        if s.startswith("U") and s[1:].isdigit():
            return f"user_{int(s[1:])}"
        return s
    if task_name == "har":
        return {"seated-breathing": "seated_breathing", "wavinghand": "waving"}.get(s, s)
    if task_name == "proximity":
        try:
            return f"{float(s)}m"
        except ValueError:
            return s
    return s


@dataclass
class SceneAttributeTable:
    """Per-sample ``(user, activity, proximity)`` labels for the co-labeled subset.

    Only samples present in every one of the three multitask tasks are kept,
    which is the same population that Level 2 ambient QA is evaluated on.
    """

    sample_ids: list[str]
    triples: list[Triple]
    splits: list[str]
    int_labels: dict[str, np.ndarray] = field(default_factory=dict)
    class_keys: dict[str, dict[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index = {sid: i for i, sid in enumerate(self.sample_ids)}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def indices_for_split(self, split: str) -> np.ndarray:
        return np.array(
            [i for i, s in enumerate(self.splits) if s == split], dtype=np.int64,
        )

    def triples_for_split(self, split: str) -> list[Triple]:
        return [self.triples[i] for i in self.indices_for_split(split)]

    def sample_ids_for_split(self, split: str) -> list[str]:
        return [self.sample_ids[i] for i in self.indices_for_split(split)]

    def triple_of(self, sample_id: str) -> Triple:
        return self.triples[self._index[sample_id]]

    def realized_triples(self, split: Optional[str] = None) -> set[Triple]:
        source = self.triples if split is None else self.triples_for_split(split)
        return set(source)

    def triple_counts(self, split: Optional[str] = None) -> Counter:
        source = self.triples if split is None else self.triples_for_split(split)
        return Counter(source)


def load_scene_attribute_table(
    task_entries: Sequence[dict[str, Any]],
    splits: Sequence[str] = ("train", "val", "test"),
    require_valid_files: bool = True,
) -> SceneAttributeTable:
    """Join the three multitask tasks into one per-sample attribute table.

    Parameters
    ----------
    task_entries
        ``config["multitask"]["tasks"]`` entries; each needs ``task_name``,
        ``task_root`` and optionally ``h5_key``.  Entries for tasks outside
        ``har`` / ``proximity`` / ``user_id`` are ignored.
    splits
        Split names to load.  A sample is assigned to the first split it is
        found in, so overlapping split files cannot double-count it.
    require_valid_files
        Drop samples whose H5 file is missing on disk, matching what the
        ambient QA evaluator does.
    """
    import json
    from pathlib import Path

    from data.metadata_loader import MetadataLoader

    entries = [
        e for e in task_entries
        if isinstance(e, dict) and e.get("task_name") in SCENE_TASK_ATTRIBUTES
    ]
    found = {e["task_name"] for e in entries}
    missing = set(TRIPLE_TASKS) - found
    if missing:
        raise ValueError(
            f"Scene attribute table needs tasks {sorted(TRIPLE_TASKS)}; "
            f"config is missing {sorted(missing)}."
        )

    loaders: dict[str, MetadataLoader] = {
        e["task_name"]: MetadataLoader(
            e["task_root"], h5_key=e.get("h5_key", "CSI_amps"),
        )
        for e in entries
    }

    class_keys: dict[str, dict[int, str]] = {}
    for task_name, loader in loaders.items():
        attr = SCENE_TASK_ATTRIBUTES[task_name]
        lm_path = Path(loader.task_root) / "metadata" / "label_mapping.json"
        if not lm_path.is_file():
            logger.warning("No label_mapping.json under %s", loader.task_root)
            continue
        with open(lm_path) as f:
            idx_to_label = json.load(f)["idx_to_label"]
        class_keys[attr] = {
            int(k): normalize_attribute_value(task_name, v)
            for k, v in idx_to_label.items()
        }

    sample_ids: list[str] = []
    sample_splits: list[str] = []
    seen: set[str] = set()

    for split in splits:
        per_task: list[set[str]] = []
        for task_name, loader in loaders.items():
            try:
                ids = loader.get_split_ids(split)
            except FileNotFoundError:
                logger.warning("Split '%s' not found for task '%s'", split, task_name)
                per_task = []
                break
            if require_valid_files:
                ids = loader.filter_valid_ids(ids)
            per_task.append(set(ids))
        if not per_task:
            continue

        shared = sorted(set.intersection(*per_task))
        for sid in shared:
            if sid in seen:
                continue
            seen.add(sid)
            sample_ids.append(sid)
            sample_splits.append(split)

    if not sample_ids:
        raise ValueError("No samples shared across har / proximity / user_id splits.")

    raw_per_task = {
        task_name: [loader.get_label(sid) for sid in sample_ids]
        for task_name, loader in loaders.items()
    }
    normalized = {
        SCENE_TASK_ATTRIBUTES[task_name]: [
            normalize_attribute_value(task_name, lbl) for lbl in labels
        ]
        for task_name, labels in raw_per_task.items()
    }

    triples: list[Triple] = list(
        zip(
            normalized["user_identity"],
            normalized["activity"],
            normalized["proximity"],
        )
    )

    int_labels: dict[str, np.ndarray] = {}
    for attr, keys in class_keys.items():
        key_to_int = {v: k for k, v in keys.items()}
        values = normalized[attr]
        unknown = sorted({v for v in values if v not in key_to_int})
        if unknown:
            raise ValueError(
                f"Attribute '{attr}' has values {unknown} absent from label_mapping.json"
            )
        int_labels[attr] = np.array([key_to_int[v] for v in values], dtype=np.int64)

    logger.info(
        "SceneAttributeTable: %d samples, %d realised triples, splits=%s",
        len(sample_ids), len(set(triples)), Counter(sample_splits),
    )

    return SceneAttributeTable(
        sample_ids=sample_ids,
        triples=triples,
        splits=sample_splits,
        int_labels=int_labels,
        class_keys=class_keys,
    )


def _entropy_bits(counts: Iterable[int]) -> float:
    counts = np.asarray([c for c in counts if c > 0], dtype=np.float64)
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _mutual_information_bits(pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    """Mutual information (bits) and its normalised form for two attributes."""
    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    total = len(pairs)
    if total == 0:
        return {"mutual_information_bits": 0.0, "normalized_mi": 0.0}

    mi = 0.0
    for (a, b), n_ab in joint.items():
        p_ab = n_ab / total
        p_a = left[a] / total
        p_b = right[b] / total
        mi += p_ab * math.log2(p_ab / (p_a * p_b))

    h_a = _entropy_bits(left.values())
    h_b = _entropy_bits(right.values())
    denom = math.sqrt(h_a * h_b)
    return {
        "mutual_information_bits": float(mi),
        "normalized_mi": float(mi / denom) if denom > 0 else 0.0,
    }


def _pair_coverage(
    triples: Sequence[Triple], i: int, j: int, n_i: int, n_j: int,
) -> dict[str, Any]:
    cells = {(t[i], t[j]) for t in triples}
    return {
        "realized_cells": len(cells),
        "total_cells": n_i * n_j,
        "coverage": len(cells) / (n_i * n_j) if n_i * n_j else 0.0,
    }


def audit_composition(table: SceneAttributeTable) -> dict[str, Any]:
    """Quantify how factorial the co-labeled subset actually is.

    The reviewer-facing question is whether Level 2's 140-way candidate set is
    effectively much smaller because only a minority of combinations occur.
    """
    n_users, n_acts, n_prox = len(USER_KEYS), len(ACTIVITY_KEYS), len(PROXIMITY_KEYS)
    total_combinations = n_users * n_acts * n_prox

    all_triples = table.triples
    realized = table.realized_triples()
    counts = table.triple_counts()

    joint_entropy = _entropy_bits(counts.values())
    marginal_entropies = {}
    for pos, attr in enumerate(TRIPLE_ATTRIBUTES):
        marginal_entropies[attr] = _entropy_bits(
            Counter(t[pos] for t in all_triples).values()
        )

    pair_stats: dict[str, Any] = {}
    positions = {attr: pos for pos, attr in enumerate(TRIPLE_ATTRIBUTES)}
    sizes = {
        "user_identity": n_users, "activity": n_acts, "proximity": n_prox,
    }
    for a, b in itertools.combinations(TRIPLE_ATTRIBUTES, 2):
        i, j = positions[a], positions[b]
        stats = _pair_coverage(all_triples, i, j, sizes[a], sizes[b])
        stats.update(_mutual_information_bits([(t[i], t[j]) for t in all_triples]))
        pair_stats[f"{a}__{b}"] = stats

    per_split: dict[str, Any] = {}
    for split in sorted(set(table.splits)):
        split_triples = table.triples_for_split(split)
        split_counts = Counter(split_triples)
        per_split[split] = {
            "num_samples": len(split_triples),
            "realized_triples": len(split_counts),
            "coverage": len(split_counts) / total_combinations,
            "min_samples_per_triple": min(split_counts.values()) if split_counts else 0,
            "median_samples_per_triple": (
                float(np.median(list(split_counts.values()))) if split_counts else 0.0
            ),
        }

    test_counts = table.triple_counts("test")
    n_test = sum(test_counts.values())
    majority_triple, majority_n = (
        test_counts.most_common(1)[0] if test_counts else (None, 0)
    )

    # Predictor that knows only the three marginals and assumes independence.
    marginal_probs = []
    for pos in range(3):
        c = Counter(t[pos] for t in table.triples_for_split("test"))
        marginal_probs.append({k: v / max(n_test, 1) for k, v in c.items()})
    best_independent_triple, best_independent_p = None, -1.0
    for u, a, p in itertools.product(USER_KEYS, ACTIVITY_KEYS, PROXIMITY_KEYS):
        prob = (
            marginal_probs[0].get(u, 0.0)
            * marginal_probs[1].get(a, 0.0)
            * marginal_probs[2].get(p, 0.0)
        )
        if prob > best_independent_p:
            best_independent_p, best_independent_triple = prob, (u, a, p)

    return {
        "total_combinations": total_combinations,
        "num_samples": len(table),
        "realized_triples": len(realized),
        "coverage": len(realized) / total_combinations,
        "effective_combinations": float(2.0 ** joint_entropy),
        "joint_entropy_bits": joint_entropy,
        "marginal_entropy_bits": marginal_entropies,
        "sum_marginal_entropy_bits": float(sum(marginal_entropies.values())),
        "total_correlation_bits": float(
            sum(marginal_entropies.values()) - joint_entropy
        ),
        "attribute_value_counts": {
            attr: dict(Counter(t[pos] for t in all_triples))
            for pos, attr in enumerate(TRIPLE_ATTRIBUTES)
        },
        "pairwise": pair_stats,
        "per_split": per_split,
        "triple_counts": {"|".join(k): v for k, v in sorted(counts.items())},
        "prior_baselines": {
            "chance_all_combinations": 1.0 / total_combinations,
            "chance_realized_only": (
                1.0 / len(realized) if realized else 0.0
            ),
            "majority_triple": (
                "|".join(majority_triple) if majority_triple else None
            ),
            "majority_triple_accuracy": (
                majority_n / n_test if n_test else 0.0
            ),
            "independent_marginal_triple": (
                "|".join(best_independent_triple) if best_independent_triple else None
            ),
            "independent_marginal_accuracy": (
                test_counts.get(best_independent_triple, 0) / n_test if n_test else 0.0
            ),
        },
    }


def format_audit_markdown(audit: dict[str, Any]) -> str:
    """Render :func:`audit_composition` output as a readable report."""
    lines: list[str] = ["# Composition Structure Audit", ""]
    lines.append(
        f"Co-labeled subset: **{audit['num_samples']}** samples with "
        f"simultaneous activity / proximity / user labels."
    )
    lines.append("")
    lines.append(
        f"**{audit['realized_triples']} of {audit['total_combinations']}** "
        f"attribute combinations actually occur "
        f"({100 * audit['coverage']:.1f}% coverage). "
        f"Effective number of combinations (2^H) = "
        f"{audit['effective_combinations']:.1f}."
    )
    lines.append("")

    lines.append("## Attribute dependence")
    lines.append("")
    lines.append(
        f"Joint entropy {audit['joint_entropy_bits']:.2f} bits vs "
        f"{audit['sum_marginal_entropy_bits']:.2f} bits if the three attributes "
        f"were independent (total correlation "
        f"{audit['total_correlation_bits']:.2f} bits)."
    )
    lines.append("")
    lines.append("| Attribute pair | Realized cells | Coverage | MI (bits) | Normalized MI |")
    lines.append("|---|---|---|---|---|")
    for pair, stats in audit["pairwise"].items():
        left, right = pair.split("__")
        lines.append(
            f"| {left} x {right} | {stats['realized_cells']} / {stats['total_cells']} "
            f"| {100 * stats['coverage']:.1f}% "
            f"| {stats['mutual_information_bits']:.3f} "
            f"| {stats['normalized_mi']:.3f} |"
        )
    lines.append("")

    lines.append("## Per split")
    lines.append("")
    lines.append("| Split | Samples | Realized triples | Coverage | Min / median samples per triple |")
    lines.append("|---|---|---|---|---|")
    for split, stats in audit["per_split"].items():
        lines.append(
            f"| {split} | {stats['num_samples']} | {stats['realized_triples']} "
            f"| {100 * stats['coverage']:.1f}% "
            f"| {stats['min_samples_per_triple']} / "
            f"{stats['median_samples_per_triple']:.0f} |"
        )
    lines.append("")

    prior = audit["prior_baselines"]
    lines.append("## Prior-only baselines on the test split")
    lines.append("")
    lines.append("| Predictor | Accuracy |")
    lines.append("|---|---|")
    lines.append(
        f"| Uniform over all {audit['total_combinations']} combinations "
        f"| {100 * prior['chance_all_combinations']:.2f}% |"
    )
    lines.append(
        f"| Uniform over the {audit['realized_triples']} realized combinations "
        f"| {100 * prior['chance_realized_only']:.2f}% |"
    )
    lines.append(
        f"| Always predict the most frequent combination ({prior['majority_triple']}) "
        f"| {100 * prior['majority_triple_accuracy']:.2f}% |"
    )
    lines.append(
        f"| Argmax of the independent marginal product "
        f"({prior['independent_marginal_triple']}) "
        f"| {100 * prior['independent_marginal_accuracy']:.2f}% |"
    )
    lines.append("")
    return "\n".join(lines)


# ======================================================================
# Embedding side: composition controls
# ======================================================================


def all_triples() -> list[Triple]:
    """Enumerate the full 140-way candidate set in ambient-QA order."""
    return [
        (user, activity, prox)
        for activity, prox, user in itertools.product(
            ACTIVITY_KEYS, PROXIMITY_KEYS, USER_KEYS,
        )
    ]


def render_triple(
    triple: Triple, template_idx: int = 0, pattern: Optional[str] = None,
) -> str:
    """Render a ``(user, activity, proximity)`` triple as a composite caption."""
    user, activity, prox = triple
    if pattern is None:
        return generate_composite_caption(
            user, activity, prox, template_idx=template_idx,
        )
    return pattern.format(
        user_desc=_USER_DESCRIPTIONS[user],
        activity=_ACTIVITY_DESCRIPTIONS[activity],
        distance=_PROXIMITY_DESCRIPTIONS[prox],
    )


def training_composite_patterns() -> list[str]:
    return list(CSIBENCH_COMPOSITE_TEMPLATES["template_patterns"])


def check_patterns_disjoint(patterns: dict[str, str]) -> dict[str, Any]:
    """Verify novel patterns render captions unseen among the training patterns.

    Compares the full rendered caption sets rather than the raw pattern
    strings, so a pattern that only differs in punctuation is still caught.
    """
    def _norm(text: str) -> str:
        return " ".join(text.strip().lower().split())

    train_captions = {
        _norm(render_triple(t, template_idx=i))
        for t in all_triples()
        for i in range(len(training_composite_patterns()))
    }

    report: dict[str, Any] = {"num_training_captions": len(train_captions), "families": {}}
    for name, pattern in patterns.items():
        rendered = {_norm(render_triple(t, pattern=pattern)) for t in all_triples()}
        overlap = sorted(rendered & train_captions)
        report["families"][name] = {
            "num_captions": len(rendered),
            "num_overlapping": len(overlap),
            "disjoint": not overlap,
            "examples_overlapping": overlap[:3],
        }
    report["all_disjoint"] = all(
        f["disjoint"] for f in report["families"].values()
    )
    return report


def ranking_metrics(
    sim: np.ndarray,
    gt_indices: np.ndarray,
    candidates: Sequence[Triple],
    gt_triples: Sequence[Triple],
) -> dict[str, Any]:
    """Top-k / MRR / per-slot accuracy for a CSI-to-candidate similarity matrix.

    ``sim`` is ``[N, C]``; ``gt_indices`` indexes into ``candidates``.  Samples
    whose ground truth is absent from the candidate set are marked with ``-1``
    and excluded.
    """
    valid = gt_indices >= 0
    n_total = int(len(gt_indices))
    if not valid.any():
        return {
            "num_samples": 0,
            "num_excluded": n_total,
            "num_candidates": len(candidates),
        }

    sim_v = sim[valid]
    gt_v = gt_indices[valid]
    gt_triples_v = [t for t, ok in zip(gt_triples, valid) if ok]

    pred_idx = sim_v.argmax(axis=1)
    # Rank of the ground truth = 1 + number of candidates scored strictly higher.
    gt_scores = sim_v[np.arange(len(gt_v)), gt_v]
    ranks = 1 + (sim_v > gt_scores[:, None]).sum(axis=1)

    pred_triples = [candidates[i] for i in pred_idx]
    decomposed = {
        attr: float(
            np.mean([p[pos] == g[pos] for p, g in zip(pred_triples, gt_triples_v)])
        )
        for pos, attr in enumerate(TRIPLE_ATTRIBUTES)
    }

    return {
        "num_samples": int(len(gt_v)),
        "num_excluded": n_total - int(len(gt_v)),
        "num_candidates": len(candidates),
        "top1_accuracy": float((pred_idx == gt_v).mean()),
        "top3_accuracy": float((ranks <= 3).mean()),
        "top5_accuracy": float((ranks <= 5).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "decomposed_per_attribute": decomposed,
    }


def _index_gt(
    candidates: Sequence[Triple], gt_triples: Sequence[Triple],
) -> np.ndarray:
    lookup = {t: i for i, t in enumerate(candidates)}
    return np.array([lookup.get(t, -1) for t in gt_triples], dtype=np.int64)


class CompositionAnalyzer:
    """Embedding-space controls that separate composition from co-occurrence."""

    def __init__(
        self,
        text_encoder: torch.nn.Module,
        text_projection: Optional[torch.nn.Module] = None,
        device: str = "cuda",
    ) -> None:
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.device = device

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        emb = self.text_encoder.encode(list(texts)).to(self.device)
        if self.text_projection is not None:
            emb = self.text_projection(emb)
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def similarities(
        self,
        csi_embeddings: torch.Tensor,
        candidates: Sequence[Triple],
        template_idx: int = 0,
        pattern: Optional[str] = None,
    ) -> np.ndarray:
        """Cosine similarity between CSI embeddings and rendered candidates."""
        texts = [
            render_triple(t, template_idx=template_idx, pattern=pattern)
            for t in candidates
        ]
        text_emb = self.encode_texts(texts)
        csi = F.normalize(csi_embeddings.to(self.device), dim=-1)
        return (csi @ text_emb.t()).cpu().numpy()

    # -------------------------------------------------------------- controls

    def candidate_set_control(
        self,
        csi_embeddings: torch.Tensor,
        gt_triples: Sequence[Triple],
        realized: Iterable[Triple],
        template_idx: int = 0,
    ) -> dict[str, Any]:
        """Compare scoring over all 140 candidates vs realized combinations only.

        If restricting the candidate set to combinations that actually occur
        barely changes accuracy, then pruning impossible combinations is not
        what produces the headline number.
        """
        full = all_triples()
        sim_full = self.similarities(csi_embeddings, full, template_idx=template_idx)

        realized_set = set(realized)
        keep = [i for i, t in enumerate(full) if t in realized_set]
        realized_candidates = [full[i] for i in keep]

        return {
            "all_combinations": ranking_metrics(
                sim_full, _index_gt(full, gt_triples), full, gt_triples,
            ),
            "realized_only": ranking_metrics(
                sim_full[:, keep],
                _index_gt(realized_candidates, gt_triples),
                realized_candidates,
                gt_triples,
            ),
        }

    def attribute_similarities(
        self,
        csi_embeddings: torch.Tensor,
        answer_embeddings: dict[str, torch.Tensor],
        answer_keys: dict[str, list[str]],
    ) -> dict[str, tuple[np.ndarray, list[str]]]:
        """Level 1 similarity of each CSI embedding to each attribute answer."""
        csi = F.normalize(csi_embeddings.to(self.device), dim=-1)
        out: dict[str, tuple[np.ndarray, list[str]]] = {}
        for attr, emb in answer_embeddings.items():
            sims = (csi @ emb.to(self.device).t()).cpu().numpy()
            out[attr] = (sims, answer_keys[attr])
        return out

    @staticmethod
    def factorized_reference(
        attribute_sims: dict[str, tuple[np.ndarray, list[str]]],
        gt_triples: Sequence[Triple],
        candidates: Optional[Sequence[Triple]] = None,
    ) -> dict[str, Any]:
        """Score triples as the sum of independent Level 1 attribute similarities.

        This is the composition-free reference for the Level 2 result: a joint
        argmax over independently-scored slots already beats three separate
        argmaxes, so the Level 2 gain over Level 1 joint accuracy is only
        evidence of dependency modelling if it also beats this.
        """
        candidates = list(candidates) if candidates is not None else all_triples()

        per_attr_index: dict[str, dict[str, int]] = {}
        for attr, (_, keys) in attribute_sims.items():
            per_attr_index[attr] = {k: i for i, k in enumerate(keys)}

        n = len(gt_triples)
        scores = np.zeros((n, len(candidates)), dtype=np.float64)
        for pos, attr in enumerate(TRIPLE_ATTRIBUTES):
            sims, _ = attribute_sims[attr]
            cols = np.array(
                [per_attr_index[attr].get(t[pos], -1) for t in candidates],
                dtype=np.int64,
            )
            if (cols < 0).any():
                unknown = {
                    candidates[i][pos] for i in np.where(cols < 0)[0]
                }
                raise ValueError(
                    f"Attribute '{attr}' has candidate values {sorted(unknown)} "
                    "with no registered Level 1 answer."
                )
            scores += sims[:, cols]

        result = ranking_metrics(
            scores, _index_gt(candidates, gt_triples), candidates, gt_triples,
        )

        # Independent per-slot argmax, i.e. the Level 1 joint accuracy.
        independent_correct = np.ones(n, dtype=bool)
        for pos, attr in enumerate(TRIPLE_ATTRIBUTES):
            sims, keys = attribute_sims[attr]
            pred = [keys[i] for i in sims.argmax(axis=1)]
            independent_correct &= np.array(
                [p == t[pos] for p, t in zip(pred, gt_triples)]
            )
        result["independent_argmax_joint_accuracy"] = float(independent_correct.mean())
        return result

    def slot_counterfactual_probe(
        self,
        csi_embeddings: torch.Tensor,
        gt_triples: Sequence[Triple],
        template_idx: int = 0,
    ) -> dict[str, Any]:
        """Vary one slot at a time with the other two fixed to ground truth.

        Isolates whether the composite sentence is read slot-wise rather than
        matched as an opaque whole-string key.
        """
        full = all_triples()
        sim = self.similarities(csi_embeddings, full, template_idx=template_idx)
        index = {t: i for i, t in enumerate(full)}

        results: dict[str, Any] = {}
        for pos, attr in enumerate(TRIPLE_ATTRIBUTES):
            values = TRIPLE_VALUE_SPACE[attr]
            correct = np.zeros(len(gt_triples), dtype=bool)
            for row, gt in enumerate(gt_triples):
                cols = []
                for value in values:
                    variant = list(gt)
                    variant[pos] = value
                    cols.append(index[tuple(variant)])
                best = values[int(np.argmax(sim[row, cols]))]
                correct[row] = best == gt[pos]
            results[attr] = {
                "num_options": len(values),
                "accuracy": float(correct.mean()),
                "chance": 1.0 / len(values),
            }
        return results

    def novel_template_control(
        self,
        csi_embeddings: torch.Tensor,
        gt_triples: Sequence[Triple],
        patterns: Optional[dict[str, str]] = None,
        candidates: Optional[Sequence[Triple]] = None,
    ) -> dict[str, Any]:
        """Re-run Level 2 with composition patterns unseen during training."""
        patterns = patterns if patterns is not None else NOVEL_COMPOSITE_PATTERNS
        candidates = list(candidates) if candidates is not None else all_triples()
        gt_idx = _index_gt(candidates, gt_triples)

        out: dict[str, Any] = {
            "disjointness": check_patterns_disjoint(patterns),
            "families": {},
        }
        for idx in range(len(training_composite_patterns())):
            sim = self.similarities(csi_embeddings, candidates, template_idx=idx)
            out["families"][f"training_pattern_{idx}"] = ranking_metrics(
                sim, gt_idx, candidates, gt_triples,
            )
        for name, pattern in patterns.items():
            sim = self.similarities(csi_embeddings, candidates, pattern=pattern)
            out["families"][name] = ranking_metrics(
                sim, gt_idx, candidates, gt_triples,
            )
        return out

    def seen_unseen_breakdown(
        self,
        csi_embeddings: torch.Tensor,
        gt_triples: Sequence[Triple],
        held_out: Iterable[Triple],
        template_idx: int = 0,
    ) -> dict[str, Any]:
        """Level 2 metrics split by whether the combination was held out.

        Reported in two settings: *generalized*, where all 140 combinations
        compete, and *closed*, where the candidate set is restricted to the
        held-out combinations (the easier, conventional compositional-ZSL
        setting).
        """
        held_out = set(held_out)
        full = all_triples()
        sim_full = self.similarities(csi_embeddings, full, template_idx=template_idx)
        gt_idx_full = _index_gt(full, gt_triples)

        is_unseen = np.array([t in held_out for t in gt_triples], dtype=bool)
        groups = {"seen": ~is_unseen, "unseen": is_unseen}

        result: dict[str, Any] = {
            "num_held_out_combinations": len(held_out),
            "held_out_combinations": sorted("|".join(t) for t in held_out),
            "generalized": {},
            "closed": {},
        }

        for name, mask in groups.items():
            if not mask.any():
                result["generalized"][name] = {"num_samples": 0}
                continue
            result["generalized"][name] = ranking_metrics(
                sim_full[mask],
                gt_idx_full[mask],
                full,
                [t for t, m in zip(gt_triples, mask) if m],
            )

        unseen_candidates = [t for t in full if t in held_out]
        if unseen_candidates and is_unseen.any():
            keep = [i for i, t in enumerate(full) if t in held_out]
            subset_gt = [t for t, m in zip(gt_triples, is_unseen) if m]
            result["closed"]["unseen"] = ranking_metrics(
                sim_full[is_unseen][:, keep],
                _index_gt(unseen_candidates, subset_gt),
                unseen_candidates,
                subset_gt,
            )

        seen_acc = result["generalized"].get("seen", {}).get("top1_accuracy")
        unseen_acc = result["generalized"].get("unseen", {}).get("top1_accuracy")
        if seen_acc and unseen_acc:
            result["harmonic_mean_top1"] = float(
                2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
            )
        return result
