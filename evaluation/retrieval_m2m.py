"""Many-to-many CSI-text retrieval evaluation.

Unlike :mod:`evaluation.retrieval`, which collapses every class into a single
mean-pooled text prototype, this module ranks against the *full* text gallery
from the pair registry and scores each query against **all** of its valid
matches.  That is the setting the dataset actually defines: one CSI sample is
paired with many captions, and one caption is paired with many CSI samples.

Relevance is graded:

* ``2`` -- the caption is on the sample's own edge list in the registry
  (this includes the Level-2 composite captions that match the sample's exact
  ``(user, activity, proximity)`` triple)
* ``1`` -- the caption belongs to the sample's class but not to that sample
* ``0`` -- otherwise

Binary metrics are reported at two relevance levels from a single ranking:

``strict``
    Only grade 2 counts.  This is the registry's own definition of a valid
    match, so a HAR sample must rank its 9 captions above the other 212,
    including the 39-odd composites that describe the same activity performed
    by a different person at a different distance.

``class_level``
    Grade >= 1 counts, i.e. any caption of the right class.  This is the
    class-level ranking view.  For tasks without Level-2 composites the two
    levels coincide, which the output records explicitly.

Graded nDCG uses the 2/1/0 weights, so a model that surfaces a sample's own
composite caption above a generic same-class caption scores higher.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.pair_registry import PairRegistry

logger = logging.getLogger(__name__)

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)
DEFAULT_MAP_KS: tuple[int, ...] = (10, 100)
DEFAULT_NDCG_KS: tuple[int, ...] = (10, 100)

#: Relevance levels scored from every ranking, mapped to their minimum grade.
RELEVANCE_LEVELS: dict[str, int] = {"strict": 2, "class_level": 1}

#: Cosine threshold above which two captions count as near-duplicates.
#:
#: Templated captions share most of their wording ("The sensor detects User 1
#: {activity} roughly {d} meters from the transmitter"), so the scaffolding
#: alone carries two captions past 0.95 even when the activity differs.  On the
#: CSI-Bench HAR registry a 0.95 threshold merges walking with running and
#: jumping; 0.99 keeps the groups inside a class while still collapsing genuine
#: paraphrases.
DEFAULT_DEDUP_COSINE: float = 0.99

#: Skip the dedup control above this gallery size (the pairwise cosine matrix
#: becomes the dominant cost).
MAX_DEDUP_GALLERY: int = 20000


def duplicate_groups(
    embeddings: np.ndarray,
    threshold: float,
    labels: Optional[Sequence] = None,
) -> list[list[int]]:
    """Group rows that a frozen encoder cannot tell apart, by leader clustering.

    Each row joins the group whose representative it is closest to, provided
    that cosine exceeds *threshold*; otherwise it becomes a new representative.
    Every member is therefore within *threshold* of its own representative,
    which single-linkage would not guarantee: chains of pairwise-similar
    captions would otherwise collapse into one implausibly large group.

    When *labels* is given, a row may only join a representative carrying the
    same label.  Two captions describing different classes are never
    interchangeable for retrieval no matter how similar their wording, so
    merging them would credit a model for confusing them.

    Rows are assumed L2-normalised.  Returns one list of indices per group.
    """
    groups: list[list[int]] = []
    leader_idx: list[int] = []
    leader_labels: list = []

    for i in range(len(embeddings)):
        if leader_idx:
            sims = embeddings[i] @ embeddings[leader_idx].T
            if labels is not None:
                eligible = np.fromiter(
                    (lbl == labels[i] for lbl in leader_labels),
                    dtype=bool, count=len(leader_labels),
                )
                sims = np.where(eligible, sims, -np.inf)
            best = int(np.argmax(sims))
            if sims[best] > threshold:
                groups[best].append(i)
                continue
        leader_idx.append(i)
        leader_labels.append(labels[i] if labels is not None else None)
        groups.append([i])

    return groups


# ---------------------------------------------------------------------------
# Chance baselines
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _harmonic_number(n: int) -> float:
    """Return the n-th harmonic number ``H_n = sum_{i=1..n} 1/i``."""
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))


def expected_random_ap(n_pos: int, n_gallery: int) -> float:
    """Exact expected average precision of a uniformly random ranking.

    For a gallery of ``G`` items containing ``P`` positives::

        E[AP] = (1/G) * (H_G + (P-1)/(G-1) * (G - H_G))

    which reduces to ``H_G / G`` for a single positive and to ``1`` when every
    gallery item is positive.
    """
    if n_pos <= 0 or n_gallery <= 0:
        return 0.0
    if n_gallery == 1:
        return 1.0
    h = _harmonic_number(n_gallery)
    return float((h + (n_pos - 1) / (n_gallery - 1) * (n_gallery - h)) / n_gallery)


def expected_random_hit_rate(n_pos: int, n_gallery: int, k: int) -> float:
    """Probability that a random top-``k`` contains at least one positive."""
    if n_pos <= 0 or n_gallery <= 0:
        return 0.0
    k = min(k, n_gallery)
    miss = 1.0
    for i in range(k):
        remaining_neg = n_gallery - n_pos - i
        if remaining_neg <= 0:
            return 1.0
        miss *= remaining_neg / (n_gallery - i)
    return float(1.0 - miss)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

class _LevelAccumulator:
    """Per-query metric buffers for one relevance level."""

    def __init__(
        self,
        threshold: int,
        n_queries: int,
        ks: Sequence[int],
        map_ks: Sequence[int],
        ndcg_ks: Sequence[int],
    ) -> None:
        self.threshold = threshold
        self.ks, self.map_ks, self.ndcg_ks = list(ks), list(map_ks), list(ndcg_ks)
        zeros = lambda: np.zeros(n_queries, dtype=np.float64)  # noqa: E731
        self.ap = zeros()
        self.r_precision = zeros()
        self.reciprocal_rank = zeros()
        self.ap_at_k = {k: zeros() for k in self.map_ks}
        self.ndcg_at_k = {k: zeros() for k in self.ndcg_ks}
        self.recall_at_k = {k: zeros() for k in self.ks}
        self.precision_at_k = {k: zeros() for k in self.ks}
        self.hit_at_k = {k: zeros() for k in self.ks}

    def score_chunk(
        self,
        start: int,
        stop: int,
        graded_sorted: np.ndarray,
        n_pos: np.ndarray,
        ranks: np.ndarray,
        discount: np.ndarray,
        n_gallery: int,
    ) -> None:
        rel_sorted = (graded_sorted >= self.threshold).astype(np.float64)
        cum_hits = np.cumsum(rel_sorted, axis=1)
        precision_curve = cum_hits / ranks[None, :]

        chunk_pos = n_pos[start:stop].astype(np.float64)
        safe_pos = np.maximum(chunk_pos, 1.0)

        self.ap[start:stop] = (precision_curve * rel_sorted).sum(axis=1) / safe_pos

        for k in self.map_ks:
            self.ap_at_k[k][start:stop] = (
                (precision_curve[:, :k] * rel_sorted[:, :k]).sum(axis=1)
                / np.minimum(safe_pos, k)
            )

        for k in self.ks:
            hits = cum_hits[:, k - 1]
            self.recall_at_k[k][start:stop] = hits / safe_pos
            self.precision_at_k[k][start:stop] = hits / k
            self.hit_at_k[k][start:stop] = (hits > 0).astype(np.float64)

        # R-Precision: precision at a cut-off equal to the number of positives.
        r_idx = np.clip(chunk_pos.astype(np.int64), 1, n_gallery) - 1
        self.r_precision[start:stop] = (
            np.take_along_axis(cum_hits, r_idx[:, None], axis=1)[:, 0] / safe_pos
        )

        first_hit = np.argmax(rel_sorted > 0, axis=1)
        has_hit = rel_sorted.any(axis=1)
        self.reciprocal_rank[start:stop] = np.where(has_hit, 1.0 / (first_hit + 1.0), 0.0)

        ideal = np.sort(rel_sorted, axis=1)[:, ::-1]
        for k in self.ndcg_ks:
            dcg = (rel_sorted[:, :k] * discount[None, :k]).sum(axis=1)
            idcg = (ideal[:, :k] * discount[None, :k]).sum(axis=1)
            self.ndcg_at_k[k][start:stop] = np.divide(
                dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0,
            )

    def summarize(
        self,
        n_pos: np.ndarray,
        valid: np.ndarray,
        query_classes: Optional[np.ndarray],
        n_gallery: int,
    ) -> dict:
        results: dict = {
            "mAP": float(self.ap[valid].mean()),
            "r_precision": float(self.r_precision[valid].mean()),
            "mrr": float(self.reciprocal_rank[valid].mean()),
        }
        for k in self.map_ks:
            results[f"mAP@{k}"] = float(self.ap_at_k[k][valid].mean())
        for k in self.ndcg_ks:
            results[f"ndcg@{k}"] = float(self.ndcg_at_k[k][valid].mean())
        for k in self.ks:
            results[f"recall@{k}"] = float(self.recall_at_k[k][valid].mean())
            results[f"precision@{k}"] = float(self.precision_at_k[k][valid].mean())
            results[f"hit_rate@{k}"] = float(self.hit_at_k[k][valid].mean())

        per_class_map: dict[str, float] = {}
        if query_classes is not None:
            class_maps: list[float] = []
            class_ndcgs: list[float] = []
            ndcg_ref = self.ndcg_at_k[self.ndcg_ks[0]] if self.ndcg_ks else None
            for cid in np.unique(query_classes[valid]):
                mask = valid & (query_classes == cid)
                if not mask.any():
                    continue
                value = float(self.ap[mask].mean())
                per_class_map[str(cid)] = value
                class_maps.append(value)
                if ndcg_ref is not None:
                    class_ndcgs.append(float(ndcg_ref[mask].mean()))
            if class_maps:
                results["mAP_macro"] = float(np.mean(class_maps))
                results["mAP_worst_class"] = float(np.min(class_maps))
                results["worst_class"] = min(per_class_map, key=per_class_map.get)
            if class_ndcgs and self.ndcg_ks:
                results[f"ndcg@{self.ndcg_ks[0]}_macro"] = float(np.mean(class_ndcgs))

        # Chance baselines for a uniformly random ranking with the same positives.
        pos_valid = n_pos[valid]
        chance: dict[str, float] = {
            "mAP": float(np.mean([expected_random_ap(int(p), n_gallery) for p in pos_valid])),
        }
        for k in self.ks:
            chance[f"recall@{k}"] = float(min(k / n_gallery, 1.0))
            chance[f"precision@{k}"] = float((pos_valid / n_gallery).mean())
            chance[f"hit_rate@{k}"] = float(
                np.mean([expected_random_hit_rate(int(p), n_gallery, k) for p in pos_valid])
            )
        results["chance"] = chance

        denom = 1.0 - chance["mAP"]
        results["normalized_mAP"] = (
            float((results["mAP"] - chance["mAP"]) / denom) if denom > 1e-12 else 0.0
        )
        if "mAP_macro" in results:
            results["normalized_mAP_macro"] = (
                float((results["mAP_macro"] - chance["mAP"]) / denom) if denom > 1e-12 else 0.0
            )

        results["per_class_map"] = per_class_map
        results["num_queries"] = int(valid.sum())
        results["num_queries_skipped"] = int((~valid).sum())
        results["gallery_size"] = int(n_gallery)
        results["avg_positives_per_query"] = float(pos_valid.mean())
        return results


def compute_ranking_metrics(
    sims: np.ndarray,
    graded_relevance: np.ndarray,
    query_classes: Optional[np.ndarray] = None,
    levels: Optional[dict[str, int]] = None,
    ks: Sequence[int] = DEFAULT_KS,
    map_ks: Sequence[int] = DEFAULT_MAP_KS,
    ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
    query_chunk: int = 2048,
) -> dict:
    """Score a ``[Q, G]`` similarity matrix against a graded relevance matrix.

    The ranking is produced once and scored at every requested relevance
    level.  Queries with no positive at a given level are dropped from that
    level (average precision is undefined for them).  Metrics are
    micro-averaged over the surviving queries; the ``*_macro`` variants first
    average within each query class so that a frequent class cannot carry the
    score.

    Args:
        sims: ``[Q, G]`` similarity scores, higher is better.
        graded_relevance: ``[Q, G]`` relevance grades (0/1/2).
        query_classes: ``[Q]`` class id per query, used for macro averaging.
        levels: ``{name: minimum_grade}``; defaults to :data:`RELEVANCE_LEVELS`.
        ks: cut-offs for recall / precision / hit-rate.
        map_ks: cut-offs for truncated mAP.
        ndcg_ks: cut-offs for nDCG.
        query_chunk: number of queries sorted at a time.

    Returns:
        ``{level_name: metrics}`` plus a ``graded`` section holding the graded
        nDCG (which does not depend on a threshold) and ``gallery_size``.
    """
    levels = dict(levels) if levels is not None else dict(RELEVANCE_LEVELS)
    n_queries, n_gallery = sims.shape

    ks = [k for k in ks if k <= n_gallery]
    map_ks = [k for k in map_ks if k <= n_gallery]
    ndcg_ks = [k for k in ndcg_ks if k <= n_gallery]

    n_pos = {
        name: (graded_relevance >= threshold).sum(axis=1)
        for name, threshold in levels.items()
    }
    valid = {name: counts > 0 for name, counts in n_pos.items()}
    if not any(v.any() for v in valid.values()):
        return {"error": "no query has a positive in the gallery"}

    accumulators = {
        name: _LevelAccumulator(threshold, n_queries, ks, map_ks, ndcg_ks)
        for name, threshold in levels.items()
    }
    ndcg_graded_at_k = {k: np.zeros(n_queries, dtype=np.float64) for k in ndcg_ks}

    ranks = np.arange(1, n_gallery + 1, dtype=np.float64)
    discount = 1.0 / np.log2(np.arange(2, n_gallery + 2, dtype=np.float64))

    for start in range(0, n_queries, query_chunk):
        stop = min(start + query_chunk, n_queries)

        order = np.argsort(-sims[start:stop], axis=1, kind="stable")
        graded_sorted = np.take_along_axis(
            graded_relevance[start:stop], order, axis=1,
        ).astype(np.float64)

        for name, acc in accumulators.items():
            acc.score_chunk(
                start, stop, graded_sorted, n_pos[name], ranks, discount, n_gallery,
            )

        # Graded nDCG: gains 2^grade - 1 => 0 / 1 / 3.
        graded_gain = np.power(2.0, graded_sorted) - 1.0
        graded_ideal = np.sort(graded_gain, axis=1)[:, ::-1]
        for k in ndcg_ks:
            g_dcg = (graded_gain[:, :k] * discount[None, :k]).sum(axis=1)
            g_idcg = (graded_ideal[:, :k] * discount[None, :k]).sum(axis=1)
            ndcg_graded_at_k[k][start:stop] = np.divide(
                g_dcg, g_idcg, out=np.zeros_like(g_dcg), where=g_idcg > 0,
            )

    results: dict = {"gallery_size": int(n_gallery)}
    for name, acc in accumulators.items():
        if not valid[name].any():
            results[name] = {"error": f"no query has a grade>={acc.threshold} positive"}
            continue
        results[name] = acc.summarize(n_pos[name], valid[name], query_classes, n_gallery)
        results[name]["min_grade"] = acc.threshold

    # Graded nDCG is averaged over every query with any positive at all, so it
    # uses the most permissive level's validity mask.
    most_permissive = min(levels, key=levels.get)
    graded: dict = {"num_queries": int(valid[most_permissive].sum())}
    for k in ndcg_ks:
        graded[f"ndcg_graded@{k}"] = float(
            ndcg_graded_at_k[k][valid[most_permissive]].mean()
        )
    results["graded"] = graded

    strict_pos, lenient_pos = n_pos.get("strict"), n_pos.get("class_level")
    if strict_pos is not None and lenient_pos is not None:
        results["levels_identical"] = bool(np.array_equal(strict_pos, lenient_pos))

    return results


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ManyToManyRetrievalEvaluator:
    """Rank CSI samples against the full caption gallery, and vice versa."""

    def __init__(
        self,
        ks: Sequence[int] = DEFAULT_KS,
        map_ks: Sequence[int] = DEFAULT_MAP_KS,
        ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
        text_batch_size: int = 256,
        query_chunk: int = 2048,
        dedup_cosine: float = DEFAULT_DEDUP_COSINE,
    ) -> None:
        self.ks = tuple(ks)
        self.map_ks = tuple(map_ks)
        self.ndcg_ks = tuple(ndcg_ks)
        self.text_batch_size = text_batch_size
        self.query_chunk = query_chunk
        self.dedup_cosine = dedup_cosine

    @torch.no_grad()
    def _encode_texts(
        self,
        texts: list[str],
        text_encoder: nn.Module,
        text_projection: Optional[nn.Module],
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions, returning ``(raw, projected)`` normalised embeddings.

        The raw (pre-projection) embeddings come from the frozen text encoder
        and are therefore identical across checkpoints, which is what makes the
        near-duplicate control comparable between methods.
        """
        raw_chunks: list[torch.Tensor] = []
        proj_chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), self.text_batch_size):
            batch = texts[start : start + self.text_batch_size]
            emb = text_encoder.encode(batch).to(device)
            raw_chunks.append(F.normalize(emb, dim=-1).cpu())
            if text_projection is not None:
                emb = text_projection(emb)
            proj_chunks.append(F.normalize(emb, dim=-1).cpu())
        return torch.cat(raw_chunks, dim=0), torch.cat(proj_chunks, dim=0)

    def _dedup_control(
        self,
        sims: np.ndarray,
        relevance: np.ndarray,
        raw_text_embs: np.ndarray,
        sample_classes: np.ndarray,
        text_classes: np.ndarray,
    ) -> tuple[dict, dict]:
        """Re-score CSI->Text after collapsing near-duplicate captions.

        A group of captions that the frozen text encoder cannot tell apart is
        merged into a single gallery entry (similarity = best member, relevance
        = best member), so a model cannot be credited several times for the
        same caption.  Grouping is confined to captions of the same class, so
        the control never forgives a model for confusing two activities.
        Returns ``(info, metrics)``; ``metrics`` is empty when there is nothing
        to collapse.
        """
        n_gallery = raw_text_embs.shape[0]
        if n_gallery > MAX_DEDUP_GALLERY:
            return {"skipped": f"gallery of {n_gallery} exceeds {MAX_DEDUP_GALLERY}"}, {}

        groups = duplicate_groups(raw_text_embs, self.dedup_cosine, labels=text_classes)
        collapsed = [g for g in groups if len(g) > 1]
        info = {
            "cosine_threshold": self.dedup_cosine,
            "class_constrained": True,
            "num_captions": n_gallery,
            "num_groups": len(groups),
            "num_captions_removed": n_gallery - len(groups),
            "num_collapsed_groups": len(collapsed),
            "largest_group": max((len(g) for g in groups), default=0),
            "num_cross_class_groups": sum(
                1 for g in groups if len({text_classes[i] for i in g}) > 1
            ),
        }
        if not collapsed:
            return info, {}

        merged_sims = np.stack([sims[:, g].max(axis=1) for g in groups], axis=1)
        merged_rel = np.stack([relevance[:, g].max(axis=1) for g in groups], axis=1)

        metrics = compute_ranking_metrics(
            np.ascontiguousarray(merged_sims), np.ascontiguousarray(merged_rel),
            sample_classes, ks=self.ks, map_ks=self.map_ks, ndcg_ks=self.ndcg_ks,
            query_chunk=self.query_chunk,
        )
        return info, metrics

    @staticmethod
    def _build_relevance(
        sample_ids: list[str],
        text_ids: list[str],
        registry: PairRegistry,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the graded ``[N, G]`` relevance matrix.

        Returns:
            ``(relevance, sample_classes, text_classes)`` where the class
            arrays hold string class ids (``""`` when unknown).
        """
        text_index = {tid: i for i, tid in enumerate(text_ids)}

        # A caption may in principle be listed under several classes; keep the
        # full set for relevance and the first for macro grouping.
        text_class_sets: list[set[str]] = [set() for _ in text_ids]
        for class_key, tids in registry.class_to_texts.items():
            for tid in tids:
                idx = text_index.get(tid)
                if idx is not None:
                    text_class_sets[idx].add(str(class_key))

        class_to_text_idx: dict[str, list[int]] = {}
        for idx, class_set in enumerate(text_class_sets):
            for class_key in class_set:
                class_to_text_idx.setdefault(class_key, []).append(idx)

        relevance = np.zeros((len(sample_ids), len(text_ids)), dtype=np.int8)
        sample_classes = np.empty(len(sample_ids), dtype=object)

        for row, sample_id in enumerate(sample_ids):
            class_key = registry.get_class_for_sample(sample_id)
            sample_classes[row] = str(class_key) if class_key is not None else ""
            if class_key is not None:
                same_class = class_to_text_idx.get(str(class_key))
                if same_class:
                    relevance[row, same_class] = 1
            for tid in registry.get_positive_texts(sample_id):
                idx = text_index.get(tid)
                if idx is not None:
                    relevance[row, idx] = 2

        text_classes = np.array(
            [sorted(s)[0] if s else "" for s in text_class_sets], dtype=object,
        )
        return relevance, sample_classes, text_classes

    @torch.no_grad()
    def evaluate(
        self,
        csi_embeddings: torch.Tensor,
        sample_ids: list[str],
        registry: PairRegistry,
        text_encoder: nn.Module,
        text_projection: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> dict:
        """Run both retrieval directions over the full many-to-many graph.

        Args:
            csi_embeddings: ``[N, D]`` projected CSI embeddings for the test split.
            sample_ids: ``[N]`` registry sample ids aligned with ``csi_embeddings``.
            registry: Pair registry holding the CSI-text bipartite graph.
            text_encoder: Frozen text encoder with ``.encode()``.
            text_projection: Optional projection applied after encoding.
            device: Target device string.

        Returns:
            Dict with ``csi_to_text``, ``text_to_csi`` and ``gallery`` sections.
        """
        known = [sid in registry.sample_to_texts for sid in sample_ids]
        n_unknown = len(known) - int(np.sum(known))
        if n_unknown:
            logger.warning(
                "%d/%d test samples are absent from the registry and were dropped",
                n_unknown, len(known),
            )
            keep = np.array(known, dtype=bool)
            csi_embeddings = csi_embeddings[torch.from_numpy(keep)]
            sample_ids = [sid for sid, ok in zip(sample_ids, known) if ok]

        if not sample_ids:
            return {"error": "no test sample could be matched to the registry"}

        text_ids = registry.get_all_text_ids()
        texts = [registry.texts[tid] for tid in text_ids]
        logger.info(
            "M2M retrieval: %d test samples x %d gallery captions",
            len(sample_ids), len(text_ids),
        )

        raw_text_embs, text_embs = self._encode_texts(
            texts, text_encoder, text_projection, device,
        )
        csi_embs = F.normalize(csi_embeddings.float().cpu(), dim=-1)
        sims = (csi_embs @ text_embs.t()).numpy().astype(np.float32)

        relevance, sample_classes, text_classes = self._build_relevance(
            sample_ids, text_ids, registry,
        )

        csi_to_text = compute_ranking_metrics(
            sims, relevance, sample_classes,
            ks=self.ks, map_ks=self.map_ks, ndcg_ks=self.ndcg_ks,
            query_chunk=self.query_chunk,
        )
        text_to_csi = compute_ranking_metrics(
            np.ascontiguousarray(sims.T), np.ascontiguousarray(relevance.T), text_classes,
            ks=self.ks, map_ks=self.map_ks, ndcg_ks=self.ndcg_ks,
            query_chunk=self.query_chunk,
        )

        dedup_info, dedup_metrics = self._dedup_control(
            sims, relevance, raw_text_embs.numpy(), sample_classes, text_classes,
        )

        results: dict = {
            "csi_to_text": csi_to_text,
            "text_to_csi": text_to_csi,
            "near_duplicate_control": dedup_info,
            "gallery": {
                "num_texts": len(text_ids),
                "num_test_samples": len(sample_ids),
                "num_classes": len(registry.class_to_texts),
                "num_samples_dropped": int(n_unknown),
                "registry_stats": registry.stats(),
            },
        }
        if dedup_metrics:
            results["csi_to_text_dedup"] = dedup_metrics
        return results


@torch.no_grad()
def run_m2m_retrieval_evaluation(
    model: nn.Module,
    dataloader,
    registry: PairRegistry,
    text_encoder: nn.Module,
    text_projection: Optional[nn.Module] = None,
    csi_projection: Optional[nn.Module] = None,
    device: str = "cuda",
    ks: Sequence[int] = DEFAULT_KS,
    map_ks: Sequence[int] = DEFAULT_MAP_KS,
    ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
    dedup_cosine: float = DEFAULT_DEDUP_COSINE,
) -> dict:
    """End-to-end many-to-many retrieval evaluation.

    Encodes the test split once, then scores both directions against the full
    caption gallery.  Requires the dataloader to emit ``sample_ids`` so that
    each CSI sample can be linked back to its registry edges.
    """
    model.eval()
    csi_encoder = model.csi_encoder if hasattr(model, "csi_encoder") else model
    if csi_projection is None and hasattr(model, "csi_projection"):
        csi_projection = model.csi_projection
    if text_projection is None and hasattr(model, "text_projection"):
        text_projection = model.text_projection

    all_embeddings: list[torch.Tensor] = []
    all_sample_ids: list[str] = []

    for batch in dataloader:
        if "sample_ids" not in batch:
            raise KeyError(
                "many-to-many retrieval needs 'sample_ids' in the batch; "
                "use CSITextDataset.collate_fn"
            )
        csi = batch["csi"].to(device)
        features = csi_encoder(csi)
        if csi_projection is not None:
            features = csi_projection(features)
        all_embeddings.append(features.cpu())
        all_sample_ids.extend(batch["sample_ids"])

    csi_embeddings = torch.cat(all_embeddings, dim=0)

    evaluator = ManyToManyRetrievalEvaluator(
        ks=ks, map_ks=map_ks, ndcg_ks=ndcg_ks, dedup_cosine=dedup_cosine,
    )
    results = evaluator.evaluate(
        csi_embeddings, all_sample_ids, registry,
        text_encoder, text_projection, device=device,
    )

    c2t = results.get("csi_to_text", {}).get("strict", {})
    t2c = results.get("text_to_csi", {}).get("strict", {})
    logger.info(
        "M2M retrieval (strict): CSI->Text mAP=%.4f (chance %.4f, norm %.4f)  "
        "Text->CSI mAP=%.4f (chance %.4f)",
        c2t.get("mAP", 0.0), c2t.get("chance", {}).get("mAP", 0.0),
        c2t.get("normalized_mAP", 0.0),
        t2c.get("mAP", 0.0), t2c.get("chance", {}).get("mAP", 0.0),
    )
    return results
