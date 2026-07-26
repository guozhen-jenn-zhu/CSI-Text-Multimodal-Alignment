#!/usr/bin/env python3
"""Validate the many-to-many retrieval metrics before running the full sweep.

Four independent checks:

1. ``compute_ranking_metrics`` against a slow per-query reference written from
   the textbook definitions.
2. The same against ``sklearn``'s ``average_precision_score`` / ``ndcg_score``.
3. ``expected_random_ap`` (closed form) against a Monte-Carlo simulation.
4. The real pair registry: relevance-matrix structure, and the fact that random
   embeddings score at the analytic chance mAP.

Usage::

    python scripts/validate_m2m_metrics.py
    python scripts/validate_m2m_metrics.py --registry ~/Downloads/pair_registry.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pair_registry import PairRegistry  # noqa: E402
from evaluation.retrieval_m2m import (  # noqa: E402
    RELEVANCE_LEVELS,
    ManyToManyRetrievalEvaluator,
    compute_ranking_metrics,
    duplicate_groups,
    expected_random_ap,
)

KS = (1, 5, 10)
MAP_KS = (10, 100)
NDCG_KS = (10, 100)

_failures: list[str] = []


def check(name: str, got: float, want: float, tol: float = 1e-9) -> None:
    ok = abs(got - want) <= tol
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}: got {got:.10f}  want {want:.10f}")
    if not ok:
        _failures.append(name)


# ---------------------------------------------------------------------------
# Reference implementation (slow, one query at a time, textbook definitions)
# ---------------------------------------------------------------------------

def reference_query_metrics(
    scores: np.ndarray, grades: np.ndarray, threshold: int = 1,
) -> dict:
    n_gallery = len(scores)
    order = sorted(range(n_gallery), key=lambda i: (-scores[i], i))
    rel = [1.0 if grades[i] >= threshold else 0.0 for i in order]
    graded = [float(grades[i]) for i in order]
    n_pos = int(sum(rel))
    if n_pos == 0:
        return {}

    hits = 0
    ap = 0.0
    for rank, r in enumerate(rel, start=1):
        if r > 0:
            hits += 1
            ap += hits / rank
    ap /= n_pos

    out = {"ap": ap}

    for k in MAP_KS:
        hits_k = 0
        ap_k = 0.0
        for rank, r in enumerate(rel[:k], start=1):
            if r > 0:
                hits_k += 1
                ap_k += hits_k / rank
        out[f"ap@{k}"] = ap_k / min(n_pos, k)

    for k in KS:
        hits_k = sum(rel[:k])
        out[f"recall@{k}"] = hits_k / n_pos
        out[f"precision@{k}"] = hits_k / k
        out[f"hit_rate@{k}"] = 1.0 if hits_k > 0 else 0.0

    out["r_precision"] = sum(rel[:n_pos]) / n_pos
    out["mrr"] = 1.0 / (rel.index(1.0) + 1)

    ideal_binary = sorted(rel, reverse=True)
    ideal_graded = sorted((2.0**g - 1.0 for g in graded), reverse=True)
    for k in NDCG_KS:
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel[:k]))
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_binary[:k]))
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
        g_dcg = sum((2.0**g - 1.0) / np.log2(i + 2) for i, g in enumerate(graded[:k]))
        g_idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_graded[:k]))
        out[f"ndcg_graded@{k}"] = g_dcg / g_idcg if g_idcg > 0 else 0.0

    return out


def test_against_reference(rng: np.random.Generator) -> None:
    print("\n[1] compute_ranking_metrics vs. brute-force reference")
    n_queries, n_gallery = 40, 137
    sims = rng.normal(size=(n_queries, n_gallery)).astype(np.float32)
    grades = rng.choice([0, 0, 0, 1, 2], size=(n_queries, n_gallery)).astype(np.int8)
    grades[3] = 0  # a query with no positive must be dropped, not crash

    got = compute_ranking_metrics(
        sims, grades, ks=KS, map_ks=MAP_KS, ndcg_ks=NDCG_KS, query_chunk=7,
    )

    for level, threshold in RELEVANCE_LEVELS.items():
        print(f"  -- level '{level}' (grade >= {threshold})")
        per_query = [
            reference_query_metrics(sims[q], grades[q], threshold)
            for q in range(n_queries)
        ]
        per_query = [m for m in per_query if m]
        block = got[level]

        def mean(key: str, rows=per_query) -> float:
            return float(np.mean([m[key] for m in rows]))

        check(f"{level}/num_queries", block["num_queries"], len(per_query))
        check(f"{level}/mAP", block["mAP"], mean("ap"), tol=1e-9)
        check(f"{level}/r_precision", block["r_precision"], mean("r_precision"), tol=1e-9)
        check(f"{level}/mrr", block["mrr"], mean("mrr"), tol=1e-9)
        for k in MAP_KS:
            check(f"{level}/mAP@{k}", block[f"mAP@{k}"], mean(f"ap@{k}"), tol=1e-9)
        for k in NDCG_KS:
            check(f"{level}/ndcg@{k}", block[f"ndcg@{k}"], mean(f"ndcg@{k}"), tol=1e-9)
        for k in KS:
            check(f"{level}/recall@{k}", block[f"recall@{k}"], mean(f"recall@{k}"), tol=1e-9)
            check(f"{level}/precision@{k}", block[f"precision@{k}"], mean(f"precision@{k}"), tol=1e-9)
            check(f"{level}/hit_rate@{k}", block[f"hit_rate@{k}"], mean(f"hit_rate@{k}"), tol=1e-9)

    print("  -- graded nDCG (threshold-independent)")
    graded_rows = [
        reference_query_metrics(sims[q], grades[q], 1) for q in range(n_queries)
    ]
    graded_rows = [m for m in graded_rows if m]
    for k in NDCG_KS:
        check(
            f"graded/ndcg_graded@{k}",
            got["graded"][f"ndcg_graded@{k}"],
            float(np.mean([m[f"ndcg_graded@{k}"] for m in graded_rows])),
            tol=1e-9,
        )


def test_against_sklearn(rng: np.random.Generator) -> None:
    print("\n[2] compute_ranking_metrics vs. sklearn")
    from sklearn.metrics import average_precision_score, ndcg_score

    n_queries, n_gallery = 25, 200
    sims = rng.normal(size=(n_queries, n_gallery)).astype(np.float64)
    grades = (rng.random((n_queries, n_gallery)) < 0.08).astype(np.int8)
    grades[grades.sum(axis=1) == 0, 0] = 1

    got = compute_ranking_metrics(sims, grades, ks=KS, map_ks=MAP_KS, ndcg_ks=NDCG_KS)
    block = got["class_level"]

    sk_ap = float(np.mean([
        average_precision_score(grades[q], sims[q]) for q in range(n_queries)
    ]))
    check("mAP (sklearn average_precision_score)", block["mAP"], sk_ap, tol=1e-8)

    for k in NDCG_KS:
        sk_ndcg = float(np.mean([
            ndcg_score(grades[q][None, :], sims[q][None, :], k=k)
            for q in range(n_queries)
        ]))
        check(f"ndcg@{k} (sklearn ndcg_score)", block[f"ndcg@{k}"], sk_ndcg, tol=1e-8)


def test_chance_baseline(rng: np.random.Generator) -> None:
    print("\n[3] expected_random_ap (closed form) vs. Monte-Carlo")
    for n_pos, n_gallery, trials in [(1, 50, 40000), (9, 221, 20000), (37, 60, 20000)]:
        aps = np.empty(trials)
        ranks = np.arange(1, n_gallery + 1)
        for t in range(trials):
            rel = np.zeros(n_gallery)
            rel[rng.choice(n_gallery, size=n_pos, replace=False)] = 1.0
            aps[t] = (np.cumsum(rel) / ranks * rel).sum() / n_pos
        mc = float(aps.mean())
        stderr = float(aps.std() / np.sqrt(trials))
        check(
            f"E[AP] P={n_pos} G={n_gallery} (MC +/- {stderr:.5f})",
            expected_random_ap(n_pos, n_gallery), mc, tol=max(5 * stderr, 1e-4),
        )
    check("E[AP] all-positive", expected_random_ap(10, 10), 1.0)
    check("E[AP] no positive", expected_random_ap(0, 10), 0.0)


def test_duplicate_groups(rng: np.random.Generator) -> None:
    print("\n[4] duplicate_groups leader clustering")
    threshold = 0.95

    # Five seeds, each with near-copies that must land in the same group.
    seeds = rng.normal(size=(5, 48))
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    rows, truth = [], []
    for s, seed in enumerate(seeds):
        for _ in range(4):
            v = seed + 0.02 * rng.normal(size=48)
            rows.append(v / np.linalg.norm(v))
            truth.append(s)
    embeddings = np.stack(rows)

    groups = duplicate_groups(embeddings, threshold)
    recovered = all(len({truth[i] for i in g}) == 1 for g in groups)
    check("near-copies never merge across seeds", float(recovered), 1.0)
    check("recovers one group per seed", float(len(groups)), 5.0)

    # Every member must be within threshold of its own representative.
    worst = min(
        float((embeddings[g] @ embeddings[g[0]]).min()) for g in groups
    )
    ok = worst > threshold
    print(f"  [{'ok  ' if ok else 'FAIL'}] min member-to-leader cosine {worst:.4f} > {threshold}")
    if not ok:
        _failures.append("member-to-leader cosine below threshold")

    # Orthogonal rows must never merge.
    identity = np.eye(6)
    check("orthogonal rows stay separate", float(len(duplicate_groups(identity, threshold))), 6.0)


def test_real_registry(registry_path: Path, rng: np.random.Generator) -> None:
    print(f"\n[5] real registry: {registry_path}")
    registry = PairRegistry.load(str(registry_path))
    stats = registry.stats()
    print(f"  {registry!r}")

    all_samples = registry.get_all_sample_ids()
    sample_ids = list(rng.choice(all_samples, size=min(3000, len(all_samples)), replace=False))
    text_ids = registry.get_all_text_ids()

    relevance, sample_classes, text_classes = ManyToManyRetrievalEvaluator._build_relevance(
        sample_ids, text_ids, registry,
    )

    n_grade2 = (relevance == 2).sum(axis=1)
    expected_grade2 = np.array([len(registry.get_positive_texts(s)) for s in sample_ids])
    check(
        "grade-2 count == registry edges per sample",
        float(np.abs(n_grade2 - expected_grade2).max()), 0.0,
    )
    check(
        "avg positives per sample matches registry stats",
        float(n_grade2.mean()), float(stats["avg_texts_per_sample"]), tol=0.5,
    )

    n_positive = (relevance >= 1).sum(axis=1)
    print(
        f"  gallery={len(text_ids)} captions | grade-2/sample={n_grade2.mean():.2f} "
        f"| grade>=1/sample={n_positive.mean():.2f} "
        f"| classes={len(np.unique(sample_classes))}"
    )
    if (n_positive < n_grade2).any():
        _failures.append("grade>=1 must be a superset of grade-2")
        print("  [FAIL] grade>=1 is not a superset of grade-2")

    print("  random embeddings should land on the analytic chance mAP:")
    dim = 64
    csi = rng.normal(size=(len(sample_ids), dim))
    txt = rng.normal(size=(len(text_ids), dim))
    csi /= np.linalg.norm(csi, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    sims = (csi @ txt.T).astype(np.float32)

    got = compute_ranking_metrics(
        sims, relevance, sample_classes, ks=KS, map_ks=MAP_KS, ndcg_ks=NDCG_KS,
    )
    for level in RELEVANCE_LEVELS:
        block = got[level]
        check(
            f"{level}/random-embedding mAP == chance mAP",
            block["mAP"], block["chance"]["mAP"], tol=0.02,
        )
        check(
            f"{level}/random-embedding recall@10 == chance recall@10",
            block["recall@10"], block["chance"]["recall@10"], tol=0.02,
        )
        print(
            f"    {level:12s} positives/query={block['avg_positives_per_query']:.2f}  "
            f"chance mAP={block['chance']['mAP']:.4f}  "
            f"chance hit_rate@1={block['chance']['hit_rate@1']:.4f}"
        )
    print(f"  levels_identical={got['levels_identical']}")

    print("  a perfect ranking should score 1.0:")
    perfect = compute_ranking_metrics(
        relevance.astype(np.float32), relevance, sample_classes,
        ks=KS, map_ks=MAP_KS, ndcg_ks=NDCG_KS,
    )
    check("perfect strict mAP", perfect["strict"]["mAP"], 1.0, tol=1e-9)
    check("perfect class_level mAP", perfect["class_level"]["mAP"], 1.0, tol=1e-9)
    check("perfect ndcg_graded@10", perfect["graded"]["ndcg_graded@10"], 1.0, tol=1e-9)
    check("perfect normalized mAP", perfect["strict"]["normalized_mAP"], 1.0, tol=1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(Path.home() / "Downloads" / "pair_registry.json"),
        help="Pair registry used for the end-to-end structural check",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    test_against_reference(rng)
    test_against_sklearn(rng)
    test_chance_baseline(rng)
    test_duplicate_groups(rng)

    registry_path = Path(args.registry).expanduser()
    if registry_path.is_file():
        test_real_registry(registry_path, rng)
    else:
        print(f"\n[5] skipped -- registry not found at {registry_path}")

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
