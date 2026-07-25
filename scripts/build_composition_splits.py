"""Build held-out-composition (HOC) task roots for the scene model.

To test whether compositional scene understanding is genuine composition rather
than memorised co-occurrence, a subset of the ``(user, activity, proximity)``
combinations is withheld from training and evaluated at test time.  Two
variants are produced, sharing the same held-out set so their numbers are
directly comparable:

``strict``
    Samples carrying a held-out combination are removed from train and val.
    The model never sees the signal condition or the composed sentence.  This
    is the conventional compositional zero-shot protocol.

``text``
    Samples stay in training but their Level 2 composite captions are stripped
    from the pair registry.  The model still sees the signal and every Level 1
    attribute caption; only the composed sentence is novel.  This isolates
    composition in text space from novelty in the signal.

Held-out combinations are chosen so that every attribute value -- and, by
default, every realised pairwise attribute cell -- still occurs in training.
An unseen combination is therefore always built from parts the model has seen,
which is precisely the claim under review.

Each variant is materialised as a set of task roots that mirror the originals
via symlinks, so nothing in the data or training pipeline needs to change:
``MultitaskCSITextDataset`` resolves ``{task_root}/splits/{split}_id.json`` and
``training/multitask_train.py`` auto-discovers ``{task_root}/pair_registry.json``.

Usage
-----
PYTHONPATH=. python scripts/build_composition_splits.py \
    --config configs/experiment/sagemaker_multitask_scene.yaml \
    --mode both --holdout_fraction 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import yaml

from data.text_generation.generate_texts import TextGenerator
from evaluation.composition import (
    SCENE_TASK_ATTRIBUTES,
    TRIPLE_ATTRIBUTES,
    SceneAttributeTable,
    Triple,
    audit_composition,
    load_scene_attribute_table,
    render_triple,
    training_composite_patterns,
)

logger = logging.getLogger(__name__)

RESERVED_DIRS = {"metadata", "splits"}


# ----------------------------------------------------------------------
# Held-out selection
# ----------------------------------------------------------------------


def _value_counts(triples: Iterable[Triple]) -> dict[str, Counter]:
    counts = {attr: Counter() for attr in TRIPLE_ATTRIBUTES}
    for t in triples:
        for pos, attr in enumerate(TRIPLE_ATTRIBUTES):
            counts[attr][t[pos]] += 1
    return counts


def _pair_counts(triples: Iterable[Triple]) -> dict[tuple[int, int], Counter]:
    positions = list(combinations(range(3), 2))
    counts = {p: Counter() for p in positions}
    for t in triples:
        for (i, j) in positions:
            counts[(i, j)][(t[i], t[j])] += 1
    return counts


def _enforceable_pairs(
    total_pair_counts: dict[tuple[int, int], Counter],
    mode: str,
) -> list[tuple[int, int]]:
    """Decide which pairwise attribute cells must survive in training.

    Enforcing coverage for a pair is only possible when every realised cell of
    that pair contains at least two combinations -- otherwise withholding any
    combination empties a cell, and no holdout exists at all.  In CSI-Bench the
    non-walking activities are recorded only at 1.5 m, so each
    ``(user, activity)`` cell holds exactly one combination and that constraint
    is vacuous; ``activity x proximity`` cells hold six and are enforceable.
    """
    if mode == "none":
        return []
    pairs = sorted(total_pair_counts)
    if mode == "all":
        return pairs
    return [p for p in pairs if min(total_pair_counts[p].values(), default=0) >= 2]


def select_held_out(
    table: SceneAttributeTable,
    holdout_fraction: float = 0.25,
    min_test_samples: int = 20,
    min_triples_per_value: int = 1,
    pair_coverage: str = "feasible",
    seed: int = 42,
) -> tuple[list[Triple], dict[str, Any]]:
    """Choose combinations to withhold from training.

    A combination is eligible when it occurs in the train split (so the
    baseline model would have learned it) and has at least *min_test_samples*
    test samples (so the unseen-composition estimate is not pure noise).
    Selection is greedy and balanced: at each step the eligible combination
    whose attribute values are least represented among the already-selected
    holdouts is taken, subject to the coverage constraints.
    """
    train_triples = sorted(table.realized_triples("train"))
    test_counts = table.triple_counts("test")

    eligible = [
        t for t in train_triples if test_counts.get(t, 0) >= min_test_samples
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)

    total_value_counts = _value_counts(train_triples)
    total_pair_counts = _pair_counts(train_triples)
    enforced_pairs = _enforceable_pairs(total_pair_counts, pair_coverage)

    selected: list[Triple] = []
    selected_value_counts = {attr: Counter() for attr in TRIPLE_ATTRIBUTES}
    selected_pair_counts = {p: Counter() for p in combinations(range(3), 2)}

    def feasible(cand: Triple) -> bool:
        for pos, attr in enumerate(TRIPLE_ATTRIBUTES):
            value = cand[pos]
            remaining = (
                total_value_counts[attr][value]
                - selected_value_counts[attr][value]
                - 1
            )
            if remaining < min_triples_per_value:
                return False
        for (i, j) in enforced_pairs:
            cell = (cand[i], cand[j])
            remaining = (
                total_pair_counts[(i, j)][cell]
                - selected_pair_counts[(i, j)][cell]
                - 1
            )
            if remaining < 1:
                return False
        return True

    target = int(round(holdout_fraction * len(train_triples)))
    remaining_pool = list(eligible)

    while len(selected) < target and remaining_pool:
        best: Optional[Triple] = None
        best_score: Optional[int] = None
        for cand in remaining_pool:
            if not feasible(cand):
                continue
            score = sum(
                selected_value_counts[attr][cand[pos]]
                for pos, attr in enumerate(TRIPLE_ATTRIBUTES)
            )
            if best_score is None or score < best_score:
                best, best_score = cand, score
        if best is None:
            break

        selected.append(best)
        remaining_pool.remove(best)
        for pos, attr in enumerate(TRIPLE_ATTRIBUTES):
            selected_value_counts[attr][best[pos]] += 1
        for (i, j) in selected_pair_counts:
            selected_pair_counts[(i, j)][(best[i], best[j])] += 1

    selected.sort()
    pair_names = {
        (i, j): f"{TRIPLE_ATTRIBUTES[i]}__{TRIPLE_ATTRIBUTES[j]}"
        for (i, j) in combinations(range(3), 2)
    }
    stats = {
        "num_train_combinations": len(train_triples),
        "num_eligible": len(eligible),
        "target_holdout": target,
        "num_held_out": len(selected),
        "holdout_fraction_achieved": (
            len(selected) / len(train_triples) if train_triples else 0.0
        ),
        "min_test_samples": min_test_samples,
        "pair_coverage": pair_coverage,
        "pair_coverage_enforced": [pair_names[p] for p in enforced_pairs],
        "pair_coverage_skipped": [
            pair_names[p] for p in pair_names if p not in enforced_pairs
        ],
        "min_cell_size_per_pair": {
            pair_names[p]: min(total_pair_counts[p].values(), default=0)
            for p in pair_names
        },
        "seed": seed,
        "held_out_per_value": {
            attr: dict(counter)
            for attr, counter in selected_value_counts.items()
        },
        "unseen_test_samples": int(
            sum(test_counts.get(t, 0) for t in selected)
        ),
        "seen_test_samples": int(
            sum(n for t, n in test_counts.items() if t not in set(selected))
        ),
    }
    if len(selected) < target:
        logger.warning(
            "Only %d of the targeted %d combinations could be withheld under "
            "the coverage constraints; relax --min_test_samples or pass "
            "--pair_coverage none to withhold more.",
            len(selected), target,
        )
    return selected, stats


# ----------------------------------------------------------------------
# Materialising task roots
# ----------------------------------------------------------------------


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(os.path.abspath(src), dst)


def _mirror_task_root(
    source: Path,
    target: Path,
    link_splits: bool,
    link_registry: bool,
) -> None:
    """Create *target* mirroring *source* through symlinks.

    ``metadata``, the H5 data subdirectories and the cached text embeddings are
    always linked.  ``splits`` and ``pair_registry.json`` are linked only when
    the variant does not override them.
    """
    target.mkdir(parents=True, exist_ok=True)

    _link(source / "metadata", target / "metadata")

    for child in source.iterdir():
        if child.is_dir() and child.name not in RESERVED_DIRS:
            _link(child, target / child.name)
        elif child.is_file() and child.name.startswith("text_embeddings_"):
            _link(child, target / child.name)

    if link_splits:
        _link(source / "splits", target / "splits")
    if link_registry and (source / "pair_registry.json").is_file():
        _link(source / "pair_registry.json", target / "pair_registry.json")


def write_strict_variant(
    table: SceneAttributeTable,
    task_entries: Sequence[dict[str, Any]],
    held_out: Sequence[Triple],
    output_root: Path,
    move_heldout_to_test: bool,
) -> dict[str, Any]:
    """Remove held-out-combination samples from the train and val splits."""
    held = set(held_out)
    drop_ids = {
        sid for sid, triple in zip(table.sample_ids, table.triples)
        if triple in held
    }

    per_task: dict[str, Any] = {}
    for entry in task_entries:
        task_name = entry["task_name"]
        source = Path(entry["task_root"])
        target = output_root / task_name
        _mirror_task_root(source, target, link_splits=False, link_registry=True)

        splits_dir = target / "splits"
        # A leftover symlink from another variant would make the writes below
        # land in the source dataset.
        if splits_dir.is_symlink():
            splits_dir.unlink()
        splits_dir.mkdir(parents=True, exist_ok=True)

        original_splits_dir = source / "splits"
        counts: dict[str, int] = {}
        moved: list[str] = []

        for split in ("train", "val"):
            ids = _read_split(original_splits_dir, split)
            kept = [sid for sid in ids if sid not in drop_ids]
            moved.extend(sid for sid in ids if sid in drop_ids)
            _write_split(splits_dir, split, kept)
            counts[split] = len(kept)

        test_ids = _read_split(original_splits_dir, "test")
        if move_heldout_to_test:
            existing = set(test_ids)
            test_ids = list(test_ids) + [
                sid for sid in moved if sid not in existing
            ]
        _write_split(splits_dir, "test", test_ids)
        counts["test"] = len(test_ids)
        counts["dropped_from_train_val"] = len(moved)

        per_task[task_name] = {"task_root": str(target), "split_sizes": counts}

    return {"variant": "strict", "output_root": str(output_root), "tasks": per_task}


def write_text_variant(
    table: SceneAttributeTable,
    task_entries: Sequence[dict[str, Any]],
    held_out: Sequence[Triple],
    output_root: Path,
) -> dict[str, Any]:
    """Strip the Level 2 captions of held-out combinations from the registries."""
    held = set(held_out)
    n_patterns = len(training_composite_patterns())
    removed_text_ids = {
        TextGenerator._make_text_id(render_triple(t, template_idx=i))
        for t in held
        for i in range(n_patterns)
    }

    per_task: dict[str, Any] = {}
    for entry in task_entries:
        task_name = entry["task_name"]
        source = Path(entry["task_root"])
        target = output_root / task_name
        _mirror_task_root(source, target, link_splits=True, link_registry=False)

        registry_path = source / "pair_registry.json"
        if not registry_path.is_file():
            raise FileNotFoundError(
                f"{registry_path} not found; the text variant needs the "
                "per-task pair registry alongside the task root."
            )
        with open(registry_path) as f:
            registry = json.load(f)

        present = removed_text_ids & set(registry.get("texts", {}))
        registry["texts"] = {
            tid: text for tid, text in registry["texts"].items()
            if tid not in removed_text_ids
        }
        registry["sample_to_texts"] = {
            sid: [tid for tid in tids if tid not in removed_text_ids]
            for sid, tids in registry["sample_to_texts"].items()
        }
        registry["class_to_texts"] = {
            cid: [tid for tid in tids if tid not in removed_text_ids]
            for cid, tids in registry.get("class_to_texts", {}).items()
        }

        empty = [
            sid for sid, tids in registry["sample_to_texts"].items() if not tids
        ]
        if empty:
            raise ValueError(
                f"{len(empty)} samples in {task_name} lost every caption "
                f"(e.g. {empty[:3]}); the Level 1 captions should have "
                "survived, so the registry does not look as expected."
            )

        out_registry = target / "pair_registry.json"
        if out_registry.is_symlink():
            out_registry.unlink()
        with open(out_registry, "w") as f:
            json.dump(registry, f, indent=2)

        per_task[task_name] = {
            "task_root": str(target),
            "expected_removed_text_ids": len(removed_text_ids),
            "removed_text_ids_found": len(present),
            "remaining_texts": len(registry["texts"]),
        }
        if len(present) != len(removed_text_ids):
            logger.warning(
                "%s: %d of %d held-out composite captions were not present in "
                "the registry; check that the registry was built with Level 2 "
                "captions for every sample.",
                task_name, len(removed_text_ids) - len(present),
                len(removed_text_ids),
            )

    return {"variant": "text", "output_root": str(output_root), "tasks": per_task}


def _held_out_sample_counts(
    table: SceneAttributeTable, held_out: Sequence[Triple],
) -> dict[str, Any]:
    """Per-split sample counts for each held-out combination."""
    held = set(held_out)
    per_split = {
        split: table.triple_counts(split) for split in sorted(set(table.splits))
    }
    return {
        "|".join(t): {split: int(counts.get(t, 0)) for split, counts in per_split.items()}
        for t in sorted(held)
    }


def _read_split(splits_dir: Path, split: str) -> list[str]:
    for name in (f"{split}_id.json", f"{split}_ids.json", f"{split}.json"):
        path = splits_dir / name
        if path.is_file():
            with open(path) as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for key in (split, "ids", "sample_ids", "samples"):
                    if key in payload:
                        payload = payload[key]
                        break
            return [str(i) for i in payload]
    raise FileNotFoundError(f"No '{split}' split file under {splits_dir}")


def _write_split(splits_dir: Path, split: str, ids: Sequence[str]) -> None:
    with open(splits_dir / f"{split}_id.json", "w") as f:
        json.dump(list(ids), f)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build held-out-composition task roots for the scene model",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/experiment/sagemaker_multitask_scene.yaml",
    )
    parser.add_argument(
        "--mode", choices=["strict", "text", "both"], default="both",
    )
    parser.add_argument("--holdout_fraction", type=float, default=0.25)
    parser.add_argument(
        "--min_test_samples", type=int, default=20,
        help="Minimum test samples a combination needs to be withheld.",
    )
    parser.add_argument(
        "--min_triples_per_value", type=int, default=1,
        help="Training combinations each attribute value must retain.",
    )
    parser.add_argument(
        "--pair_coverage", choices=["feasible", "all", "none"], default="feasible",
        help="Which pairwise attribute cells must survive in training. "
             "'feasible' (default) enforces every pair whose cells are large "
             "enough for the constraint to be satisfiable, 'all' enforces all "
             "three pairs, 'none' only keeps each attribute value.",
    )
    parser.add_argument(
        "--move_heldout_to_test", action="store_true",
        help="Append the dropped train/val samples to the test split instead "
             "of discarding them. Increases the unseen sample count but makes "
             "the test population differ from the baseline run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict_root", type=str, default=None,
        help="Output root for the strict variant "
             "(default: sibling '<multitask>_hoc' of the task roots).",
    )
    parser.add_argument(
        "--text_root", type=str, default=None,
        help="Output root for the text variant "
             "(default: sibling '<multitask>_hoc_text').",
    )
    parser.add_argument(
        "--manifest_dir", type=str, default="paper_results/composition",
        help="Where to also copy the split manifest for bookkeeping.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing output roots before writing.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only choose the held-out combinations and write the manifest. "
             "Useful for inspecting the selection from a metadata-only copy "
             "of the dataset.",
    )
    parser.add_argument(
        "--check_files", action="store_true",
        help="Verify H5 files exist while building the attribute table.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task_entries = [
        e for e in config.get("multitask", {}).get("tasks", [])
        if isinstance(e, dict) and e.get("task_name") in SCENE_TASK_ATTRIBUTES
    ]
    if len(task_entries) != 3:
        raise SystemExit(
            f"{args.config} must define the har, proximity and user_id "
            "multitask entries."
        )

    table = load_scene_attribute_table(
        task_entries, require_valid_files=args.check_files,
    )
    audit = audit_composition(table)
    logger.info(
        "Realised %d of %d combinations across %d co-labeled samples.",
        audit["realized_triples"], audit["total_combinations"], len(table),
    )

    held_out, stats = select_held_out(
        table,
        holdout_fraction=args.holdout_fraction,
        min_test_samples=args.min_test_samples,
        min_triples_per_value=args.min_triples_per_value,
        pair_coverage=args.pair_coverage,
        seed=args.seed,
    )
    if not held_out:
        raise SystemExit(
            "No combination satisfied the holdout constraints. Lower "
            "--min_test_samples or pass --pair_coverage none."
        )

    base_parent = Path(task_entries[0]["task_root"]).parent
    strict_root = Path(
        args.strict_root or f"{base_parent}_hoc"
    )
    text_root = Path(
        args.text_root or f"{base_parent}_hoc_text"
    )

    manifest: dict[str, Any] = {
        "source_config": args.config,
        "dry_run": bool(args.dry_run),
        "held_out_triples": ["|".join(t) for t in held_out],
        "selection": stats,
        "audit": {
            "realized_triples": audit["realized_triples"],
            "total_combinations": audit["total_combinations"],
            "coverage": audit["coverage"],
        },
        "variants": {},
    }

    manifest["held_out_sample_counts"] = _held_out_sample_counts(table, held_out)

    if args.dry_run:
        logger.info("Dry run: no task roots written.")
    else:
        if args.mode in ("strict", "both"):
            if args.force and strict_root.exists():
                shutil.rmtree(strict_root)
            manifest["variants"]["strict"] = write_strict_variant(
                table, task_entries, held_out, strict_root,
                move_heldout_to_test=args.move_heldout_to_test,
            )

        if args.mode in ("text", "both"):
            if args.force and text_root.exists():
                shutil.rmtree(text_root)
            manifest["variants"]["text"] = write_text_variant(
                table, task_entries, held_out, text_root,
            )

        for variant in manifest["variants"].values():
            root = Path(variant["output_root"])
            with open(root / "composition_split.json", "w") as f:
                json.dump(manifest, f, indent=2)

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "composition_split.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Withheld {stats['num_held_out']} of {stats['num_train_combinations']} "
        f"training combinations "
        f"({100 * stats['holdout_fraction_achieved']:.0f}%), covering "
        f"{stats['unseen_test_samples']} unseen and "
        f"{stats['seen_test_samples']} seen test samples."
    )
    for name, variant in manifest["variants"].items():
        print(f"  {name}: {variant['output_root']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
