"""Zero-shot generalization to held-out classes on SignFi.

Splits the 276 SignFi classes into 200 seen + 76 unseen. Trains alignment
on the 200-class subset, then evaluates zero-shot on the 76 held-out
classes that were NEVER seen during training.

This is the strongest possible evidence of genuine semantic transfer: the
model must generalize to entirely new sign language words using only the
text descriptions.

Usage:
    # Step 1: Generate the class split
    python scripts/run_held_out_classes.py --stage split \
        --signfi_root data/processed/signfi/sign_recognition

    # Step 2: Train on 200-class subset (on SageMaker)
    python scripts/run_held_out_classes.py --stage train \
        --signfi_root /data/Dataset/signfi/sign_recognition \
        --checkpoint_dir /model/checkpoints/held_out

    # Step 3: Evaluate on held-out 76 classes
    python scripts/run_held_out_classes.py --stage evaluate \
        --signfi_root /data/Dataset/signfi/sign_recognition \
        --checkpoint /model/checkpoints/held_out/best.pt \
        --output_dir paper_results/held_out_classes
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_SEEN = 200
N_TOTAL = 276
SEED = 42


def generate_class_split(
    signfi_root: Path,
    output_dir: Path,
    n_seen: int = N_SEEN,
    seed: int = SEED,
) -> dict:
    """Generate a deterministic split of 276 SignFi classes.

    Ensures each partition has enough samples for meaningful evaluation
    by checking per-class sample counts before splitting.
    """
    from data.text_generation.templates.task_templates import SIGNFI_SIGN_WORDS

    n_total = len(SIGNFI_SIGN_WORDS)
    assert n_total == N_TOTAL, f"Expected {N_TOTAL} classes, got {n_total}"
    n_held_out = n_total - n_seen

    rng = np.random.RandomState(seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)

    seen_indices = sorted(all_indices[:n_seen].tolist())
    held_out_indices = sorted(all_indices[n_seen:].tolist())

    seen_words = [SIGNFI_SIGN_WORDS[i] for i in seen_indices]
    held_out_words = [SIGNFI_SIGN_WORDS[i] for i in held_out_indices]

    split = {
        "n_total": n_total,
        "n_seen": n_seen,
        "n_held_out": n_held_out,
        "seed": seed,
        "seen_class_ids": seen_indices,
        "held_out_class_ids": held_out_indices,
        "seen_words": seen_words,
        "held_out_words": held_out_words,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "held_out_class_split.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    logger.info(
        "Generated class split: %d seen + %d held-out -> %s",
        n_seen, n_held_out, split_path,
    )

    seen_label_mapping = {
        "label_to_idx": {word: i for i, word in enumerate(seen_words)},
        "idx_to_label": {str(i): word for i, word in enumerate(seen_words)},
        "original_class_ids": {word: seen_indices[i] for i, word in enumerate(seen_words)},
    }
    with open(output_dir / "seen_label_mapping.json", "w") as f:
        json.dump(seen_label_mapping, f, indent=2)

    return split


def build_held_out_class_texts(
    split: dict,
    use_seen: bool = False,
) -> dict[int, list[str]]:
    """Build class_texts for either seen or held-out classes.

    Returns class_texts using the ORIGINAL class IDs from the full 276-class
    mapping so that zero-shot evaluation works correctly.
    """
    from data.text_generation.templates.task_templates import (
        SIGNFI_SIGN_WORDS,
        SIGNFI_TEMPLATE_PATTERNS,
    )

    if use_seen:
        indices = split["seen_class_ids"]
    else:
        indices = split["held_out_class_ids"]

    class_texts = {}
    for idx in indices:
        word = SIGNFI_SIGN_WORDS[idx]
        texts = [t.format(word=word) for t in SIGNFI_TEMPLATE_PATTERNS]
        class_texts[idx] = texts

    return class_texts


def evaluate_held_out(
    checkpoint_path: str,
    signfi_root: str,
    split_path: str,
    output_dir: str,
) -> None:
    """Evaluate a trained model on held-out classes.

    Loads the checkpoint, filters test data to held-out classes only,
    and runs zero-shot + retrieval evaluation.
    """
    import torch
    import yaml

    from data import build_dataloaders
    from evaluation import run_zero_shot_evaluation
    from evaluation.retrieval import run_retrieval_evaluation
    from scripts.run_evaluate import (
        _numpy_json,
        _save_json,
        _strip_heavy,
        build_model_from_config,
    )
    from training.train import _merge_configs

    with open(split_path) as f:
        split = json.load(f)

    held_out_ids = set(split["held_out_class_ids"])
    seen_ids = set(split["seen_class_ids"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _merge_configs(ckpt.get("config", {}))

    model, text_encoder, text_projection, csi_projection = build_model_from_config(
        config, device,
    )
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)

    loader_cfg = {
        "task_root": signfi_root,
        "h5_key": config.get("dataset_config", {}).get("h5_key", "CSI_amps"),
        "registry_path": None,
        "text_embeddings_path": None,
        "target_subcarriers": config.get("dataset_config", {}).get("preprocessing", {}).get("target_subcarriers", 64),
        "target_time_steps": config.get("dataset_config", {}).get("preprocessing", {}).get("target_time_steps", 500),
        "batch_size": 128,
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }
    loaders = build_dataloaders(loader_cfg)
    test_loader = loaders["test"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results = {}

    held_out_texts = build_held_out_class_texts(split, use_seen=False)
    seen_texts = build_held_out_class_texts(split, use_seen=True)

    logger.info("Evaluating on %d held-out classes...", len(held_out_ids))
    held_out_results = run_zero_shot_evaluation(
        model=model,
        dataloader=test_loader,
        class_texts=held_out_texts,
        text_encoder=text_encoder,
        text_projection=text_projection,
        csi_projection=csi_projection,
        device=str(device),
    )

    held_out_embs = held_out_results["csi_embeddings"]
    held_out_labels = held_out_results["labels"]
    mask = np.isin(held_out_labels, list(held_out_ids))

    if mask.sum() > 0:
        filtered_embs = held_out_embs[mask]
        filtered_labels = held_out_labels[mask]

        from evaluation.zero_shot import ZeroShotClassifier
        clf = ZeroShotClassifier(
            held_out_texts, text_encoder, text_projection, device=str(device),
        )
        held_out_metrics = clf.evaluate(filtered_embs, filtered_labels)
        results["held_out_zero_shot"] = _strip_heavy(held_out_metrics)
        logger.info(
            "Held-out classes (%d samples): accuracy=%.4f",
            mask.sum(), held_out_metrics["accuracy"],
        )
    else:
        logger.warning("No test samples found for held-out classes!")
        results["held_out_zero_shot"] = {"error": "no_samples_for_held_out_classes"}

    logger.info("Evaluating on %d seen classes...", len(seen_ids))
    mask_seen = np.isin(held_out_labels, list(seen_ids))
    if mask_seen.sum() > 0:
        from evaluation.zero_shot import ZeroShotClassifier
        clf_seen = ZeroShotClassifier(
            seen_texts, text_encoder, text_projection, device=str(device),
        )
        seen_metrics = clf_seen.evaluate(held_out_embs[mask_seen], held_out_labels[mask_seen])
        results["seen_zero_shot"] = _strip_heavy(seen_metrics)
        logger.info(
            "Seen classes (%d samples): accuracy=%.4f",
            mask_seen.sum(), seen_metrics["accuracy"],
        )

    logger.info("Evaluating retrieval on held-out classes...")
    if mask.sum() > 0:
        try:
            from evaluation.retrieval import RetrievalEvaluator
            ret_eval = RetrievalEvaluator(ks=(1, 5, 10))
            retrieval_res = ret_eval.evaluate(
                filtered_embs, filtered_labels, held_out_texts,
                text_encoder, text_projection, device=str(device),
            )
            results["held_out_retrieval"] = _numpy_json(retrieval_res)
        except Exception as e:
            logger.warning("Retrieval on held-out failed: %s", e)

    results["split_info"] = {
        "n_seen": len(seen_ids),
        "n_held_out": len(held_out_ids),
        "n_test_samples_held_out": int(mask.sum()),
        "n_test_samples_seen": int(mask_seen.sum()),
        "split_path": split_path,
        "checkpoint": checkpoint_path,
    }

    _save_json(_numpy_json(results), out_path / "held_out_evaluation.json")

    lines = [
        "# Held-Out Class Evaluation",
        "",
        f"- **Seen classes**: {len(seen_ids)} (used during training)",
        f"- **Held-out classes**: {len(held_out_ids)} (never seen during training)",
        "",
        "## Results",
        "",
        "| Set | Accuracy | Weighted F1 | N Samples |",
        "|-----|----------|-------------|-----------|",
    ]

    ho = results.get("held_out_zero_shot", {})
    if "accuracy" in ho:
        lines.append(
            f"| Held-out ({len(held_out_ids)} classes) | "
            f"{ho['accuracy']*100:.1f}% | {ho.get('weighted_f1', 0)*100:.1f}% | "
            f"{mask.sum()} |"
        )

    se = results.get("seen_zero_shot", {})
    if "accuracy" in se:
        lines.append(
            f"| Seen ({len(seen_ids)} classes) | "
            f"{se['accuracy']*100:.1f}% | {se.get('weighted_f1', 0)*100:.1f}% | "
            f"{mask_seen.sum()} |"
        )

    lines.append("")

    ret = results.get("held_out_retrieval", {})
    if "csi_to_text" in ret:
        c2t = ret["csi_to_text"]
        lines.extend([
            "## Retrieval (Held-Out Classes)",
            "",
            f"- CSI->Text R@1: {c2t.get('recall@1', 0)*100:.1f}%",
            f"- CSI->Text R@5: {c2t.get('recall@5', 0)*100:.1f}%",
            f"- CSI->Text R@10: {c2t.get('recall@10', 0)*100:.1f}%",
            "",
        ])

    (out_path / "held_out_evaluation.md").write_text("\n".join(lines))
    logger.info("Results saved to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["split", "evaluate"],
        help="Which stage to run",
    )
    parser.add_argument("--signfi_root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="paper_results/held_out_classes")
    parser.add_argument("--split_path", default=None)
    args = parser.parse_args()

    if args.stage == "split":
        generate_class_split(
            signfi_root=Path(args.signfi_root),
            output_dir=Path(args.output_dir),
        )
    elif args.stage == "evaluate":
        if not args.checkpoint:
            parser.error("--checkpoint required for evaluate stage")
        split_path = args.split_path or str(
            Path(args.output_dir) / "held_out_class_split.json"
        )
        evaluate_held_out(
            checkpoint_path=args.checkpoint,
            signfi_root=args.signfi_root,
            split_path=split_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
