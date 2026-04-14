"""Generate text pair registries for all datasets and tasks.

This script is meant for running **locally on SageMaker** (or any machine
where the source data is mounted under a ``data_root`` directory).  It reads
``label_mapping.json`` from each task's metadata folder and passes the
``task_name`` to ``TextGenerator`` so that task-local class IDs are used.

For building the dataset from a laptop via S3, use ``build_dataset_from_s3.py``
instead.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "csibench": "configs/dataset/csibench.yaml",
    "xrf55": "configs/dataset/xrf55.yaml",
    "signfi": "configs/dataset/signfi.yaml",
}


def _load_label_mapping(task_root: Path, fine_grained: bool = False) -> dict[str, int]:
    """Load label_mapping.json and return ``{raw_label_str: class_id}``.

    When *fine_grained* is True (motion_source 4-class override) we build
    the mapping from the raw labels in sample_metadata instead.
    """
    lm_path = task_root / "metadata" / "label_mapping.json"
    if not lm_path.exists():
        raise FileNotFoundError(f"label_mapping.json not found at {lm_path}")
    with open(lm_path) as f:
        lm = json.load(f)

    if fine_grained:
        return None  # caller will build from raw labels
    return {str(k): int(v) for k, v in lm["label_to_idx"].items()}


def _normalize_csibench_metadata(meta: dict) -> dict | None:
    """Convert raw CSV metadata to template-compatible keys for Level 2 texts."""
    user_raw = str(meta.get("user", ""))
    act_raw = str(meta.get("activity", ""))
    dist_raw = meta.get("distance")

    if not user_raw or not act_raw or dist_raw is None:
        return None

    if user_raw.startswith("U") and user_raw[1:].isdigit():
        user_key = f"user_{int(user_raw[1:])}"
    else:
        user_key = user_raw

    act_map = {"seated-breathing": "seated_breathing", "wavinghand": "waving"}
    activity = act_map.get(act_raw, act_raw)
    proximity = f"{float(dist_raw)}m"

    return {"user_id": user_key, "activity": activity, "proximity": proximity}


def generate_for_task(
    dataset_name: str,
    task_name: str,
    task_root: str,
    h5_key: str,
    output_dir: str,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    num_paraphrases: int = 5,
    fine_grained_labels: bool = False,
) -> Path:
    """Generate a pair registry for a single task."""
    from data.metadata_loader import MetadataLoader
    from data.text_generation.generate_texts import TextGenerator

    task_root_path = Path(task_root)
    loader = MetadataLoader(task_root=task_root, h5_key=h5_key)
    all_ids = loader.get_all_sample_ids()

    label_to_idx = _load_label_mapping(task_root_path, fine_grained=fine_grained_labels)

    if fine_grained_labels or label_to_idx is None:
        raw_labels = sorted(set(str(loader.get_label(sid)) for sid in all_ids))
        label_to_idx = {label: idx for idx, label in enumerate(raw_labels)}
        logger.info("Fine-grained labels (%d classes): %s", len(label_to_idx), label_to_idx)

    sample_labels: dict[str, int] = {}
    for sid in all_ids:
        raw = str(loader.get_label(sid))
        if raw in label_to_idx:
            sample_labels[sid] = label_to_idx[raw]
        else:
            logger.warning("Unmapped label %r for sample %s — skipping", raw, sid)
    all_ids = [sid for sid in all_ids if sid in sample_labels]

    sample_metadata = None
    if dataset_name == "csibench":
        sample_metadata = {}
        for sid in all_ids:
            meta = loader.get_metadata(sid)
            normed = _normalize_csibench_metadata(meta)
            if normed:
                sample_metadata[sid] = normed

    gen = TextGenerator(
        dataset_name=dataset_name,
        task_name=task_name,
        use_llm_paraphrase=use_llm,
        llm_model=llm_model,
        num_paraphrases=num_paraphrases,
    )

    registry = gen.build_pair_registry(
        sample_ids=all_ids,
        sample_labels=sample_labels,
        sample_metadata=sample_metadata if sample_metadata else None,
    )

    if fine_grained_labels and dataset_name == "csibench" and task_name == "motion_source":
        registry["benchmark_class_groups"] = {
            "nonhuman": [idx for label, idx in label_to_idx.items() if label.lower() != "human"],
            "human": [idx for label, idx in label_to_idx.items() if label.lower() == "human"],
        }

    if dataset_name == "csibench" and task_name == "user_id":
        iid_users = {"U01", "U03", "U04", "U05", "U06"}
        ood_holdout = {"U02"}
        new_users = {"U22"}
        registry["benchmark_class_groups"] = {
            "iid_5class": sorted(idx for label, idx in label_to_idx.items() if label in iid_users),
            "ood_holdout": sorted(idx for label, idx in label_to_idx.items() if label in ood_holdout),
            "new_user": sorted(idx for label, idx in label_to_idx.items() if label in new_users),
            "full_7class": sorted(label_to_idx.values()),
        }

    out_path = Path(output_dir) / dataset_name / task_name / "pair_registry.json"
    gen.save_registry(registry, str(out_path))

    logger.info("Generated: %s (%d samples)", out_path, len(all_ids))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text registries for all datasets")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Process only these tasks (default: all)",
    )
    parser.add_argument("--data_root", type=str, default="/data/fm_downstream",
                        help="Root data directory (SageMaker mount)")
    parser.add_argument("--output_dir", type=str, default="data/generated_texts")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num_paraphrases", type=int, default=5)
    parser.add_argument("--s3_sync", action="store_true", help="Upload results to S3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    generated = []

    for dataset_name in args.datasets:
        config_path = DATASET_CONFIGS[dataset_name]
        if not Path(config_path).is_file():
            logger.warning("Config not found: %s -- skipping %s", config_path, dataset_name)
            continue

        with open(config_path) as f:
            ds_cfg = yaml.safe_load(f)

        source_prefix = ds_cfg.get("source_prefix", dataset_name)

        # --- Standalone tasks ---
        standalone_tasks = ds_cfg.get("tasks", {})
        if isinstance(standalone_tasks, dict):
            for task_name, task_cfg in standalone_tasks.items():
                if args.tasks and task_name not in args.tasks:
                    logger.info("Skipping %s/%s (not in --tasks filter)", dataset_name, task_name)
                    continue

                folder = task_cfg.get("folder", task_name)
                h5_key = task_cfg.get("h5_key", "CSI_amps")
                fine_grained = task_cfg.get("fine_grained_labels", False)
                task_root = str(Path(args.data_root) / source_prefix / folder) if source_prefix else str(Path(args.data_root) / folder)

                if not Path(task_root).is_dir():
                    logger.warning("Task root not found: %s -- skipping", task_root)
                    continue

                out = generate_for_task(
                    dataset_name=dataset_name,
                    task_name=task_name,
                    task_root=task_root,
                    h5_key=h5_key,
                    output_dir=args.output_dir,
                    use_llm=args.use_llm,
                    llm_model=args.llm_model,
                    num_paraphrases=args.num_paraphrases,
                    fine_grained_labels=fine_grained,
                )
                generated.append(out)

        # --- Multitask ---
        mt_cfg = ds_cfg.get("multitask")
        if mt_cfg:
            mt_folder = mt_cfg["folder"]
            h5_key = mt_cfg.get("h5_key", "CSI_amps")
            sub_tasks = mt_cfg.get("tasks", {})

            for sub_task_name, sub_cfg in sub_tasks.items():
                if args.tasks and sub_task_name not in args.tasks:
                    logger.info("Skipping multitask/%s (not in --tasks filter)", sub_task_name)
                    continue

                meta_folder = sub_cfg.get("metadata_folder", sub_task_name)
                task_root = str(Path(args.data_root) / source_prefix / mt_folder / meta_folder) if source_prefix else str(Path(args.data_root) / mt_folder / meta_folder)

                if not Path(task_root).is_dir():
                    logger.warning("Task root not found: %s -- skipping", task_root)
                    continue

                out = generate_for_task(
                    dataset_name=dataset_name,
                    task_name=sub_task_name,
                    task_root=task_root,
                    h5_key=h5_key,
                    output_dir=args.output_dir,
                    use_llm=args.use_llm,
                    llm_model=args.llm_model,
                    num_paraphrases=args.num_paraphrases,
                )
                generated.append(out)

    if args.s3_sync and generated:
        try:
            from utils.s3_sync import S3Syncer
            syncer = S3Syncer(bucket="rnd-sagemaker", prefix="Multimodal_CSI_Text")
            syncer.upload_dataset(args.output_dir)
            logger.info("Synced generated texts to S3")
        except Exception as e:
            logger.warning("S3 sync failed: %s", e)

    print(f"\nGenerated {len(generated)} pair registries under {args.output_dir}/")


if __name__ == "__main__":
    main()
