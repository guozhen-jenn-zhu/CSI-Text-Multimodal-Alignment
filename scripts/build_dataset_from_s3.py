#!/usr/bin/env python
"""Build the multimodal CSI-Text dataset from S3 without local bulk storage.

Reads source CSI data from s3://rnd-sagemaker/Data/fm_downstream/,
generates text pair registries locally (only metadata is downloaded),
then copies CSI files server-side within S3 and uploads the registries
to s3://rnd-sagemaker/Multimodal_CSI_Text/Dataset/.

Registry output uses schema v2 with task-local class IDs and leverages
each task's ``label_mapping.json`` for label normalisation.

The resulting dataset layout at the destination is self-contained:

    s3://rnd-sagemaker/Multimodal_CSI_Text/Dataset/
    ├── csibench/
    │   ├── fall_detection/
    │   │   ├── pair_registry.json
    │   │   ├── metadata/  splits/  sub_Human/...
    │   ├── motion_source/  ...
    │   └── multitask/
    │       ├── sub_Human/...  (shared H5s, copied ONCE)
    │       ├── har/  pair_registry.json  metadata/  splits/
    │       ├── proximity/  ...
    │       └── user_id/  ...
    ├── xrf55/
    │   └── action_recognition/...
    └── signfi/
        └── sign_recognition/...

Usage:
    # Text generation only (very fast, no CSI copy)
    python scripts/build_dataset_from_s3.py --skip_copy

    # Full dataset build (text gen + server-side S3 copy of H5 files)
    python scripts/build_dataset_from_s3.py

    # Single dataset
    python scripts/build_dataset_from_s3.py --datasets csibench

    # Specific tasks
    python scripts/build_dataset_from_s3.py --datasets csibench --tasks har proximity

    # With LLM paraphrasing
    python scripts/build_dataset_from_s3.py --datasets csibench --use_llm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import yaml

logger = logging.getLogger(__name__)

BUCKET = "rnd-sagemaker"
SOURCE_PREFIX = "Data/fm_downstream"
DEST_PREFIX = "Multimodal_CSI_Text/Dataset"

DATASET_CONFIGS = {
    "csibench": "configs/dataset/csibench.yaml",
    "xrf55": "configs/dataset/xrf55.yaml",
    "signfi": "configs/dataset/signfi.yaml",
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def s3_list_keys(prefix: str) -> list[str]:
    """List all object keys under *prefix* in the bucket."""
    s3 = _get_s3()
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def s3_download(s3_key: str, local_path: str | Path) -> None:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _get_s3().download_file(BUCKET, s3_key, str(local_path))


def s3_upload(local_path: str | Path, s3_key: str) -> None:
    _get_s3().upload_file(str(local_path), BUCKET, s3_key)
    logger.info("Uploaded -> s3://%s/%s", BUCKET, s3_key)


def s3_copy(source_key: str, dest_key: str) -> None:
    """Server-side copy within the same bucket (no local traffic)."""
    _get_s3().copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": source_key},
        Key=dest_key,
    )


# ---------------------------------------------------------------------------
# Download only lightweight metadata
# ---------------------------------------------------------------------------

def download_metadata(s3_task_prefix: str, local_task_root: Path) -> None:
    """Download metadata CSV, label_mapping, and split JSONs for a task."""
    total = 0
    for subdir in ("metadata", "splits"):
        keys = s3_list_keys(f"{s3_task_prefix}/{subdir}/")
        for key in keys:
            rel = key[len(s3_task_prefix) + 1:]
            s3_download(key, local_task_root / rel)
        total += len(keys)
    logger.info("Downloaded %d metadata/split files from %s", total, s3_task_prefix)


# ---------------------------------------------------------------------------
# Server-side S3 copy
# ---------------------------------------------------------------------------

def copy_s3_prefix(
    source_prefix: str,
    dest_prefix: str,
    max_workers: int = 32,
) -> int:
    """Copy all objects under *source_prefix* to *dest_prefix* within S3.

    Returns the number of files copied.
    """
    all_keys = s3_list_keys(source_prefix)
    if not all_keys:
        logger.warning("No objects found under %s — skipping copy", source_prefix)
        return 0

    logger.info(
        "Copying %d files: s3://%s/%s -> s3://%s/%s",
        len(all_keys), BUCKET, source_prefix, BUCKET, dest_prefix,
    )

    def _copy_one(key: str) -> str:
        rel = key[len(source_prefix):]
        dest_key = f"{dest_prefix}{rel}"
        s3_copy(key, dest_key)
        return dest_key

    copied = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_copy_one, k): k for k in all_keys}
        for future in as_completed(futures):
            src = futures[future]
            try:
                future.result()
                copied += 1
            except Exception as exc:
                logger.error("Failed to copy %s: %s", src, exc)
                failed += 1
            if copied % 1000 == 0 and copied > 0:
                logger.info("  copied %d / %d files ...", copied, len(all_keys))

    logger.info("Copy complete: %d copied, %d failed", copied, failed)
    return copied


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def load_label_mapping(local_task_root: Path) -> dict:
    """Load label_mapping.json from a task's metadata directory."""
    lm_path = local_task_root / "metadata" / "label_mapping.json"
    if not lm_path.exists():
        raise FileNotFoundError(f"label_mapping.json not found at {lm_path}")
    with open(lm_path) as f:
        return json.load(f)


def build_fine_grained_label_mapping(raw_labels: list[str]) -> dict:
    """Build a 0-indexed mapping from unique raw labels in sorted order.

    Used for motion_source fine-grained override (4-class instead of binary).
    """
    unique = sorted(set(str(l) for l in raw_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique)}
    idx_to_label = {str(idx): label for label, idx in label_to_idx.items()}
    return {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "num_classes": len(unique),
    }


def normalize_csibench_metadata(meta: dict) -> dict | None:
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


# ---------------------------------------------------------------------------
# Text pair registry generation
# ---------------------------------------------------------------------------

def generate_pair_registry(
    dataset_name: str,
    task_name: str,
    local_task_root: Path,
    h5_key: str,
    use_llm: bool,
    llm_model: str,
    num_paraphrases: int,
    fine_grained_labels: bool = False,
) -> dict:
    """Generate a pair registry using only metadata (no H5 data needed).

    Returns the v2 registry dict.
    """
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from data.metadata_loader import MetadataLoader
    from data.text_generation.generate_texts import TextGenerator

    loader = MetadataLoader(task_root=str(local_task_root), h5_key=h5_key)
    all_ids = loader.get_all_sample_ids()

    label_mapping = load_label_mapping(local_task_root)
    label_to_idx = label_mapping["label_to_idx"]

    if fine_grained_labels:
        raw_labels = [str(loader.get_label(sid)) for sid in all_ids]
        label_mapping = build_fine_grained_label_mapping(raw_labels)
        label_to_idx = label_mapping["label_to_idx"]
        logger.info(
            "Using fine-grained labels (%d classes): %s",
            label_mapping["num_classes"],
            label_to_idx,
        )

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
            normed = normalize_csibench_metadata(meta)
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
            "nonhuman": [
                idx for label, idx in label_to_idx.items()
                if label.lower() != "human"
            ],
            "human": [
                idx for label, idx in label_to_idx.items()
                if label.lower() == "human"
            ],
        }

    if dataset_name == "csibench" and task_name == "user_id":
        iid_users = {"U01", "U03", "U04", "U05", "U06"}
        ood_holdout = {"U02"}
        new_users = {"U22"}
        registry["benchmark_class_groups"] = {
            "iid_5class": sorted(
                idx for label, idx in label_to_idx.items() if label in iid_users
            ),
            "ood_holdout": sorted(
                idx for label, idx in label_to_idx.items() if label in ood_holdout
            ),
            "new_user": sorted(
                idx for label, idx in label_to_idx.items() if label in new_users
            ),
            "full_7class": sorted(label_to_idx.values()),
        }

    logger.info(
        "Generated registry for %s/%s: %d samples, %d unique texts, %d total pairs",
        dataset_name,
        task_name,
        len(registry["sample_to_texts"]),
        len(registry["texts"]),
        sum(len(v) for v in registry["sample_to_texts"].values()),
    )
    return registry


# ---------------------------------------------------------------------------
# Process standalone task
# ---------------------------------------------------------------------------

def process_standalone_task(
    dataset_name: str,
    task_name: str,
    task_cfg: dict,
    source_prefix_ds: str,
    args: argparse.Namespace,
) -> None:
    """Process a standalone task: generate texts, optionally copy data."""
    task_folder = task_cfg.get("folder", task_name)
    h5_key = task_cfg.get("h5_key", "CSI_amps")
    fine_grained = task_cfg.get("fine_grained_labels", False)
    if source_prefix_ds:
        s3_task_prefix = f"{SOURCE_PREFIX}/{source_prefix_ds}/{task_folder}"
    else:
        s3_task_prefix = f"{SOURCE_PREFIX}/{task_folder}"

    logger.info("=" * 60)
    logger.info("Processing %s / %s  (source: %s)", dataset_name, task_name, s3_task_prefix)
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory(prefix=f"csi_{dataset_name}_{task_name}_") as tmpdir:
        local_task_root = Path(tmpdir)
        download_metadata(s3_task_prefix, local_task_root)

        registry = generate_pair_registry(
            dataset_name=dataset_name,
            task_name=task_name,
            local_task_root=local_task_root,
            h5_key=h5_key,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            num_paraphrases=args.num_paraphrases,
            fine_grained_labels=fine_grained,
        )

        reg_local = local_task_root / "pair_registry.json"
        with open(reg_local, "w") as f:
            json.dump(registry, f, indent=2)

        reg_s3_key = f"{DEST_PREFIX}/{dataset_name}/{task_name}/pair_registry.json"
        s3_upload(reg_local, reg_s3_key)

    if not args.skip_copy:
        src_pfx = f"{SOURCE_PREFIX}/{source_prefix_ds}/{task_folder}/" if source_prefix_ds else f"{SOURCE_PREFIX}/{task_folder}/"
        copy_s3_prefix(
            source_prefix=src_pfx,
            dest_prefix=f"{DEST_PREFIX}/{dataset_name}/{task_name}/",
            max_workers=args.copy_workers,
        )
    else:
        logger.info("Skipping S3 data copy (--skip_copy)")


# ---------------------------------------------------------------------------
# Process multitask dataset
# ---------------------------------------------------------------------------

def process_multitask(
    dataset_name: str,
    mt_cfg: dict,
    source_prefix_ds: str,
    args: argparse.Namespace,
) -> int:
    """Process the Multitask dataset: shared H5s + per-task registries.

    Returns number of tasks processed.
    """
    mt_folder = mt_cfg["folder"]
    h5_key = mt_cfg.get("h5_key", "CSI_amps")
    h5_subdir = mt_cfg.get("h5_subdir", "sub_Human")
    sub_tasks = mt_cfg.get("tasks", {})

    if source_prefix_ds:
        s3_mt_prefix = f"{SOURCE_PREFIX}/{source_prefix_ds}/{mt_folder}"
    else:
        s3_mt_prefix = f"{SOURCE_PREFIX}/{mt_folder}"

    count = 0
    for sub_task_name, sub_cfg in sub_tasks.items():
        if args.tasks and sub_task_name not in args.tasks:
            logger.info("Skipping multitask/%s (not in --tasks filter)", sub_task_name)
            continue

        meta_folder = sub_cfg.get("metadata_folder", sub_task_name)
        s3_task_prefix = f"{s3_mt_prefix}/{meta_folder}"

        logger.info("=" * 60)
        logger.info("Processing %s / multitask / %s  (source: %s)", dataset_name, sub_task_name, s3_task_prefix)
        logger.info("=" * 60)

        with tempfile.TemporaryDirectory(prefix=f"csi_{dataset_name}_mt_{sub_task_name}_") as tmpdir:
            local_task_root = Path(tmpdir)
            download_metadata(s3_task_prefix, local_task_root)

            registry = generate_pair_registry(
                dataset_name=dataset_name,
                task_name=sub_task_name,
                local_task_root=local_task_root,
                h5_key=h5_key,
                use_llm=args.use_llm,
                llm_model=args.llm_model,
                num_paraphrases=args.num_paraphrases,
            )

            reg_local = local_task_root / "pair_registry.json"
            with open(reg_local, "w") as f:
                json.dump(registry, f, indent=2)

            reg_s3_key = f"{DEST_PREFIX}/{dataset_name}/multitask/{sub_task_name}/pair_registry.json"
            s3_upload(reg_local, reg_s3_key)

        if not args.skip_copy:
            copy_s3_prefix(
                source_prefix=f"{s3_task_prefix}/metadata/",
                dest_prefix=f"{DEST_PREFIX}/{dataset_name}/multitask/{sub_task_name}/metadata/",
                max_workers=args.copy_workers,
            )
            copy_s3_prefix(
                source_prefix=f"{s3_task_prefix}/splits/",
                dest_prefix=f"{DEST_PREFIX}/{dataset_name}/multitask/{sub_task_name}/splits/",
                max_workers=args.copy_workers,
            )

        count += 1

    if not args.skip_copy:
        any_task_requested = not args.tasks or any(t in args.tasks for t in sub_tasks)
        if any_task_requested:
            logger.info("Copying shared H5 files for multitask/%s (once) ...", h5_subdir)
            copy_s3_prefix(
                source_prefix=f"{s3_mt_prefix}/{h5_subdir}/",
                dest_prefix=f"{DEST_PREFIX}/{dataset_name}/multitask/{h5_subdir}/",
                max_workers=args.copy_workers,
            )

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multimodal CSI-Text dataset from S3 source data",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
        help="Which datasets to process (default: all)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Process only these tasks within each dataset (default: all)",
    )
    parser.add_argument(
        "--skip_copy", action="store_true",
        help="Skip server-side S3 copy of CSI H5 files (text gen only)",
    )
    parser.add_argument(
        "--use_llm", action="store_true",
        help="Enable LLM-based paraphrasing for richer text diversity",
    )
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num_paraphrases", type=int, default=5)
    parser.add_argument(
        "--copy_workers", type=int, default=32,
        help="Number of parallel threads for S3 server-side copy",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    total_tasks = 0

    for dataset_name in args.datasets:
        config_path = DATASET_CONFIGS.get(dataset_name)
        if not config_path or not Path(config_path).is_file():
            logger.warning("Config not found: %s — skipping %s", config_path, dataset_name)
            continue

        with open(config_path) as f:
            ds_cfg = yaml.safe_load(f)

        source_prefix_ds = ds_cfg.get("source_prefix", dataset_name)

        standalone_tasks = ds_cfg.get("tasks", {})
        if isinstance(standalone_tasks, dict):
            for task_name, task_cfg in standalone_tasks.items():
                if args.tasks and task_name not in args.tasks:
                    logger.info("Skipping %s/%s (not in --tasks filter)", dataset_name, task_name)
                    continue
                process_standalone_task(
                    dataset_name=dataset_name,
                    task_name=task_name,
                    task_cfg=task_cfg,
                    source_prefix_ds=source_prefix_ds,
                    args=args,
                )
                total_tasks += 1

        mt_cfg = ds_cfg.get("multitask")
        if mt_cfg:
            total_tasks += process_multitask(
                dataset_name=dataset_name,
                mt_cfg=mt_cfg,
                source_prefix_ds=source_prefix_ds,
                args=args,
            )

    logger.info("-" * 60)
    logger.info(
        "All done.  %d task(s) processed.  Dataset at s3://%s/%s/",
        total_tasks, BUCKET, DEST_PREFIX,
    )


if __name__ == "__main__":
    main()
