#!/usr/bin/env python
"""Build a small sample of CSI-TextBench for NeurIPS dataset review.

Reads from s3://rnd-sagemaker/Multimodal_CSI_Text/Dataset/,
performs class-stratified sampling, filters all metadata artifacts
(CSV, splits, pair_registry), and copies only the sampled H5 files
(server-side within S3) to s3://rnd-sagemaker/Multimodal_CSI_Text/Sample_CSI_TextBench/.

The output preserves the identical directory structure as the full dataset
but is self-contained: every referenced sample ID has a corresponding H5 file.

Usage:
    # Full build (default ~2.8 GB)
    python scripts/build_sample_dataset.py

    # Dry-run (metadata only, skip H5 copy)
    python scripts/build_sample_dataset.py --dry_run

    # Custom rates
    python scripts/build_sample_dataset.py --csibench_rate 0.03 --xrf55_rate 0.02

    # Single dataset
    python scripts/build_sample_dataset.py --datasets xrf55
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

logger = logging.getLogger(__name__)

BUCKET = "rnd-sagemaker"
SRC_PREFIX = "Multimodal_CSI_Text/Dataset"
DEST_PREFIX = "Multimodal_CSI_Text/Sample_CSI_TextBench"

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def s3_download_text(key: str) -> str:
    resp = _get_s3().get_object(Bucket=BUCKET, Key=key)
    return resp["Body"].read().decode("utf-8")


def s3_upload_text(content: str, key: str) -> None:
    _get_s3().put_object(Bucket=BUCKET, Key=key, Body=content.encode("utf-8"))
    logger.info("Uploaded -> s3://%s/%s", BUCKET, key)


def s3_copy(src_key: str, dest_key: str) -> None:
    _get_s3().copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": src_key},
        Key=dest_key,
    )


def s3_list_keys(prefix: str) -> list[str]:
    paginator = _get_s3().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def parse_csv(text: str) -> tuple[list[dict], list[str]]:
    """Parse CSV text, returning (rows, fieldnames)."""
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    fieldnames = reader.fieldnames or []
    return rows, list(fieldnames)


def rows_to_csv(rows: list[dict], fieldnames: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    rows: list[dict],
    rate: float,
    label_col: str = "label",
    seed: int = 42,
    min_per_class: int = 2,
) -> list[dict]:
    """Class-stratified random sampling of metadata rows."""
    rng = random.Random(seed)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row[label_col]].append(row)

    sampled: list[dict] = []
    for cls in sorted(by_class):
        cls_rows = by_class[cls]
        n = max(min_per_class, round(len(cls_rows) * rate))
        n = min(n, len(cls_rows))
        sampled.extend(rng.sample(cls_rows, n))

    return sampled


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_pair_registry(registry_text: str, sampled_ids: set[str]) -> str:
    """Filter pair_registry.json to only include sampled sample IDs.

    Keeps all class_to_texts intact so class-level text descriptions
    remain complete for zero-shot evaluation.
    """
    reg = json.loads(registry_text)

    reg["sample_to_texts"] = {
        k: v for k, v in reg.get("sample_to_texts", {}).items()
        if k in sampled_ids
    }
    reg["sample_to_class"] = {
        k: v for k, v in reg.get("sample_to_class", {}).items()
        if k in sampled_ids
    }

    referenced_text_ids: set[str] = set()
    for text_ids in reg["sample_to_texts"].values():
        referenced_text_ids.update(text_ids)
    for text_ids in reg.get("class_to_texts", {}).values():
        referenced_text_ids.update(text_ids)

    reg["texts"] = {
        k: v for k, v in reg.get("texts", {}).items()
        if k in referenced_text_ids
    }

    return json.dumps(reg, indent=2)


def filter_split_json(split_text: str, sampled_ids: set[str]) -> str:
    ids = json.loads(split_text)
    return json.dumps([sid for sid in ids if sid in sampled_ids], indent=2)


# ---------------------------------------------------------------------------
# H5 copy
# ---------------------------------------------------------------------------

def copy_h5_files(
    pairs: list[tuple[str, str]],
    max_workers: int = 32,
) -> int:
    """Server-side S3 copy of (src_key, dest_key) pairs. Returns count copied."""
    if not pairs:
        return 0

    total = len(pairs)
    logger.info("Copying %d H5 files (server-side) ...", total)

    copied = 0
    failed = 0

    def _do_copy(src: str, dest: str) -> str:
        s3_copy(src, dest)
        return dest

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_do_copy, s, d): s for s, d in pairs}
        for future in as_completed(futures):
            try:
                future.result()
                copied += 1
            except Exception as exc:
                logger.error("Failed to copy %s: %s", futures[future], exc)
                failed += 1
            if copied % 500 == 0 and copied > 0:
                logger.info("  H5 copy progress: %d / %d", copied, total)

    logger.info("H5 copy complete: %d copied, %d failed", copied, failed)
    return copied


# ---------------------------------------------------------------------------
# Standalone task
# ---------------------------------------------------------------------------

def process_standalone_task(
    dataset: str,
    task: str,
    rate: float,
    seed: int,
    max_workers: int,
    dry_run: bool,
) -> set[str]:
    """Sample and copy a standalone task. Returns the set of sampled IDs."""
    logger.info("=" * 60)
    logger.info("Standalone  %s / %s  (rate=%.2f)", dataset, task, rate)
    logger.info("=" * 60)

    src_pfx = f"{SRC_PREFIX}/{dataset}/{task}"
    dst_pfx = f"{DEST_PREFIX}/{dataset}/{task}"

    # --- metadata CSV ---
    csv_text = s3_download_text(f"{src_pfx}/metadata/sample_metadata.csv")
    rows, fieldnames = parse_csv(csv_text)
    logger.info("  Total samples in CSV: %d", len(rows))

    sampled = stratified_sample(rows, rate, seed=seed)
    sampled_ids = {r["id"] for r in sampled}
    logger.info("  Sampled: %d", len(sampled_ids))

    s3_upload_text(
        rows_to_csv(sampled, fieldnames),
        f"{dst_pfx}/metadata/sample_metadata.csv",
    )

    # --- label_mapping + other small metadata JSONs ---
    for key in s3_list_keys(f"{src_pfx}/metadata/"):
        if key.endswith(".json"):
            rel = key[len(src_pfx):]
            s3_copy(key, f"{dst_pfx}{rel}")
            logger.info("  Copied %s", key.split("/")[-1])

    # --- splits ---
    for key in s3_list_keys(f"{src_pfx}/splits/"):
        split_text = s3_download_text(key)
        filtered = filter_split_json(split_text, sampled_ids)
        rel = key[len(src_pfx):]
        s3_upload_text(filtered, f"{dst_pfx}{rel}")

    # --- pair_registry.json ---
    reg_key = f"{src_pfx}/pair_registry.json"
    try:
        reg_text = s3_download_text(reg_key)
        s3_upload_text(
            filter_pair_registry(reg_text, sampled_ids),
            f"{dst_pfx}/pair_registry.json",
        )
    except Exception as exc:
        logger.warning("  pair_registry.json skipped: %s", exc)

    # --- H5 files ---
    if not dry_run:
        h5_pairs: list[tuple[str, str]] = []
        for row in sampled:
            stripped = row["file_path"].lstrip("./")
            h5_pairs.append((
                f"{src_pfx}/{stripped}",
                f"{dst_pfx}/{stripped}",
            ))
        copy_h5_files(h5_pairs, max_workers)

    return sampled_ids


# ---------------------------------------------------------------------------
# Multitask
# ---------------------------------------------------------------------------

def process_multitask(
    dataset: str,
    subtasks: list[str],
    rate: float,
    seed: int,
    max_workers: int,
    dry_run: bool,
) -> None:
    """Sample multitask subtasks that share a common H5 pool."""
    mt_src = f"{SRC_PREFIX}/{dataset}/multitask"
    mt_dst = f"{DEST_PREFIX}/{dataset}/multitask"

    # Accumulate (stripped_path -> (src_key, dest_key)) across subtasks
    # so shared H5 files are copied only once.
    shared_h5: dict[str, tuple[str, str]] = {}

    for subtask in subtasks:
        logger.info("=" * 60)
        logger.info("Multitask   %s / %s  (rate=%.2f)", dataset, subtask, rate)
        logger.info("=" * 60)

        sub_src = f"{mt_src}/{subtask}"
        sub_dst = f"{mt_dst}/{subtask}"

        # --- metadata CSV ---
        csv_text = s3_download_text(f"{sub_src}/metadata/sample_metadata.csv")
        rows, fieldnames = parse_csv(csv_text)
        logger.info("  Total samples in CSV: %d", len(rows))

        sampled = stratified_sample(rows, rate, seed=seed)
        sampled_ids = {r["id"] for r in sampled}
        logger.info("  Sampled: %d", len(sampled_ids))

        s3_upload_text(
            rows_to_csv(sampled, fieldnames),
            f"{sub_dst}/metadata/sample_metadata.csv",
        )

        # --- label_mapping JSONs ---
        for key in s3_list_keys(f"{sub_src}/metadata/"):
            if key.endswith(".json"):
                rel = key[len(sub_src):]
                s3_copy(key, f"{sub_dst}{rel}")

        # --- splits ---
        for key in s3_list_keys(f"{sub_src}/splits/"):
            split_text = s3_download_text(key)
            filtered = filter_split_json(split_text, sampled_ids)
            rel = key[len(sub_src):]
            s3_upload_text(filtered, f"{sub_dst}{rel}")

        # --- pair_registry.json ---
        reg_key = f"{sub_src}/pair_registry.json"
        try:
            reg_text = s3_download_text(reg_key)
            s3_upload_text(
                filter_pair_registry(reg_text, sampled_ids),
                f"{sub_dst}/pair_registry.json",
            )
        except Exception as exc:
            logger.warning("  pair_registry.json skipped: %s", exc)

        # --- collect H5 paths (shared pool) ---
        for row in sampled:
            # file_path like "../../sub_Human/user_.../session_...h5"
            # lstrip("./") strips all leading dots+slashes -> "sub_Human/..."
            stripped = row["file_path"].lstrip("./")
            if stripped not in shared_h5:
                shared_h5[stripped] = (
                    f"{mt_src}/{stripped}",
                    f"{mt_dst}/{stripped}",
                )

    # --- copy shared H5 files once ---
    if not dry_run:
        logger.info(
            "Copying %d unique shared multitask H5 files ...",
            len(shared_h5),
        )
        copy_h5_files(list(shared_h5.values()), max_workers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_DEFS = {
    "csibench": {
        "standalone": ["fall_detection", "motion_source"],
        "multitask": ["har", "proximity", "user_id"],
    },
    "xrf55": {
        "standalone": ["action_recognition"],
    },
    "signfi": {
        "standalone": ["sign_recognition"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a <4 GB sample of CSI-TextBench for NeurIPS review",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--csibench_rate", type=float, default=0.05,
                        help="Sampling rate for CSI-Bench tasks (default: 5%%)")
    parser.add_argument("--xrf55_rate", type=float, default=0.03,
                        help="Sampling rate for XRF55 (default: 3%%)")
    parser.add_argument("--signfi_rate", type=float, default=0.50,
                        help="Sampling rate for SignFi (default: 50%%)")
    parser.add_argument("--copy_workers", type=int, default=32,
                        help="Parallel threads for S3 server-side copy")
    parser.add_argument("--dry_run", action="store_true",
                        help="Upload filtered metadata only; skip H5 copy")
    parser.add_argument("--datasets", nargs="+",
                        default=["csibench", "xrf55", "signfi"],
                        choices=["csibench", "xrf55", "signfi"],
                        help="Which datasets to include (default: all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    rates = {
        "csibench": args.csibench_rate,
        "xrf55": args.xrf55_rate,
        "signfi": args.signfi_rate,
    }

    for dataset in args.datasets:
        rate = rates[dataset]
        defs = TASK_DEFS[dataset]

        for task in defs.get("standalone", []):
            process_standalone_task(
                dataset, task, rate, args.seed, args.copy_workers, args.dry_run,
            )

        mt_subtasks = defs.get("multitask")
        if mt_subtasks:
            process_multitask(
                dataset, mt_subtasks, rate, args.seed,
                args.copy_workers, args.dry_run,
            )

    logger.info("-" * 60)
    logger.info(
        "Done!  Sample dataset at s3://%s/%s/",
        BUCKET, DEST_PREFIX,
    )


if __name__ == "__main__":
    main()
