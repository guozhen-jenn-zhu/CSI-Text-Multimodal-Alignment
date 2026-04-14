#!/usr/bin/env python3
"""
Anonymize S3 file paths for FallDetection/sub_Human dataset.

Replaces personal names in user_ and env_ folder segments with anonymous IDs
based on the mapping from "Benchmark Paper Stats - Summary.pdf".

Usage:
    python scripts/anonymize_s3_falldetection.py --dry-run   # Preview changes
    python scripts/anonymize_s3_falldetection.py              # Execute renames
"""

import argparse
import boto3
from collections import defaultdict

BUCKET = "rnd-sagemaker"
PREFIX = "Data/fm_downstream/FallDetection/sub_Human/"

USER_MAP = {
    "user_Avinash":   "user_U07",
    "user_Brendon":   "user_U08",
    "user_Brian":     "user_U09",
    "user_Dan":       "user_U10",
    "user_Deepika":   "user_U11",
    "user_Huilang":   "user_U12",
    "user_Lin":       "user_U13",
    "user_Pranav":    "user_U14",
    "user_Quoc":      "user_U15",
    "user_Sean":      "user_U16",
    "user_Tim":       "user_U17",
    "user_Vamshika":  "user_U18",
    "user_WeiHsiang": "user_U06",
    "user_Wenchao":   "user_U19",
    "user_Yi":        "user_U20",
    "user_Yuqian":    "user_U21",
    "user_Zahid":     "user_U22",
    "user_Fall":      "user_U11",
}

ENV_MAP = {
    "env_Airbnb":     "env_E24",
    "env_DeepikaApt": "env_E21",
    "env_OWoffice":   "env_E23",
    "env_QuocHome":   "env_E26",
    "env_Testhouse":  "env_E22",
    "env_Testhouse2": "env_E25",
}


def build_new_key(old_key: str) -> str:
    """Replace user and env path segments with anonymized IDs."""
    parts = old_key.split("/")
    new_parts = []
    for part in parts:
        if part in USER_MAP:
            new_parts.append(USER_MAP[part])
        elif part in ENV_MAP:
            new_parts.append(ENV_MAP[part])
        else:
            new_parts.append(part)
    return "/".join(new_parts)


def main():
    parser = argparse.ArgumentParser(description="Anonymize S3 FallDetection paths")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print planned renames without executing")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    all_objects = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            all_objects.append(obj["Key"])

    print(f"Found {len(all_objects)} objects under s3://{BUCKET}/{PREFIX}\n")

    rename_plan = []
    unmapped_users = set()
    unmapped_envs = set()

    for old_key in all_objects:
        new_key = build_new_key(old_key)
        if new_key != old_key:
            rename_plan.append((old_key, new_key))

        parts = old_key.split("/")
        for part in parts:
            if part.startswith("user_") and part not in USER_MAP:
                unmapped_users.add(part)
            if part.startswith("env_") and part not in ENV_MAP:
                unmapped_envs.add(part)

    if unmapped_users:
        print(f"WARNING: Unmapped users found: {unmapped_users}")
    if unmapped_envs:
        print(f"WARNING: Unmapped envs found: {unmapped_envs}")

    unchanged = len(all_objects) - len(rename_plan)
    print(f"Files to rename: {len(rename_plan)}")
    print(f"Files unchanged: {unchanged}\n")

    summary = defaultdict(int)
    for old_key, new_key in rename_plan:
        old_parts = old_key.split("/")
        new_parts = new_key.split("/")
        for o, n in zip(old_parts, new_parts):
            if o != n:
                summary[(o, n)] += 1

    print("=== Mapping Summary ===")
    for (old_seg, new_seg), count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {old_seg:25s} -> {new_seg:15s}  ({count} files)")
    print()

    if args.dry_run:
        print("=== Sample Renames (first 10) ===")
        for old_key, new_key in rename_plan[:10]:
            print(f"  {old_key}")
            print(f"  -> {new_key}\n")
        print("DRY RUN complete. No changes made.")
        return

    print("=== Executing Renames ===")
    success = 0
    errors = 0
    for i, (old_key, new_key) in enumerate(rename_plan):
        try:
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={"Bucket": BUCKET, "Key": old_key},
                Key=new_key,
            )
            s3.delete_object(Bucket=BUCKET, Key=old_key)
            success += 1
        except Exception as e:
            print(f"  ERROR on {old_key}: {e}")
            errors += 1

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(rename_plan)} "
                  f"(success={success}, errors={errors})")

    print(f"\nDone! success={success}, errors={errors}")


if __name__ == "__main__":
    main()
