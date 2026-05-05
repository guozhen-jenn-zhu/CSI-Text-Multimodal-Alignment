"""
Reorganize the upstream XRF55 release into the canonical user/activity/env layout.

Source layout (upstream):
    <source>/Scene<N>/WiFi/<user>_<act>_<repeat>.npy

Target layout:
    <target>/user_<XX>/act_<XX>/env_<N>/<user>_<act>_<repeat>.npy

Defaults:
    --source : <xrf55_root>/raw
    --target : <xrf55_root>/organized

The XRF55 root defaults to ``<repo_root>/data/Dataset/xrf55/action_recognition``;
override with ``--root`` or ``$XRF55_ROOT`` (see ``_paths.py``).
"""

import argparse
import os
import shutil
from pathlib import Path
import re

from _paths import organized_dir, raw_dir, xrf55_root

# Pattern to match filename: user_activity_repeat.npy
filename_pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.npy')


def main(root=None, source=None, target=None):
    print(f"XRF55 root: {xrf55_root(root)}")
    # Source and target directories
    source_dir = Path(source).expanduser().resolve() if source else raw_dir(root)
    target_base = Path(target).expanduser().resolve() if target else organized_dir(root)
    print(f"Source: {source_dir}")
    print(f"Target: {target_base}")

    if not source_dir.is_dir():
        raise SystemExit(f"Source folder not found: {source_dir}")
    target_base.mkdir(parents=True, exist_ok=True)

    # Process all Scene folders
    for scene_folder in source_dir.iterdir():
        if not scene_folder.is_dir() or not scene_folder.name.startswith('Scene'):
            continue

        # Extract environment number from Scene folder (e.g., Scene1 -> 1)
        env_match = re.search(r'Scene(\d+)', scene_folder.name)
        if not env_match:
            continue

        env_num = env_match.group(1)
        wifi_folder = scene_folder / "WiFi"

        if not wifi_folder.exists():
            continue

        print(f"Processing {scene_folder.name} (Environment {env_num})...")

        # Process all .npy files in the WiFi folder
        npy_files = list(wifi_folder.glob("*.npy"))
        total_files = len(npy_files)

        for idx, npy_file in enumerate(npy_files, 1):
            # Parse filename
            match = filename_pattern.match(npy_file.name)
            if not match:
                print(f"Warning: Could not parse filename {npy_file.name}, skipping...")
                continue

            user_num = match.group(1)
            activity_num = match.group(2)
            repeat_num = match.group(3)

            # Create target directory structure
            target_dir = target_base / f"user_{user_num:0>2}" / f"act_{activity_num:0>2}" / f"env_{env_num}"
            target_dir.mkdir(parents=True, exist_ok=True)

            # Move file to target location
            target_file = target_dir / npy_file.name
            shutil.move(str(npy_file), str(target_file))

            if idx % 1000 == 0:
                print(f"  Processed {idx}/{total_files} files...")

        print(f"Completed {scene_folder.name}: {total_files} files moved\n")

    print("Dataset reorganization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize raw XRF55 .npy files into user/activity/env subfolders.")
    parser.add_argument("--source", default=None, help="Source dir holding upstream Scene<N>/WiFi/*.npy (default: <xrf55_root>/raw)")
    parser.add_argument("--target", default=None, help="Target dir for the user/act/env layout (default: <xrf55_root>/organized)")
    parser.add_argument(
        "--root",
        default=None,
        help="XRF55 root directory (overrides $XRF55_ROOT and the default <repo_root>/data/Dataset/xrf55/action_recognition)",
    )
    args = parser.parse_args()
    main(root=args.root, source=args.source, target=args.target)
