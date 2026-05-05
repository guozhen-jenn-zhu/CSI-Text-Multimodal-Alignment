"""
Move .h5 files into subfolders by sign label for all dataset folders under sub_Human.

Example:
  <signfi_root>/sub_Human/dataset_home_276/csid_dataset_home_276_1_277.h5
  -> <signfi_root>/sub_Human/dataset_home_276/sign001/csid_dataset_home_276_1_277.h5

  <signfi_root>/sub_Human/dataset_home_276/csid_dataset_home_276_18_570.h5
  -> <signfi_root>/sub_Human/dataset_home_276/sign018/csid_dataset_home_276_18_570.h5

Filename format: {prefix}_{basename}_{label}_{sample}.h5  ->  move to sign{label:03d}/

The SignFi root defaults to ``<repo_root>/data/Dataset/signfi/sign_recognition``;
override with ``--root`` or ``$SIGNFI_ROOT`` (see ``_paths.py``).
"""

import argparse
import os
import shutil

from _paths import signfi_root, sub_human_dir


def parse_label_from_h5_filename(filename):
    """
    From {prefix}_{basename}_{label}_{sample}.h5 return label as int, or None if unparseable.
    """
    if not filename.endswith(".h5"):
        return None
    stem = filename[:-4]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def organize_by_label(dataset_dir):
    """Move .h5 files in dataset_dir into sign{label:03d}/ subfolders. Skip files already in sign***."""
    moved = 0
    errors = []
    for name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.endswith(".h5"):
            continue
        lab = parse_label_from_h5_filename(name)
        if lab is None:
            errors.append((name, "could not parse label"))
            continue
        subfolder = f"sign{lab:03d}"
        dest_dir = os.path.join(dataset_dir, subfolder)
        dest_path = os.path.join(dest_dir, name)
        if path == dest_path:
            continue
        try:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(path, dest_path)
            moved += 1
        except Exception as e:
            errors.append((name, str(e)))
    return moved, errors


def main(root=None):
    sub_human = sub_human_dir(root)
    print(f"SignFi root: {signfi_root(root)}")
    if not sub_human.is_dir():
        raise SystemExit(f"sub_Human folder not found: {sub_human}")

    dataset_dirs = [
        d for d in os.listdir(sub_human)
        if (sub_human / d).is_dir() and not d.startswith("sign")
    ]
    if not dataset_dirs:
        print(f"No dataset subfolders found under {sub_human}")
        return

    total_moved = 0
    for d in sorted(dataset_dirs):
        n, errs = organize_by_label(str(sub_human / d))
        total_moved += n
        if n or errs:
            status = f"moved {n} files" if n else "no files to move"
            if errs:
                status += f", {len(errs)} issues"
            print(f"  {d}: {status}")
            for fname, msg in errs[:5]:
                print(f"    - {fname}: {msg}")
            if len(errs) > 5:
                print(f"    ... and {len(errs) - 5} more")

    print(f"Done. Total files moved: {total_moved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize SignFi .h5 files into sign{NNN}/ folders by label.")
    parser.add_argument(
        "--root",
        default=None,
        help="SignFi root directory (overrides $SIGNFI_ROOT and the default <repo_root>/data/Dataset/signfi/sign_recognition)",
    )
    args = parser.parse_args()
    main(root=args.root)
