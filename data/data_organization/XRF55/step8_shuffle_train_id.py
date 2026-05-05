"""
Shuffle ``train_id.json`` in place (or to ``--output``) so the training loader
does not iterate samples in a deterministic user/activity/repeat order.

Default: ``<xrf55_root>/splits/train_id.json``, seed 42.

The XRF55 root defaults to ``<repo_root>/data/Dataset/xrf55/action_recognition``;
override with ``--root`` or ``$XRF55_ROOT`` (see ``_paths.py``).
"""

import argparse
import json
import random
from pathlib import Path

from _paths import splits_dir, xrf55_root


def shuffle_train_id(train_id_file, seed=None, output_file=None):
    """
    Shuffle the list of train IDs in train_id.json.

    Args:
        train_id_file: Path to train_id.json
        seed: Optional random seed for reproducibility
        output_file: Optional path to save shuffled result (default: overwrite input)
    """
    train_path = Path(train_id_file)

    if not train_path.exists():
        print(f"Error: File not found: {train_path}")
        return

    print("=" * 60)
    print("Shuffling train_id.json")
    print("=" * 60)
    print(f"Input file: {train_path}")

    # Load IDs
    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = json.load(f)

    n = len(train_ids)
    print(f"Loaded {n} train IDs")

    if seed is not None:
        random.seed(seed)
        print(f"Random seed: {seed}")

    # Shuffle in place
    random.shuffle(train_ids)

    # Output path
    out_path = Path(output_file) if output_file else train_path
    out_path = Path(out_path)

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2, ensure_ascii=False)

    print(f"Saved shuffled IDs to: {out_path}")
    print("Sample (first 5 after shuffle):")
    for i, id_ in enumerate(train_ids[:5], 1):
        print(f"  {i}. {id_}")
    print("=" * 60)


def main(root=None, train_id=None, seed=42, output=None):
    print(f"XRF55 root: {xrf55_root(root)}")
    train_id_path = str(Path(train_id).expanduser().resolve()) if train_id else str(splits_dir(root) / "train_id.json")
    output_path = str(Path(output).expanduser().resolve()) if output else None
    shuffle_train_id(train_id_path, seed=seed, output_file=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle train_id.json so the loader does not see samples in a deterministic order.")
    parser.add_argument("--train-id", default=None, help="Path to train_id.json (default: <xrf55_root>/splits/train_id.json)")
    parser.add_argument("--output", default=None, help="Where to write the shuffled JSON (default: overwrite the input file)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42; pass a negative value for a non-deterministic shuffle)")
    parser.add_argument(
        "--root",
        default=None,
        help="XRF55 root directory (overrides $XRF55_ROOT and the default <repo_root>/data/Dataset/xrf55/action_recognition)",
    )
    args = parser.parse_args()
    seed_value = args.seed if args.seed is not None and args.seed >= 0 else None
    main(root=args.root, train_id=args.train_id, seed=seed_value, output=args.output)
