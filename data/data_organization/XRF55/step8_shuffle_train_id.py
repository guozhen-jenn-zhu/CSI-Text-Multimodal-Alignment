import json
import random
from pathlib import Path

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

if __name__ == "__main__":
    import sys

    # Path to train_id.json
    train_id_path = r"E:\XRF55\XRF55V2\splits\train_id.json"

    # Optional: seed for reproducibility (e.g. 42). Set to None for different shuffle each run.
    random_seed = 42

    # Optional: write to a different file. None = overwrite train_id.json
    output_path = None

    # Allow overrides from command line
    # Usage: python shuffle_train_id.py [train_id.json] [seed]
    if len(sys.argv) > 1:
        train_id_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            random_seed = int(sys.argv[2])
        except ValueError:
            random_seed = None

    shuffle_train_id(train_id_path, seed=random_seed, output_file=output_path)
