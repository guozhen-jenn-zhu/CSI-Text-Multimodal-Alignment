"""
Generate ``train_id.json`` / ``val_id.json`` / ``test_id.json`` from the XRF55
metadata CSV using the official repeat-based protocol::

    train: repeats 1-12
    val:   repeats 13-14
    test:  repeats 15-20

Defaults:
    --metadata : <xrf55_root>/metadata/sample_metadata.csv
    --output   : <xrf55_root>/splits

The XRF55 root defaults to ``<repo_root>/data/Dataset/xrf55/action_recognition``;
override with ``--root`` or ``$XRF55_ROOT`` (see ``_paths.py``).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from _paths import metadata_csv_path, splits_dir, xrf55_root

def generate_splits(metadata_file, output_dir):
    """
    Generate train, validation, and test split JSON files based on metadata.
    
    Split logic:
    - Training: repeats 1-12 (for each user, activity, environment)
    - Validation: repeats 13-14
    - Testing: repeats 15-20
    
    Args:
        metadata_file: Path to the metadata CSV file
        output_dir: Directory to save the split JSON files
    """
    metadata_path = Path(metadata_file)
    output_path = Path(output_dir)
    
    print("=" * 60)
    print("Generating Data Splits")
    print("=" * 60)
    print(f"Metadata file: {metadata_path}")
    print(f"Output directory: {output_path}")
    print()
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read metadata
    print("Reading metadata...")
    train_ids = []
    val_ids = []
    test_ids = []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row['id']
            repeat = int(row['repeat'])
            user = row['user']
            activity = int(row['activity'])
            environment = int(row['environment'])
            
            # Split based on repeat number
            if 1 <= repeat <= 12:
                train_ids.append(file_id)
            elif 13 <= repeat <= 14:
                val_ids.append(file_id)
            elif 15 <= repeat <= 20:
                test_ids.append(file_id)
            else:
                print(f"Warning: Repeat {repeat} for {file_id} is outside expected range (1-20)")
    
    # Sort IDs for consistency
    train_ids.sort()
    val_ids.sort()
    test_ids.sort()
    
    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    print(f"Testing samples: {len(test_ids)}")
    print(f"Total samples: {len(train_ids) + len(val_ids) + len(test_ids)}")
    print()
    
    # Verify split distribution
    print("Verifying split distribution...")
    # Group by user, activity, environment to verify splits
    split_verification = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repeat = int(row['repeat'])
            key = (row['user'], int(row['activity']), int(row['environment']))
            
            if 1 <= repeat <= 12:
                split_verification[key]['train'] += 1
            elif 13 <= repeat <= 14:
                split_verification[key]['val'] += 1
            elif 15 <= repeat <= 20:
                split_verification[key]['test'] += 1
    
    # Display sample verification
    print("Sample split distribution (first 5 user-activity-env combinations):")
    for i, (key, counts) in enumerate(list(split_verification.items())[:5], 1):
        user, activity, env = key
        print(f"  {i}. User {user}, Activity {activity}, Env {env}: "
              f"Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")
    print()
    
    # Write JSON files
    train_file = output_path / "train_id.json"
    val_file = output_path / "val_id.json"
    test_file = output_path / "test_id.json"
    
    print(f"Writing split files...")
    print(f"  Training: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_ids, f, indent=2, ensure_ascii=False)
    
    print(f"  Validation: {val_file}")
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_ids, f, indent=2, ensure_ascii=False)
    
    print(f"  Testing: {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_ids, f, indent=2, ensure_ascii=False)
    
    print()
    print("Sample IDs from each split:")
    print(f"  Training (first 5): {train_ids[:5]}")
    print(f"  Validation (first 5): {val_ids[:5]}")
    print(f"  Testing (first 5): {test_ids[:5]}")
    print()
    print("=" * 60)
    print("Split generation complete!")
    print("=" * 60)

def main(root=None, metadata=None, output=None):
    print(f"XRF55 root: {xrf55_root(root)}")
    metadata_file = str(Path(metadata).expanduser().resolve()) if metadata else str(metadata_csv_path(root))
    output_directory = str(Path(output).expanduser().resolve()) if output else str(splits_dir(root))
    generate_splits(metadata_file, output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate XRF55 train/val/test split JSON files from sample_metadata.csv.")
    parser.add_argument("--metadata", default=None, help="Path to the metadata CSV (default: <xrf55_root>/metadata/sample_metadata.csv)")
    parser.add_argument("--output", default=None, help="Directory to write the *_id.json split files into (default: <xrf55_root>/splits)")
    parser.add_argument(
        "--root",
        default=None,
        help="XRF55 root directory (overrides $XRF55_ROOT and the default <repo_root>/data/Dataset/xrf55/action_recognition)",
    )
    args = parser.parse_args()
    main(root=args.root, metadata=args.metadata, output=args.output)
