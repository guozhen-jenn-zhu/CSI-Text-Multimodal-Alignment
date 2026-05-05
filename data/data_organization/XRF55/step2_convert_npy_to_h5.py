"""
Convert XRF55 .npy files into .h5 files while preserving array shape and dtype.

By default the source is taken from the organized layout produced by step1, falling
back to the raw dump if no organized files are present:
    --source : <xrf55_root>/organized  (else <xrf55_root>/raw)
    --target : same directory tree as source (each .npy gets a sibling .h5)

The XRF55 root defaults to ``<repo_root>/data/Dataset/xrf55/action_recognition``;
override with ``--root`` or ``$XRF55_ROOT`` (see ``_paths.py``).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from _paths import organized_dir, raw_dir, xrf55_root


def convert_npy_to_h5(npy_path, h5_path):
    """
    Convert a .npy file to .h5 file while preserving dimensions.
    
    Args:
        npy_path: Path to the input .npy file
        h5_path: Path to the output .h5 file
    """
    # Load the .npy file
    data = np.load(npy_path)
    
    # Display .npy data dimensions
    print(f"  .npy file: {npy_path.name}")
    print(f"    Shape: {data.shape}")
    print(f"    Dtype: {data.dtype}")
    print(f"    Size: {data.size} elements")
    
    # Create parent directory if it doesn't exist
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to .h5 file
    with h5py.File(h5_path, 'w') as f:
        # Create a dataset with the same name as the file (without extension)
        dataset_name = h5_path.stem
        f.create_dataset(dataset_name, data=data)
    
    # Verify the .h5 file dimensions
    with h5py.File(h5_path, 'r') as f:
        h5_data = f[dataset_name]
        print(f"  .h5 file: {h5_path.name}")
        print(f"    Shape: {h5_data.shape}")
        print(f"    Dtype: {h5_data.dtype}")
        print(f"    Size: {h5_data.size} elements")
        
        # Verify dimensions match
        if data.shape == h5_data.shape:
            print(f"    [OK] Dimensions match!")
        else:
            print(f"    [ERROR] Dimension mismatch!")
    
    print()

def convert_directory(source_dir, target_dir=None, recursive=True):
    """
    Convert all .npy files in a directory to .h5 files.
    
    Args:
        source_dir: Source directory containing .npy files
        target_dir: Target directory for .h5 files (if None, saves in same location)
        recursive: Whether to search recursively
    """
    source_path = Path(source_dir)
    
    if target_dir is None:
        target_path = source_path
    else:
        target_path = Path(target_dir)
    
    # Find all .npy files
    if recursive:
        npy_files = list(source_path.rglob("*.npy"))
    else:
        npy_files = list(source_path.glob("*.npy"))
    
    total_files = len(npy_files)
    print(f"Found {total_files} .npy files to convert\n")
    
    if total_files == 0:
        print("No .npy files found!")
        return
    
    # Process each file
    for idx, npy_file in enumerate(npy_files, 1):
        print(f"[{idx}/{total_files}] Processing: {npy_file}")
        
        # Determine output path
        if target_dir is None:
            # Save in same directory, just change extension
            h5_file = npy_file.with_suffix('.h5')
        else:
            # Maintain relative path structure in target directory
            relative_path = npy_file.relative_to(source_path)
            h5_file = target_path / relative_path.with_suffix('.h5')
        
        try:
            convert_npy_to_h5(npy_file, h5_file)
        except Exception as e:
            print(f"  [ERROR] Error converting {npy_file}: {e}\n")
            continue
        
        # Show progress every 100 files
        if idx % 100 == 0:
            print(f"Progress: {idx}/{total_files} files converted\n")


def main(root=None, source=None, target=None, test=False):
    print(f"XRF55 root: {xrf55_root(root)}")

    if source is not None:
        source_directory = str(Path(source).expanduser().resolve())
    else:
        # Check organized directory first, then fall back to raw directory
        organized_path = organized_dir(root)
        raw_path = raw_dir(root)
        if organized_path.exists() and any(organized_path.rglob("*.npy")):
            source_directory = str(organized_path)
            print("Using organized directory as source")
        elif raw_path.exists() and any(raw_path.rglob("*.npy")):
            source_directory = str(raw_path)
            print("Using raw directory as source")
        else:
            source_directory = str(organized_path)  # Default
            print("Warning: No .npy files found in expected locations")

    target_directory = str(Path(target).expanduser().resolve()) if target else None

    print("=" * 60)
    print("Converting .npy files to .h5 format")
    if test:
        print("TEST MODE: Converting only the first file")
    print("=" * 60)
    print(f"Source directory: {source_directory}")
    print(f"Target directory: {target_directory if target_directory else 'Same as source (replacing .npy with .h5)'}")
    print()

    if test:
        # Test with a single file
        source_path = Path(source_directory)
        npy_files = list(source_path.rglob("*.npy"))
        if npy_files:
            test_file = npy_files[0]
            print(f"Testing with file: {test_file}\n")
            h5_file = test_file.with_suffix('.h5')
            convert_npy_to_h5(test_file, h5_file)
            print("Test conversion successful!")
        else:
            print("No .npy files found for testing!")
    else:
        # Convert all .npy files
        # Set target_directory=None to save .h5 files in the same location as .npy files
        convert_directory(source_directory, target_directory, recursive=True)

    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XRF55 .npy files to .h5 (one .h5 per .npy, same shape).")
    parser.add_argument("--source", default=None, help="Source dir containing .npy files (default: <xrf55_root>/organized, fallback <xrf55_root>/raw)")
    parser.add_argument("--target", default=None, help="Target dir for .h5 files (default: alongside the .npy files)")
    parser.add_argument("--test", "-t", action="store_true", help="Convert only the first .npy file (smoke test)")
    parser.add_argument(
        "--root",
        default=None,
        help="XRF55 root directory (overrides $XRF55_ROOT and the default <repo_root>/data/Dataset/xrf55/action_recognition)",
    )
    args = parser.parse_args()
    main(root=args.root, source=args.source, target=args.target, test=args.test)
