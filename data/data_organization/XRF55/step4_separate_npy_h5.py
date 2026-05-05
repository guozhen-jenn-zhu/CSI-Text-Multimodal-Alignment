"""
Move every .npy out of the organized tree into a parallel ``organized_npy`` tree
so the canonical layout retains only the .h5 outputs from step2.

Defaults:
    --source : <xrf55_root>/organized
    --target : <xrf55_root>/organized_npy

The XRF55 root defaults to ``<repo_root>/data/Dataset/xrf55/action_recognition``;
override with ``--root`` or ``$XRF55_ROOT`` (see ``_paths.py``).
"""

import argparse
import shutil
from pathlib import Path

from _paths import organized_dir, organized_npy_dir, xrf55_root

def move_npy_files(source_dir, target_dir):
    """
    Move all .npy files from source directory to target directory,
    preserving the directory structure.
    
    Args:
        source_dir: Source directory containing .npy files
        target_dir: Target directory where .npy files will be moved
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Find all .npy files
    npy_files = list(source_path.rglob("*.npy"))
    total_files = len(npy_files)
    
    print("=" * 60)
    print("Separating .npy and .h5 files")
    print("=" * 60)
    print(f"Source directory: {source_path}")
    print(f"Target directory: {target_path}")
    print(f"Found {total_files} .npy files to move")
    print()
    
    if total_files == 0:
        print("No .npy files found!")
        return
    
    moved_count = 0
    error_count = 0
    
    # Process each file
    for idx, npy_file in enumerate(npy_files, 1):
        # Get relative path from source directory
        relative_path = npy_file.relative_to(source_path)
        
        # Create target file path
        target_file = target_path / relative_path
        
        # Create parent directory if it doesn't exist
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Move the file
            shutil.move(str(npy_file), str(target_file))
            moved_count += 1
            
            if idx % 1000 == 0:
                print(f"Progress: {idx}/{total_files} files moved...")
                
        except Exception as e:
            print(f"Error moving {npy_file}: {e}")
            error_count += 1
            continue
    
    print()
    print("=" * 60)
    print(f"Separation complete!")
    print(f"  Successfully moved: {moved_count} files")
    print(f"  Errors: {error_count} files")
    print("=" * 60)
    
    # Verify: count remaining .npy files in source
    remaining_npy = list(source_path.rglob("*.npy"))
    remaining_h5 = list(source_path.rglob("*.h5"))
    
    print()
    print("Verification:")
    print(f"  .npy files remaining in source: {len(remaining_npy)}")
    print(f"  .h5 files in source: {len(remaining_h5)}")
    print(f"  .npy files in target: {len(list(target_path.rglob('*.npy')))}")

def main(root=None, source=None, target=None):
    print(f"XRF55 root: {xrf55_root(root)}")
    source_directory = str(Path(source).expanduser().resolve()) if source else str(organized_dir(root))
    target_directory = str(Path(target).expanduser().resolve()) if target else str(organized_npy_dir(root))
    move_npy_files(source_directory, target_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the organized XRF55 tree by moving .npy files into a sibling directory.")
    parser.add_argument("--source", default=None, help="Mixed .npy/.h5 source dir (default: <xrf55_root>/organized)")
    parser.add_argument("--target", default=None, help="Destination dir for the .npy files (default: <xrf55_root>/organized_npy)")
    parser.add_argument(
        "--root",
        default=None,
        help="XRF55 root directory (overrides $XRF55_ROOT and the default <repo_root>/data/Dataset/xrf55/action_recognition)",
    )
    args = parser.parse_args()
    main(root=args.root, source=args.source, target=args.target)
