import shutil
from pathlib import Path

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

if __name__ == "__main__":
    # Source directory (where .npy and .h5 files currently are)
    source_directory = r"E:\XRF55\organized"
    
    # Target directory (where .npy files will be moved)
    target_directory = r"E:\XRF55\organized_npy"
    
    # Move all .npy files
    move_npy_files(source_directory, target_directory)
