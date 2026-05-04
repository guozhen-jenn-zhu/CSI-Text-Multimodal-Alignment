import re
from pathlib import Path

def parse_and_rename_file(file_path, base_dir):
    """
    Parse file information and generate new filename with environment.
    
    Args:
        file_path: Full path to the .h5 file
        base_dir: Base directory
    
    Returns:
        Tuple of (old_path, new_path) or None if parsing fails
    """
    file_path = Path(file_path)
    base_dir = Path(base_dir)
    
    # Parse folder structure to get environment
    relative_path = file_path.relative_to(base_dir)
    env_match = re.search(r'env_(\d+)', str(relative_path))
    
    if not env_match:
        print(f"  Warning: Could not find environment in path: {relative_path}")
        return None
    
    env_num = env_match.group(1)
    
    # Get file extension
    file_ext = file_path.suffix  # .h5 or .npy
    
    # Check if file is already in the new format
    new_format_match = re.match(r'user(\d+)_act(\d+)_repeat(\d+)_env(\d+)' + re.escape(file_ext), file_path.name)
    if new_format_match:
        # File is already in the new format, skip it
        return None
    
    # Parse filename: 01_02_03.h5 or 01_02_03.npy -> user=01, activity=02, repeat=03
    filename_match = re.match(r'(\d+)_(\d+)_(\d+)' + re.escape(file_ext), file_path.name)
    
    if not filename_match:
        # File is in neither format, skip it
        return None
    
    user_num = filename_match.group(1)
    activity_num = filename_match.group(2)
    repeat_num = filename_match.group(3)
    
    # Generate new filename: user01_act01_repeat01_env1.h5 or .npy
    new_filename = f"user{user_num}_act{activity_num}_repeat{repeat_num}_env{env_num}{file_ext}"
    new_path = file_path.parent / new_filename
    
    return (file_path, new_path)

def rename_files_in_directory(source_dir, dry_run=False):
    """
    Rename all .h5 files in the directory to include environment information.
    
    Args:
        source_dir: Directory containing .h5 files
        dry_run: If True, only show what would be renamed without actually renaming
    """
    source_path = Path(source_dir)
    
    # Find all .h5 and .npy files
    h5_files = list(source_path.rglob("*.h5"))
    npy_files = list(source_path.rglob("*.npy"))
    all_files = h5_files + npy_files
    total_files = len(all_files)
    
    print("=" * 60)
    print("Renaming files with environment information")
    print("=" * 60)
    print(f"Source directory: {source_path}")
    print(f"Found {len(h5_files)} .h5 files and {len(npy_files)} .npy files")
    print(f"Total files: {total_files}")
    if dry_run:
        print("DRY RUN MODE: No files will be renamed")
    print()
    
    if total_files == 0:
        print("No files found!")
        return
    
    rename_operations = []
    errors = []
    
    # Process each file
    skipped_count = 0
    for idx, h5_file in enumerate(all_files, 1):
        result = parse_and_rename_file(h5_file, source_path)
        
        if result:
            old_path, new_path = result
            # Skip if new filename is the same as old (already renamed)
            if old_path.name != new_path.name:
                rename_operations.append((old_path, new_path))
            else:
                skipped_count += 1
        else:
            # Check if it's already in new format (should be skipped silently)
            if re.match(r'user\d+_act\d+_repeat\d+_env\d+\.(h5|npy)', h5_file.name):
                skipped_count += 1
            else:
                errors.append(h5_file)
        
        if idx % 1000 == 0:
            print(f"Processed {idx}/{total_files} files...")
    
    print(f"\nPrepared {len(rename_operations)} rename operations")
    if skipped_count > 0:
        print(f"  {skipped_count} files already in new format (skipped)")
    if errors:
        print(f"  {len(errors)} files had parsing errors")
    
    # Show sample renames
    print("\nSample renames (first 5):")
    for old_path, new_path in rename_operations[:5]:
        print(f"  {old_path.name}")
        print(f"    -> {new_path.name}")
    
    # Check for conflicts (if new filename already exists)
    conflicts = []
    for old_path, new_path in rename_operations:
        if new_path.exists() and new_path != old_path:
            conflicts.append((old_path, new_path))
    
    if conflicts:
        print(f"\nWarning: {len(conflicts)} naming conflicts detected!")
        print("Conflicting files (first 5):")
        for old_path, new_path in conflicts[:5]:
            print(f"  {old_path} -> {new_path} (already exists)")
    
    if not dry_run:
        # Perform renames
        print(f"\nRenaming files...")
        success_count = 0
        error_count = 0
        
        for old_path, new_path in rename_operations:
            try:
                # Skip if it's the same name
                if old_path == new_path:
                    continue
                
                old_path.rename(new_path)
                success_count += 1
            except Exception as e:
                print(f"  Error renaming {old_path}: {e}")
                error_count += 1
        
        print(f"\nRenaming complete!")
        print(f"  Successfully renamed: {success_count} files")
        print(f"  Errors: {error_count} files")
    else:
        print(f"\nDry run complete. {len(rename_operations)} files would be renamed.")
    
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    # Check for dry run mode
    dry_run = "--dry-run" in sys.argv or "-d" in sys.argv
    
    # Source directory
    source_directory = r"E:\XRF55\XRF55V2"
    
    rename_files_in_directory(source_directory, dry_run=dry_run)
