import csv
import re
from pathlib import Path

def parse_file_info(file_path, base_dir):
    """
    Parse file information from path and filename.
    Filename format: user01_act01_repeat01_env1.h5
    
    Args:
        file_path: Full path to the .h5 file
        base_dir: Base directory to calculate relative path
    
    Returns:
        Dictionary with parsed information
    """
    file_path = Path(file_path)
    base_dir = Path(base_dir)
    
    # Get relative path from base directory
    relative_path = file_path.relative_to(base_dir)
    
    # Parse filename: user01_act01_repeat01_env1.h5
    filename_match = re.match(r'user(\d+)_act(\d+)_repeat(\d+)_env(\d+)\.h5', file_path.name)
    
    if not filename_match:
        # Try old format as fallback: 01_01_01.h5
        old_format_match = re.match(r'(\d+)_(\d+)_(\d+)\.h5', file_path.name)
        if old_format_match:
            # Extract from folder structure if old format
            user_match = re.search(r'user_(\d+)', str(relative_path))
            env_match = re.search(r'env_(\d+)', str(relative_path))
            user_num = old_format_match.group(1)
            activity_num = old_format_match.group(2)
            repeat_num = old_format_match.group(3)
            env_num = env_match.group(1) if env_match else None
        else:
            raise ValueError(f"Could not parse filename: {file_path.name}")
    else:
        user_num = filename_match.group(1)
        activity_num = filename_match.group(2)
        repeat_num = filename_match.group(3)
        env_num = filename_match.group(4)
    
    # Use filename (without extension) as id
    id_str = file_path.stem
    
    # Format file_path as relative path with ./ prefix
    file_path_str = f"./{relative_path.as_posix()}"
    
    return {
        'id': id_str,
        'file_path': file_path_str,
        'subject': 'Human',
        'user': f"U{user_num:0>2}" if user_num else None,
        'activity': int(activity_num) if activity_num else None,
        'environment': int(env_num) if env_num else None,
        'device': '5300',
        'repeat': int(repeat_num) if repeat_num else None,
        'label': int(activity_num) if activity_num else None
    }

def generate_metadata(source_dir, output_dir):
    """
    Generate metadata CSV file for all .h5 files in the source directory.
    
    Args:
        source_dir: Directory containing .h5 files
        output_dir: Directory to save metadata CSV
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Find all .h5 files
    h5_files = list(source_path.rglob("*.h5"))
    total_files = len(h5_files)
    
    print("=" * 60)
    print("Generating Metadata")
    print("=" * 60)
    print(f"Source directory: {source_path}")
    print(f"Output directory: {output_path}")
    print(f"Found {total_files} .h5 files")
    print()
    
    if total_files == 0:
        print("No .h5 files found!")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata rows
    metadata_rows = []
    
    for idx, h5_file in enumerate(h5_files, 1):
        try:
            file_info = parse_file_info(h5_file, source_path)
            metadata_rows.append(file_info)
            
            if idx % 1000 == 0:
                print(f"Processed {idx}/{total_files} files...")
                
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue
    
    # Sort by user, activity, environment, repeat
    metadata_rows.sort(key=lambda x: (
        x.get('user', ''),
        x.get('activity', 0),
        x.get('environment', 0),
        x.get('repeat', 0)
    ))
    
    # Write to CSV
    output_file = output_path / "metadata.csv"
    
    # Define column order
    columns = ['id', 'file_path', 'subject', 'user', 'activity', 'environment', 'device', 'repeat', 'label']
    
    print(f"\nWriting metadata to: {output_file}")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metadata_rows)
    
    print(f"Metadata generated successfully!")
    print(f"Total records: {len(metadata_rows)}")
    print()
    
    # Display sample records
    print("Sample records (first 5):")
    for i, row in enumerate(metadata_rows[:5], 1):
        print(f"  {i}. {row['id']} -> {row['file_path']}")
    
    print()
    print("=" * 60)
    print("Metadata generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Source directory containing .h5 files
    source_directory = r"E:\XRF55\downstream"
    
    # Output directory for metadata
    output_directory = r"E:\XRF55\downstream\metadata"
    
    generate_metadata(source_directory, output_directory)
