import numpy as np
import h5py
from pathlib import Path
import os

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

if __name__ == "__main__":
    import sys
    
    # Check for test mode (convert only first file)
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    # You can modify these paths as needed
    # Check organized directory first, then fall back to dataset directory
    organized_dir = Path(r"E:\XRF55\organized")
    dataset_dir = Path(r"E:\XRF55\dataset")
    
    # Determine which directory to use
    if organized_dir.exists() and list(organized_dir.rglob("*.npy")):
        source_directory = str(organized_dir)
        print("Using organized directory as source")
    elif dataset_dir.exists() and list(dataset_dir.rglob("*.npy")):
        source_directory = str(dataset_dir)
        print("Using dataset directory as source")
    else:
        source_directory = str(organized_dir)  # Default
        print("Warning: No .npy files found in expected locations")
    
    # Save .h5 files in the same directory structure (replace .npy with .h5)
    target_directory = None  # Set to None to save in same location, or specify a different path
    
    print("=" * 60)
    print("Converting .npy files to .h5 format")
    if test_mode:
        print("TEST MODE: Converting only the first file")
    print("=" * 60)
    print(f"Source directory: {source_directory}")
    print(f"Target directory: {target_directory if target_directory else 'Same as source (replacing .npy with .h5)'}")
    print()
    
    if test_mode:
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
