"""
Reshape XRF55 .h5 samples from (270, 1000) into the canonical (270, 1000, 1) layout
and reorder the antenna/subcarrier axis with the documented stride-3 permutation.

By default reads from ``<xrf55_root>/organized`` and writes the reshaped .h5 files
into ``<xrf55_root>`` (mirrored relative paths). Pass ``--in-place`` to overwrite
the source files instead, or ``--target`` to send the output elsewhere.

The XRF55 root defaults to ``<repo_root>/data/Dataset/xrf55/action_recognition``;
override with ``--root`` or ``$XRF55_ROOT`` (see ``_paths.py``).
"""

import argparse
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from _paths import organized_dir, xrf55_root

def reorder_indices_270():
    """
    Generate reordering indices for 270 elements.
    Pattern: [1, 4, 7, 10, 2, 5, 8, 3, 6, 9] for [1,2,3,...,10]
    This means: collect all indices where index % 3 == 0, then all where index % 3 == 1, 
    then all where index % 3 == 2.
    For 270 elements, we'll apply the same pattern.
    """
    indices = []
    group_size = 3
    
    # Collect all indices grouped by their modulo 3 value
    # First: all indices where index % 3 == 0
    # Then: all indices where index % 3 == 1
    # Finally: all indices where index % 3 == 2
    for pos in range(group_size):
        idx = pos
        while idx < 270:
            indices.append(idx)
            idx += group_size
    
    return np.array(indices)

def plot_reshaped_data(data, file_path):
    """
    Plot the reshaped data for visualization.
    
    Args:
        data: The reshaped numpy array
        file_path: Path to the file (for title)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: 2D heatmap of the data (squeeze the last dimension if it's 1)
    if len(data.shape) == 3 and data.shape[2] == 1:
        plot_data = data[:, :, 0]
    else:
        plot_data = data.squeeze()
    
    im1 = axes[0].imshow(plot_data, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'Reshaped Data Heatmap\n{Path(file_path).name}\nShape: {data.shape}')
    axes[0].set_xlabel(f'Time/Feature Dimension ({data.shape[1]})')
    axes[0].set_ylabel('Reordered Index (270)')
    plt.colorbar(im1, ax=axes[0], label='Value')
    
    # Plot 2: Sample of first few rows
    num_rows_to_plot = min(10, plot_data.shape[0])
    for i in range(num_rows_to_plot):
        axes[1].plot(plot_data[i, :], label=f'Row {i}', alpha=0.7, linewidth=1)
    axes[1].set_title(f'Sample of First {num_rows_to_plot} Rows')
    axes[1].set_xlabel('Time/Feature Index')
    axes[1].set_ylabel('Value')
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()  # Display but don't save

def reshape_h5_data(h5_path, output_path=None, in_place=True, plot_first=False):
    """
    Reshape .h5 data according to specifications:
    1. Add dimension to make (270, 1000) -> (270, 1000, 1)
    2. Reorder the first dimension (270) according to pattern
    3. Downsample (DISABLED): would reduce (270, 1000, 1) -> (270, 500, 1)
    4. Final shape: (270, 1000, 1) with downsample disabled
    
    Args:
        h5_path: Path to input .h5 file
        output_path: Path to output .h5 file (if None and not in_place, creates new file)
        in_place: If True, overwrites the original file (default: True)
        plot_first: If True, plot the data after reshaping (default: False)
    """
    h5_path = Path(h5_path)
    
    # Read the .h5 file
    with h5py.File(h5_path, 'r') as f:
        # Get the dataset (assuming it's named after the file stem)
        dataset_name = h5_path.stem
        if dataset_name not in f.keys():
            # Try to get the first dataset if name doesn't match
            dataset_name = list(f.keys())[0]
        
        data = f[dataset_name][:]
        original_shape = data.shape
        print(f"  Original shape: {original_shape}")
        print(f"  Original dtype: {data.dtype}")
    
    # Step 1: Add dimension to make (270, 1000) -> (270, 1000, 1)
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
        print(f"  After adding dimension: {data.shape}")
    
    # Step 2: Reorder the first dimension (270)
    if data.shape[0] == 270:
        reorder_indices = reorder_indices_270()
        data = data[reorder_indices, :, :]
        print(f"  After reordering first dimension: {data.shape}")
    else:
        print(f"  Warning: First dimension is {data.shape[0]}, not 270. Skipping reordering.")
    
    # Step 3: Downsample the second dimension by factor of 2 (DISABLED)
    # if data.shape[1] == 1000:
    #     data = data[:, ::2, :]
    #     print(f"  After downsampling second dimension by factor of 2: {data.shape}")
    # else:
    #     print(f"  Warning: Second dimension is {data.shape[1]}, not 1000. Skipping downsampling.")
    
    # Step 4: Verify final shape (270, 1000, 1) when downsample disabled; (270, 500, 1) when enabled
    final_shape = data.shape
    print(f"  Final shape: {final_shape}")
    
    # Determine output path
    if in_place:
        output_path = h5_path
        # Create temporary file first
        temp_path = h5_path.with_suffix('.h5.tmp')
        write_path = temp_path
    elif output_path is None:
        output_path = h5_path.with_name(h5_path.stem + '_reshaped.h5')
        write_path = output_path
    else:
        write_path = Path(output_path)
    
    # Create parent directory if needed
    write_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the reshaped data
    with h5py.File(write_path, 'w') as f:
        f.create_dataset(dataset_name, data=data)
    
    # If in_place, replace original with temp file
    if in_place:
        h5_path.unlink()
        temp_path.rename(h5_path)
        print(f"  Updated file in place: {h5_path}")
    else:
        print(f"  Saved to: {write_path}")
    
    # Verify the output
    with h5py.File(write_path if not in_place else h5_path, 'r') as f:
        verify_data = f[dataset_name][:]
        print(f"  Verified shape: {verify_data.shape}")
        print(f"  Verified dtype: {verify_data.dtype}")
    
    # Plot if requested
    if plot_first:
        print(f"  Plotting reshaped data...")
        plot_reshaped_data(verify_data, h5_path)
    
    print()
    return final_shape

def reshape_directory(source_dir, target_dir=None, in_place=True, recursive=True, plot_first=False):
    """
    Reshape all .h5 files in a directory.
    
    Args:
        source_dir: Source directory containing .h5 files
        target_dir: Target directory for reshaped files (if None and not in_place, saves with _reshaped suffix)
        in_place: If True, overwrites original files (default: True)
        recursive: Whether to search recursively
        plot_first: If True, plot the first reshaped file (default: False)
    """
    source_path = Path(source_dir)
    
    # Find all .h5 files
    if recursive:
        h5_files = list(source_path.rglob("*.h5"))
    else:
        h5_files = list(source_path.glob("*.h5"))
    
    total_files = len(h5_files)
    print("=" * 60)
    print("Reshaping .h5 files")
    print("=" * 60)
    print(f"Source directory: {source_path}")
    if target_dir:
        print(f"Target directory: {target_dir}")
    elif in_place:
        print("Mode: In-place (overwriting original files)")
    else:
        print("Mode: Creating new files with _reshaped suffix")
    print(f"Found {total_files} .h5 files to reshape")
    print()
    
    if total_files == 0:
        print("No .h5 files found!")
        return
    
    success_count = 0
    error_count = 0
    first_file_plotted = False
    
    # Process each file
    for idx, h5_file in enumerate(h5_files, 1):
        print(f"[{idx}/{total_files}] Processing: {h5_file}")
        
        try:
            if target_dir:
                # Maintain relative path structure in target directory
                relative_path = h5_file.relative_to(source_path)
                output_file = Path(target_dir) / relative_path
            else:
                output_file = None
            
            # Plot only the first file if requested
            should_plot = plot_first and not first_file_plotted
            reshape_h5_data(h5_file, output_file, in_place, plot_first=should_plot)
            if should_plot:
                first_file_plotted = True
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Error reshaping {h5_file}: {e}\n")
            error_count += 1
            continue
        
        # Show progress every 100 files
        if idx % 100 == 0:
            print(f"Progress: {idx}/{total_files} files processed\n")
    
    print("=" * 60)
    print(f"Reshaping complete!")
    print(f"  Successfully processed: {success_count} files")
    print(f"  Errors: {error_count} files")
    print("=" * 60)

def main(root=None, source=None, target=None, in_place=False, no_plot=False, test=False):
    print(f"XRF55 root: {xrf55_root(root)}")
    source_directory = str(Path(source).expanduser().resolve()) if source else str(organized_dir(root))
    target_directory = str(Path(target).expanduser().resolve()) if target else str(xrf55_root(root))

    plot_first = not no_plot

    print("=" * 60)
    print("Reshaping .h5 files")
    if test:
        print("TEST MODE: Processing only the first file")
    if in_place:
        print("IN-PLACE MODE: Will overwrite original files")
    elif not test:
        print(f"TARGET MODE: Will save to {target_directory}")
    if plot_first or test:
        print("PLOTTING: Will plot the first reshaped file")
    print("=" * 60)
    print()

    if test:
        # Test with a single file
        source_path = Path(source_directory)
        h5_files = list(source_path.rglob("*.h5"))
        if h5_files:
            test_file = h5_files[0]
            print(f"Testing with file: {test_file}\n")
            if target_directory and not in_place:
                relative_path = test_file.relative_to(source_path)
                test_output = Path(target_directory) / relative_path
            else:
                test_output = None
            reshape_h5_data(test_file, test_output, in_place=in_place, plot_first=True)
            print("Test reshaping successful!")
        else:
            print("No .h5 files found for testing!")
    else:
        reshape_directory(
            source_directory,
            target_directory if not in_place else None,
            in_place=in_place,
            recursive=True,
            plot_first=plot_first,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshape XRF55 .h5 files: add channel axis and reorder the 270-dim axis.")
    parser.add_argument("--source", default=None, help="Source dir of .h5 files (default: <xrf55_root>/organized)")
    parser.add_argument("--target", default=None, help="Output dir for reshaped .h5 files (default: <xrf55_root>); ignored with --in-place")
    parser.add_argument("--in-place", "-i", action="store_true", help="Overwrite the source .h5 files instead of writing to --target")
    parser.add_argument("--no-plot", action="store_true", help="Do not show the matplotlib visualization of the first file")
    parser.add_argument("--test", "-t", action="store_true", help="Process only the first .h5 file (smoke test)")
    parser.add_argument(
        "--root",
        default=None,
        help="XRF55 root directory (overrides $XRF55_ROOT and the default <repo_root>/data/Dataset/xrf55/action_recognition)",
    )
    args = parser.parse_args()
    main(
        root=args.root,
        source=args.source,
        target=args.target,
        in_place=args.in_place,
        no_plot=args.no_plot,
        test=args.test,
    )
