import os
import shutil
from pathlib import Path
import re

# Source and target directories
source_dir = Path(r"E:\XRF55\dataset")
target_base = Path(r"E:\XRF55\organized")

# Pattern to match filename: user_activity_repeat.npy
filename_pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.npy')

# Process all Scene folders
for scene_folder in source_dir.iterdir():
    if not scene_folder.is_dir() or not scene_folder.name.startswith('Scene'):
        continue
    
    # Extract environment number from Scene folder (e.g., Scene1 -> 1)
    env_match = re.search(r'Scene(\d+)', scene_folder.name)
    if not env_match:
        continue
    
    env_num = env_match.group(1)
    wifi_folder = scene_folder / "WiFi"
    
    if not wifi_folder.exists():
        continue
    
    print(f"Processing {scene_folder.name} (Environment {env_num})...")
    
    # Process all .npy files in the WiFi folder
    npy_files = list(wifi_folder.glob("*.npy"))
    total_files = len(npy_files)
    
    for idx, npy_file in enumerate(npy_files, 1):
        # Parse filename
        match = filename_pattern.match(npy_file.name)
        if not match:
            print(f"Warning: Could not parse filename {npy_file.name}, skipping...")
            continue
        
        user_num = match.group(1)
        activity_num = match.group(2)
        repeat_num = match.group(3)
        
        # Create target directory structure
        target_dir = target_base / f"user_{user_num:0>2}" / f"act_{activity_num:0>2}" / f"env_{env_num}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file to target location
        target_file = target_dir / npy_file.name
        shutil.move(str(npy_file), str(target_file))
        
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{total_files} files...")
    
    print(f"Completed {scene_folder.name}: {total_files} files moved\n")

print("Dataset reorganization complete!")
