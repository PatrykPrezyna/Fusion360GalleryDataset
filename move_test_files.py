import os
import shutil
import time
from pathlib import Path

# Define paths
test_folder = Path("output_data/iso_10000_mechanical/test")
source_dir = Path("output_data/14_views_10000_mechanical")
dest_dir = Path("output_data/14_views_10000_mechanical/test")

# Ensure destination directory exists
dest_dir.mkdir(parents=True, exist_ok=True)

# Get all test IDs from the test folder
print("Scanning test folder for IDs...")
test_ids = set()
for png_file in test_folder.glob("*.png"):
    # Extract the base ID (everything before _isometric_00)
    base_name = png_file.stem
    if "_isometric_00" in base_name:
        test_id = base_name.replace("_isometric_00", "")
        test_ids.add(test_id)

print(f"Found {len(test_ids)} unique test IDs")

# First, count total files to move
print("Counting files to move...")
total_files = 0
files_to_move = []
for test_id in test_ids:
    matching_files = list(source_dir.glob(f"{test_id}*.png"))
    for file in matching_files:
        if file.parent != dest_dir:  # Skip if already in test folder
            total_files += 1
            files_to_move.append(file)

print(f"Found {total_files} files to move\n")
print("=" * 60)

# Move files with progress
moved_count = 0
error_count = 0
start_time = time.time()

for idx, file in enumerate(files_to_move, 1):
    dest_path = dest_dir / file.name
    try:
        shutil.move(str(file), str(dest_path))
        moved_count += 1
        
        # Show progress every 10 files or on every file for first 50
        if idx <= 50 or idx % 10 == 0 or idx == total_files:
            progress = (idx / total_files) * 100
            elapsed = time.time() - start_time
            if idx > 0:
                avg_time_per_file = elapsed / idx
                remaining = avg_time_per_file * (total_files - idx)
                print(f"[{idx:5d}/{total_files}] ({progress:5.1f}%) "
                      f"Moved: {file.name[:50]:<50} "
                      f"ETA: {remaining:.1f}s", end='\r')
    except Exception as e:
        error_count += 1
        print(f"\nError moving {file.name}: {e}")

elapsed_total = time.time() - start_time
print("\n" + "=" * 60)
print(f"\nCompleted!")
print(f"  Total files moved: {moved_count}")
print(f"  Errors: {error_count}")
print(f"  Time elapsed: {elapsed_total:.2f} seconds")
if moved_count > 0:
    print(f"  Average speed: {moved_count/elapsed_total:.1f} files/second")

