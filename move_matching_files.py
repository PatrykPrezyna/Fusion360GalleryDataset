"""
Script to move files from 14_views_10000_mechanical and STEP files from input_data
to Test_pool_DINO_14_views based on base names found in Test_pool_DINO folder.

For each file in Test_pool_DINO, extracts the base name (UUID part before first underscore)
and finds:
1. All PNG files in 14_views_10000_mechanical with the same base name
2. All STEP files in input_data (and subfolders) with the same base name
Then moves them to Test_pool_DINO_14_views.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


def extract_base_name(filename: str) -> str:
    """
    Extract the base name (UUID) from a filename.
    Base name is the part before the first underscore.
    
    Example: "4c3a6c54-05cb-11ec-9a2b-0a23aa40f633_isometric_00.png" 
    -> "4c3a6c54-05cb-11ec-9a2b-0a23aa40f633"
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    if '_' in base_name:
        return base_name.split('_')[0]
    return base_name


def main():
    # Define paths
    base_dir = Path("output_data")
    input_data_dir = Path("input_data")
    test_pool_dir = base_dir / "Test_pool_DINO"
    source_dir = base_dir / "14_views_10000_mechanical"
    target_dir = base_dir / "Test_pool_DINO_14_views"
    
    # Check if directories exist
    if not test_pool_dir.exists():
        print(f"Error: Test_pool_DINO directory not found: {test_pool_dir}")
        return
    
    if not source_dir.exists():
        print(f"Error: 14_views_10000_mechanical directory not found: {source_dir}")
        return
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}")
    
    # Get all files from Test_pool_DINO and extract base names
    print(f"\nScanning files in {test_pool_dir}...")
    test_pool_files = list(test_pool_dir.glob("*.png"))
    
    if len(test_pool_files) == 0:
        print(f"Warning: No PNG files found in {test_pool_dir}")
        return
    
    base_names = set()
    for file in test_pool_files:
        base_name = extract_base_name(file.name)
        base_names.add(base_name)
        print(f"  Found base name: {base_name} (from {file.name})")
    
    print(f"\nFound {len(base_names)} unique base names in Test_pool_DINO")
    
    # Find all files in source directory that match these base names
    print(f"\nSearching for matching files in {source_dir}...")
    matching_files = []
    
    # Get all PNG files from source directory (including subdirectories)
    source_files = list(source_dir.rglob("*.png"))
    print(f"  Scanning {len(source_files)} files in source directory...")
    
    for source_file in source_files:
        source_base_name = extract_base_name(source_file.name)
        if source_base_name in base_names:
            matching_files.append(source_file)
            print(f"  Match found: {source_file.name} (base: {source_base_name})")
    
    print(f"\nFound {len(matching_files)} matching PNG files to move")
    
    # Also search for STEP files in input_data
    print(f"\nSearching for matching STEP files in {input_data_dir}...")
    matching_step_files = []
    
    if input_data_dir.exists():
        # Get all STEP files from input_data directory (including subdirectories)
        step_files = list(input_data_dir.rglob("*.step"))
        print(f"  Scanning {len(step_files)} STEP files in input_data directory...")
        
        for step_file in step_files:
            step_base_name = extract_base_name(step_file.name)
            if step_base_name in base_names:
                matching_step_files.append(step_file)
                print(f"  Match found: {step_file.name} (base: {step_base_name})")
        
        print(f"\nFound {len(matching_step_files)} matching STEP files to move")
    else:
        print(f"  Warning: input_data directory not found: {input_data_dir}")
    
    total_files = len(matching_files) + len(matching_step_files)
    
    if total_files == 0:
        print("No matching files found. Nothing to move.")
        return
    
    # Ask for confirmation
    print(f"\nReady to move {total_files} files:")
    print(f"  - {len(matching_files)} PNG files from {source_dir}")
    if len(matching_step_files) > 0:
        print(f"  - {len(matching_step_files)} STEP files from {input_data_dir}")
    print(f"  Target: {target_dir}")
    
    response = input("\nProceed with moving files? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    # Move PNG files
    print(f"\nMoving PNG files...")
    moved_count = 0
    skipped_count = 0
    error_count = 0
    
    for source_file in matching_files:
        try:
            target_path = target_dir / source_file.name
            
            # Check if file already exists in target
            if target_path.exists():
                print(f"  Skipping {source_file.name} (already exists in target)")
                skipped_count += 1
                continue
            
            # Move the file
            shutil.move(str(source_file), str(target_path))
            print(f"  Moved: {source_file.name}")
            moved_count += 1
            
        except Exception as e:
            print(f"  Error moving {source_file.name}: {e}")
            error_count += 1
    
    # Move STEP files
    if len(matching_step_files) > 0:
        print(f"\nMoving STEP files...")
        step_moved_count = 0
        step_skipped_count = 0
        step_error_count = 0
        
        for step_file in matching_step_files:
            try:
                target_path = target_dir / step_file.name
                
                # Check if file already exists in target
                if target_path.exists():
                    print(f"  Skipping {step_file.name} (already exists in target)")
                    step_skipped_count += 1
                    continue
                
                # Move the file
                shutil.move(str(step_file), str(target_path))
                print(f"  Moved: {step_file.name}")
                step_moved_count += 1
                
            except Exception as e:
                print(f"  Error moving {step_file.name}: {e}")
                step_error_count += 1
        
        # Update totals
        moved_count += step_moved_count
        skipped_count += step_skipped_count
        error_count += step_error_count
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Base names found in Test_pool_DINO: {len(base_names)}")
    print(f"\nPNG Files:")
    print(f"  Files found matching base names: {len(matching_files)}")
    if len(matching_step_files) > 0:
        print(f"\nSTEP Files:")
        print(f"  Files found matching base names: {len(matching_step_files)}")
    print(f"\nTotal files successfully moved: {moved_count}")
    print(f"Total files skipped (already exist): {skipped_count}")
    print(f"Total errors: {error_count}")
    print(f"\nTarget directory: {target_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
