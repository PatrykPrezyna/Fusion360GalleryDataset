import os
import math
import json
import argparse
from PIL import Image
from OCC.Extend.DataExchange import read_step_file
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Dir, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.V3d import V3d_Ypos, V3d_Xpos, V3d_Zpos

# Matte mid-gray
MODEL_COLOR = Quantity_Color(0.4, 0.4, 0.4, Quantity_TOC_RGB)

# Standardized rotations: Each entry is a tuple of (rotation_steps, view_name)
# rotation_steps is a list of (angle_deg, axis_direction) tuples applied sequentially
# The default view (no rotation) is isometric, so we rotate the object to get other views
STANDARD_ROTATIONS = [
    ([], "isometric"),                           # Isometric view (no rotation - default view)
    # Standard orthogonal views - all six sides
    # Note: "front" is the same as isometric (default view), so we use orthogonal rotations for other sides
    ([(90, gp_Dir(0, 1, 0))], "left"),           # Left side: rotate 90° around Y-axis
    ([(180, gp_Dir(0, 1, 0))], "back"),          # Back view: rotate 180° around Y-axis
    ([(270, gp_Dir(0, 1, 0))], "right"),         # Right side: rotate 270° around Y-axis
    ([(90, gp_Dir(1, 0, 0))], "top"),            # Top view: rotate 90° around X-axis
    ([(270, gp_Dir(1, 0, 0))], "bottom"),        # Bottom view: rotate 270° around X-axis
    # Oblique/angled views for better visualization
    ([(45, gp_Dir(0, 1, 0))], "front_left"),     # Between front and left: 45° around Y-axis
    ([(135, gp_Dir(0, 1, 0))], "back_left"),     # Between back and left: 135° around Y-axis
    ([(225, gp_Dir(0, 1, 0))], "back_right"),    # Between back and right: 225° around Y-axis
    ([(315, gp_Dir(0, 1, 0))], "front_right"),   # Between front and right: 315° around Y-axis
    # Isometric views with compound rotations (standard isometric: 45° horizontal + ~35° vertical)
    ([(45, gp_Dir(0, 1, 0)), (35.264, gp_Dir(1, 0, 0))], "isometric_1"),  # Standard isometric
    ([(135, gp_Dir(0, 1, 0)), (35.264, gp_Dir(1, 0, 0))], "isometric_2"), # Isometric from back-left
    ([(225, gp_Dir(0, 1, 0)), (35.264, gp_Dir(1, 0, 0))], "isometric_3"), # Isometric from back-right
    ([(315, gp_Dir(0, 1, 0)), (35.264, gp_Dir(1, 0, 0))], "isometric_4"), # Isometric from front-right
]

def rotate_shape(shape, rotation_steps):
    """
    Apply one or more rotations to a shape sequentially.
    
    Args:
        shape: The shape to rotate
        rotation_steps: List of (angle_deg, axis_direction) tuples to apply sequentially
    
    Returns:
        The rotated shape
    """
    result = shape
    for angle_deg, axis in rotation_steps:
        trsf = gp_Trsf()
        angle_rad = math.radians(angle_deg)
        rot_axis = gp_Ax1(gp_Pnt(0, 0, 0), axis)
        trsf.SetRotation(rot_axis, angle_rad)
        result = BRepBuilderAPI_Transform(result, trsf).Shape()
    return result

def has_category(step_file_path, target_category):
    """
    Check if the STEP file belongs to an assembly with the specified category.
    Looks for assembly.json in the same directory as the STEP file.
    
    Args:
        step_file_path: Path to the STEP file
        target_category: Category name to check for (e.g., "Mechanical Engineering")
    
    Returns:
        True if the assembly has the target category, False otherwise
    """
    step_dir = os.path.dirname(step_file_path)
    assembly_json_path = os.path.join(step_dir, "assembly.json")
    
    if not os.path.exists(assembly_json_path):
        # If assembly.json doesn't exist, we can't determine the category
        # Return False to skip it
        return False
    
    try:
        with open(assembly_json_path, 'r', encoding='utf-8') as f:
            assembly_data = json.load(f)
        
        # Check if properties exists and has categories
        properties = assembly_data.get("properties", {})
        categories = properties.get("categories", [])
        
        # Check if target_category is in the categories list
        return target_category in categories
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Error reading assembly.json at {assembly_json_path}: {e}")
        return False

def process_step_file(step_path, out_folder, display):
    shape = read_step_file(step_path)
    if not shape:
        print(f"Error: Could not load {step_path}")
        return
    base_name = os.path.splitext(os.path.basename(step_path))[0]
    
    for i, (rotation_steps, view_name) in enumerate(STANDARD_ROTATIONS):
        # Rotate the shape only if rotation_steps is not empty; otherwise use the original
        if not rotation_steps:
            rotated_shape = shape
        else:
            rotated_shape = rotate_shape(shape, rotation_steps)

        display.EraseAll()
        display.DisplayShape(rotated_shape, update=True, color=MODEL_COLOR, transparency=0.0)
        display.FitAll()
        
        image_file = os.path.join(out_folder, f"{base_name}_{view_name}_{i:02d}.png")
        try:
            display.ExportToImage(image_file)
            # Convert to grayscale (black and white) and resize to 515x512 pixels
            img = Image.open(image_file)
            # Convert to grayscale (black and white) only if needed
            if img.mode != 'L':
                img = img.convert('L')
            # Resize to exactly 515x512 pixels only if size differs
            if img.size != (515, 512):
                img = img.resize((515, 512), Image.Resampling.LANCZOS)
            img.save(image_file)
            print(f"Saved: {image_file}")
        except Exception as e:
            print(f"Error exporting image: {e}")
    display.EraseAll()


def main(max_parts=None, output_folder="output_data", filter_category=None):
    """
    Process STEP files found under the input_data directory, stopping after
    max_parts files if specified.
    
    Args:
        max_parts: Maximum number of STEP files to process (None = all)
        output_folder: Output folder for generated images
        filter_category: Optional category name to filter by (None = no filtering)
    """
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the OpenCascade display only once to avoid heavy per-file setup
    display, start_display, _, _ = init_display()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])

    # Get the input_data folder path
    input_data_folder = os.path.join(os.path.dirname(__file__), "input_data")
    if not os.path.exists(input_data_folder):
        print(f"Input data folder not found: {input_data_folder}")
        return

    # Collect all STEP files first
    print("Collecting all STEP files...")
    all_step_files = []
    for root, dirs, files in os.walk(input_data_folder):
        step_files = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith('.step')
            and 'assembly' not in os.path.splitext(f)[0].lower()
        ]
        all_step_files.extend(step_files)

    total_files = len(all_step_files)
    print(f"Found {total_files} STEP files in total")

    # Optionally filter STEP files by category
    if filter_category:
        print(f"Filtering STEP files by '{filter_category}' category...")
        filtered_step_files = []
        skipped_count = 0
        for step_file in all_step_files:
            if has_category(step_file, filter_category):
                filtered_step_files.append(step_file)
            else:
                skipped_count += 1
        
        print(f"Found {len(filtered_step_files)} STEP files with '{filter_category}' category")
        print(f"Skipped {skipped_count} STEP files from other categories")
        
        all_step_files = filtered_step_files
    else:
        print("No category filter applied - processing all STEP files")

    if max_parts is not None:
        all_step_files = all_step_files[:max_parts]
        print(f"Will process at most {max_parts} STEP files (parts)")

    print(f"Processing {len(all_step_files)} STEP files...")

    total_processed = 0
    for i, step_file in enumerate(all_step_files, start=1):
        print(f"\n[{i}/{len(all_step_files)}] Processing: {os.path.basename(step_file)}")
        process_step_file(step_file, output_folder, display)
        total_processed += 1

    print(f"\nProcessing completed!")
    print(f"Processed {total_processed} files")
    print(f"All images saved to {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process STEP files in batches")
    parser.add_argument("-n", "--num-parts", type=int, default=None,
                        help="Maximum number of STEP files (parts) to process. "
                             "If omitted, all found STEP files are processed.")
    parser.add_argument("-o", "--output-folder", type=str, default="output_data",
                        help="Output folder to save the images (default: output_data)")
    parser.add_argument("-c", "--category", type=str, default=None,
                        help="Optional category filter (e.g., 'Mechanical Engineering'). "
                             "If specified, only processes STEP files from assemblies with this category. "
                             "If omitted, all STEP files are processed.")
    args = parser.parse_args()
    main(max_parts=args.num_parts, output_folder=args.output_folder, filter_category=args.category)
