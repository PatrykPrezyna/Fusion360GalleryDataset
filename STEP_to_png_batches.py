import os
import math
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

# Standardized rotations: (angle_deg, axis_direction, view_name)
# The default view (no rotation) is isometric, so we rotate the object to get other views
STANDARD_ROTATIONS = [
    (0, gp_Dir(0, 0, 1), "isometric"),      # Isometric view (no rotation - default view)
    # (90, gp_Dir(0, 1, 0), "front"),         # Front view: rotate 90° around Y-axis
    # (90, gp_Dir(0, 1, 0), "right_side"),     # Right side view (90° around Y-axis)
    # (90, gp_Dir(1, 0, 0), "top"),            # Top view (90° around X-axis)
    # (90, gp_Dir(1, 1, 0), "isometric_1"),   # Isometric view variant
    # (90, gp_Dir(0, 1, 1), "isometric_2"),   # Isometric view variant
    # (90, gp_Dir(1, 0, 1), "isometric_3"),   # Isometric view variant
    # (45, gp_Dir(0, 1, 0), "angle_1"),        # Angled view
    # (45, gp_Dir(1, 0, 0), "angle_2"),        # Angled view
    # (45, gp_Dir(1, 1, 0), "angle_3"),        # Angled view
    # (45, gp_Dir(0, 1, 1), "angle_4"),        # Angled view
    # (45, gp_Dir(1, 0, 1), "angle_5"),       # Angled view
    # (45, gp_Dir(1, 1, 1), "angle_6"),        # Angled view
]

def rotate_shape(shape, angle_deg, axis):
    trsf = gp_Trsf()
    angle_rad = math.radians(angle_deg)
    rot_axis = gp_Ax1(gp_Pnt(0, 0, 0), axis)
    trsf.SetRotation(rot_axis, angle_rad)
    return BRepBuilderAPI_Transform(shape, trsf).Shape()

def process_step_file(step_path, out_folder):
    shape = read_step_file(step_path)
    if not shape:
        print(f"Error: Could not load {step_path}")
        return
    display, start_display, _, _ = init_display()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    base_name = os.path.splitext(os.path.basename(step_path))[0]
    
    for i, (angle, axis, view_name) in enumerate(STANDARD_ROTATIONS):
        # Rotate the shape only if angle is non-zero; otherwise use the original
        if angle == 0:
            rotated_shape = shape
        else:
            rotated_shape = rotate_shape(shape, angle, axis)

        display.EraseAll()
        display.DisplayShape(rotated_shape, update=True, color=MODEL_COLOR, transparency=0.0)
        display.FitAll()
        
        image_file = os.path.join(out_folder, f"{base_name}_{view_name}_{i:02d}.png")
        try:
            display.ExportToImage(image_file)
            # Convert to grayscale (black and white) and resize to 515x512 pixels
            img = Image.open(image_file)
            img_gray = img.convert('L')
            # Resize to exactly 515x512 pixels
            img_resized = img_gray.resize((515, 512), Image.Resampling.LANCZOS)
            img_resized.save(image_file)
            print(f"Saved: {image_file}")
        except Exception as e:
            print(f"Error exporting image: {e}")
    display.EraseAll()


def main(max_parts=None, output_folder="output_data"):
    """
    Process STEP files found under the input_data directory, stopping after
    max_parts files if specified.
    """
    os.makedirs(output_folder, exist_ok=True)

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
    print(f"Found {total_files} STEP files to process in total")

    if max_parts is not None:
        all_step_files = all_step_files[:max_parts]
        print(f"Will process at most {max_parts} STEP files (parts)")

    print(f"Processing {len(all_step_files)} STEP files...")

    total_processed = 0
    for i, step_file in enumerate(all_step_files, start=1):
        print(f"\n[{i}/{len(all_step_files)}] Processing: {os.path.basename(step_file)}")
        process_step_file(step_file, output_folder)
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
    args = parser.parse_args()
    main(max_parts=args.num_parts, output_folder=args.output_folder)
