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
    (90, gp_Dir(0, 1, 0), "front"),         # Front view: rotate 90° around Y-axis
    (90, gp_Dir(0, 1, 0), "right_side"),     # Right side view (90° around Y-axis)
    (90, gp_Dir(1, 0, 0), "top"),            # Top view (90° around X-axis)
    (90, gp_Dir(1, 1, 0), "isometric_1"),   # Isometric view variant
    (90, gp_Dir(0, 1, 1), "isometric_2"),   # Isometric view variant
    (90, gp_Dir(1, 0, 1), "isometric_3"),   # Isometric view variant
    (45, gp_Dir(0, 1, 0), "angle_1"),        # Angled view
    (45, gp_Dir(1, 0, 0), "angle_2"),        # Angled view
    (45, gp_Dir(1, 1, 0), "angle_3"),        # Angled view
    (45, gp_Dir(0, 1, 1), "angle_4"),        # Angled view
    (45, gp_Dir(1, 0, 1), "angle_5"),       # Angled view
    (45, gp_Dir(1, 1, 1), "angle_6"),        # Angled view
]

def rotate_shape(shape, angle_deg, axis):
    trsf = gp_Trsf()
    angle_rad = math.radians(angle_deg)
    rot_axis = gp_Ax1(gp_Pnt(0, 0, 0), axis)
    trsf.SetRotation(rot_axis, angle_rad)
    return BRepBuilderAPI_Transform(shape, trsf).Shape()

def process_step_file(step_path, out_folder, include_front_view=True):
    shape = read_step_file(step_path)
    if not shape:
        print(f"Error: Could not load {step_path}")
        return
    display, start_display, _, _ = init_display()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    base_name = os.path.splitext(os.path.basename(step_path))[0]
    
    # Display the shape once
    display.DisplayShape(shape, update=True, color=MODEL_COLOR, transparency=0.0)
    display.FitAll()
    
    for i, (angle, axis, view_name) in enumerate(STANDARD_ROTATIONS):
        # Skip front view if not requested
        if view_name == "front" and not include_front_view:
            continue
        
        # For front view, use display view orientation instead of rotating the object
        if view_name == "front":
            try:
                # Set view to front (looking along +Y axis)
                view = display.GetView()
                view.SetProj(V3d_Ypos)
                view.FitAll()
            except:
                # Fallback: rotate object -90° around X-axis then 90° around Z-axis for front view
                trsf1 = gp_Trsf()
                trsf1.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(-90))
                trsf2 = gp_Trsf()
                trsf2.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(90))
                trsf = gp_Trsf()
                trsf.SetTransformation(trsf2.Transformation().Multiplied(trsf1.Transformation()))
                rotated_shape = BRepBuilderAPI_Transform(shape, trsf).Shape()
                display.EraseAll()
                display.DisplayShape(rotated_shape, update=True, color=MODEL_COLOR, transparency=0.0)
                display.FitAll()
        elif angle == 0:
            # For isometric (no rotation), use the already displayed shape
            display.EraseAll()
            display.DisplayShape(shape, update=True, color=MODEL_COLOR, transparency=0.0)
            display.FitAll()
        else:
            # For other views, rotate the object
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

def main(batch_size=2, start_batch=0, include_front_view=True):
    # Create output folder once for all images
    output_folder = "output_data"
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
        step_files = [os.path.join(root, f) for f in files 
                     if f.lower().endswith('.step') and 'assembly' not in os.path.splitext(f)[0].lower()]
        all_step_files.extend(step_files)
    
    total_files = len(all_step_files)
    print(f"Found {total_files} STEP files to process")
    
    # Split into batches
    num_batches = (total_files + batch_size - 1) // batch_size
    print(f"Files will be processed in {num_batches} batches of {batch_size} files each")
    
    if start_batch >= num_batches:
        print(f"Error: Start batch {start_batch} is out of range. There are only {num_batches} batches.")
        return
    
    # Process the specified batch
    start_idx = start_batch * batch_size
    end_idx = min(start_idx + batch_size, total_files)
    batch_files = all_step_files[start_idx:end_idx]
    
    print(f"\nProcessing batch {start_batch} (files {start_idx+1} to {end_idx} of {total_files})")
    print(f"Files in this batch: {len(batch_files)}")
    
    total_processed = 0
    for i, step_file in enumerate(batch_files, start=1):
        print(f"\n[{i}/{len(batch_files)}] Processing: {os.path.basename(step_file)}")
        process_step_file(step_file, output_folder, include_front_view=include_front_view)
        total_processed += 1
    
    print(f"\nBatch {start_batch} completed!")
    print(f"Processed {total_processed} files in this batch")
    print(f"All images saved to {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process STEP files in batches")
    parser.add_argument("-b", "--batch", type=int, default=0,
                        help="Batch number to start from (0-indexed, default: 0)")
    parser.add_argument("-s", "--batch-size", type=int, default=200,
                        help="Number of files per batch (default: 200)")
    parser.add_argument("--no-front-view", action="store_true",
                        help="Exclude front view from the generated images")
    args = parser.parse_args()
    main(batch_size=args.batch_size, start_batch=args.batch, include_front_view=not args.no_front_view)
