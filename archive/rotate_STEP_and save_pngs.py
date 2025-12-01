import os
import sys
import json
import random
import math
import argparse
from OCC.Extend.DataExchange import read_step_file
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Dir, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

# Matte mid-gray
MODEL_COLOR = Quantity_Color(0.4, 0.4, 0.4, Quantity_TOC_RGB)
random.seed(42)

def random_unit_vector():
    phi = random.uniform(0, 2 * math.pi)
    cos_theta = random.uniform(-1, 1)
    sin_theta = math.sqrt(1 - cos_theta ** 2)
    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)
    z = cos_theta
    return gp_Dir(x, y, z)

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
    for i in range(10):
        angle = random.uniform(15, 345)
        axis = random_unit_vector()
        rotated_shape = rotate_shape(shape, angle, axis)
        display.EraseAll()
        display.DisplayShape(rotated_shape, update=True, color=MODEL_COLOR, transparency=0.0)
        display.FitAll()
        image_file = os.path.join(out_folder, f"{base_name}_rot_{i:02d}.png")
        try:
            display.ExportToImage(image_file)
            print(f"Saved: {image_file}")
        except Exception as e:
            print(f"Error exporting image: {e}")
    display.EraseAll()

def main(indexes):
    # Load JSON file (assume "picture_info.json" in script directory)
    json_path = os.path.join(os.path.dirname(__file__), "picture_info.json")
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return
    with open(json_path, "r") as f:
        data = json.load(f)
    for index in indexes:
        entry = next((item for item in data if item['index'] == index), None)
        if not entry:
            print(f"No entry with index {index} in JSON.")
            continue
        # Always drop the filename
        parent_folder = os.path.dirname(entry["file_path"])
        parent_folder_name = os.path.basename(parent_folder)
        # Check if path is directory or single file
        if os.path.isdir(parent_folder):
            step_files = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if f.lower().endswith('.step')]
        elif parent_folder.lower().endswith('.step'):
            step_files = [parent_folder]
        else:
            print(f"Path is neither a directory nor a STEP file: {parent_folder}")
            continue
        output_folder = os.path.join("output_data", parent_folder_name)
        os.makedirs(output_folder, exist_ok=True)
        if not step_files:
            print("No STEP files found.")
            continue
        for step_file in step_files:
            process_step_file(step_file, output_folder)
        print(f"All images saved to {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process STEP files by index")
    parser.add_argument("-i", "--indexes", nargs="+", type=int, default=[2, 9, 14, 40, 65, 68, 82, 119, 142, 144, 182, 220, 217, 230, 228, 236, 239, 270, 322, 362, 386, 405, 447, 486, 528, 598, 632, 713],
                        help="Indexes in picture_info.json")
    args = parser.parse_args()
    main(args.indexes)
