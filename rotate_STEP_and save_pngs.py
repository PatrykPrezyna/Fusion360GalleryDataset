from OCC.Extend.DataExchange import read_step_file
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Dir, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
import math
import random

# --- Configuration ---
path_to_file = 'input_data/51a00400-0573-11ec-9601-06368d9f66a5.step'  # Use the path to your STEP file
output_image_file_template = "output_data/random_rotated_view_{:02d}.png"
# Matte mid-gray (matches your sample image)
MODEL_COLOR = Quantity_Color(0.4, 0.4, 0.4, Quantity_TOC_RGB)
random.seed(42)
# ---------------------

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

def display_and_save_rotations():
    shape = read_step_file(path_to_file)
    if not shape:
        print("Error: Could not load the STEP file.")
        return

    display, start_display, _, _ = init_display()
    # Set white background (to match your sample)
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])

    for i in range(10):
        angle = random.uniform(15, 345)  # Random angle between 15 and 345 degrees
        axis = random_unit_vector()
        rotated_shape = rotate_shape(shape, angle, axis)
        display.EraseAll()
        display.DisplayShape(rotated_shape, update=True, color=MODEL_COLOR, transparency=0.0)
        display.FitAll()

        image_file = output_image_file_template.format(i)
        try:
            display.ExportToImage(image_file)
            print(f"Saved: {image_file}")
        except Exception as e:
            print(f"Error exporting image: {e}")

    print("Random rotations and export complete. Close the viewer window to exit.")
    start_display()

if __name__ == '__main__':
    display_and_save_rotations()
