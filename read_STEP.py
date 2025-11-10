from OCC.Extend.DataExchange import read_step_file
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Display.SimpleGui import init_display 
from OCC.Core.gp import gp_Dir, gp_Trsf, gp_Vec, gp_Ax1, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform # Required for moving Part 2
import math # Needed for converting degrees to radians

# --- Configuration ---
path_to_file_1 = '257ba290-05aa-11ec-9425-065da05e55cf.step' 
path_to_file_2 = '257c8cde-05aa-11ec-bd3c-065da05e55cf.step' 

output_image_file = "parameterized_assembly_view.png"
PART1_COLOR = 'RED'
PART2_COLOR = 'GREEN'
# ---------------------

def position_part(shape, dx, dy, dz, rot_angle_deg, rot_axis):
    """
    Applies a translation and rotation to a given shape.

    Args:
        shape (TopoDS_Shape): The part to be moved.
        dx, dy, dz (float): Translation distances along X, Y, Z axes.
        rot_angle_deg (float): Rotation angle in degrees.
        rot_axis (gp_Dir): The axis of rotation (e.g., gp_Dir(1, 0, 0) for X).

    Returns:
        TopoDS_Shape: The transformed part.
    """
    a_trsf = gp_Trsf()
    
    # 1. Apply Translation
    translation_vector = gp_Vec(dx, dy, dz)
    a_trsf.SetTranslation(translation_vector) 
    
    # 2. Apply Rotation (around the origin)
    if rot_angle_deg != 0:
        rot_angle_rad = math.radians(rot_angle_deg)
        # Create an axis of rotation (Axis passing through origin (0,0,0) with direction rot_axis)
        rot_axis_of_interest = gp_Ax1(gp_Pnt(0, 0, 0), rot_axis) 
        a_trsf.SetRotation(rot_axis_of_interest, rot_angle_rad) 
    
    # Apply the combined transformation
    transformed_shape = BRepBuilderAPI_Transform(shape, a_trsf).Shape()
    return transformed_shape

def create_and_display_assembly():
    
    # 1. Read the two parts
    part1_base = read_step_file(path_to_file_1)
    part2_base = read_step_file(path_to_file_2)
    
    if not part1_base or not part2_base:
        print("Error: Could not load one or both STEP files. Check paths and file names.")
        return 

    # 2. PARAMETERIZE THE CONNECTION
    # ----------------------------------------------------------------------
    # Define how Part 2 should be positioned relative to Part 1 (at origin)
    
    # Movement (X, Y, Z)
    TRANSLATION_X = 150.0  # Move 150 units along X for clear separation
    TRANSLATION_Y = 0.0
    TRANSLATION_Z = 0.0

    # Rotation (Angle in degrees and Axis)
    ROTATION_ANGLE_DEG = 45.0 # Example: Rotate 45 degrees
    ROTATION_AXIS = gp_Dir(0, 0, 1) # Rotate around the Z-axis (0, 0, 1)
    
    part2_transformed = position_part(
        part2_base, 
        TRANSLATION_X, 
        TRANSLATION_Y, 
        TRANSLATION_Z, 
        ROTATION_ANGLE_DEG, 
        ROTATION_AXIS
    )
    # ----------------------------------------------------------------------
    
    print(f"File '{path_to_file_1}' loaded (Part 1, at origin).")
    print(f"File '{path_to_file_2}' loaded (Part 2) and translated by ({TRANSLATION_X}, {TRANSLATION_Y}, {TRANSLATION_Z}).")
    print(f"Part 2 also rotated by {ROTATION_ANGLE_DEG} degrees around the Z-axis.")


    # 3. Calculate Properties (Example)
    prop = GProp_GProps()
    brepgprop_VolumeProperties(part1_base, prop, 1e-5) 
    volume = prop.Mass()
    print(f"Calculated Volume for Part 1: {volume}")

    # 4. Visualization and Export
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Display both parts separately with different colors to form the assembly view
    display.DisplayShape(part1_base, update=False, color=PART1_COLOR) 
    display.DisplayShape(part2_transformed, update=True, color=PART2_COLOR)

    # Adjusts the camera to show everything
    display.FitAll()
    
    # Export the current view to an image file
    try:
        display.ExportToImage(output_image_file)
        print(f"Successfully exported assembly view to '{output_image_file}'")
    except Exception as e:
        print(f"Error exporting image: {e}")
        
    # 5. Start the interactive viewer loop
    print("\nStarting interactive viewer. Close the window to exit the program.")
    start_display() 

if __name__ == '__main__':
    create_and_display_assembly()