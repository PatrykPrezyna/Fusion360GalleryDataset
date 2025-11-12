import json
import math
from OCC.Extend.DataExchange import read_step_file
from OCC.Display.SimpleGui import init_display
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.gp import gp_Dir, gp_Trsf, gp_Vec, gp_Ax1, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from typing import Dict, Any, Tuple

# --- Configuration ---
ASSEMBLY_JSON_FILE = 'input_data/assembly.json'
OUTPUT_IMAGE_FILE = "output_data/full_json_assembly_view.png"
ROOT_COLOR = 'RED'
PART_COLORS = ['GREEN', 'BLUE', 'YELLOW', 'CYAN', 'MAGENTA'] 

# --- Data Loading and Helpers (Unchanged) ---

def load_assembly_data(json_path: str) -> Dict[str, Any] | None:
    """Loads the assembly structure from the specified JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading assembly file '{json_path}': {e}")
        return None

def calculate_mate_trsf(geom_data: Dict[str, Any]) -> Tuple[float, float, float, float, gp_Dir]:
    """
    Calculates the transformation (translation and zero rotation) required 
    to mate entity_two's min XZ corner to entity_one's XZ center.
    """
    # entity_one is the fixed part (Target)
    bbox1 = geom_data["entity_one"]["bounding_box"]
    target_x = (bbox1["max_point"]["x"] + bbox1["min_point"]["x"]) / 2.0
    target_z = (bbox1["max_point"]["z"] + bbox1["min_point"]["z"]) / 2.0
    target_y = bbox1["max_point"]["y"] 

    # entity_two is the part to be moved (Source)
    bbox2 = geom_data["entity_two"]["bounding_box"]
    source_x = bbox2["min_point"]["x"]
    source_y = bbox2["min_point"]["y"]
    source_z = bbox2["min_point"]["z"]

    # Calculate Translation Vector
    dx = target_x - source_x
    dy = target_y - source_y
    dz = target_z - source_z
    
    # Simple planar face-to-face mate, rotation is assumed to be 0 for this constraint.
    ROTATION_ANGLE_DEG = 0.0
    ROTATION_AXIS = gp_Dir(0, 0, 1) 

    return dx, dy, dz, ROTATION_ANGLE_DEG, ROTATION_AXIS

def position_part(shape, dx, dy, dz, rot_angle_deg, rot_axis):
    """Applies a translation and rotation to a given shape."""
    a_trsf = gp_Trsf()
    
    # 1. Apply Rotation (always applied before translation if we transform from origin)
    if rot_angle_deg != 0:
        rot_angle_rad = math.radians(rot_angle_deg)
        rot_axis_of_interest = gp_Ax1(gp_Pnt(0, 0, 0), rot_axis) 
        a_trsf.SetRotation(rot_axis_of_interest, rot_angle_rad) 
        
    # 2. Apply Translation
    translation_vector = gp_Vec(dx, dy, dz)
    a_trsf.SetTranslation(translation_vector) 
    
    transformed_shape = BRepBuilderAPI_Transform(shape, a_trsf).Shape()
    return transformed_shape

# --- Main Assembly Logic (Viewer3d Fix Applied) ---

def create_and_display_assembly():
    
    assembly_data = load_assembly_data(ASSEMBLY_JSON_FILE)
    if not assembly_data:
        return

    # 1. Identify all bodies and their STEP files
    component_id = assembly_data.get('root', {}).get('component')
    if not component_id:
        print("Error: Root component ID is missing.")
        return

    all_body_ids = assembly_data.get('components', {}).get(component_id, {}).get('bodies', [])
    
    if not all_body_ids:
        print("Error: No bodies found in the root component.")
        return

    # Map Body ID to STEP file path and load all parts
    loaded_shapes = {}
    body_id_to_path = {
        uid: data.get("step") for uid, data in assembly_data.get("bodies", {}).items()
    }
    
    for uid in all_body_ids:
        step_path = body_id_to_path.get(uid)
        if step_path:
            # Assumes step files are in the same folder as the script/json
            shape = read_step_file("input_data/" + step_path) 
            if shape:
                loaded_shapes[uid] = shape
            else:
                print(f"Warning: Failed to load STEP file for body ID {uid} ({step_path}). Skipping.")

    if not loaded_shapes:
        print("Error: No shapes successfully loaded.")
        return

    # 2. Initialize Assembly State
    root_body_id = all_body_ids[0]
    final_assembly_shapes = {root_body_id: loaded_shapes.get(root_body_id)}
    unpositioned_bodies = set(all_body_ids[1:])
    
    print(f"Root body **{root_body_id}** is fixed at the origin.")

    # 3. Iteratively apply contacts until all desired parts are positioned
    
    contacts = assembly_data.get("contacts", [])
    iteration = 0
    
    while unpositioned_bodies and iteration < len(contacts) * 2: 
        
        bodies_to_position_in_this_pass = set()

        for contact in contacts:
            body1_id = contact["entity_one"]["body"]
            body2_id = contact["entity_two"]["body"]
            
            fixed_id, moving_id = None, None
            mate_data = None
            
            # Case 1: Body 1 is fixed, Body 2 is moving
            if body1_id in final_assembly_shapes and body2_id in unpositioned_bodies:
                fixed_id = body1_id
                moving_id = body2_id
                mate_data = {
                    "entity_one": contact["entity_one"], # Target Face (Fixed Part)
                    "entity_two": contact["entity_two"]  # Source Face (Moving Part)
                }
                
            # Case 2: Body 2 is fixed, Body 1 is moving
            elif body2_id in final_assembly_shapes and body1_id in unpositioned_bodies:
                fixed_id = body2_id
                moving_id = body1_id
                mate_data = {
                    "entity_one": contact["entity_two"], # Target Face (Fixed Part)
                    "entity_two": contact["entity_one"]  # Source Face (Moving Part)
                }
            
            if moving_id and moving_id in loaded_shapes and mate_data:
                
                # a. Calculate the required transformation
                dx, dy, dz, rot_angle_deg, rot_axis = calculate_mate_trsf(mate_data)
                
                # b. Apply the transformation to the moving part's base shape
                transformed_shape = position_part(
                    loaded_shapes[moving_id], 
                    dx, dy, dz, 
                    rot_angle_deg, 
                    rot_axis
                )
                
                # c. Update assembly state
                final_assembly_shapes[moving_id] = transformed_shape
                bodies_to_position_in_this_pass.add(moving_id)
                
                print(f"Mate applied: **{moving_id}** -> **{fixed_id}**. Translation: ({dx:.2f}, {dy:.2f}, {dz:.2f})")
                
        # Update the list of unpositioned bodies for the next pass
        unpositioned_bodies -= bodies_to_position_in_this_pass
        
        if not bodies_to_position_in_this_pass and unpositioned_bodies:
            print("\nWarning: Assembly stalled. Unable to find a contact to position the remaining bodies.")
            break
            
        iteration += 1

    # 4. Visualization and Export
    display, start_display, _, _ = init_display()

    print("\n--- Displaying Assembly ---")
    color_index = 0
    
    for uid, shape in final_assembly_shapes.items():
        color = ROOT_COLOR if uid == root_body_id else PART_COLORS[color_index % len(PART_COLORS)]
        
        # --- FIX: Removed the unsupported 'label' argument ---
        display.DisplayShape(shape, update=False, color=color) 
        
        color_index += 1

    display.FitAll()
    
    try:
        display.ExportToImage(OUTPUT_IMAGE_FILE)
        print(f"Successfully exported assembly view to '{OUTPUT_IMAGE_FILE}'")
    except Exception:
        pass 
        
    # 5. Start the interactive viewer loop
    print("\nStarting interactive viewer. Close the window to exit the program.")
    start_display() 

if __name__ == '__main__':
    create_and_display_assembly()