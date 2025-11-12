import os
import json
from PIL import Image

def check_conditions(assembly_json_path, category="Mechanical Engineering", max_bodies=3):
    try:
        with open(assembly_json_path, 'r') as f:
            data = json.load(f)
        categories = data.get('properties', {}).get('categories', [])
        body_count = data.get('properties', {}).get('body_count', 0)
        if category in categories and body_count < max_bodies:
            return True
    except Exception as e:
        print(f"Error reading {assembly_json_path}: {e}")
    return False

def find_assemblies(root_dir, category="Mechanical Engineering", max_bodies=3):
    assembly_pngs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "assembly.json" in filenames:
            json_path = os.path.join(dirpath, "assembly.json")
            if check_conditions(json_path, category, max_bodies):
                png_path = os.path.join(dirpath, "assembly.png")
                if os.path.exists(png_path):
                    assembly_pngs.append(png_path)
    return assembly_pngs

def create_composite(images_paths, output_path):
    images = [Image.open(p) for p in images_paths]
    if not images:
        print("No images to display.")
        return
    # Find max width and height for the grid (let's use up to 4 per row)
    per_row = 4
    thumb_width = max([img.width for img in images])
    thumb_height = max([img.height for img in images])
    grid_rows = (len(images) + per_row - 1) // per_row
    # Create a blank canvas
    composite = Image.new('RGBA', (per_row * thumb_width, grid_rows * thumb_height), (255, 255, 255, 0))
    for idx, img in enumerate(images):
        row = idx // per_row
        col = idx % per_row
        x = col * thumb_width
        y = row * thumb_height
        composite.paste(img, (x, y))
    composite.save(output_path)
    #composite.show()

if __name__ == "__main__":
    root_dir = "../a1.0.0_02"      # Change to your directory
    output_path = "composite_2.png"
    # see../docs/tags.json for all categories
    assembly_images = find_assemblies(root_dir, category="Mechanical Engineering", max_bodies=4)
    create_composite(assembly_images, output_path)
