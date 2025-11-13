import os
import json
from PIL import Image, ImageDraw, ImageFont

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
    assembly_info = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "assembly.json" in filenames:
            json_path = os.path.join(dirpath, "assembly.json")
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                categories = data.get('properties', {}).get('categories', [])
                body_count = data.get('properties', {}).get('body_count', 0)
            except:
                continue
            # if category in categories and body_count < max_bodies:
            if body_count == max_bodies and category in categories:
                png_path = os.path.join(dirpath, "assembly.png")
                if os.path.exists(png_path):
                    assembly_info.append({
                        "png_path": os.path.abspath(png_path),
                        "categories": categories,
                        "body_count": body_count
                    })
    return assembly_info

def create_composite(assemblies_info, output_path, info_path="picture_info.json", thumb_size=(128, 128)):
    images = [Image.open(info["png_path"]).convert("RGBA").resize(thumb_size, Image.LANCZOS) for info in assemblies_info]
    if not images:
        print("No images to display.")
        return
    per_row = 4
    thumb_width, thumb_height = thumb_size
    label_height = 40  # Room for text under each image
    grid_rows = (len(images) + per_row - 1) // per_row
    composite = Image.new('RGBA', (per_row * thumb_width, grid_rows * (thumb_height + label_height)), (255, 255, 255, 0))
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        index_font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        index_font = ImageFont.load_default()
        
    draw = ImageDraw.Draw(composite)
    picture_info = []
    for idx, (img, info) in enumerate(zip(images, assemblies_info)):
        row = idx // per_row
        col = idx % per_row
        x = col * thumb_width
        y = row * (thumb_height + label_height)
        composite.paste(img, (x, y))
        # Draw index at top-left (with slight padding)
        draw.rectangle([(x+2, y+2), (x+28, y+22)], fill=(255,255,255,180))
        draw.text((x+6, y+6), f"#{idx}", font=index_font, fill=(0,0,0))
        # Write category and body count below image
        label = f"{','.join(info['categories'])}\nBodies: {info['body_count']}"
        draw.multiline_text((x + 5, y + thumb_height + 5), label, font=font, fill=(0, 0, 0))
        picture_info.append({
            "index": idx,
            "row": row,
            "col": col,
            "file_path": info["png_path"],
            "categories": info["categories"],
            "body_count": info["body_count"]
        })
    composite.save(output_path)
    with open(info_path, 'w') as f:
        json.dump(picture_info, f, indent=2)


if __name__ == "__main__":
    root_dir = "input_data"
    output_path = "composite_annotated.png"
    info_path = "picture_info.json"
    category = "Mechanical Engineering"
    max_bodies = 2
    assemblies_info = find_assemblies(root_dir, category, max_bodies)
    create_composite(assemblies_info, output_path, info_path)
