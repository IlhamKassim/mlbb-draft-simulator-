import os
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_item_image(item_name, output_path):
    # Create a 100x100 image with a dark background
    img = Image.new('RGB', (100, 100), '#2a3142')
    draw = ImageDraw.Draw(img)
    
    # Draw a gold border
    draw.rectangle([(0, 0), (99, 99)], outline='#c8aa6e', width=2)
    
    # Add item name as text
    wrapped_text = textwrap.fill(item_name, width=10)
    
    try:
        # Try to use a better font if available
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Calculate text size and position
    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (100 - text_width) // 2
    y = (100 - text_height) // 2
    
    # Draw text in gold color
    draw.text((x, y), wrapped_text, fill='#c8aa6e', font=font)
    
    # Save the image
    img.save(output_path)

def main():
    # Create items directory if it doesn't exist
    items_dir = 'static/img/items'
    os.makedirs(items_dir, exist_ok=True)
    
    # Read item builds
    with open('static/data/hero_item_builds.json', 'r') as f:
        builds = json.load(f)
    
    # Collect unique items
    unique_items = set()
    for hero in builds.values():
        for build_type in hero.values():
            unique_items.update(build_type)
    
    # Create placeholder image for each item
    for item in unique_items:
        filename = item.lower().replace("'", "").replace(" ", "_") + '.png'
        output_path = os.path.join(items_dir, filename)
        if not os.path.exists(output_path):  # Only create if doesn't exist
            create_item_image(item, output_path)
            print(f"Created placeholder for: {item}")
        else:
            print(f"Skipped existing placeholder for: {item}")

if __name__ == '__main__':
    main() 