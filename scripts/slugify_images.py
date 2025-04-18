#!/usr/bin/env python3
"""
Script to rename hero images using slugified names.
This ensures consistent naming across the application.
"""
import os
import re
from pathlib import Path

def slugify(text: str) -> str:
    """Convert text to lowercase, replace special chars with underscores."""
    # Convert to lowercase and replace spaces/special chars with underscore
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    return text

def rename_images(img_dir: str) -> None:
    """Rename all PNG files in the directory using slugified names."""
    img_path = Path(img_dir)
    if not img_path.exists():
        print(f"Error: Directory {img_dir} does not exist")
        return

    # Get all PNG files
    png_files = list(img_path.glob('*.png'))
    print(f"Found {len(png_files)} PNG files")

    # Process each file
    for file_path in png_files:
        # Get original name without extension
        original_name = file_path.stem
        # Create new slugified name
        new_name = slugify(original_name) + '.png'
        new_path = file_path.parent / new_name

        # Skip if file already has correct name
        if file_path.name == new_name:
            print(f"Skipping {file_path.name} (already correct)")
            continue

        # Rename file
        try:
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {file_path.name}: {e}")

if __name__ == '__main__':
    # Get the absolute path to the static/img directory
    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / 'static' / 'img'
    
    print(f"Processing images in: {img_dir}")
    rename_images(str(img_dir))
    print("Done!") 