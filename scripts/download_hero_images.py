import os
import requests
import json
from pathlib import Path
import re
from PIL import Image, ImageDraw
from io import BytesIO
from bs4 import BeautifulSoup

def create_directories():
    base_dir = Path(__file__).parent.parent
    img_dir = base_dir / 'static' / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)
    items_dir = img_dir / 'items'
    items_dir.mkdir(parents=True, exist_ok=True)
    return img_dir

def create_placeholder_image(path, size=(200, 200)):
    """Create a simple placeholder image"""
    img = Image.new('RGB', size, color='#2A2D3E')  # Dark blue-gray background
    draw = ImageDraw.Draw(img)
    # Draw border
    draw.rectangle([0, 0, size[0]-1, size[1]-1], outline='#6A6F8A', width=2)
    # Draw hero icon placeholder
    icon_size = min(size) // 2
    x1 = (size[0] - icon_size) // 2
    y1 = (size[1] - icon_size) // 2
    x2 = x1 + icon_size
    y2 = y1 + icon_size
    draw.ellipse([x1, y1, x2, y2], outline='#6A6F8A', width=2)
    # Draw cross lines
    draw.line([x1, y1, x2, y2], fill='#6A6F8A', width=2)
    draw.line([x2, y1, x1, y2], fill='#6A6F8A', width=2)
    img.save(path)

def load_hero_roles():
    base_dir = Path(__file__).parent.parent
    with open(base_dir / 'hero_roles.json', 'r') as f:
        return json.load(f)

def sanitize_filename(name):
    """Convert hero name to filename format"""
    return name.lower().replace("'", "").replace(" ", "_").replace(".", "").replace("-", "_")

def get_hero_image_url(hero_name):
    """Get image URL from Mobile Legends Game Guide"""
    # Special cases for heroes with different name formats
    name_mappings = {
        "Lou Yi": "Luo Yi",
        "Rynn": "Arlott",  # Rynn is also known as Arlott in some regions
        "X.Borg": "X-Borg",
        "Chang'e": "Change"
    }
    
    # Use mapped name if available
    search_name = name_mappings.get(hero_name, hero_name)
    
    # Format hero name for URL
    formatted_name = search_name.lower().replace(" ", "-").replace("'", "").replace(".", "")
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    try:
        # Try different variations of the hero name
        variations = [
            formatted_name,
            formatted_name.replace("-", ""),
            search_name.replace(" ", "").replace("'", "").replace("-", "").lower(),
            hero_name.replace(" ", "").replace("'", "").replace("-", "").lower()
        ]
        
        for name_variant in variations:
            # Try Mobile Legends Wiki
            guide_url = f"https://mobile-legends.fandom.com/wiki/{search_name.replace(' ', '_')}"
            response = requests.get(guide_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for the hero image in different locations
                img_tags = soup.select('img.pi-image-thumbnail, img.thumbimage, img[alt*="Portrait"], img[alt*="portrait"], img[alt*="Hero"]')
                
                for img in img_tags:
                    src = img.get('src', '')
                    if src and ('portrait' in src.lower() or 'hero' in src.lower()):
                        # Clean up the URL
                        if src.startswith('//'):
                            src = 'https:' + src
                        # Remove any scaling parameters
                        src = re.sub(r'/scale-to-width-down/\d+', '', src)
                        src = re.sub(r'/scale-to-height-down/\d+', '', src)
                        # Remove revision parameter
                        src = re.sub(r'\?cb=\d+', '', src)
                        return src
                        
            # Try alternative URLs if the first one fails
            alt_urls = [
                f"https://mlbb.fandom.com/wiki/{search_name.replace(' ', '_')}",
                f"https://mobilelegendsbangbang.fandom.com/wiki/{search_name.replace(' ', '_')}",
                f"https://mobile-legends.fandom.com/wiki/Category:{search_name.replace(' ', '_')}"
            ]
            
            for alt_url in alt_urls:
                try:
                    response = requests.get(alt_url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        img_tags = soup.select('img.pi-image-thumbnail, img.thumbimage, img[alt*="Portrait"], img[alt*="portrait"], img[alt*="Hero"]')
                        
                        for img in img_tags:
                            src = img.get('src', '')
                            if src and ('portrait' in src.lower() or 'hero' in src.lower()):
                                if src.startswith('//'):
                                    src = 'https:' + src
                                src = re.sub(r'/scale-to-width-down/\d+', '', src)
                                src = re.sub(r'/scale-to-height-down/\d+', '', src)
                                src = re.sub(r'\?cb=\d+', '', src)
                                return src
                except Exception as e:
                    print(f"Error checking alternative URL for {hero_name}: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"Error fetching hero page for {hero_name}: {str(e)}")
    
    return None

def download_hero_images():
    img_dir = create_directories()
    hero_roles = load_hero_roles()
    
    # Create placeholder images
    placeholder_path = img_dir / 'placeholder.png'
    items_placeholder_path = img_dir / 'items' / 'placeholder.png'
    
    if not placeholder_path.exists():
        print("Creating hero placeholder image...")
        create_placeholder_image(placeholder_path)
    
    if not items_placeholder_path.exists():
        print("Creating items placeholder image...")
        create_placeholder_image(items_placeholder_path, size=(100, 100))
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    # Collect all heroes
    all_heroes = []
    for role, heroes in hero_roles.items():
        all_heroes.extend(heroes)
    
    print(f"Found {len(all_heroes)} heroes")
    print("Starting download...")

    # Download each hero image
    for hero in all_heroes:
        filename = sanitize_filename(hero) + ".png"
        filepath = img_dir / filename
        
        if filepath.exists():
            print(f"Skipping {hero} - image already exists")
            continue
        
        print(f"Downloading {hero}...")
        
        # Try to get hero image
        image_url = get_hero_image_url(hero)
        if image_url:
            try:
                response = requests.get(image_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    # Convert image to PNG if needed
                    img = Image.open(BytesIO(response.content))
                    img.save(filepath, 'PNG')
                    print(f"Successfully downloaded {hero}")
                    continue
            except Exception as e:
                print(f"Error downloading {hero}: {str(e)}")
        
        print(f"Failed to download {hero}, using placeholder")
        # Copy placeholder for failed downloads
        if placeholder_path.exists():
            from shutil import copyfile
            copyfile(placeholder_path, filepath)

if __name__ == "__main__":
    download_hero_images()