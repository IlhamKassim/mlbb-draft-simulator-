import os
import requests
from bs4 import BeautifulSoup
import json

def clean_item_name(item_name):
    """Convert item name to wiki format"""
    return item_name.lower().replace("'", "").replace(" ", "_")

def download_image(url, output_path):
    """Download image from URL and save to output path"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {output_path}")
            return True
        else:
            print(f"Failed to download {url}: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create items directory if it doesn't exist
    items_dir = 'static/img/items'
    os.makedirs(items_dir, exist_ok=True)
    
    # Read item builds to get list of items
    with open('static/data/hero_item_builds.json', 'r') as f:
        builds = json.load(f)
    
    # Collect unique items
    unique_items = set()
    for hero in builds.values():
        for build_type in hero.values():
            unique_items.update(build_type)
    
    # Base URL for Mobile Legends Wiki
    base_url = "https://mobile-legends.fandom.com/wiki/"
    
    # Download each item's image
    for item in unique_items:
        clean_name = clean_item_name(item)
        output_path = os.path.join(items_dir, f"{clean_name}.png")
        
        # Skip if image already exists
        if os.path.exists(output_path):
            print(f"Skipped existing image: {item}")
            continue
        
        # Try to get the item page
        try:
            page_url = base_url + item.replace(" ", "_")
            response = requests.get(page_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find the item image (usually in the infobox)
                img_tag = soup.find('img', {'class': 'pi-image-thumbnail'})
                if img_tag and img_tag.get('src'):
                    img_url = img_tag['src']
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    download_image(img_url, output_path)
                else:
                    print(f"Could not find image for: {item}")
            else:
                print(f"Failed to get page for: {item}")
        except Exception as e:
            print(f"Error processing {item}: {str(e)}")

if __name__ == '__main__':
    main() 