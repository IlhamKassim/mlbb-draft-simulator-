#!/usr/bin/env python3
"""
MLBB Hero Icon Downloader
Downloads official hero icons from various sources for the MLBB Draft Simulator.
"""

import os
import sys
import time
import requests
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('hero_icon_downloader')

class MLBBHeroIconDownloader:
    def __init__(self, output_dir='static/hero-icons', size=(256, 256)):
        """Initialize the hero icon downloader
        
        Args:
            output_dir: Directory to save hero icons
            size: Target size for icons (width, height)
        """
        self.output_dir = Path(output_dir)
        self.size = size
        self.success_count = 0
        self.failed_heroes = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Known MLBB Sources (multiple patterns tried for each hero)
        self.sources = [
            # Mobile Legends Wiki - Hero portraits 
            lambda name: f"https://static.wikia.nocookie.net/mobile-legends/images/thumb/{name[0].upper()}/{name[:2].lower()}/{name}/{name}_Portrait.png/256px-{name}_Portrait.png",
            
            # Mobile Legends Wiki - Hero icons
            lambda name: f"https://static.wikia.nocookie.net/mobile-legends/images/{name[0].upper()}/{name[:2].lower()}/{name}_icon.png",
            
            # Mobile Legends Official Site (pattern 1)
            lambda name: f"https://mlbb.fandom.com/wiki/Special:FilePath/{name}.png",
            
            # Mobile Legends Official Site (pattern 2)
            lambda name: f"https://mlbb.fandom.com/wiki/Special:FilePath/{name}_icon.png",
            
            # Mobile Legends Wiki - Alternative pattern
            lambda name: f"https://static.wikia.nocookie.net/mobile-legends/images/{name[0].upper()}/{name[:2].lower()}/{name}.jpg/revision/latest/scale-to-width-down/256",
           
            # MLBB Heroes Wiki
            lambda name: f"https://mlbbheroes.com/wp-content/uploads/{name.lower()}.png",
            
            # Alternative fandom site
            lambda name: f"https://mlbb.fandom.com/wiki/Special:FilePath/{name}_portrait.png",
        ]
        
    def normalize_name(self, name):
        """Convert hero name to lowercase and replace spaces with hyphens
        
        Args:
            name: Hero name
            
        Returns:
            Normalized filename
        """
        normalized = name.lower().replace(' ', '-')
        # Remove special characters except hyphens and alphanumerics
        normalized = re.sub(r'[^a-z0-9-]', '', normalized)
        return normalized
    
    def format_name_for_url(self, name):
        """Format hero name for URL patterns
        
        Args:
            name: Hero name
        
        Returns:
            URL-formatted name
        """
        # For URLs, typically replace spaces with underscores and maintain capitalization
        formatted = name.replace(' ', '_')
        return formatted
    
    def process_image(self, img):
        """Process image to standard size with transparent background
        
        Args:
            img: PIL Image object
        
        Returns:
            Processed PIL Image
        """
        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize to target size
        if img.size != self.size:
            img = img.resize(self.size, Image.LANCZOS)
        
        # If the image has a solid background color, make it transparent
        # This is a simplified approach - more advanced background removal would require
        # more complex algorithms
        return img
    
    def create_placeholder(self, hero_name, output_path):
        """Create a placeholder image for heroes that can't be downloaded
        
        Args:
            hero_name: Name of the hero
            output_path: Path to save the image
        """
        # Create a gradient background
        img = Image.new('RGBA', self.size, (240, 240, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw a circle with the hero's initial
        draw.ellipse([(40, 40), (216, 216)], fill=(100, 149, 237, 255), outline=(25, 25, 112, 255))
        
        # Try to use a nice font for the hero name
        try:
            font = ImageFont.truetype('Arial', 32)
        except:
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', 32)
            except:
                font = ImageFont.load_default()
        
        # Draw the hero name in the center
        name_to_display = hero_name
        if len(name_to_display) > 10:
            name_to_display = name_to_display[:10] + "..."
        
        draw.text(
            (128, 128),
            name_to_display,
            fill=(255, 255, 255, 255),
            font=font,
            anchor="mm"
        )
        
        # Save the placeholder
        img.save(output_path)
        logger.info(f"Created placeholder for {hero_name}")
    
    def download_hero_icon(self, hero_name, allow_placeholder=True):
        """Download hero icon from various sources
        
        Args:
            hero_name: Name of the hero
            allow_placeholder: Whether to create placeholder if download fails
        
        Returns:
            bool: Whether the download was successful
        """
        normalized_name = self.normalize_name(hero_name)
        formatted_name = self.format_name_for_url(hero_name)
        output_path = self.output_dir / f"{normalized_name}.png"
        
        logger.info(f"Downloading icon for {hero_name}")
        
        # Try each source
        for i, source_func in enumerate(self.sources):
            try:
                url = source_func(formatted_name)
                logger.info(f"Trying source {i+1}: {url}")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # Check if we got HTML instead of an image
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    logger.warning(f"Source {i+1} returned HTML instead of an image")
                    continue
                
                # Try to open the image
                try:
                    img = Image.open(BytesIO(response.content))
                    
                    # Process the image
                    img = self.process_image(img)
                    
                    # Save the image
                    img.save(output_path)
                    logger.info(f"✅ Successfully saved {hero_name} icon")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to process image from source {i+1}: {e}")
                    continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Source {i+1} request failed: {e}")
                continue
        
        # If all sources failed, create a placeholder
        if allow_placeholder:
            logger.warning(f"All sources failed for {hero_name}, creating placeholder")
            self.create_placeholder(hero_name, output_path)
        
        return False
    
    def download_all_heroes(self, hero_list):
        """Download icons for all heroes in the list
        
        Args:
            hero_list: List of hero names
        """
        self.success_count = 0
        self.failed_heroes = []
        
        for i, hero_name in enumerate(hero_list):
            logger.info(f"[{i+1}/{len(hero_list)}] Processing {hero_name}...")
            
            # Add delay to avoid rate limiting
            if i > 0:
                time.sleep(0.5)
            
            # Try to download
            success = self.download_hero_icon(hero_name)
            if success:
                self.success_count += 1
            else:
                self.failed_heroes.append(hero_name)
        
        self.print_report()
    
    def print_report(self):
        """Print report of download results"""
        logger.info("=" * 50)
        logger.info(f"Download completed: {self.success_count} heroes successful")
        
        if self.failed_heroes:
            logger.warning(f"Failed to download {len(self.failed_heroes)} heroes: {', '.join(self.failed_heroes)}")
            logger.warning("Placeholders were created for these heroes")
        else:
            logger.info("All hero icons were successfully downloaded!")
        
        logger.info("=" * 50)
        logger.info("Verifying files...")
        
        # Verify files exist
        all_files = list(self.output_dir.glob("*.png"))
        for i, filepath in enumerate(all_files):
            hero_name = filepath.stem
            file_size = filepath.stat().st_size / 1024  # KB
            logger.info(f"{i+1}. {hero_name}.png ({file_size:.1f} KB)")
        
        logger.info("=" * 50)
        logger.info(f"Total hero icons in directory: {len(all_files)}")
        logger.info(f"Icons are located in: {self.output_dir.absolute()}")

def get_heroes_from_api():
    """Get hero list from the API
    
    Returns:
        list: List of hero names
    """
    try:
        response = requests.get("http://localhost:8008/draft/heroes")
        if response.status_code == 200:
            heroes_data = response.json()
            return [hero["name"] for hero in heroes_data]
        else:
            logger.error(f"API returned status code {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Failed to fetch heroes from API: {e}")
        return []

def get_heroes_from_backup():
    """Backup hero list if API fails
    
    Returns:
        list: List of common MLBB heroes
    """
    return [
        "Alucard", "Balmond", "Eudora", "Fanny", "Franco", 
        "Gusion", "Hanzo", "Hayabusa", "Helcurt", "Lancelot",
        "Layla", "Ling", "Miya", "Nana", "Tigreal", "Zilong",
        "Lesley", "Gusion", "Chou", "Granger", "Alice", "Angela",
        "Selena", "Gord", "Aldous", "Pharsa", "Guinevere", "Lunox",
        "Esmeralda", "Kimmy", "Baxia", "Carmilla", "Cecilion", "Atlas"
    ]

if __name__ == "__main__":
    # Output directory - use absolute path for reliability
    base_dir = Path(__file__).parent.parent.absolute()
    output_dir = base_dir / "static" / "hero-icons"
    
    downloader = MLBBHeroIconDownloader(output_dir=output_dir)
    
    # Get hero list from API or use backup
    heroes = get_heroes_from_api()
    
    if not heroes:
        logger.warning("Could not get heroes from API, using backup list")
        heroes = get_heroes_from_backup()
    
    logger.info(f"Found {len(heroes)} heroes to download")
    
    # Start downloading
    downloader.download_all_heroes(heroes)
