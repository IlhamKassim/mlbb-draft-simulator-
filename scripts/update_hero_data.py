"""
Automated hero data update script.
Fetches and validates hero data from multiple sources.
"""
import json
import logging
import os
import requests
import time
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def send_notification(self, title: str, message: str, color: int = 0x00ff00):
        """Send a Discord notification via webhook."""
        try:
            data = {
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            requests.post(self.webhook_url, json=data)
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

class HeroDataUpdater:
    def __init__(self, data_dir: str, discord_webhook: str = None):
        self.data_dir = Path(data_dir)
        self.api_base_url = "https://api.mobalytics.gg/mlbb/v1"
        self.backup_sources = [
            "https://mobile-legends.fandom.com/api.php",
            "https://mlbb.fandom.com/api.php"
        ]
        self.notifier = DiscordNotifier(discord_webhook) if discord_webhook else None
        
        # Pre-defined hero roles and specialties for better data consistency
        self.valid_roles = {'Tank', 'Fighter', 'Assassin', 'Mage', 'Marksman', 'Support'}
        self.valid_specialties = {
            'Burst', 'Charge', 'Chase', 'Control', 'Damage', 
            'Guard', 'Initiator', 'Push', 'Poke', 'Reap', 'Sustain'
        }
        self.valid_difficulties = {'Easy', 'Normal', 'Hard'}
        
        # Create necessary directories if they don't exist
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure hero-icons directory exists
        hero_icons_dir = Path(self.data_dir.parent, "hero-icons")
        hero_icons_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Hero icons directory: {hero_icons_dir}")
        
    def notify(self, title: str, message: str, is_error: bool = False):
        """Send notification if Discord webhook is configured."""
        if self.notifier:
            self.notifier.send_notification(title, message, 0xff0000 if is_error else 0x00ff00)

    def update_all(self):
        """Update all hero-related data files."""
        try:
            # Fetch new data
            hero_data = self.fetch_hero_data()
            hero_roles = self.extract_hero_roles(hero_data)
            
            # Validate data
            if not self.validate_data(hero_data, hero_roles):
                error_msg = "Data validation failed"
                logger.error(error_msg)
                self.notify("Hero Data Update Failed", error_msg, True)
                return False
                
            # Backup existing files
            self.backup_existing_files()
            
            # Save new data
            self.save_json(hero_data, 'hero_data.json')
            self.save_json(hero_roles, 'hero_roles.json')
            
            # Download hero icons
            self.download_hero_icons(hero_data)
            
            # Send success notification with stats
            success_msg = f"Updated {len(hero_data)} heroes\nRoles updated: {len(hero_roles)}"
            logger.info(success_msg)
            self.notify("Hero Data Update Success", success_msg)
            return True
            
        except Exception as e:
            error_msg = f"Update failed: {str(e)}"
            logger.error(error_msg)
            self.notify("Hero Data Update Error", error_msg, True)
            return False
            
    def fetch_hero_data(self) -> Dict[str, Any]:
        """Fetch hero data from primary and backup sources."""
        # Try to load hero data from local file first (for debugging and as fallback)
        try:
            local_file = self.data_dir / "hero_data.json"
            if local_file.exists():
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data and len(data) > 5:  # Basic sanity check
                        logger.info(f"Loaded {len(data)} heroes from local file")
                        return data
        except Exception as e:
            logger.warning(f"Failed to load local hero data: {str(e)}")
        
        # Manually defined base hero data as a last resort
        base_heroes = {
            "Layla": {"name": "Layla", "roles": ["Marksman"], "specialty": "Damage", "difficulty": "Easy"},
            "Miya": {"name": "Miya", "roles": ["Marksman"], "specialty": "Damage", "difficulty": "Easy"},
            "Tigreal": {"name": "Tigreal", "roles": ["Tank"], "specialty": "Control", "difficulty": "Easy"},
            "Alucard": {"name": "Alucard", "roles": ["Fighter"], "specialty": "Chase", "difficulty": "Normal"},
            "Nana": {"name": "Nana", "roles": ["Mage", "Support"], "specialty": "Control", "difficulty": "Easy"},
            "Eudora": {"name": "Eudora", "roles": ["Mage"], "specialty": "Burst", "difficulty": "Easy"},
            "Zilong": {"name": "Zilong", "roles": ["Fighter", "Assassin"], "specialty": "Chase", "difficulty": "Easy"},
            "Franco": {"name": "Franco", "roles": ["Tank"], "specialty": "Control", "difficulty": "Normal"},
            "Gord": {"name": "Gord", "roles": ["Mage"], "specialty": "Poke", "difficulty": "Normal"},
            "Karina": {"name": "Karina", "roles": ["Assassin"], "specialty": "Reap", "difficulty": "Easy"},
            "Akai": {"name": "Akai", "roles": ["Tank"], "specialty": "Control", "difficulty": "Normal"},
            "Saber": {"name": "Saber", "roles": ["Assassin"], "specialty": "Burst", "difficulty": "Normal"},
            "Balmond": {"name": "Balmond", "roles": ["Fighter"], "specialty": "Damage", "difficulty": "Easy"},
            "Bruno": {"name": "Bruno", "roles": ["Marksman"], "specialty": "Damage", "difficulty": "Normal"},
            "Fanny": {"name": "Fanny", "roles": ["Assassin"], "specialty": "Charge", "difficulty": "Hard"}
        }
        
        # Add descriptions to base heroes
        for name, hero in base_heroes.items():
            hero["description"] = f"{name} is a {'/'.join(hero['roles'])} hero with {hero['specialty']} specialty."
        
        # Try to augment base data with data from API sources
        try:
            augmented = self.fetch_from_api(base_heroes)
            if augmented and len(augmented) > len(base_heroes):
                return augmented
        except Exception as e:
            logger.warning(f"API augmentation failed: {e}")
            
        # Return base heroes as a last resort
        logger.warning(f"Using base hero data with {len(base_heroes)} heroes as fallback")
        return base_heroes
        
    def fetch_from_api(self, base_heroes: Dict[str, Any]) -> Dict[str, Any]:
        """Try to fetch hero data from public APIs to augment base data"""
        # Potential API endpoints - these are placeholders and would need to be adjusted
        # for real APIs with proper authentication
        api_endpoints = [
            "https://api.mobalytics.gg/mlbb/v1/heroes",  # Example API
            "https://api.mlbbmeta.com/heroes",           # Example API
            "https://mlbb-api.herokuapp.com/heroes"      # Example API
        ]
        
        for endpoint in api_endpoints:
            try:
                logger.info(f"Trying API endpoint: {endpoint}")
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        # Convert list response to our dictionary format
                        heroes = {}
                        for hero in data:
                            if 'name' in hero:
                                name = hero['name']
                                heroes[name] = {
                                    "name": name,
                                    "roles": hero.get('roles', []),
                                    "specialty": hero.get('specialty', 'Damage'),
                                    "difficulty": hero.get('difficulty', 'Normal'),
                                    "description": hero.get('description', f"{name} is a powerful hero in Mobile Legends."),
                                    "last_update": time.strftime('%Y-%m-%d')
                                }
                        
                        # If we got a substantial number of heroes, use this data
                        if len(heroes) > 20:  # Arbitrary threshold
                            logger.info(f"Fetched {len(heroes)} heroes from {endpoint}")
                            return heroes
                            
                    elif isinstance(data, dict) and len(data) > 0:
                        # Already in dictionary format, validate and use
                        if all('name' in hero for hero in data.values()):
                            logger.info(f"Fetched {len(data)} heroes from {endpoint}")
                            # Add last_update field
                            for hero in data.values():
                                hero["last_update"] = time.strftime('%Y-%m-%d')
                            return data
            except Exception as e:
                logger.warning(f"API endpoint {endpoint} failed: {e}")
                
        # If all APIs failed, augment base heroes with publicly available data
        # This section would implement scraping or other data collection methods
        # For now, just return the base heroes
        return base_heroes
        
    def validate_data(self, hero_data: Dict[str, Any], hero_roles: Dict[str, List[str]]) -> bool:
        """Validate the structure and content of hero data."""
        if not hero_data or not hero_roles:
            logger.error("Hero data or roles is empty")
            return False
            
        # Check required fields
        required_fields = ['name', 'roles', 'specialty', 'difficulty']
        for hero_name, hero in hero_data.items():
            missing_fields = [field for field in required_fields if field not in hero]
            if missing_fields:
                logger.error(f"Hero {hero_name} is missing required fields: {missing_fields}")
                return False
                
        # Validate roles
        for hero_name, roles in hero_roles.items():
            invalid_roles = [r for r in roles if r not in self.valid_roles]
            if invalid_roles:
                logger.warning(f"Hero {hero_name} has invalid roles: {invalid_roles}")
                # Fix invalid roles instead of failing
                hero_roles[hero_name] = [r for r in roles if r in self.valid_roles]
                if not hero_roles[hero_name]:
                    hero_roles[hero_name] = ["Fighter"]  # Default role
                
        # Add role if missing
        for hero_name, hero in hero_data.items():
            if not hero.get('roles'):
                logger.warning(f"Hero {hero_name} has no roles, setting default")
                hero['roles'] = ["Fighter"]
                
        return True
        
    def backup_existing_files(self):
        """Create backups of existing data files."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        for filename in ['hero_data.json', 'hero_roles.json']:
            src = self.data_dir / filename
            if src.exists():
                dst = self.data_dir / f"{filename}.{timestamp}.bak"
                src.rename(dst)
                logger.info(f"Backed up {src} to {dst}")
                
    def save_json(self, data: Dict[str, Any], filename: str):
        """Save data to JSON file."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} items to {filepath}")
            
    def extract_hero_roles(self, hero_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract role information from hero data."""
        roles = {}
        for hero_name, hero_info in hero_data.items():
            roles[hero_name] = hero_info.get('roles', [])
        return roles

    def download_hero_icons(self, hero_data: Dict[str, Any]):
        """Download hero icons for all heroes in the data."""
        hero_icons_dir = Path(self.data_dir.parent, "hero-icons")
        hero_icons_dir.mkdir(parents=True, exist_ok=True)
        
        # Track success/failure stats
        success_count = 0
        failure_count = 0
        
        for hero_name in hero_data.keys():
            try:
                # Create a URL-friendly version of hero name
                safe_name = self.get_safe_filename(hero_name)
                file_path = hero_icons_dir / f"{safe_name}.png"
                
                # Skip if the icon already exists
                if file_path.exists():
                    logger.info(f"Icon for {hero_name} already exists at {file_path}")
                    success_count += 1
                    continue
                    
                # Try to download the icon from multiple sources
                if self.try_download_icon(hero_name, file_path):
                    success_count += 1
                else:
                    # Create a placeholder SVG as fallback
                    self.create_placeholder_svg(hero_name, hero_data[hero_name].get('roles', []), 
                                              hero_icons_dir / f"{safe_name}.svg")
                    failure_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to download icon for {hero_name}: {e}")
                failure_count += 1
                
        logger.info(f"Hero icons: {success_count} downloaded, {failure_count} placeholders created")
        
    def try_download_icon(self, hero_name: str, output_path: Path) -> bool:
        """Try to download hero icon from multiple sources."""
        sources = [
            f"https://mobalytics.gg/mlbb/heroes/{hero_name.lower().replace(' ', '-')}/icon.png",
            f"https://mobile-legends.net/hero-icons/{hero_name.lower().replace(' ', '-')}.png",
            f"https://mlbb-heroes.com/images/heroes/{hero_name.lower().replace(' ', '_')}.png"
        ]
        
        for source_url in sources:
            try:
                logger.info(f"Trying to download {hero_name} icon from {source_url}")
                response = requests.get(source_url, timeout=10, stream=True)
                if response.status_code == 200:
                    # Check if it's actually an image
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                        logger.info(f"Downloaded {hero_name} icon from {source_url}")
                        return True
            except Exception as e:
                logger.warning(f"Failed to download {hero_name} from {source_url}: {e}")
                
        logger.warning(f"Could not download icon for {hero_name} from any source")
        return False
        
    def create_placeholder_svg(self, hero_name: str, roles: List[str], output_path: Path):
        """Create a placeholder SVG with hero name and role."""
        # Choose color based on primary role
        role_colors = {
            "Tank": "#3F51B5",      # Blue
            "Fighter": "#F44336",   # Red
            "Assassin": "#9C27B0",  # Purple
            "Mage": "#00BCD4",      # Cyan
            "Marksman": "#FF9800",  # Orange
            "Support": "#4CAF50"    # Green
        }
        
        primary_role = roles[0] if roles else "Fighter"
        color = role_colors.get(primary_role, "#757575")
        
        # Create SVG with hero name and role
        svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <rect width="256" height="256" fill="{color}40" rx="20" ry="20"/>
  <text x="128" y="120" font-family="Arial, sans-serif" font-size="24" text-anchor="middle" fill="#212121">{hero_name}</text>
  <text x="128" y="150" font-family="Arial, sans-serif" font-size="20" text-anchor="middle" fill="#212121">{primary_role}</text>
  <circle cx="128" cy="60" r="30" fill="{color}" opacity="0.8"/>
</svg>
"""
        
        # Save SVG file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg)
        logger.info(f"Created placeholder SVG for {hero_name} at {output_path}")
        
    def get_safe_filename(self, name: str) -> str:
        """Convert hero name to a safe filename format."""
        # Remove any character not a letter, digit, dash, or underscore
        safe_name = re.sub(r'[^\w\-]', '-', name.lower())
        # Remove multiple consecutive dashes
        safe_name = re.sub(r'--+', '-', safe_name)
        # Remove leading or trailing dashes
        safe_name = safe_name.strip('-')
        return safe_name
        
if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'static' / 'data'
    
    # Get Discord webhook URL from environment variable
    discord_webhook = os.getenv('MLBB_DISCORD_WEBHOOK')
    
    updater = HeroDataUpdater(data_dir, discord_webhook)
    if updater.update_all():
        logger.info("Hero data updated successfully")
    else:
        logger.error("Failed to update hero data")
        
    # Make sure to run the download_hero_icons function separately in case
    # the main update fails but we still want to try getting icons
    try:
        logger.info("Starting hero icon download")
        with open(data_dir / 'hero_data.json', 'r', encoding='utf-8') as f:
            hero_data = json.load(f)
        updater.download_hero_icons(hero_data)
    except Exception as e:
        logger.error(f"Failed to download hero icons: {e}")