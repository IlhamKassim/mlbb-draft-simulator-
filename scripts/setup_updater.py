#!/usr/bin/env python3
"""
Setup script for the MLBB hero data update mechanism.
Tests the updater and configures environment variables.
"""
import os
import sys
from pathlib import Path
from update_hero_data import HeroDataUpdater
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables for the updater."""
    webhook_url = input("Enter Discord webhook URL (press Enter to skip): ").strip()
    
    if webhook_url:
        # Create or update environment file
        env_path = Path(__file__).parent.parent / '.env'
        with open(env_path, 'a') as f:
            f.write(f"\nMLBB_DISCORD_WEBHOOK={webhook_url}\n")
        os.environ['MLBB_DISCORD_WEBHOOK'] = webhook_url
        logger.info("Discord webhook configured")
    else:
        logger.warning("Skipping Discord notifications setup")

def test_updater():
    """Test the hero data update mechanism."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'static' / 'data'
    
    # Create test instance with current environment
    webhook_url = os.getenv('MLBB_DISCORD_WEBHOOK')
    updater = HeroDataUpdater(data_dir, webhook_url)
    
    # Try updating hero data
    logger.info("Testing hero data update...")
    success = updater.update_all()
    
    if success:
        logger.info("✅ Update mechanism test successful")
        return True
    else:
        logger.error("❌ Update mechanism test failed")
        return False

def main():
    print("MLBB Hero Data Updater Setup")
    print("-" * 30)
    
    # Setup environment
    setup_environment()
    
    # Test the updater
    if test_updater():
        print("\nSetup completed successfully!")
        print("The updater will run daily at midnight.")
        print("You can also run it manually with: python scripts/update_hero_data.py")
    else:
        print("\nSetup encountered errors. Please check the logs and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()