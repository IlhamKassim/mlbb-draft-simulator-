"""
Data loading utilities for the MLBB Draft Simulator.

This module handles loading and caching of game data from JSON files.
"""
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from flask import Flask, current_app
from .errors import DataLoadError

class DataLoader:
    """Handles loading and caching of game data."""
    
    _hero_data: Optional[Dict[str, Any]] = None
    _hero_roles: Optional[Dict[str, List[str]]] = None
    _item_builds: Optional[Dict[str, Any]] = None
    _pro_stats: Optional[Dict[str, Any]] = None
    _last_load_time: Optional[datetime] = None
    
    @classmethod
    def initialize(cls, app: Flask) -> None:
        """Initialize the data loader with the application context.
        
        Args:
            app: Flask application instance.
        """
        try:
            cls._load_all_data(app)
        except Exception as e:
            current_app.logger.error(f"Failed to initialize DataLoader: {e}")
            raise DataLoadError(f"Failed to initialize data: {e}")
    
    @classmethod
    def _load_all_data(cls, app: Flask) -> None:
        """Load all game data from JSON files.
        
        Args:
            app: Flask application instance.
        
        Raises:
            DataLoadError: If any required data file cannot be loaded.
        """
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'static', 'data')
            
            # Load hero data
            hero_data_path = os.path.join(data_dir, 'hero_data.json')
            with open(hero_data_path, 'r', encoding='utf-8') as f:
                cls._hero_data = json.load(f)
            
            # Load hero roles
            hero_roles_path = os.path.join(data_dir, 'hero_roles.json')
            with open(hero_roles_path, 'r', encoding='utf-8') as f:
                cls._hero_roles = json.load(f)
            
            # Load item builds
            item_builds_path = os.path.join(data_dir, 'hero_item_builds.json')
            with open(item_builds_path, 'r', encoding='utf-8') as f:
                cls._item_builds = json.load(f)
            
            # Load pro stats
            pro_stats_path = os.path.join(data_dir, 'pro_stats.json')
            with open(pro_stats_path, 'r', encoding='utf-8') as f:
                cls._pro_stats = json.load(f)
            
            cls._last_load_time = datetime.now()
            
            app.logger.info('Successfully loaded all game data')
        except Exception as e:
            app.logger.error(f'Error loading game data: {e}')
            raise DataLoadError(f'Failed to load game data: {e}')
    
    @staticmethod
    def _load_json_file(filepath: str) -> Dict[str, Any]:
        """Load and parse a JSON file.
        
        Args:
            filepath: Path to the JSON file.
            
        Returns:
            Parsed JSON data as a dictionary.
            
        Raises:
            DataLoadError: If the file cannot be read or parsed.
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise DataLoadError(f'Data file not found: {filepath}')
        except json.JSONDecodeError:
            raise DataLoadError(f'Invalid JSON in file: {filepath}')
    
    @classmethod
    def _check_reload_needed(cls) -> bool:
        """Check if data needs to be reloaded based on file modifications.
        
        Returns:
            True if any data file has been modified since last load.
        """
        if not cls._last_load_time:
            return True
            
        data_dir = current_app.config['DATA_DIR']
        files_to_check = [
            current_app.config['HERO_DATA_FILE'],
            current_app.config['ITEM_BUILDS_FILE'],
            current_app.config['PRO_STATS_FILE']
        ]
        
        for filename in files_to_check:
            filepath = os.path.join(data_dir, filename)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if mtime > cls._last_load_time:
                    return True
            except OSError:
                current_app.logger.warning(f'Could not check mtime for {filepath}')
                return True
        
        return False
    
    @classmethod
    def get_hero_data(cls) -> Dict[str, Any]:
        """Get hero data, reloading if necessary.
        
        Returns:
            Dictionary containing hero data.
        """
        if cls._check_reload_needed():
            cls._load_all_data(current_app)
        return cls._hero_data or {}
    
    @classmethod
    def get_hero_roles(cls) -> Dict[str, List[str]]:
        """Get hero roles, reloading if necessary.
        
        Returns:
            Dictionary containing hero roles.
        """
        if cls._check_reload_needed():
            cls._load_all_data(current_app)
        return cls._hero_roles or {}
    
    @classmethod
    def get_item_builds(cls) -> Dict[str, Any]:
        """Get item build data, reloading if necessary.
        
        Returns:
            Dictionary containing item build data.
        """
        if cls._check_reload_needed():
            cls._load_all_data(current_app)
        return cls._item_builds or {}
    
    @classmethod
    def get_pro_stats(cls) -> Dict[str, Any]:
        """Get pro statistics data, reloading if necessary.
        
        Returns:
            Dictionary containing pro statistics data.
        """
        if cls._check_reload_needed():
            cls._load_all_data(current_app)
        return cls._pro_stats or {}

    @classmethod
    def load_hero_data(cls) -> Dict[str, Any]:
        """Load hero data from JSON file.
        
        Returns:
            dict: Hero data dictionary
        """
        if not cls._hero_data:
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                   'static', 'data', 'hero_data.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cls._hero_data = json.load(f)
            except Exception as e:
                print(f"Error loading hero data: {e}")
                cls._hero_data = {}
        
        return cls._hero_data
    
    @classmethod
    def get_heroes_by_role(cls, role: str) -> Dict[str, Any]:
        """Get heroes filtered by role.
        
        Args:
            role: Hero role to filter by
            
        Returns:
            dict: Filtered hero data
        """
        heroes = cls.load_hero_data()
        return {
            name: data for name, data in heroes.items()
            if data.get('role', '') == role
        }
    
    @classmethod
    def search_heroes(cls, query: str, role: str = None) -> Dict[str, Any]:
        """Search heroes by name and optionally filter by role.
        
        Args:
            query: Search query string
            role: Optional role filter
            
        Returns:
            dict: Filtered hero data
        """
        heroes = cls.load_hero_data()
        filtered_heroes = {}
        
        for name, data in heroes.items():
            if query.lower() in name.lower():
                if not role or data.get('role', '').lower() == role.lower():
                    filtered_heroes[name] = data
        
        return filtered_heroes 