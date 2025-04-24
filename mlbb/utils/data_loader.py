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

"""
Data loading utilities for the MLBB Draft Simulator.
"""
import json
import os
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from .errors import DataLoadError

class MLBBDataLoader:
    """Handles loading and processing of MLBB match data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.matches_df = None
        self.hero_stats = {}
        self.patch_stats = {}
    
    def load_matches(self, file_pattern: str = "**/*.csv") -> None:
        """Load match data from CSV files.
        
        Args:
            file_pattern: Glob pattern for match data files.
        """
        try:
            # Load all CSV files in data directory
            all_files = []
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith('.csv'):
                        all_files.append(os.path.join(root, file))
            
            if not all_files:
                raise DataLoadError(f"No CSV files found in {self.data_dir}")
            
            # Read and concatenate all files
            dfs = []
            for file in all_files:
                df = pd.read_csv(file)
                # Convert string lists to actual lists
                list_cols = ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']
                for col in list_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(eval)
                dfs.append(df)
            
            self.matches_df = pd.concat(dfs, ignore_index=True)
            
            # Compute patch-specific statistics
            self._compute_patch_stats()
            
        except Exception as e:
            raise DataLoadError(f"Failed to load match data: {e}")
    
    def _compute_patch_stats(self) -> None:
        """Compute hero statistics for each patch version."""
        if self.matches_df is None:
            return
        
        for patch in self.matches_df['patch_version'].unique():
            patch_matches = self.matches_df[self.matches_df['patch_version'] == patch]
            
            hero_stats = {}
            for hero in self._get_all_heroes():
                # Calculate pick rate
                blue_picks = sum(hero in picks for picks in patch_matches['blue_picks'])
                red_picks = sum(hero in picks for picks in patch_matches['red_picks'])
                total_picks = blue_picks + red_picks
                pick_rate = total_picks / len(patch_matches)
                
                # Calculate ban rate
                blue_bans = sum(hero in bans for bans in patch_matches['blue_bans'])
                red_bans = sum(hero in bans for bans in patch_matches['red_bans'])
                total_bans = blue_bans + red_bans
                ban_rate = total_bans / len(patch_matches)
                
                # Calculate win rate
                blue_wins = sum((hero in picks and winner == 'blue')
                              for picks, winner in zip(patch_matches['blue_picks'],
                                                     patch_matches['winner']))
                red_wins = sum((hero in picks and winner == 'red')
                             for picks, winner in zip(patch_matches['red_picks'],
                                                    patch_matches['winner']))
                
                if total_picks > 0:
                    win_rate = (blue_wins + red_wins) / total_picks
                else:
                    win_rate = 0.0
                
                hero_stats[hero] = {
                    'pick_rate': pick_rate,
                    'ban_rate': ban_rate,
                    'win_rate': win_rate,
                    'total_picks': total_picks,
                    'total_bans': total_bans
                }
            
            self.patch_stats[patch] = hero_stats
    
    def _get_all_heroes(self) -> set:
        """Get set of all heroes appearing in matches."""
        heroes = set()
        for col in ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']:
            heroes.update(*self.matches_df[col].tolist())
        return heroes
    
    def get_patch_stats(self, patch: str = None) -> Dict[str, Dict[str, float]]:
        """Get hero statistics for a specific patch.
        
        Args:
            patch: Patch version. If None, returns latest patch stats.
            
        Returns:
            Dictionary of hero statistics for the patch.
        """
        if not self.patch_stats:
            return {}
        
        if patch is None:
            # Get latest patch
            patch = max(self.patch_stats.keys())
        
        return self.patch_stats.get(patch, {})
    
    def compute_hero_features(self) -> Dict[str, Any]:
        """Compute hero features from match data.
        
        Returns:
            Dictionary containing hero features including:
            - patches: List of patch versions
            - heroes: List of all heroes
            - patch_stats: Per-patch hero statistics
            - role_synergies: Role synergy matrix
        """
        if self.matches_df is None:
            raise DataLoadError("No match data loaded. Call load_matches() first.")
            
        # Get sorted list of patches
        patches = sorted(self.matches_df['patch_version'].unique())
        
        # Get all heroes that appear in the dataset
        heroes = list(self._get_all_heroes())
        
        # Compute role synergies
        roles = ['Tank', 'Fighter', 'Assassin', 'Marksman', 'Mage', 'Support']
        role_synergies = {}
        
        for role1 in roles:
            role_synergies[role1] = {}
            for role2 in roles:
                # Calculate win rate when these roles are on the same team
                win_rate = self._compute_role_synergy(role1, role2)
                role_synergies[role1][role2] = win_rate
        
        return {
            'patches': patches,
            'heroes': heroes,
            'patch_stats': self.patch_stats,
            'role_synergies': role_synergies
        }
    
    def _compute_role_synergy(self, role1: str, role2: str) -> float:
        """Compute synergy score between two roles.
        
        Args:
            role1: First role
            role2: Second role
            
        Returns:
            Win rate when both roles are on the same team
        """
        total_games = 0
        won_games = 0
        
        # Load hero roles if not already loaded
        if not hasattr(self, '_hero_roles'):
            try:
                with open(os.path.join(os.path.dirname(self.data_dir), 'static/data/hero_roles.json'), 'r') as f:
                    self._hero_roles = json.load(f)
            except Exception:
                return 0.0
                
        for _, match in self.matches_df.iterrows():
            # Check blue team
            blue_roles = set()
            for hero in match['blue_picks']:
                if hero in self._hero_roles:
                    blue_roles.update(self._hero_roles[hero])
                    
            # Check red team
            red_roles = set()
            for hero in match['red_picks']:
                if hero in self._hero_roles:
                    red_roles.update(self._hero_roles[hero])
                    
            # Check if roles appear together on either team
            if (role1 in blue_roles and role2 in blue_roles):
                total_games += 1
                if match['winner'] == 'blue':
                    won_games += 1
            if (role1 in red_roles and role2 in red_roles):
                total_games += 1
                if match['winner'] == 'red':
                    won_games += 1
                    
        return won_games / total_games if total_games > 0 else 0.0

    def compute_role_trends(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute pick, ban, and win rates by role across patches.
        
        Returns:
            Dict with structure:
            {
                'patch_id': {
                    'role': {
                        'pick_rate': float,
                        'ban_rate': float,
                        'win_rate': float
                    }
                }
            }
        """
        role_trends = {}
        
        # Load hero role mappings
        with open('static/data/hero_roles.json', 'r') as f:
            role_data = json.load(f)
            roles = role_data['roles'].keys()
        
        # Get hero to role mapping
        with open('static/data/hero_data.json', 'r') as f:
            hero_data = json.load(f)
            hero_roles = {hero['name']: hero['primary_role'] for hero in hero_data}
        
        for patch in self.patch_stats.keys():
            role_trends[patch] = {role: {'pick_rate': 0.0, 'ban_rate': 0.0, 'win_rate': 0.0} 
                                for role in roles}
            
            patch_matches = self.matches_df[self.matches_df['patch'] == patch]
            total_matches = len(patch_matches)
            
            if total_matches == 0:
                continue
                
            # Aggregate stats by role
            for role in roles:
                role_heroes = [hero for hero, hero_role in hero_roles.items() 
                             if hero_role == role]
                
                role_picks = 0
                role_wins = 0
                role_bans = 0
                
                for hero in role_heroes:
                    if hero in self.patch_stats[patch]:
                        stats = self.patch_stats[patch][hero]
                        role_picks += stats.get('picks', 0)
                        role_wins += stats.get('wins', 0)
                        role_bans += stats.get('bans', 0)
                
                # Calculate rates
                if role_picks > 0:
                    role_trends[patch][role]['pick_rate'] = role_picks / (total_matches * 2)
                    role_trends[patch][role]['win_rate'] = role_wins / role_picks
                if total_matches > 0:
                    role_trends[patch][role]['ban_rate'] = role_bans / (total_matches * 2)
        
        return role_trends