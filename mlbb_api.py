"""
MLBB Hero Analytics API integration with caching and fallback functionality.
"""
import requests
import time
import json
import os
from flask import current_app

_cache = {"heroes": None, "details": {}, "timestamp": 0}
CACHE_TTL = 3600  # seconds

def get_all_heroes():
    """Get a list of all heroes, with caching and fallback to static data.
    
    Returns:
        list: List of hero names
    """
    now = time.time()
    if _cache["heroes"] and now - _cache["timestamp"] < CACHE_TTL:
        return _cache["heroes"]
    
    try:
        resp = requests.get("https://api.mlbbhero.com/v1/heroes", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        heroes = [h["name"] for h in data]
        _cache.update({"heroes": heroes, "timestamp": now})
        return heroes
    except Exception as e:
        current_app.logger.warning(f"MLBB API failed, falling back to static data: {e}")
        # fallback to static file
        file_path = os.path.join(current_app.root_path, 'static', 'data', 'hero_data.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return list(json.load(f).keys())
        except Exception as file_err:
            current_app.logger.error(f"Static data fallback failed: {file_err}")
            return []

def get_hero_details(hero_name):
    """Get detailed information about a specific hero, with caching and fallback.
    
    Args:
        hero_name (str): Name of the hero to get details for
        
    Returns:
        dict: Hero details including description, roles, etc.
    """
    # Check cache first
    if hero_name in _cache["details"] and time.time() - _cache["timestamp"] < CACHE_TTL:
        return _cache["details"][hero_name]
    
    try:
        # Convert hero name to API-friendly format
        slug = hero_name.lower().replace(" ", "-").replace("'", "")
        resp = requests.get(f"https://api.mlbbhero.com/v1/heroes/{slug}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        # Cache the result
        _cache["details"][hero_name] = data
        _cache["timestamp"] = time.time()
        
        return data
    except Exception as e:
        current_app.logger.warning(f"MLBB API details failed for {hero_name}, falling back to static data: {e}")
        # fallback to static data
        file_path = os.path.join(current_app.root_path, 'static', 'data', 'hero_data.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if hero_name in data:
                    return {
                        "name": hero_name,
                        "description": data[hero_name].get("description", ""),
                        "roles": data[hero_name].get("roles", []),
                        "specialty": data[hero_name].get("specialty", ""),
                        "difficulty": data[hero_name].get("difficulty", "")
                    }
                return {"name": hero_name, "description": "No information available."}
        except Exception as file_err:
            current_app.logger.error(f"Static data fallback failed for {hero_name}: {file_err}")
            return {"name": hero_name, "description": "Error loading hero information."} 