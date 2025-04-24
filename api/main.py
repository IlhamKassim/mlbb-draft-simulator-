"""
FastAPI application for MLBB draft prediction.
"""
import logging
from typing import List, Optional, Dict
from pathlib import Path
import json

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_loader import MLBBDataLoader
from simulator.mcts import DraftState, MCTSDraftSimulator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLBB Draft Analytics API",
    description="API for Mobile Legends: Bang Bang draft prediction and analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load model and data loader
MODEL_PATH = Path("models/baseline/baseline.joblib")
DATA_PATH = Path("data/raw")
HERO_PATH = Path("static/data/heroes.json")

try:
    print("Starting API initialization...")
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully: {type(model)}")
    
    print(f"Initializing data loader with path {DATA_PATH}")
    data_loader = MLBBDataLoader(DATA_PATH)
    print("Data loader initialized, loading matches...")
    
    print("Looking for sample_matches.csv...")
    if (DATA_PATH / "sample_matches.csv").exists():
        print("Sample matches file found, loading...")
    else:
        print("WARNING: sample_matches.csv not found in data/raw!")
    
    try:
        data_loader.load_matches()
        print("Matches loaded successfully!")
    except Exception as match_error:
        print(f"Error loading matches: {match_error}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Computing hero features...")
    hero_features = data_loader.compute_hero_features()
    available_heroes = hero_features['heroes']
    print(f"Found {len(available_heroes)} heroes: {available_heroes[:5]}...")
    
    print("Initializing simulator...")
    simulator = MCTSDraftSimulator(
        model=model,
        data_loader=data_loader,
        available_heroes=available_heroes
    )
    print("Simulator initialized successfully!")
except Exception as e:
    print(f"Error loading model or data: {e}")
    import traceback
    traceback.print_exc()
    model = None
    data_loader = None
    simulator = None
    print("API initialization failed. Endpoints requiring data_loader will return 503 errors.")

class DraftState(BaseModel):
    """Current state of a draft."""
    blue_picks: List[str]
    red_picks: List[str]
    blue_bans: List[str]
    red_bans: List[str]
    patch_version: Optional[str] = None

class HeroRecommendation(BaseModel):
    """Hero recommendation with win probability."""
    hero: str
    win_probability: float
    description: Optional[str] = None

class PredictionResponse(BaseModel):
    """Model prediction response."""
    blue_win_probability: float
    features_used: Dict[str, float]
    recommendations: List[HeroRecommendation]
    current_phase: str
    blue_turn: bool

@app.get("/")
async def root():
    """API root endpoint."""
    return {"status": "ok", "message": "MLBB Draft Analytics API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_draft(draft: DraftState):
    """
    Predict win probability and recommend next action for current draft state.
    
    The endpoint will:
    1. Compute current blue team win probability
    2. Use MCTS to simulate possible draft completions
    3. Return top hero recommendations for next pick/ban
    4. Include feature importance for interpretability
    """
    if not all([model, data_loader, simulator]):
        raise HTTPException(
            status_code=503,
            detail="Model, data loader, or simulator not initialized"
        )
    
    try:
        # Validate draft state
        for hero_list in [draft.blue_picks, draft.red_picks, draft.blue_bans, draft.red_bans]:
            # Ensure lists are not None
            if hero_list is None:
                raise ValueError("Hero lists in draft state cannot be None")
            
            # Check for unknown heroes
            unknown_heroes = [h for h in hero_list if h not in simulator.available_heroes]
            if unknown_heroes:
                raise ValueError(f"Unknown heroes in draft: {', '.join(unknown_heroes)}")
                
        # Validate draft state integrity (proper number of picks/bans)
        if len(draft.blue_picks) > 5 or len(draft.red_picks) > 5:
            raise ValueError("Teams cannot have more than 5 picks")
            
        if len(draft.blue_bans) > 3 or len(draft.red_bans) > 3:
            raise ValueError("Teams cannot have more than 3 bans")
            
        # Check for duplicate heroes
        all_heroes = draft.blue_picks + draft.red_picks + draft.blue_bans + draft.red_bans
        if len(all_heroes) != len(set(all_heroes)):
            raise ValueError("Draft contains duplicate heroes")
            
        # Get feature vector and base prediction
        features = data_loader.prepare_model_features(
            draft.blue_picks,
            draft.red_picks,
            draft.blue_bans,
            draft.red_bans
        )
        win_prob = float(model.predict_proba([features])[0][1])
        
        # Get feature importance
        try:
            feature_names = data_loader.get_feature_names()
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            # Only include features that are actually used in the draft
            filtered_importance = {}
            for name, value in feature_importance.items():
                parts = name.split('_', 2)
                if len(parts) == 3:
                    team, action, hero = parts
                    if ((team == 'blue' and action == 'pick' and hero in draft.blue_picks) or
                        (team == 'red' and action == 'pick' and hero in draft.red_picks) or
                        (team == 'blue' and action == 'ban' and hero in draft.blue_bans) or
                        (team == 'red' and action == 'ban' and hero in draft.red_bans)):
                        filtered_importance[name] = value
            # If no features matched, use top 10 overall
            if not filtered_importance:
                filtered_importance = dict(sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10])
        except Exception as e:
            # Fallback if feature importance fails
            logger.warning(f"Error getting feature importance: {e}")
            filtered_importance = {"error": 0.0}
        
        # Determine current phase and turn
        is_pick_phase = len(draft.blue_bans) + len(draft.red_bans) >= 6
        blue_picks_remaining = 5 - len(draft.blue_picks)
        red_picks_remaining = 5 - len(draft.red_picks)
        blue_turn = (
            (not is_pick_phase and len(draft.blue_bans) <= len(draft.red_bans)) or
            (is_pick_phase and blue_picks_remaining > red_picks_remaining)
        )
        
        # Create draft state for simulator
        sim_state = simulator.DraftState(
            blue_picks=draft.blue_picks,
            red_picks=draft.red_picks,
            blue_bans=draft.blue_bans,
            red_bans=draft.red_bans,
            blue_turn=blue_turn,
            is_pick_phase=is_pick_phase
        )
        
        # Get top 5 recommendations
        recommendations = []
        try:
            rankings = simulator.get_action_rankings(sim_state, top_k=5)
            recommendations = [
                HeroRecommendation(
                    hero=hero,
                    win_probability=prob,
                    description=f"{'Pick' if is_pick_phase else 'Ban'} {hero} "
                               f"({prob:.1%} win probability)"
                )
                for hero, prob in rankings
            ]
        except Exception as e:
            # Fallback recommendations if simulation fails
            logger.error(f"Error generating recommendations: {e}")
            legal_actions = simulator.get_legal_actions(sim_state)
            recommendations = [
                HeroRecommendation(
                    hero=hero,
                    win_probability=0.5,
                    description=f"{'Pick' if is_pick_phase else 'Ban'} {hero}"
                )
                for hero in legal_actions[:5]
            ]
        
        phase = "PICK" if is_pick_phase else "BAN"
        turn = "BLUE" if blue_turn else "RED"
        
        return PredictionResponse(
            blue_win_probability=win_prob,
            features_used=filtered_importance,
            recommendations=recommendations,
            current_phase=f"{phase} PHASE",
            blue_turn=blue_turn
        )
        
    except Exception as e:
        logger.error(f"Error processing draft: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error processing draft: {str(e)}"
        )
        
@app.get("/stats")
async def get_hero_stats(patch_version: Optional[str] = None):
    """Get hero statistics for the current/specified patch."""
    if not data_loader:
        raise HTTPException(
            status_code=503,
            detail="Data loader not initialized"
        )
    
    try:
        hero_features = data_loader.compute_hero_features(patch_version)
        return {
            "patch_version": patch_version,
            "hero_stats": hero_features
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error computing hero stats: {str(e)}"
        )

@app.get("/side-bias")
async def get_side_bias(
    min_games: int = 10,
    top_n: Optional[int] = None
):
    """Get hero side bias analysis."""
    if not data_loader:
        raise HTTPException(
            status_code=503,
            detail="Data loader not initialized"
        )
    
    try:
        biases = data_loader.rank_side_bias(
            min_games=min_games,
            top_n=top_n
        )
        return {
            "min_games": min_games,
            "heroes_analyzed": len(biases),
            "bias_data": [bias._asdict() for bias in biases]
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error computing side bias: {str(e)}"
        )

@app.get("/stats/synergy")
async def get_synergy_matrix(min_games: int = 10):
    """Get hero synergy matrix data."""
    if not data_loader:
        raise HTTPException(
            status_code=503,
            detail="Data loader not initialized"
        )
    
    try:
        hero_features = data_loader.compute_hero_features()
        heroes = hero_features['heroes']
        matrix = hero_features['synergy_matrix']
        
        # Filter by minimum games
        if 'raw_stats' in hero_features:
            stats = hero_features['raw_stats']
            valid_heroes = [h for h in heroes 
                          if stats[h]['picks'] >= min_games]
            hero_idx = [heroes.index(h) for h in valid_heroes]
            filtered_matrix = matrix[np.ix_(hero_idx, hero_idx)]
        else:
            valid_heroes = heroes
            filtered_matrix = matrix.tolist()  # Convert to list for JSON
        
        return {
            "heroes": valid_heroes,
            "matrix": filtered_matrix.tolist(),
            "min_games": min_games
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error computing synergy matrix: {str(e)}"
        )

@app.get("/stats/counter")
async def get_counter_matrix(min_games: int = 10):
    """Get hero counter matrix data."""
    if not data_loader:
        raise HTTPException(
            status_code=503,
            detail="Data loader not initialized"
        )
    
    try:
        hero_features = data_loader.compute_hero_features()
        heroes = hero_features['heroes']
        matrix = hero_features['counter_matrix']
        
        # Filter by minimum games
        if 'raw_stats' in hero_features:
            stats = hero_features['raw_stats']
            valid_heroes = [h for h in heroes 
                          if stats[h]['picks'] >= min_games]
            hero_idx = [heroes.index(h) for h in valid_heroes]
            filtered_matrix = matrix[np.ix_(hero_idx, hero_idx)]
        else:
            valid_heroes = heroes
            filtered_matrix = matrix
        
        return {
            "heroes": valid_heroes,
            "matrix": filtered_matrix.tolist(),
            "min_games": min_games
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error computing counter matrix: {str(e)}"
        )

@app.get("/draft/heroes")
async def get_draft_heroes():
    """Get all available heroes for draft selection with complete metadata."""
    if not data_loader:
        raise HTTPException(
            status_code=503,
            detail="Data loader not initialized"
        )
    
    try:
        # Get heroes list from data_loader
        hero_features = data_loader.compute_hero_features()
        heroes_from_data = hero_features.get('heroes', [])
        print(f"Found {len(heroes_from_data)} heroes in data_loader")
        
        # Try to get detailed hero data from JSON file
        hero_data = {}
        hero_roles = {}
        
        try:
            hero_data_path = Path("static/data/hero_data.json")
            if hero_data_path.exists():
                with open(hero_data_path, "r") as f:
                    hero_data = json.load(f)
                print(f"Loaded hero_data.json with {len(hero_data)} heroes")
            
            hero_roles_path = Path("static/data/hero_roles.json")
            if hero_roles_path.exists():
                with open(hero_roles_path, "r") as f:
                    hero_roles = json.load(f)
                print(f"Loaded hero_roles.json with {len(hero_roles)} entries")
        except Exception as e:
            print(f"Warning: Could not load hero data files: {e}")
        
        # Calculate hero stats
        win_rates = hero_features.get('win_rates', {})
        pick_rates = hero_features.get('pick_rates', {})
        ban_rates = hero_features.get('ban_rates', {})
        
        # Merge data from all sources
        hero_list = []
        
        # Process heroes from data_loader
        for hero_name in heroes_from_data:
            # Base hero object with mandatory fields
            hero_obj = {
                "id": hero_name.lower().replace(" ", "-"),
                "name": hero_name,
                "roles": [],
                "specialty": "Unknown",
                "difficulty": "Normal",
                "description": f"{hero_name} is a hero in Mobile Legends: Bang Bang.",
                "imageUrl": f"/static/hero-icons/{hero_name.lower().replace(' ', '-')}.png",
                "winRate": win_rates.get(hero_name, 0.5),
                "pickRate": pick_rates.get(hero_name, 0.0),
                "banRate": ban_rates.get(hero_name, 0.0)
            }
            
            # Add data from hero_data.json if available
            if hero_name in hero_data:
                hero_info = hero_data[hero_name]
                hero_obj.update({
                    "roles": hero_info.get("roles", []),
                    "specialty": hero_info.get("specialty", "Unknown"),
                    "difficulty": hero_info.get("difficulty", "Normal"),
                    "description": hero_info.get("description", hero_obj["description"])
                })
            # Add roles from hero_roles.json if available
            elif hero_name in hero_roles:
                hero_obj["roles"] = hero_roles[hero_name]
            
            # Ensure there's at least one role
            if not hero_obj["roles"]:
                hero_obj["roles"] = ["Fighter"]
                
            hero_list.append(hero_obj)
            
        # Add "Category:" entries from hero_data.json as they represent roles
        for name, info in hero_data.items():
            if name.startswith("Category:") and name not in [h["name"] for h in hero_list]:
                category_name = name.replace("Category:", "").strip()
                if category_name in ["Fighter", "Tank", "Assassin", "Mage", "Marksman", "Support"]:
                    # This is a role category, can be used for filtering
                    continue
                    
        # Sort heroes by name
        hero_list.sort(key=lambda x: x["name"])
        
        print(f"Returning {len(hero_list)} heroes with metadata")
        if hero_list:
            # Print sample hero for debugging
            print(f"Sample hero: {hero_list[0]}")
            
        return hero_list
    except Exception as e:
        print(f"Error in get_draft_heroes: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving heroes: {str(e)}"
        )

@app.get("/api/hero_roles")
async def get_hero_roles():
    """Get hero roles data."""
    try:
        hero_roles_path = Path("static/data/hero_roles.json")
        if not hero_roles_path.exists():
            logger.warning(f"Hero roles file not found at {hero_roles_path}")
            return {
                "roles": {},
                "role_counters": {}
            }
        
        with open(hero_roles_path, "r") as f:
            hero_roles = json.loads(f.read())
            
        return {
            "roles": hero_roles
        }
    except Exception as e:
        logger.error(f"Error loading hero roles: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading hero roles: {str(e)}"
        )

# Mount static files for hero icons
# This must be placed at the end of the file after all routes
try:
    # Use absolute path for more reliable static file serving
    base_dir = Path(__file__).parent.parent.absolute()
    static_dir = base_dir / "static"
    if not static_dir.exists():
        static_dir.mkdir(exist_ok=True)
        logger.info(f"Created static directory at {static_dir}")
    
    # Create hero-icons directory if it doesn't exist
    hero_icons_dir = static_dir / "hero-icons"
    if not hero_icons_dir.exists():
        hero_icons_dir.mkdir(exist_ok=True)
        logger.info(f"Created hero-icons directory at {hero_icons_dir}")
    
    logger.info(f"Mounting static files from {static_dir}")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Log available directories for debugging
    logger.info(f"Available directories in static: {[d.name for d in static_dir.iterdir() if d.is_dir()]}")
except Exception as e:
    logger.error(f"Failed to mount static files: {e}")