"""
Tests for MLBB Analytics API endpoints.
"""
import json
from typing import List
import pytest
from fastapi.testclient import TestClient
import numpy as np

from api.main import app
from simulator.mcts import DraftState

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "MLBB Draft Analytics API"
    }

def test_predict_draft():
    """Test draft prediction endpoint."""
    draft_state = {
        "blue_picks": ["Chou", "Gusion"],
        "red_picks": ["Franco", "Fanny"],
        "blue_bans": ["Ling", "Lancelot", "Hayabusa"],
        "red_bans": ["Wanwan", "Beatrix", "Karrie"],
        "patch_version": "1.6.44"
    }
    
    response = client.post("/predict", json=draft_state)
    assert response.status_code == 200
    
    data = response.json()
    assert "blue_win_probability" in data
    assert isinstance(data["blue_win_probability"], float)
    assert 0 <= data["blue_win_probability"] <= 1
    
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5  # Should return top 5 or fewer
    
    # Verify recommendation format
    for rec in data["recommendations"]:
        assert "hero" in rec
        assert "win_probability" in rec
        assert "description" in rec
        assert 0 <= rec["win_probability"] <= 1
        
    # Verify current phase
    assert data["current_phase"] in ["PICK PHASE", "BAN PHASE"]
    assert isinstance(data["blue_turn"], bool)

def test_invalid_draft():
    """Test error handling for invalid draft state."""
    invalid_draft = {
        "blue_picks": ["InvalidHero"],
        "red_picks": [],
        "blue_bans": [],
        "red_bans": [],
    }
    
    response = client.post("/predict", json=invalid_draft)
    assert response.status_code == 400
    assert "error" in response.json()["detail"].lower()

def test_hero_stats():
    """Test hero statistics endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "hero_stats" in data
    stats = data["hero_stats"]
    
    # Check required statistics are present
    assert "heroes" in stats
    assert "pick_rates" in stats
    assert "ban_rates" in stats
    assert "win_rates" in stats
    
    # Verify rate bounds
    for hero in stats["heroes"]:
        assert 0 <= stats["pick_rates"][hero] <= 1
        assert 0 <= stats["ban_rates"][hero] <= 1
        assert 0 <= stats["win_rates"][hero] <= 1

def test_side_bias():
    """Test side bias analysis endpoint."""
    response = client.get("/side-bias", params={"min_games": 10})
    assert response.status_code == 200
    
    data = response.json()
    assert "min_games" in data
    assert "heroes_analyzed" in data
    assert "bias_data" in data
    
    # Check bias data format
    for bias in data["bias_data"]:
        assert "hero" in bias
        assert "effect_size" in bias
        assert "category" in bias
        assert "blue_rate" in bias
        assert "red_rate" in bias
        assert -1 <= bias["effect_size"] <= 1  # Cohen's h bounds

def test_error_handling():
    """Test API error handling."""
    # Test missing required fields
    incomplete_draft = {
        "blue_picks": ["Chou"]
        # Missing other required fields
    }
    response = client.post("/predict", json=incomplete_draft)
    assert response.status_code == 422  # Validation error
    
    # Test invalid patch version
    response = client.get("/stats", params={"patch_version": "invalid"})
    assert response.status_code == 400
    
    # Test invalid min_games parameter
    response = client.get("/side-bias", params={"min_games": -1})
    assert response.status_code == 400

def test_root_endpoint():
    """Test the root API endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_draft_predict_endpoint_valid():
    """Test the /predict endpoint with valid data."""
    test_data = {
        "blue_picks": ["Layla"],
        "red_picks": ["Tigreal"],
        "blue_bans": ["Fanny"],
        "red_bans": ["Gusion"]
    }
    response = client.post("/predict", json=test_data)
    
    # Assert response status and structure
    assert response.status_code == 200
    data = response.json()
    assert "blue_win_probability" in data
    assert isinstance(data["blue_win_probability"], float)
    assert 0 <= data["blue_win_probability"] <= 1
    
    # Check recommendations
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    if data["recommendations"]:
        assert "hero" in data["recommendations"][0]
        assert "win_probability" in data["recommendations"][0]

def test_draft_predict_endpoint_invalid():
    """Test the /predict endpoint with invalid data."""
    # Missing required fields
    test_data = {
        "blue_picks": ["Layla"]
        # Missing other required fields
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error
    
    # Invalid hero name
    test_data = {
        "blue_picks": ["NonExistentHero"],
        "red_picks": [],
        "blue_bans": [],
        "red_bans": []
    }
    response = client.post("/predict", json=test_data)
    # Either 400 (application error) or 503 (service unavailable) is acceptable
    # depending on whether the data_loader is initialized
    assert response.status_code in [400, 503]
    
    # Duplicate heroes
    test_data = {
        "blue_picks": ["Layla", "Layla"],
        "red_picks": [],
        "blue_bans": [],
        "red_bans": []
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400
    assert "duplicate" in response.json()["detail"].lower()

def test_draft_heroes_endpoint():
    """Test the /draft/heroes endpoint."""
    response = client.get("/draft/heroes")
    
    # Either 200 (success) or 503 (service unavailable) is acceptable
    # depending on whether the data_loader is initialized
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            hero = data[0]
            assert "id" in hero
            assert "name" in hero
            assert "roles" in hero
            assert isinstance(hero["roles"], list)
    else:
        assert response.status_code == 503
        assert "data loader" in response.json()["detail"].lower()

def test_stats_endpoint():
    """Test the /stats endpoint."""
    response = client.get("/stats")
    
    # Either 200 (success) or 503 (service unavailable) is acceptable
    if response.status_code == 200:
        data = response.json()
        assert "hero_stats" in data
        hero_stats = data["hero_stats"]
        assert "heroes" in hero_stats
        assert isinstance(hero_stats["heroes"], list)
    else:
        assert response.status_code == 503