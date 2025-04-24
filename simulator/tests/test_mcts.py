"""
Tests for MCTS draft simulator.
"""
import pytest
import numpy as np
from typing import List
from unittest.mock import Mock

from simulator.mcts import DraftState, MCTSDraftSimulator

# Mock model prediction
class MockModel:
    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])  # Always predict 60% blue win

# Mock data loader
class MockDataLoader:
    def prepare_model_features(self, blue_picks: List[str], red_picks: List[str],
                             blue_bans: List[str], red_bans: List[str]):
        return np.zeros(100)  # Mock feature vector

@pytest.fixture
def simulator():
    """Create simulator with mock model and data loader."""
    model = MockModel()
    data_loader = MockDataLoader()
    available_heroes = [f"hero_{i}" for i in range(20)]  # 20 test heroes
    return MCTSDraftSimulator(
        model=model,
        data_loader=data_loader,
        available_heroes=available_heroes,
        iterations=100  # Reduced for testing
    )

@pytest.fixture
def empty_state():
    """Create empty draft state."""
    return DraftState(
        blue_picks=[],
        red_picks=[],
        blue_bans=[],
        red_bans=[],
        blue_turn=True,
        is_pick_phase=False  # Start with bans
    )

def test_draft_state_initialization(empty_state):
    """Test draft state initialization."""
    assert len(empty_state.blue_picks) == 0
    assert len(empty_state.red_picks) == 0
    assert len(empty_state.blue_bans) == 0
    assert len(empty_state.red_bans) == 0
    assert empty_state.blue_turn is True
    assert empty_state.is_pick_phase is False

def test_legal_actions(simulator, empty_state):
    """Test legal action generation."""
    actions = simulator.get_legal_actions(empty_state)
    assert len(actions) == 20  # All heroes available
    
    # After some picks and bans
    state = DraftState(
        blue_picks=["hero_0", "hero_1"],
        red_picks=["hero_2", "hero_3"],
        blue_bans=["hero_4"],
        red_bans=["hero_5"],
        blue_turn=True,
        is_pick_phase=True
    )
    actions = simulator.get_legal_actions(state)
    assert len(actions) == 14  # 20 - 6 used heroes
    assert "hero_0" not in actions  # Used heroes not available
    assert "hero_5" not in actions

def test_terminal_state(simulator):
    """Test terminal state detection."""
    state = DraftState(
        blue_picks=["hero_0", "hero_1", "hero_2", "hero_3", "hero_4"],
        red_picks=["hero_5", "hero_6", "hero_7", "hero_8", "hero_9"],
        blue_bans=["hero_10", "hero_11", "hero_12"],
        red_bans=["hero_13", "hero_14", "hero_15"],
        blue_turn=True,
        is_pick_phase=True
    )
    assert simulator.is_terminal(state)

def test_make_move(simulator, empty_state):
    """Test move application."""
    # Test ban phase
    new_state = simulator.make_move(empty_state, "hero_0")
    assert len(new_state.blue_bans) == 1
    assert new_state.blue_bans[0] == "hero_0"
    assert new_state.blue_turn is False
    
    # Test transition to pick phase
    state = DraftState(
        blue_picks=[],
        red_picks=[],
        blue_bans=["hero_0", "hero_1", "hero_2"],
        red_bans=["hero_3", "hero_4", "hero_5"],
        blue_turn=True,
        is_pick_phase=False
    )
    new_state = simulator.make_move(state, "hero_6")
    assert new_state.is_pick_phase is True
    assert len(new_state.blue_picks) == 1
    assert new_state.blue_picks[0] == "hero_6"

def test_best_action(simulator, empty_state):
    """Test best action selection."""
    action, win_prob = simulator.get_best_action(empty_state)
    assert action in simulator.available_heroes
    assert 0 <= win_prob <= 1

def test_action_rankings(simulator, empty_state):
    """Test action ranking generation."""
    rankings = simulator.get_action_rankings(empty_state, top_k=5)
    assert len(rankings) == 5
    assert all(0 <= prob <= 1 for _, prob in rankings)
    assert all(action in simulator.available_heroes for action, _ in rankings)
    
    # Check sorting
    probs = [prob for _, prob in rankings]
    assert probs == sorted(probs, reverse=True)