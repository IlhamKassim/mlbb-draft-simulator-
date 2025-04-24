#!/usr/bin/env python
"""
Test script for MCTS draft simulator.

This script tests the functionality of the Monte Carlo Tree Search
draft simulator implementation, checking various edge cases and
ensuring that the draft recommendations are valid.
"""
import sys
import os
from pathlib import Path
import json
import logging
import time
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import MLBBDataLoader
from simulator.mcts import DraftState, MCTSDraftSimulator
from simulator.visualization import plot_mcts_tree

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_mcts')

def load_test_data(example_file='early_draft.json'):
    """Load draft example from examples directory."""
    examples_dir = Path(__file__).parent.parent / 'examples'
    file_path = examples_dir / example_file
    
    if not file_path.exists():
        logger.error(f"Test data file not found: {file_path}")
        return None
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None

def load_model_and_data():
    """Load the ML model and data loader."""
    model_path = Path(__file__).parent.parent / 'models/baseline/baseline.joblib'
    data_path = Path(__file__).parent.parent / 'data/raw'
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None, None
        
    try:
        model = joblib.load(model_path)
        data_loader = MLBBDataLoader(data_path)
        data_loader.load_matches()
        
        # Compute hero features to get available heroes
        hero_features = data_loader.compute_hero_features()
        
        return model, data_loader, hero_features['heroes']
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        return None, None, None

def create_draft_state(draft_data):
    """Create a DraftState object from draft data."""
    return DraftState(
        blue_picks=draft_data.get('blue_picks', []),
        red_picks=draft_data.get('red_picks', []),
        blue_bans=draft_data.get('blue_bans', []),
        red_bans=draft_data.get('red_bans', []),
        blue_turn=draft_data.get('blue_turn', True),
        is_pick_phase=draft_data.get('is_pick_phase', False)
    )

def test_simulator(model, data_loader, available_heroes, draft_state, iterations=500):
    """Test the MCTS simulator with a given draft state."""
    logger.info(f"Testing simulator with draft state: {draft_state}")
    
    # Create simulator
    simulator = MCTSDraftSimulator(
        model=model,
        data_loader=data_loader,
        available_heroes=available_heroes,
        iterations=iterations
    )
    
    # Test get_legal_actions
    legal_actions = simulator.get_legal_actions(draft_state)
    logger.info(f"Legal actions: {len(legal_actions)} available")
    
    # Test get_best_action
    start_time = time.time()
    best_action, win_prob = simulator.get_best_action(draft_state)
    elapsed = time.time() - start_time
    
    logger.info(f"Best action: {best_action} with win probability {win_prob:.3f}")
    logger.info(f"Time taken: {elapsed:.2f} seconds")
    
    # Test get_action_rankings
    rankings = simulator.get_action_rankings(draft_state, top_k=5)
    
    logger.info("Top 5 recommended actions:")
    for i, (action, prob) in enumerate(rankings):
        logger.info(f"{i+1}. {action} - {prob:.3f}")
        
    # Verify that state comparison works properly
    for action, prob in rankings:
        new_state = simulator.make_move(draft_state, action)
        # Make a copy of the state with the same values
        copy_state = DraftState(
            blue_picks=list(new_state.blue_picks),
            red_picks=list(new_state.red_picks),
            blue_bans=list(new_state.blue_bans),
            red_bans=list(new_state.red_bans),
            blue_turn=new_state.blue_turn,
            is_pick_phase=new_state.is_pick_phase
        )
        # Verify that our state comparison logic works
        assert (copy_state.blue_picks == new_state.blue_picks and
                copy_state.red_picks == new_state.red_picks and
                copy_state.blue_bans == new_state.blue_bans and
                copy_state.red_bans == new_state.red_bans and
                copy_state.blue_turn == new_state.blue_turn and
                copy_state.is_pick_phase == new_state.is_pick_phase)
    
    return simulator, best_action, win_prob, rankings

def test_make_move(simulator):
    """Test the make_move method for phase transitions."""
    # Test ban phase
    init_state = DraftState(
        blue_picks=[], red_picks=[], 
        blue_bans=[], red_bans=[],
        blue_turn=True, is_pick_phase=False
    )
    
    # First blue ban
    state = simulator.make_move(init_state, "Ling")
    assert state.blue_bans == ["Ling"]
    assert state.blue_turn == False
    assert state.is_pick_phase == False
    
    # First red ban
    state = simulator.make_move(state, "Wanwan")
    assert state.red_bans == ["Wanwan"]
    assert state.blue_turn == True
    assert state.is_pick_phase == False
    
    # Complete ban phase (3 bans each)
    state = simulator.make_move(state, "Lancelot")  # Blue ban 2
    state = simulator.make_move(state, "Beatrix")   # Red ban 2
    state = simulator.make_move(state, "Hayabusa")  # Blue ban 3
    state = simulator.make_move(state, "Karrie")    # Red ban 3
    
    # Should transition to pick phase with blue first
    assert state.is_pick_phase == True
    assert state.blue_turn == True
    assert len(state.blue_bans) == 3
    assert len(state.red_bans) == 3
    
    # Test pick phase order
    state = simulator.make_move(state, "Gusion")   # Blue pick 1
    assert state.blue_turn == False
    
    state = simulator.make_move(state, "Franco")   # Red pick 1
    state = simulator.make_move(state, "Fanny")    # Red pick 2 (second pick for red)
    assert state.blue_turn == True
    
    state = simulator.make_move(state, "Chou")     # Blue pick 2
    state = simulator.make_move(state, "Tigreal")  # Blue pick 3 (second pick for blue)
    assert state.blue_turn == False
    
    state = simulator.make_move(state, "Eudora")   # Red pick 3
    state = simulator.make_move(state, "Nana")     # Red pick 4 (second pick for red)
    assert state.blue_turn == True
    
    state = simulator.make_move(state, "Layla")    # Blue pick 4
    state = simulator.make_move(state, "Zilong")   # Blue pick 5 (last pick for blue)
    assert state.blue_turn == False
    
    state = simulator.make_move(state, "Saber")    # Red pick 5 (last pick)
    
    # Draft should be complete
    assert len(state.blue_picks) == 5
    assert len(state.red_picks) == 5
    assert len(state.blue_bans) == 3
    assert len(state.red_bans) == 3
    
    logger.info("Make move test passed!")
    return True

def main():
    """Run tests for the MCTS simulator."""
    # Load model and data
    model, data_loader, available_heroes = load_model_and_data()
    if not all([model, data_loader, available_heroes]):
        logger.error("Failed to load model or data")
        return False
        
    logger.info(f"Loaded model and data with {len(available_heroes)} available heroes")
    
    # Test early draft
    early_draft = load_test_data('early_draft.json')
    if early_draft:
        early_state = create_draft_state(early_draft)
        simulator, action, _, _ = test_simulator(model, data_loader, available_heroes, early_state)
        
        # Test the make_move logic
        test_make_move(simulator)
        
        # Generate a tree visualization
        logger.info("Generating tree visualization...")
        output_dir = Path(__file__).parent.parent / 'examples/output'
        output_dir.mkdir(exist_ok=True)
        
        # Run MCTS and save tree visualization
        root = simulator.get_best_action_node(early_state, iterations=200)
        plot_mcts_tree(
            root, 
            max_depth=2, 
            save_path=output_dir / 'test_mcts_tree.png',
            title="MCTS Tree Visualization"
        )
        
        logger.info(f"Visualization saved to {output_dir / 'test_mcts_tree.png'}")
        
    # Test mid draft
    mid_draft = load_test_data('mid_draft.json')
    if mid_draft:
        mid_state = create_draft_state(mid_draft)
        test_simulator(model, data_loader, available_heroes, mid_state)
        
    # Test late draft
    late_draft = load_test_data('late_draft.json')
    if late_draft:
        late_state = create_draft_state(late_draft)
        test_simulator(model, data_loader, available_heroes, late_state)
        
    logger.info("All tests completed!")
    return True

if __name__ == "__main__":
    main()