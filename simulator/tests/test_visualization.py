"""
Tests for MCTS tree visualization.
"""
import pytest
import tempfile
import json
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from simulator.mcts import MCTSNode, DraftState
from simulator.visualization import (
    create_node_label,
    get_node_color,
    create_edge_label,
    plot_mcts_tree,
    plot_value_distribution,
    export_tree_data,
    plot_hero_matrix,
    plot_synergy_matrix,
    plot_counter_matrix
)

# Skip visualization tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not PLOT_AVAILABLE,
    reason="Plotting dependencies not available"
)

@pytest.fixture
def sample_state():
    """Create sample draft state."""
    return DraftState(
        blue_picks=["Chou", "Gusion"],
        red_picks=["Franco"],
        blue_bans=["Ling"],
        red_bans=["Wanwan"],
        blue_turn=True,
        is_pick_phase=True
    )

@pytest.fixture
def sample_node(sample_state):
    """Create sample MCTS node."""
    node = MCTSNode(state=sample_state)
    node.wins = 30
    node.visits = 50
    
    # Add some children
    child1 = MCTSNode(
        state=DraftState(
            blue_picks=["Chou", "Gusion", "Lancelot"],
            red_picks=["Franco"],
            blue_bans=["Ling"],
            red_bans=["Wanwan"],
            blue_turn=False,
            is_pick_phase=True
        ),
        parent=node
    )
    child1.wins = 20
    child1.visits = 30
    
    child2 = MCTSNode(
        state=DraftState(
            blue_picks=["Chou", "Gusion", "Hayabusa"],
            red_picks=["Franco"],
            blue_bans=["Ling"],
            red_bans=["Wanwan"],
            blue_turn=False,
            is_pick_phase=True
        ),
        parent=node
    )
    child2.wins = 10
    child2.visits = 20
    
    node.children = [child1, child2]
    return node

def test_node_label(sample_node):
    """Test node label creation."""
    label = create_node_label(sample_node)
    
    # Check label content
    assert "Pick (Blue)" in label
    assert "B: Chou,Gusion" in label
    assert "R: Franco" in label
    assert "W/V: 30/50" in label
    assert "WR: 60.00%" in label

def test_node_color(sample_node):
    """Test node color calculation."""
    color = get_node_color(sample_node)
    assert color == 0.6  # 30/50
    
    # Test unvisited node
    empty_node = MCTSNode(state=sample_node.state)
    assert get_node_color(empty_node) == 0.5  # Neutral color

def test_edge_label(sample_node):
    """Test edge label creation."""
    parent_state = sample_node.state
    child_state = sample_node.children[0].state
    
    label = create_edge_label(parent_state, child_state)
    assert "Blue Pick" in label
    assert "Lancelot" in label

def test_plot_mcts_tree(sample_node):
    """Test tree plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tree.png"
        
        # Test with both layout options
        plot_mcts_tree(sample_node, save_path=output_path, layout='spring')
        assert output_path.exists()
        output_path.unlink()
        
        plot_mcts_tree(sample_node, save_path=output_path, layout='kamada_kawai')
        assert output_path.exists()

def test_plot_value_distribution(sample_node):
    """Test value distribution plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "dist.png"
        plot_value_distribution(sample_node, save_path=output_path)
        assert output_path.exists()

def test_export_tree_data(sample_node):
    """Test tree data export."""
    data = export_tree_data(sample_node)
    
    # Check root node data
    assert data['wins'] == 30
    assert data['visits'] == 50
    assert len(data['children']) == 2
    
    # Check child nodes
    child = data['children'][0]
    assert child['wins'] == 20
    assert child['visits'] == 30
    
    # Check state data
    assert len(data['state']['blue_picks']) == 2
    assert len(data['state']['red_picks']) == 1
    assert data['state']['is_pick_phase'] is True

def test_plot_large_tree(sample_node):
    """Test plotting with larger tree depths."""
    # Create a deeper tree
    def add_children(node, depth=0, max_depth=3):
        if depth >= max_depth:
            return
            
        for i in range(2):
            new_picks = node.state.blue_picks + [f"Hero_{depth}_{i}"]
            child = MCTSNode(
                state=DraftState(
                    blue_picks=new_picks,
                    red_picks=node.state.red_picks,
                    blue_bans=node.state.blue_bans,
                    red_bans=node.state.red_bans,
                    blue_turn=not node.state.blue_turn,
                    is_pick_phase=True
                ),
                parent=node
            )
            child.wins = np.random.randint(0, 100)
            child.visits = 100
            node.children.append(child)
            add_children(child, depth + 1, max_depth)
    
    add_children(sample_node)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "large_tree.png"
        plot_mcts_tree(sample_node, max_depth=3, save_path=output_path)
        assert output_path.exists()

def test_hero_matrix():
    """Test hero interaction matrix plotting."""
    heroes = ["Hero1", "Hero2", "Hero3"]
    matrix = np.array([
        [0.0, 0.5, -0.3],
        [0.5, 0.0, 0.2],
        [-0.3, 0.2, 0.0]
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "matrix.png"
        plot_hero_matrix(
            matrix=matrix,
            heroes=heroes,
            title="Test Matrix",
            save_path=output_path,
            highlight_threshold=0.4
        )
        assert output_path.exists()

def test_synergy_matrix():
    """Test synergy matrix plotting."""
    hero_features = {
        'heroes': ["Hero1", "Hero2", "Hero3"],
        'synergy_matrix': np.array([
            [0.0, 0.5, -0.3],
            [0.5, 0.0, 0.2],
            [-0.3, 0.2, 0.0]
        ]),
        'raw_stats': {
            'Hero1': {'picks': 100},
            'Hero2': {'picks': 50},
            'Hero3': {'picks': 5}
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "synergy.png"
        
        # Test with minimum games filter
        plot_synergy_matrix(
            hero_features,
            min_games=10,
            save_path=output_path
        )
        assert output_path.exists()

def test_counter_matrix():
    """Test counter matrix plotting."""
    hero_features = {
        'heroes': ["Hero1", "Hero2", "Hero3"],
        'counter_matrix': np.array([
            [0.0, 0.5, -0.3],
            [-0.5, 0.0, 0.2],
            [0.3, -0.2, 0.0]
        ]),
        'raw_stats': {
            'Hero1': {'picks': 100},
            'Hero2': {'picks': 50},
            'Hero3': {'picks': 5}
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "counter.png"
        
        # Test with minimum games filter
        plot_counter_matrix(
            hero_features,
            min_games=10,
            save_path=output_path
        )
        assert output_path.exists()

def test_matrix_filtering():
    """Test matrix filtering by minimum games."""
    hero_features = {
        'heroes': ["Hero1", "Hero2", "Hero3"],
        'synergy_matrix': np.array([
            [0.0, 0.5, -0.3],
            [0.5, 0.0, 0.2],
            [-0.3, 0.2, 0.0]
        ]),
        'raw_stats': {
            'Hero1': {'picks': 100},
            'Hero2': {'picks': 5},
            'Hero3': {'picks': 50}
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "filtered.png"
        
        # Only Hero1 and Hero3 should be included
        plot_synergy_matrix(
            hero_features,
            min_games=10,
            save_path=output_path
        )
        assert output_path.exists()