"""
Visualization utilities for MLBB match and draft analysis.
"""
from typing import Optional, Tuple, Dict, Any, List, Set, Union
import math
import json
import os
from pathlib import Path

try:
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy import stats
    import pandas as pd
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from .mcts import DraftState

# Define standard roles
ROLES = ['Tank', 'Fighter', 'Assassin', 'Marksman', 'Mage', 'Support']

# Load hero roles
def load_hero_roles():
    """Load hero roles from JSON file with robust path handling."""
    paths_to_try = [
        Path("static/data/hero_roles.json"),
        Path("../static/data/hero_roles.json"),
        Path(__file__).parent.parent / "static" / "data" / "hero_roles.json",
    ]
    
    for path in paths_to_try:
        try:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load hero roles from {path}: {e}")
    
    # Return empty default if loading fails
    print("Warning: Could not load hero roles file. Using empty default.")
    return {"roles": {}}

# Load hero roles with robust error handling
try:
    HERO_ROLES = load_hero_roles()
except Exception as e:
    print(f"Error loading hero roles: {e}")
    HERO_ROLES = {"roles": {}}

def _check_plot_dependencies():
    """Check if plotting dependencies are available."""
    if not PLOT_AVAILABLE:
        raise ImportError(
            "Plotting dependencies not available. Please install: "
            "numpy, networkx, matplotlib, plotly"
        )

def filter_hero_data(hero_features: Dict[str, Any],
                   roles: Optional[List[str]] = None,
                   min_pick_rate: Optional[float] = None,
                   min_win_rate: Optional[float] = None) -> Tuple[List[str], np.ndarray]:
    """
    Filter heroes based on roles and performance metrics.
    """
    heroes = hero_features['heroes']
    matrix = np.array(hero_features.get('synergy_matrix', []))
    
    # Apply filters
    valid_indices = []
    for i, hero in enumerate(heroes):
        stats = hero_features['patch_stats'].get(hero, {})
        
        if roles and not any(role in hero_features['hero_roles'].get(hero, []) for role in roles):
            continue
            
        if min_pick_rate and stats.get('pick_rate', 0) < min_pick_rate:
            continue
            
        if min_win_rate and stats.get('win_rate', 0) < min_win_rate:
            continue
            
        valid_indices.append(i)
    
    filtered_heroes = [heroes[i] for i in valid_indices]
    filtered_matrix = matrix[np.ix_(valid_indices, valid_indices)] if len(matrix) > 0 else np.array([])
    
    return filtered_heroes, filtered_matrix

def load_hero_data() -> Dict[str, List[str]]:
    """Load hero role mappings from hero data."""
    try:
        with open("static/data/hero_data.json") as f:
            data = json.load(f)
            
        # Create role -> heroes mapping
        role_map: Dict[str, List[str]] = {}
        for hero in data['heroes']:
            for role in hero['roles']:
                if role not in role_map:
                    role_map[role] = []
                role_map[role].append(hero['name'])
                
        return role_map
    except Exception as e:
        print(f"Warning: Could not load hero data: {e}")
        return {}

# Initialize hero role mappings
HERO_ROLE_MAP = load_hero_data()

def get_heroes_by_role(role: str) -> Set[str]:
    """Get set of heroes for a given role."""
    return set(HERO_ROLE_MAP.get(role, []))

def plot_hero_matrix_plotly(matrix: np.ndarray,
                          heroes: List[str],
                          title: str = 'Hero Matrix',
                          colorscale: str = 'RdBu',
                          width: int = 800,
                          height: int = 800) -> go.Figure:
    """
    Create an interactive Plotly heatmap for hero interactions.
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=heroes,
        y=heroes,
        colorscale=colorscale,
        zmid=0,
        text=np.round(matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='%{x} vs %{y}<br>Score: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_tickangle=-45,
        yaxis_autorange='reversed'
    )
    
    return fig

def plot_synergy_matrix_plotly(hero_features: Dict[str, Any],
                            roles: Optional[List[str]] = None,
                            min_games: int = 10,
                            min_pick_rate: Optional[float] = None,
                            min_win_rate: Optional[float] = None,
                            width: int = 800,
                            height: int = 800,
                            **kwargs) -> go.Figure:
    """
    Create interactive synergy matrix visualization.
    
    Args:
        hero_features: Dictionary from compute_hero_features()
        roles: Optional list of roles to filter by
        min_games: Minimum games for hero to be included
        min_pick_rate: Optional minimum pick rate to include hero
        min_win_rate: Optional minimum win rate to include hero
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    heroes, matrix = filter_hero_data(
        hero_features,
        roles=roles,
        min_pick_rate=min_pick_rate or (min_games/hero_features.get('total_games', 1000)),
        min_win_rate=min_win_rate
    )
    
    return plot_hero_matrix_plotly(
        matrix,
        heroes,
        title=f'Hero Synergy Matrix {" (" + ", ".join(roles) + ")" if roles else ""}',
        width=width,
        height=height,
        **kwargs
    )

def plot_counter_matrix_plotly(hero_features: Dict[str, Any],
                            roles: Optional[List[str]] = None,
                            min_games: int = 10,
                            min_pick_rate: Optional[float] = None,
                            min_win_rate: Optional[float] = None,
                            width: int = 800,
                            height: int = 800,
                            **kwargs) -> go.Figure:
    """
    Create interactive counter matrix visualization.
    
    Args:
        hero_features: Dictionary from compute_hero_features()
        roles: Optional list of roles to filter by
        min_games: Minimum games for hero to be included
        min_pick_rate: Optional minimum pick rate to include hero
        min_win_rate: Optional minimum win rate to include hero
        width: Plot width in pixels
        height: int = 800,
        **kwargs) -> go.Figure:
    """
    heroes, matrix = filter_hero_data(
        hero_features,
        roles=roles,
        min_pick_rate=min_pick_rate or (min_games/hero_features.get('total_games', 1000)),
        min_win_rate=min_win_rate
    )
    
    # For counter matrix, we transpose to show "X counters Y"
    matrix = matrix.T if len(matrix) > 0 else matrix
    
    return plot_hero_matrix_plotly(
        matrix,
        heroes,
        title=f'Hero Counter Matrix {" (" + ", ".join(roles) + ")" if roles else ""}',
        width=width,
        height=height,
        **kwargs
    )

def create_node_label(node, max_heroes: int = 3) -> str:
    """Create a readable label for a node in the MCTS tree.
    
    Args:
        node: MCTSNode instance
        max_heroes: Maximum number of heroes to show in label
        
    Returns:
        String label for the node
    """
    state = node.state
    win_rate = node.wins / node.visits if node.visits > 0 else 0
    
    # Show last few heroes for each team
    blue = state.blue_picks[-max_heroes:] if state.blue_picks else []
    red = state.red_picks[-max_heroes:] if state.red_picks else []
    
    # Add ... if we truncated
    blue_prefix = "..." if len(state.blue_picks) > max_heroes else ""
    red_prefix = "..." if len(state.red_picks) > max_heroes else ""
    
    phase = "Pick" if state.is_pick_phase else "Ban"
    turn = "Blue" if state.blue_turn else "Red"
    
    return (f"{phase} ({turn})\n"
            f"B: {blue_prefix}{','.join(blue)}\n"
            f"R: {red_prefix}{','.join(red)}\n"
            f"W/V: {node.wins}/{node.visits}\n"
            f"WR: {win_rate:.2%}")

def get_node_color(node) -> float:
    """Get node color value based on win rate.
    
    Args:
        node: MCTSNode instance
        
    Returns:
        Value between 0-1 for colormapping
    """
    if node.visits == 0:
        return 0.5  # Neutral color for unvisited
    return node.wins / node.visits

def create_edge_label(parent_state, child_state) -> str:
    """Create label for edge between nodes.
    
    Args:
        parent_state: DraftState of parent node
        child_state: DraftState of child node
        
    Returns:
        String describing the action taken
    """
    # Find the hero that was added
    for field in ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']:
        parent_list = getattr(parent_state, field)
        child_list = getattr(child_state, field)
        if len(child_list) > len(parent_list):
            hero = child_list[-1]  # Last hero added
            action = "Pick" if field.endswith('picks') else "Ban"
            team = "Blue" if field.startswith('blue') else "Red"
            return f"{team} {action}\n{hero}"
    return ""

def plot_mcts_tree(root_node, max_depth: int = 3,
                   figsize: Tuple[int, int] = (12, 8),
                   node_size_scale: float = 1000,
                   save_path: Optional[Path] = None,
                   title: Optional[str] = None,
                   layout: str = 'kamada_kawai',
                   colormap: str = 'RdYlBu') -> None:
    """
    Visualize MCTS tree starting from root_node.
    
    Args:
        root_node: Root MCTSNode to start visualization from
        max_depth: Maximum depth to visualize (default: 3)
        figsize: Figure size as (width, height) tuple (default: (12,8))
        node_size_scale: Scaling factor for node sizes (default: 1000)
        save_path: Optional path to save figure
        title: Optional title for the plot
        layout: Graph layout algorithm ('kamada_kawai' or 'spring')
        colormap: Matplotlib colormap name for win rates
        
    Notes:
        - Nodes are sized by visit count
        - Node color indicates win rate (blue=high, red=low)
        - Edge labels show actions taken
        - Layout algorithms:
          - kamada_kawai: Better for small trees (slower but more stable)
          - spring: Better for large trees (faster but more chaotic)
    """
    _check_plot_dependencies()
    
    # Create graph
    G = nx.DiGraph()
    node_colors = []
    node_sizes = []
    
    # BFS to build graph up to max_depth
    queue = [(root_node, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
            
        # Add node
        node_id = id(node)  # Use object id as unique identifier
        G.add_node(node_id)
        
        # Store visualization attributes
        node_colors.append(get_node_color(node))
        node_sizes.append(math.sqrt(node.visits + 1) * node_size_scale)
        
        # Add children if not at max depth
        if depth < max_depth:
            for child in node.children:
                child_id = id(child)
                G.add_node(child_id)
                G.add_edge(node_id, child_id)
                queue.append((child, depth + 1))
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.get_cmap(colormap),
        vmin=0, vmax=1
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    
    # Add node labels
    labels = {
        node_id: create_node_label(node)
        for node_id, node in zip(G.nodes(), [n for n, _ in queue])
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add edge labels
    edge_labels = {}
    for u, v in G.edges():
        u_node = next(n for n, _ in queue if id(n) == u)
        v_node = next(n for n, _ in queue if id(n) == v)
        edge_labels[(u, v)] = create_edge_label(u_node.state, v_node.state)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(colormap),
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    plt.colorbar(sm, label='Win Rate')
    
    # Add title
    if title:
        plt.title(title)
    else:
        stats = (f"Tree Depth: {max_depth}, "
                f"Total Nodes: {len(G.nodes())}, "
                f"Root Visits: {root_node.visits}")
        plt.title(stats)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_value_distribution(node, num_bins: int = 30,
                          figsize: Tuple[int, int] = (8, 6),
                          save_path: Optional[Path] = None) -> None:
    """
    Plot distribution of node values (win rates) in the tree.
    
    Args:
        node: Root MCTSNode to analyze
        num_bins: Number of histogram bins (default: 30)
        figsize: Figure size as (width, height)
        save_path: Optional path to save figure
    """
    _check_plot_dependencies()
    
    values = []
    queue = [node]
    
    # Collect all node values
    while queue:
        n = queue.pop(0)
        if n.visits > 0:
            values.append(n.wins / n.visits)
        queue.extend(n.children)
    
    # Create histogram
    plt.figure(figsize=figsize)
    plt.hist(values, bins=num_bins, density=True)
    plt.xlabel('Win Rate')
    plt.ylabel('Density')
    plt.title(f'Distribution of Node Values\n(n={len(values)} nodes)')
    
    # Add summary statistics
    if values:
        mean = np.mean(values)
        std = np.std(values)
        plt.axvline(mean, color='red', linestyle='--',
                   label=f'Mean: {mean:.3f}')
        plt.axvline(mean + std, color='gray', linestyle=':',
                   label=f'±1 SD: {std:.3f}')
        plt.axvline(mean - std, color='gray', linestyle=':')
        plt.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def export_tree_data(root_node, max_depth: int = None) -> Dict[str, Any]:
    """
    Export tree data in a format suitable for external visualization.
    
    Args:
        root_node: Root MCTSNode to export
        max_depth: Optional maximum depth to export
        
    Returns:
        Dictionary with tree data in a format suitable for D3.js or similar
    """
    def process_node(node, depth: int = 0) -> dict:
        if max_depth is not None and depth > max_depth:
            return None
            
        data = {
            'name': f"Node {id(node)}",
            'wins': node.wins,
            'visits': node.visits,
            'win_rate': node.wins / node.visits if node.visits > 0 else 0,
            'state': {
                'blue_picks': node.state.blue_picks,
                'red_picks': node.state.red_picks,
                'blue_bans': node.state.blue_bans,
                'red_bans': node.state.red_bans,
                'blue_turn': node.state.blue_turn,
                'is_pick_phase': node.state.is_pick_phase
            },
            'children': []
        }
        
        for child in node.children:
            child_data = process_node(child, depth + 1)
            if child_data:
                data['children'].append(child_data)
                
        return data
    
    return process_node(root_node)

def plot_draft_sequence(states: List[DraftState],
                       win_probs: List[float],
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: Optional[Path] = None) -> None:
    """
    Plot the sequence of draft states with win probabilities.
    
    Args:
        states: List of DraftState objects in sequence
        win_probs: List of win probabilities for each state
        figsize: Figure size as (width, height)
        save_path: Optional path to save figure
    """
    _check_plot_dependencies()
    
    plt.figure(figsize=figsize)
    
    # Create two subplots
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])  # Draft visualization
    ax2 = plt.subplot(gs[1])  # Win probability
    
    # Plot draft sequence
    n_states = len(states)
    blue_picks = []
    red_picks = []
    blue_bans = []
    red_bans = []
    
    for i, state in enumerate(states):
        # Add picks and bans with x-coordinate for sequence
        for hero in state.blue_picks:
            if hero not in blue_picks:
                blue_picks.append(hero)
                ax1.scatter(i, 4, color='blue', s=100)
                ax1.annotate(hero, (i, 4), rotation=45, ha='right')
                
        for hero in state.red_picks:
            if hero not in red_picks:
                red_picks.append(hero)
                ax1.scatter(i, 3, color='red', s=100)
                ax1.annotate(hero, (i, 3), rotation=45, ha='right')
                
        for hero in state.blue_bans:
            if hero not in blue_bans:
                blue_bans.append(hero)
                ax1.scatter(i, 2, color='blue', alpha=0.5, s=100)
                ax1.annotate(f"BAN {hero}", (i, 2), rotation=45, ha='right')
                
        for hero in state.red_bans:
            if hero not in red_bans:
                red_bans.append(hero)
                ax1.scatter(i, 1, color='red', alpha=0.5, s=100)
                ax1.annotate(f"BAN {hero}", (i, 1), rotation=45, ha='right')
    
    # Customize draft plot
    ax1.set_yticks([1, 2, 3, 4])
    ax1.set_yticklabels(['Red Bans', 'Blue Bans', 'Red Picks', 'Blue Picks'])
    ax1.set_xlim(-0.5, n_states - 0.5)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.set_title("Draft Sequence")
    
    # Plot win probability
    x = range(len(win_probs))
    ax2.plot(x, win_probs, 'b-', label='Blue Win Probability')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-0.5, n_states - 0.5)
    ax2.set_xlabel("Draft Step")
    ax2.set_ylabel("Win Probability")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_win_probability_heatmap(root_node,
                               depth: int = 3,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[Path] = None) -> None:
    """
    Create a heatmap of win probabilities for different draft paths.
    
    Args:
        root_node: Root MCTSNode to analyze
        depth: Depth of paths to analyze
        figsize: Figure size as (width, height)
        save_path: Optional path to save figure
    """
    _check_plot_dependencies()
    
    paths = []
    win_rates = []
    
    def collect_paths(node, current_path=None, current_depth=0):
        if current_path is None:
            current_path = []
            
        if current_depth == depth or not node.children:
            if node.visits > 0:
                paths.append(current_path)
                win_rates.append(node.wins / node.visits)
            return
            
        for child in node.children:
            # Get the action that led to this child
            action = None
            for field in ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']:
                parent_list = getattr(node.state, field)
                child_list = getattr(child.state, field)
                if len(child_list) > len(parent_list):
                    action = child_list[-1]
                    break
                    
            if action:
                collect_paths(child, 
                            current_path + [action],
                            current_depth + 1)
    
    # Collect all paths
    collect_paths(root_node)
    
    if not paths:
        return
        
    # Create matrix for heatmap
    n_paths = len(paths)
    matrix = np.zeros((n_paths, depth))
    matrix.fill(np.nan)
    
    # Fill matrix with win rates
    for i, (path, win_rate) in enumerate(zip(paths, win_rates)):
        for j, action in enumerate(path):
            matrix[i, j] = win_rate
    
    # Create heatmap
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, aspect='auto', cmap='RdYlBu')
    
    # Add colorbar
    plt.colorbar(im, label='Win Probability')
    
    # Customize appearance
    plt.xlabel("Draft Step")
    plt.ylabel("Path")
    plt.title("Win Probabilities Across Draft Paths")
    
    # Add path labels
    path_labels = [' → '.join(p) for p in paths]
    plt.yticks(range(n_paths), path_labels)
    
    # Rotate x-axis labels
    plt.xticks(range(depth), range(1, depth + 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_hero_matrix(matrix: np.ndarray,
                    heroes: List[str],
                    title: str,
                    figsize: Tuple[int, int] = (12, 10),
                    cmap: str = 'RdBu',
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None,
                    annotate: bool = True,
                    highlight_threshold: Optional[float] = None,
                    save_path: Optional[Path] = None) -> None:
    """
    Plot a hero interaction matrix (synergy or counter).
    
    Args:
        matrix: 2D numpy array of hero interactions
        heroes: List of hero names corresponding to matrix indices
        title: Plot title
        figsize: Figure size as (width, height)
        cmap: Colormap name (default: 'RdBu')
        vmin: Optional minimum value for colormap
        vmax: Optional maximum value for colormap
        annotate: Whether to add value annotations (default: True)
        highlight_threshold: Optional threshold to highlight strong interactions
        save_path: Optional path to save figure
    """
    _check_plot_dependencies()
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add hero labels
    plt.xticks(range(len(heroes)), heroes, rotation=45, ha='right')
    plt.yticks(range(len(heroes)), heroes)
    
    # Add title
    plt.title(title)
    
    # Add value annotations
    if annotate:
        for i in range(len(heroes)):
            for j in range(len(heroes)):
                value = matrix[i, j]
                if highlight_threshold and abs(value) >= highlight_threshold:
                    color = 'white' if abs(value) > 0.7 else 'black'
                    weight = 'bold' if abs(value) >= highlight_threshold else 'normal'
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=color, weight=weight)
                elif annotate:
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color='black', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_synergy_matrix(hero_features: Dict[str, Any],
                       min_games: int = 10,
                       highlight_threshold: float = 0.3,
                       **kwargs) -> None:
    """
    Plot hero synergy matrix.
    
    Args:
        hero_features: Dictionary from compute_hero_features()
        min_games: Minimum games for hero to be included
        highlight_threshold: Threshold for highlighting strong synergies
        **kwargs: Additional arguments passed to plot_hero_matrix()
    """
    heroes = hero_features['heroes']
    matrix = hero_features['synergy_matrix']
    
    # Filter by minimum games if raw_stats available
    if 'raw_stats' in hero_features:
        stats = hero_features['raw_stats']
        valid_heroes = [h for h in heroes 
                       if stats[h]['picks'] >= min_games]
        hero_idx = [heroes.index(h) for h in valid_heroes]
        filtered_matrix = matrix[np.ix_(hero_idx, hero_idx)]
    else:
        valid_heroes = heroes
        filtered_matrix = matrix
    
    plot_hero_matrix(
        filtered_matrix,
        valid_heroes,
        title='Hero Synergy Matrix',
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        highlight_threshold=highlight_threshold,
        **kwargs
    )

def plot_counter_matrix(hero_features: Dict[str, Any],
                       min_games: int = 10,
                       highlight_threshold: float = 0.3,
                       **kwargs) -> None:
    """
    Plot hero counter matrix.
    
    Args:
        hero_features: Dictionary from compute_hero_features()
        min_games: Minimum games for hero to be included
        highlight_threshold: Threshold for highlighting strong counters
        **kwargs: Additional arguments passed to plot_hero_matrix()
    """
    heroes = hero_features['heroes']
    matrix = hero_features['counter_matrix']
    
    # Filter by minimum games if raw_stats available
    if 'raw_stats' in hero_features:
        stats = hero_features['raw_stats']
        valid_heroes = [h for h in heroes 
                       if stats[h]['picks'] >= min_games]
        hero_idx = [heroes.index(h) for h in valid_heroes]
        filtered_matrix = matrix[np.ix_(hero_idx, hero_idx)]
    else:
        valid_heroes = heroes
        filtered_matrix = matrix
    
    plot_hero_matrix(
        filtered_matrix,
        valid_heroes,
        title='Hero Counter Matrix',
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        highlight_threshold=highlight_threshold,
        **kwargs
    )

def plot_hero_network(hero_features: Dict[str, Any],
                     min_games: int = 10,
                     min_synergy: float = 0.3,
                     roles: Optional[List[str]] = None,
                     width: int = 800,
                     height: int = 800) -> go.Figure:
    """
    Create an interactive network graph of hero synergies.
    
    Args:
        hero_features: Dictionary from compute_hero_features()
        min_games: Minimum games for hero to be included
        min_synergy: Minimum synergy score to show edge
        roles: Optional list of roles to filter by
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    _check_plot_dependencies()
    
    # Get filtered hero data
    heroes = hero_features['heroes']
    matrix = hero_features['synergy_matrix']
    
    filtered_heroes, mask = filter_hero_data(
        hero_features,
        roles=roles,
        min_pick_rate=None,
        min_win_rate=None
    )
    
    if 'raw_stats' in hero_features:
        stats = hero_features['raw_stats']
        games_mask = np.array([stats[h]['picks'] >= min_games for h in heroes])
        mask &= games_mask
        filtered_heroes = [h for h, m in zip(heroes, mask) if m]
    
    filtered_matrix = matrix[np.ix_(mask, mask)]
    
    # Create network layout
    G = nx.Graph()
    
    # Add nodes
    for i, hero in enumerate(filtered_heroes):
        G.add_node(hero)
    
    # Add edges for strong synergies
    edges = []
    edge_weights = []
    for i in range(len(filtered_heroes)):
        for j in range(i + 1, len(filtered_heroes)):
            synergy = filtered_matrix[i, j]
            if abs(synergy) >= min_synergy:
                edges.append((filtered_heroes[i], filtered_heroes[j]))
                edge_weights.append(synergy)
    
    G.add_edges_from(edges)
    
    # Get position layout
    pos = nx.spring_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_colors = []
    
    for (node1, node2), weight in zip(edges, edge_weights):
        x0, y0 = pos[node1]
        x1, y1 = pos[node2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_colors.extend([weight, weight, weight])
    
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color=edge_colors,
                 colorscale='RdBu',
                 cmin=-1, cmax=1),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for hero in G.nodes():
        x, y = pos[hero]
        node_x.append(x)
        node_y.append(y)
        node_text.append(hero)
        if 'raw_stats' in hero_features:
            size = np.sqrt(stats[hero]['picks']) * 10
        else:
            size = 20
        node_size.append(size)
    
    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            size=node_size,
            line_width=2
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edges_trace, nodes_trace],
        layout=go.Layout(
            title='Hero Synergy Network',
            titlefont_size=16,
            showlegend=False,
            width=width,
            height=height,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def compute_patch_stats(hero_features: Dict[str, Any],
                       patches: Optional[List[str]] = None) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute hero statistics across patches.
    
    Args:
        hero_features: Dictionary from compute_hero_features()
        patches: Optional list of patches to include, in chronological order
        
    Returns:
        Dictionary mapping heroes to their stats across patches
    """
    if 'raw_stats' not in hero_features or 'patch_stats' not in hero_features['raw_stats']:
        return {}
        
    patch_data = hero_features['raw_stats']['patch_stats']
    if not patches:
        patches = sorted(list(set(p for h in patch_data.values() 
                               for p in h.keys())))
    
    hero_stats = {}
    for hero in hero_features['heroes']:
        if hero not in patch_data:
            continue
            
        # Initialize stats
        hero_stats[hero] = {
            'pick_rate': [],
            'win_rate': [],
            'ban_rate': []
        }
        
        # Collect stats for each patch
        for patch in patches:
            patch_stats = patch_data[hero].get(patch, {})
            total_games = patch_stats.get('total_games', 0)
            if total_games > 0:
                pick_rate = patch_stats.get('picks', 0) / total_games
                ban_rate = patch_stats.get('bans', 0) / total_games
                wins = patch_stats.get('wins', 0)
                picks = patch_stats.get('picks', 0)
                win_rate = wins / picks if picks > 0 else 0
                
                hero_stats[hero]['pick_rate'].append(pick_rate)
                hero_stats[hero]['win_rate'].append(win_rate)
                hero_stats[hero]['ban_rate'].append(ban_rate)
            else:
                # Fill with zeros if no data for this patch
                hero_stats[hero]['pick_rate'].append(0)
                hero_stats[hero]['win_rate'].append(0)
                hero_stats[hero]['ban_rate'].append(0)
    
    return hero_stats

def plot_hero_trend(hero_features: Dict[str, Any],
                   heroes: List[str],
                   stat: str = 'win_rate',
                   patches: Optional[List[str]] = None,
                   width: int = 800,
                   height: int = 500) -> go.Figure:
    """Create an interactive line plot of hero statistics over time."""
    _check_plot_dependencies()
    
    # Get patch statistics
    patch_stats = compute_patch_stats(hero_features, patches)
    if not patch_stats:
        raise ValueError("No patch statistics available")
    
    # Create figure
    fig = go.Figure()
    
    # Add line for each hero
    for hero in heroes:
        if hero in patch_stats:
            fig.add_trace(go.Scatter(
                x=patches if patches else sorted(list(set(p for h in patch_stats.values() 
                                                      for p in h[stat]))),
                y=patch_stats[hero][stat],
                name=hero,
                mode='lines+markers',
                customdata=[hero] * len(patches if patches else patch_stats[hero][stat]),
                hovertemplate='%{customdata}<br>%{x}<br>' + f'{stat}: ' + '%{y:.1%}<extra></extra>'
            ))
    
    # Update layout
    stat_titles = {
        'win_rate': 'Win Rate',
        'pick_rate': 'Pick Rate',
        'ban_rate': 'Ban Rate'
    }
    
    fig.update_layout(
        title=f'Hero {stat_titles[stat]} Across Patches',
        width=width,
        height=height,
        xaxis_title='Patch',
        yaxis_title=stat_titles[stat],
        yaxis_tickformat=',.0%',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_meta_snapshot(hero_features: Dict[str, Any],
                      patch: str,
                      min_games: int = 50,
                      width: int = 1000,
                      height: int = 600) -> go.Figure:
    """
    Create an interactive scatter plot showing hero pick rates vs win rates.
    
    Args:
        hero_features: Dictionary from compute_hero_features()
        patch: Specific patch to analyze
        min_games: Minimum games for a hero to be included
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    _check_plot_dependencies()
    
    if 'raw_stats' not in hero_features or 'patch_stats' not in hero_features['raw_stats']:
        raise ValueError("No patch statistics available")
        
    patch_data = hero_features['raw_stats']['patch_stats']
    
    # Collect data points
    x_data = []  # Pick rates
    y_data = []  # Win rates
    sizes = []   # Number of games
    names = []   # Hero names
    roles = []   # Hero roles
    
    for hero in hero_features['heroes']:
        if hero in patch_data and patch in patch_data[hero]:
            stats = patch_data[hero][patch]
            total_games = stats.get('total_games', 0)
            picks = stats.get('picks', 0)
            
            if picks >= min_games:
                pick_rate = picks / total_games
                win_rate = stats.get('wins', 0) / picks
                
                x_data.append(pick_rate)
                y_data.append(win_rate)
                sizes.append(np.sqrt(picks) * 10)
                names.append(hero)
                # Get primary role - you'll need to implement this
                roles.append(get_primary_role(hero, hero_features))
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add reference lines
    avg_wr = np.mean(y_data)
    avg_pr = np.mean(x_data)
    
    fig.add_hline(y=avg_wr, line_dash="dash", line_color="gray",
                  annotation_text="Average Win Rate")
    fig.add_vline(x=avg_pr, line_dash="dash", line_color="gray",
                  annotation_text="Average Pick Rate")
    
    # Add points for each role
    for role in set(roles):
        mask = [r == role for r in roles]
        fig.add_trace(go.Scatter(
            x=[x_data[i] for i in range(len(x_data)) if mask[i]],
            y=[y_data[i] for i in range(len(y_data)) if mask[i]],
            mode='markers+text',
            name=role,
            text=[names[i] for i in range(len(names)) if mask[i]],
            textposition="top center",
            marker=dict(
                size=[sizes[i] for i in range(len(sizes)) if mask[i]],
                sizemode='area',
                sizeref=2.*max(sizes)/(40.**2),
                sizemin=4
            ),
            hovertemplate="%{text}<br>" +
                         "Pick Rate: %{x:.1%}<br>" +
                         "Win Rate: %{y:.1%}<br>" +
                         "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Hero Meta Snapshot - Patch {patch}',
        width=width,
        height=height,
        xaxis_title='Pick Rate',
        yaxis_title='Win Rate',
        xaxis_tickformat=',.0%',
        yaxis_tickformat=',.0%',
        xaxis_range=[-0.01, max(x_data) * 1.1],
        yaxis_range=[0.4, 0.6],
        legend_title='Role',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add quadrant labels
    fig.add_annotation(x=max(x_data), y=0.55,
                      text="Meta Picks",
                      showarrow=False)
    fig.add_annotation(x=min(x_data), y=0.55,
                      text="Situational Strong",
                      showarrow=False)
    fig.add_annotation(x=max(x_data), y=0.45,
                      text="Overused",
                      showarrow=False)
    fig.add_annotation(x=min(x_data), y=0.45,
                      text="Underperforming",
                      showarrow=False)
    
    return fig

def get_primary_role(hero: str, hero_features: Dict[str, Any]) -> str:
    """Get primary role for a hero from hero data.
    
    Handles different formats of hero data that might be present in the system.
    
    Args:
        hero: Hero name
        hero_features: Dictionary containing hero data
        
    Returns:
        Primary role as string or 'Unknown' if not found
    """
    # Try to find role in HERO_ROLES first
    if HERO_ROLES and 'roles' in HERO_ROLES:
        if hero in HERO_ROLES['roles']:
            roles = HERO_ROLES['roles'].get(hero, [])
            if roles:
                return roles[0]
    
    # Format 1: hero_features has 'heroes' as list of dicts
    if 'heroes' in hero_features and isinstance(hero_features['heroes'], list):
        hero_data = next((h for h in hero_features['heroes'] 
                        if h.get('name') == hero), None)
        if hero_data and 'roles' in hero_data and hero_data['roles']:
            return hero_data['roles'][0]
    
    # Format 2: hero_features['heroes'] is a list of strings and roles in hero_features['hero_roles']
    if ('heroes' in hero_features and isinstance(hero_features['heroes'], list) and 
            isinstance(hero_features['heroes'][0], str) and 'hero_roles' in hero_features):
        roles = hero_features['hero_roles'].get(hero, [])
        if roles:
            return roles[0]
            
    # Format 3: direct dictionary mapping in hero_features['raw_stats']
    if 'raw_stats' in hero_features and hero in hero_features['raw_stats']:
        hero_stats = hero_features['raw_stats'][hero]
        if 'roles' in hero_stats and hero_stats['roles']:
            return hero_stats['roles'][0]
    
    # Try using HERO_ROLE_MAP as fallback
    for role, heroes in HERO_ROLE_MAP.items():
        if hero in heroes:
            return role
            
    return 'Unknown'

def plot_time_series(df: pd.DataFrame,
                    heroes: Optional[List[str]] = None,
                    roles: Optional[List[str]] = None, 
                    metrics: List[str] = ['pick_rate', 'ban_rate', 'win_rate'],
                    patches: Optional[List[str]] = None,
                    ci_level: float = 0.95,
                    interactive: bool = True,
                    save_path: Optional[Path] = None) -> Union[None, go.Figure]:
    """
    Create time series visualization of hero performance metrics across patches.
    
    Args:
        df: DataFrame with columns 'hero', 'patch_version', and metric columns
        heroes: Optional list of specific heroes to plot
        roles: Optional list of roles to filter heroes by
        metrics: List of metrics to plot (pick_rate, ban_rate, win_rate)
        patches: Optional list of specific patches to include
        ci_level: Confidence interval level for error bands (0-1)
        interactive: If True, use Plotly, else use Matplotlib
        save_path: Optional path to save the plot
        
    Returns:
        None if save_path provided, else returns the figure object
    """
    _check_plot_dependencies()
    
    # Filter data
    plot_df = df.copy()
    if heroes:
        plot_df = plot_df[plot_df['hero'].isin(heroes)]
    if roles:
        role_heroes = set()
        for role in roles:
            role_heroes.update(get_heroes_by_role(role))
        plot_df = plot_df[plot_df['hero'].isin(role_heroes)]
    if patches:
        plot_df = plot_df[plot_df['patch_version'].isin(patches)]
        
    if interactive:
        fig = go.Figure()
        
        # Add traces for each metric
        for metric in metrics:
            for hero in plot_df['hero'].unique():
                hero_data = plot_df[plot_df['hero'] == hero]
                
                # Calculate Wilson confidence intervals
                n = hero_data['total_games']
                p = hero_data[metric]
                z = stats.norm.ppf((1 + ci_level) / 2)
                ci_low = (p + z*z/(2*n) - z*np.sqrt((p*(1-p) + z*z/(4*n))/n))/(1 + z*z/n)
                ci_high = (p + z*z/(2*n) + z*np.sqrt((p*(1-p) + z*z/(4*n))/n))/(1 + z*z/n)
                
                # Add main line
                fig.add_trace(
                    go.Scatter(
                        x=hero_data['patch_version'],
                        y=hero_data[metric],
                        name=f"{hero} - {metric}",
                        mode='lines+markers',
                        line=dict(width=2),
                        hovertemplate=(
                            "Hero: %{text}<br>" +
                            "Patch: %{x}<br>" +
                            f"{metric}: %{{y:.1%}}<br>" +
                            "<extra></extra>"
                        ),
                        text=[hero]*len(hero_data)
                    )
                )
                
                # Add confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=hero_data['patch_version'].tolist() + hero_data['patch_version'].tolist()[::-1],
                        y=ci_high.tolist() + ci_low.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )
        
        # Update layout
        fig.update_layout(
            title='Hero Performance Trends',
            xaxis_title='Patch Version',
            yaxis_title='Rate',
            hovermode='x unified',
            yaxis=dict(
                tickformat='.0%',
                range=[0, plot_df[metrics].max().max() * 1.1]
            )
        )
        
        if save_path:
            fig.write_html(str(save_path))
            return None
        return fig
        
    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for metric in metrics:
            for hero in plot_df['hero'].unique():
                hero_data = plot_df[plot_df['hero'] == hero]
                
                # Calculate confidence intervals
                n = hero_data['total_games']
                p = hero_data[metric]
                z = stats.norm.ppf((1 + ci_level) / 2)
                ci_low = (p + z*z/(2*n) - z*np.sqrt((p*(1-p) + z*z/(4*n))/n))/(1 + z*z/n)
                ci_high = (p + z*z/(2*n) + z*np.sqrt((p*(1-p) + z*z/(4*n))/n))/(1 + z*z/n)
                
                line = ax.plot(hero_data['patch_version'], 
                             hero_data[metric],
                             label=f"{hero} - {metric}",
                             marker='o')
                
                color = line[0].get_color()
                ax.fill_between(hero_data['patch_version'],
                              ci_low,
                              ci_high,
                              color=color,
                              alpha=0.2)
        
        ax.set_title('Hero Performance Trends')
        ax.set_xlabel('Patch Version')
        ax.set_ylabel('Rate')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return None
        return fig

def plot_role_time_series(role_stats: Dict[str, Dict[str, List[float]]],
                         patches: List[str],
                         metric: str = 'win_rate',
                         title: Optional[str] = None) -> go.Figure:
    """
    Plot time series of role statistics across patches.
    
    Args:
        role_stats: Dictionary of role stats with structure:
            {role: {'win_rate': [...], 'pick_rate': [...], 'ban_rate': [...]}}
        patches: List of patch versions
        metric: Which metric to plot ('win_rate', 'pick_rate', or 'ban_rate')
        title: Optional plot title
    
    Returns:
        Plotly Figure object
    """
    _check_plot_dependencies()
    
    fig = go.Figure()
    
    for role in role_stats:
        fig.add_trace(go.Scatter(
            x=patches,
            y=role_stats[role][metric],
            name=role,
            mode='lines+markers'
        ))
    
    metric_label = ' '.join(metric.split('_')).title()
    fig.update_layout(
        title=title or f'Role {metric_label} Over Time',
        xaxis_title='Patch',
        yaxis_title=metric_label,
        hovermode='x unified'
    )
    
    return fig

def plot_role_matchup_heatmap(role_vs_role: Dict[str, Dict[str, float]],
                            title: str = 'Role Matchup Win Rates') -> go.Figure:
    """
    Create a heatmap showing win rates between different roles.
    
    Args:
        role_vs_role: Dictionary of {attacker_role: {defender_role: win_rate}}
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    _check_plot_dependencies()
    
    roles = list(role_vs_role.keys())
    matrix = np.array([[role_vs_role[atk][def_] for def_ in roles] for atk in roles])
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=roles,
        y=roles,
        colorscale='RdBu',
        text=[[f'{val:.1%}' for val in row] for row in matrix],
        hoverongaps=False,
        hovertemplate='%{y} vs %{x}<br>Win Rate: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Defender Role',
        yaxis_title='Attacker Role',
        xaxis={'side': 'bottom', 'tickangle': 45},
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

def detect_meta_shifts(role_stats: Dict[str, Dict[str, List[float]]],
                      threshold: float = 0.05) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect significant changes in role statistics between patches.
    
    Args:
        role_stats: Dictionary of role stats over time
        threshold: Minimum change to be considered significant
        
    Returns:
        Dictionary of significant changes by role
    """
    shifts = {}
    
    for role, metrics in role_stats.items():
        shifts[role] = []
        for metric, values in metrics.items():
            for i in range(1, len(values)):
                change = values[i] - values[i-1]
                if abs(change) >= threshold:
                    shifts[role].append({
                        'metric': metric,
                        'patch_index': i,
                        'change': change
                    })
    
    return shifts