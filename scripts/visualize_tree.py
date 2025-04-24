"""
Script to demonstrate MCTS tree visualization.
"""
import argparse
from pathlib import Path
import json
import joblib

from simulator.mcts import MCTSDraftSimulator, DraftState
from simulator.visualization import (
    plot_mcts_tree,
    plot_value_distribution,
    export_tree_data
)
from data_loader import MLBBDataLoader

def main(data_dir: str, model_path: str, draft_json: str,
         output_dir: str, max_depth: int = 3,
         iterations: int = 1000):
    """
    Visualize MCTS tree for a given draft state.
    
    Args:
        data_dir: Path to data directory
        model_path: Path to trained model
        draft_json: Path to JSON file with draft state
        output_dir: Directory to save visualizations
        max_depth: Maximum tree depth to visualize
        iterations: Number of MCTS iterations
    """
    # Load model and data
    print("Loading model and data...")
    model = joblib.load(model_path)
    data_loader = MLBBDataLoader(data_dir)
    data_loader.load_matches()
    
    # Get available heroes
    hero_features = data_loader.compute_hero_features()
    available_heroes = hero_features['heroes']
    
    # Load draft state
    print("Loading draft state...")
    with open(draft_json) as f:
        draft = json.load(f)
    
    state = DraftState(
        blue_picks=draft['blue_picks'],
        red_picks=draft['red_picks'],
        blue_bans=draft['blue_bans'],
        red_bans=draft['red_bans'],
        blue_turn=draft.get('blue_turn', True),
        is_pick_phase=draft.get('is_pick_phase', False)
    )
    
    # Initialize simulator
    print(f"Running MCTS with {iterations} iterations...")
    simulator = MCTSDraftSimulator(
        model=model,
        data_loader=data_loader,
        available_heroes=available_heroes,
        iterations=iterations
    )
    
    # Run simulation
    action, win_prob = simulator.get_best_action(state)
    print(f"\nBest action: {action} (win probability: {win_prob:.1%})")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Tree visualization
    plot_mcts_tree(
        simulator.root,
        max_depth=max_depth,
        save_path=output_dir / "mcts_tree.png",
        title=f"MCTS Tree (depth={max_depth}, iterations={iterations})"
    )
    print("- Saved tree visualization to mcts_tree.png")
    
    # 2. Value distribution
    plot_value_distribution(
        simulator.root,
        save_path=output_dir / "value_dist.png"
    )
    print("- Saved value distribution to value_dist.png")
    
    # 3. Export tree data for external viz
    tree_data = export_tree_data(simulator.root, max_depth=max_depth)
    with open(output_dir / "tree_data.json", 'w') as f:
        json.dump(tree_data, f, indent=2)
    print("- Exported tree data to tree_data.json")
    
    # 4. Print top recommendations
    print("\nTop 5 recommended actions:")
    rankings = simulator.get_action_rankings(state, top_k=5)
    for hero, prob in rankings:
        print(f"- {hero:<15} {prob:>6.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MCTS tree for MLBB drafting")
    parser.add_argument('--data', required=True, help="Path to data directory")
    parser.add_argument('--model', required=True, help="Path to trained model")
    parser.add_argument('--draft', required=True, help="Path to draft state JSON")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--depth', type=int, default=3, help="Maximum tree depth")
    parser.add_argument('--iterations', type=int, default=1000,
                       help="Number of MCTS iterations")
    
    args = parser.parse_args()
    main(args.data, args.model, args.draft, args.output,
         args.depth, args.iterations)