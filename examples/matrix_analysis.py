"""
Example script demonstrating filtered hero interaction analysis.

This script shows how to:
1. Filter hero matrices by role, pick rate, and win rate
2. Generate interactive Plotly visualizations
3. Create network graphs of hero synergies
4. Export filtered data as HTML and JSON
"""
import json
from pathlib import Path

from data_loader import MLBBDataLoader
from simulator.visualization import (
    plot_synergy_matrix_plotly,
    plot_counter_matrix_plotly,
    plot_hero_network
)

def main():
    # Setup paths
    data_dir = Path("data/raw")
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading match data...")
    loader = MLBBDataLoader(data_dir)
    loader.load_matches()
    
    # Compute hero features
    hero_features = loader.compute_hero_features()
    
    # 1. Role-based analysis
    print("\n1. Analyzing tank synergies...")
    tank_fig = plot_synergy_matrix_plotly(
        hero_features,
        roles=["Tank"],
        min_games=50,
        width=800,
        height=800
    )
    tank_fig.write_html(output_dir / "tank_synergies.html")
    
    # Also create network graph for tanks
    tank_network = plot_hero_network(
        hero_features,
        roles=["Tank"],
        min_games=50,
        min_synergy=0.3,
        width=800,
        height=800
    )
    tank_network.write_html(output_dir / "tank_network.html")
    
    print("\n2. Analyzing marksman counters...")
    marksman_fig = plot_counter_matrix_plotly(
        hero_features,
        roles=["Marksman"],
        min_games=50,
        width=800,
        height=800
    )
    marksman_fig.write_html(output_dir / "marksman_counters.html")
    
    # 2. Performance-filtered analysis
    print("\n3. Analyzing high win-rate hero synergies...")
    winrate_fig = plot_synergy_matrix_plotly(
        hero_features,
        min_win_rate=0.55,  # Only heroes with >55% win rate
        min_games=100,
        width=1000,
        height=800
    )
    winrate_fig.write_html(output_dir / "high_winrate_synergies.html")
    
    # Create network for high win-rate heroes
    winrate_network = plot_hero_network(
        hero_features,
        min_games=100,
        min_synergy=0.3,
        width=1000,
        height=800
    )
    winrate_network.write_html(output_dir / "winrate_network.html")
    
    # 3. Meta analysis (high pick rate)
    print("\n4. Analyzing meta hero interactions...")
    meta_fig = plot_counter_matrix_plotly(
        hero_features,
        min_pick_rate=0.1,  # Only heroes picked in >10% of games
        min_games=100,
        width=1000,
        height=800
    )
    meta_fig.write_html(output_dir / "meta_counters.html")
    
    print("\nAll visualizations have been saved to examples/output/")
    print("Generated files:")
    print("- tank_synergies.html (matrix)")
    print("- tank_network.html (network graph)")
    print("- marksman_counters.html")
    print("- high_winrate_synergies.html (matrix)")
    print("- winrate_network.html (network graph)")
    print("- meta_counters.html")

if __name__ == "__main__":
    main()