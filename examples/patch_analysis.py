"""
Example script demonstrating time-series analysis features.

This script shows how to:
1. Track hero performance across patches
2. Generate meta snapshots for specific patches
3. Compare role performance over time
4. Export interactive visualizations
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mlbb.utils.data_loader import MLBBDataLoader
    from simulator.visualization import (
        plot_hero_trend,
        plot_meta_snapshot,
        plot_hero_network,
        plot_hero_matrix
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have installed all requirements:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def analyze_meta_trends(hero_features: Dict[str, Any], 
                       output_dir: Path,
                       min_games: int = 50) -> None:
    """Generate meta trend visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Track top performers across patches
    top_heroes = [
        'Beatrix', 'Ling', 'Yu Zhong', 'Mathilda', 
        'Valentina', 'Xavier', 'Julian', 'Joy'
    ]
    
    # Plot win rates
    win_trend = plot_hero_trend(
        hero_features,
        heroes=top_heroes,
        stat='win_rate',
        width=1000,
        height=600,
        title='Win Rate Trends Across Patches'
    )
    win_trend.write_html(output_dir / "win_rate_trends.html")
    
    # Plot pick rates
    pick_trend = plot_hero_trend(
        hero_features,
        heroes=top_heroes,
        stat='pick_rate',
        width=1000,
        height=600,
        title='Pick Rate Trends Across Patches'
    )
    pick_trend.write_html(output_dir / "pick_rate_trends.html")
    
    # Plot ban rates
    ban_trend = plot_hero_trend(
        hero_features,
        heroes=top_heroes,
        stat='ban_rate',
        width=1000,
        height=600,
        title='Ban Rate Trends Across Patches'
    )
    ban_trend.write_html(output_dir / "ban_rate_trends.html")
    
    # 2. Create meta snapshot for latest patch
    latest_patch = get_latest_patch(hero_features)
    meta_snap = plot_meta_snapshot(
        hero_features,
        patch=latest_patch,
        min_games=min_games,
        width=1200,
        height=800
    )
    meta_snap.write_html(output_dir / "current_meta.html")
    
    # 3. Role synergy analysis for current patch
    roles = ['Tank', 'Fighter', 'Assassin', 'Marksman', 'Mage', 'Support']
    
    # Create role synergy matrix
    synergy_matrix = plot_hero_matrix(
        hero_features['role_synergies'],
        roles,
        title=f'Role Synergy Matrix - Patch {latest_patch}',
        figsize=(10, 8),
        cmap='RdBu',
        annotate=True
    )
    synergy_matrix.write_html(output_dir / "role_synergy_matrix.html")
    
    # Create role network graphs
    for role in roles:
        network = plot_hero_network(
            hero_features,
            roles=[role],
            min_games=min_games,
            min_synergy=0.2,
            width=800,
            height=800,
            title=f'{role} Hero Network - Patch {latest_patch}'
        )
        network.write_html(output_dir / f"{role.lower()}_network.html")

def get_latest_patch(hero_features: Dict[str, Any]) -> str:
    """Get the most recent patch version from hero features."""
    patches = hero_features.get('patches', [])
    return patches[-1] if patches else "unknown"

def analyze_patch_changes(hero_features: Dict[str, Any],
                        patch1: str,
                        patch2: str,
                        min_games: int = 50) -> pd.DataFrame:
    """Analyze hero performance changes between two patches."""
    if not all(p in hero_features['patch_stats'] for p in [patch1, patch2]):
        raise ValueError("One or both patches not found in data")
        
    changes = []
    for hero in hero_features['heroes']:
        stats1 = hero_features['patch_stats'][patch1].get(hero, {})
        stats2 = hero_features['patch_stats'][patch2].get(hero, {})
        
        if (stats1.get('total_games', 0) >= min_games and 
            stats2.get('total_games', 0) >= min_games):
            
            change = {
                'hero': hero,
                'win_rate_change': stats2['win_rate'] - stats1['win_rate'],
                'pick_rate_change': stats2['pick_rate'] - stats1['pick_rate'],
                'ban_rate_change': stats2['ban_rate'] - stats1['ban_rate'],
                'games_played': stats2['total_games']
            }
            changes.append(change)
    
    return pd.DataFrame(changes)

def plot_patch_comparison(changes_df: pd.DataFrame,
                        patch1: str,
                        patch2: str,
                        output_dir: Path) -> None:
    """Create visualization comparing hero performance between patches."""
    # Create subplot figure
    fig = make_subplots(rows=1, cols=3,
                       subplot_titles=['Win Rate Changes',
                                     'Pick Rate Changes',
                                     'Ban Rate Changes'])
    
    metrics = ['win_rate_change', 'pick_rate_change', 'ban_rate_change']
    for i, metric in enumerate(metrics, 1):
        # Sort by absolute change
        df_sorted = changes_df.sort_values(metric, key=abs, ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=df_sorted['hero'],
                y=df_sorted[metric],
                name=metric.replace('_', ' ').title(),
                marker_color=['red' if x < 0 else 'green' for x in df_sorted[metric]]
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        title=f'Hero Performance Changes: {patch1} → {patch2}',
        height=600,
        width=1800,
        showlegend=False,
        barmode='relative'
    )
    
    fig.write_html(output_dir / "patch_comparison.html")

def plot_hero_trend(hero_features, hero_name):
    """Plot win rate trends for a specific hero across patches."""
    patches = hero_features['patches']
    stats = hero_features['patch_stats']
    
    win_rates = [stats[patch].get(hero_name, {}).get('win_rate', 0) for patch in patches]
    pick_rates = [stats[patch].get(hero_name, {}).get('pick_rate', 0) for patch in patches]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(patches, win_rates, 'b-', label='Win Rate')
    ax2.plot(patches, pick_rates, 'r-', label='Pick Rate')
    
    ax1.set_xlabel('Patch')
    ax1.set_ylabel('Win Rate', color='b')
    ax2.set_ylabel('Pick Rate', color='r')
    
    plt.title(f'{hero_name} Performance Trends')
    fig.tight_layout()
    return fig

def plot_role_synergies(hero_features):
    """Plot role synergy heatmap."""
    synergies = hero_features['role_synergies']
    roles = list(synergies.keys())
    
    matrix = [[synergies[r1][r2] for r2 in roles] for r1 in roles]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', 
                xticklabels=roles, yticklabels=roles,
                cmap='RdYlBu')
    plt.title('Role Synergy Matrix')
    return plt.gcf()

def main():
    """Run patch analysis example."""
    try:
        # Initialize data loader
        data_dir = project_root / 'data' / 'raw'
        loader = MLBBDataLoader(str(data_dir))
        loader.load_matches()
        
        # Compute hero features
        hero_features = loader.compute_hero_features()
        
        # Create output directory
        output_dir = project_root / 'examples' / 'output' / 'meta_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate trend visualizations
        analyze_meta_trends(hero_features, output_dir)
        
        # Compare latest two patches
        patches = sorted(hero_features['patches'])
        if len(patches) >= 2:
            prev_patch, latest_patch = patches[-2:]
            changes_df = analyze_patch_changes(
                hero_features,
                prev_patch,
                latest_patch
            )
            plot_patch_comparison(changes_df, prev_patch, latest_patch, output_dir)
            
            # Save raw changes data
            changes_df.to_csv(output_dir / "patch_changes.csv", index=False)
            
        # Plot trends for top picked heroes
        latest_patch = hero_features['patches'][-1]
        top_heroes = sorted(
            hero_features['patch_stats'][latest_patch].items(),
            key=lambda x: x[1]['pick_rate'],
            reverse=True
        )[:5]
        
        for hero_name, _ in top_heroes:
            fig = plot_hero_trend(hero_features, hero_name)
            fig.savefig(os.path.join(output_dir, f'{hero_name}_trends.png'))
            plt.close(fig)
        
        # Plot role synergies
        fig = plot_role_synergies(hero_features)
        fig.savefig(os.path.join(output_dir, 'role_synergies.png'))
        plt.close(fig)
            
    except Exception as e:
        print(f"Error running patch analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()