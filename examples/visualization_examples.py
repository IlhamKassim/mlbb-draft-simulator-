"""
Examples of using the MLBB draft analytics visualization features.

This script demonstrates:
1. MCTS tree visualization
2. Draft sequence analysis
3. Win probability heatmaps
4. Hero synergy/counter matrices
5. Value distribution analysis
"""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List

from data_loader import MLBBDataLoader
from simulator.mcts import MCTSDraftSimulator, DraftState
from simulator.visualization import (
    plot_mcts_tree,
    plot_draft_sequence,
    plot_win_probability_heatmap,
    plot_value_distribution,
    plot_synergy_matrix,
    plot_counter_matrix
)

def load_draft_state(path: str) -> DraftState:
    """Load draft state from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return DraftState(
        blue_picks=data['blue_picks'],
        red_picks=data['red_picks'],
        blue_bans=data['blue_bans'],
        red_bans=data['red_bans'],
        blue_turn=data.get('blue_turn', True),
        is_pick_phase=data.get('is_pick_phase', False)
    )

def plot_hero_trend(hero_stats: Dict[str, Dict], patches: List[str], metric: str = 'pick_rate', figsize=(12, 6)):
    """Plot trend of a specific metric across patches for heroes."""
    plt.figure(figsize=figsize)
    
    df = pd.DataFrame({
        'Patch': patches * len(hero_stats),
        'Hero': [hero for hero in hero_stats.keys() for _ in patches],
        'Value': [hero_stats[hero][patch][metric] for hero in hero_stats.keys() for patch in patches]
    })
    
    sns.lineplot(data=df, x='Patch', y='Value', hue='Hero')
    plt.title(f'Hero {metric.replace("_", " ").title()} Trends Across Patches')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()

def plot_meta_summary(patch_stats: Dict[str, Dict], patches: List[str], top_n: int = 10):
    """Plot summary of meta changes across patches."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['pick_rate', 'ban_rate', 'win_rate']
    
    for patch in patches:
        heroes = list(patch_stats[patch].keys())
        for metric in metrics:
            values = [patch_stats[patch][hero][metric] for hero in heroes]
            # Sort heroes by metric value
            sorted_pairs = sorted(zip(heroes, values), key=lambda x: x[1], reverse=True)
            top_heroes = [x[0] for x in sorted_pairs[:top_n]]
            top_values = [x[1] for x in sorted_pairs[:top_n]]
            
            # Plot bar charts for top heroes
            if metric == 'pick_rate':
                ax = axs[0, 0]
            elif metric == 'ban_rate':
                ax = axs[0, 1]
            else:  # win_rate
                ax = axs[1, 0]
                
            ax.bar(top_heroes, top_values)
            ax.set_title(f'Top {top_n} Heroes by {metric.replace("_", " ").title()}')
            ax.tick_params(axis='x', rotation=45)
    
    # Add patch summary text
    summary_text = []
    for patch in patches:
        n_picked = sum(1 for h in patch_stats[patch].values() if h['pick_rate'] > 0.1)
        n_banned = sum(1 for h in patch_stats[patch].values() if h['ban_rate'] > 0.1)
        summary_text.append(f'Patch {patch}:\n{n_picked} heroes picked >10%\n{n_banned} heroes banned >10%')
    
    axs[1, 1].axis('off')
    axs[1, 1].text(0.1, 0.5, '\n\n'.join(summary_text), fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_role_distribution(patch_stats: Dict[str, Dict], patches: List[str], roles: List[str]):
    """Plot role distribution changes across patches."""
    plt.figure(figsize=(12, 6))
    
    role_counts = {patch: {role: 0 for role in roles} for patch in patches}
    for patch in patches:
        for hero, stats in patch_stats[patch].items():
            if stats['pick_rate'] > 0.1:  # Consider heroes with >10% pick rate
                role_counts[patch][stats['role']] += 1
    
    df = pd.DataFrame(role_counts).T
    df.plot(kind='bar', stacked=True)
    plt.title('Role Distribution Across Patches')
    plt.xlabel('Patch')
    plt.ylabel('Number of Heroes with >10% Pick Rate')
    plt.legend(title='Roles', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    return plt.gcf()

def plot_role_trends(patch_stats: Dict[str, Dict], patches: List[str], role_data: Dict, figsize=(15, 10)):
    """
    Plot comprehensive role trend analysis across patches.
    
    Args:
        patch_stats: Dictionary of patch statistics
        patches: List of patch versions
        role_data: Role configuration data
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Prepare role-based statistics
    role_stats = {patch: {} for patch in patches}
    for patch in patches:
        role_counts = {role: 0 for role in role_data['roles'].keys()}
        total_picks = 0
        
        for hero, stats in patch_stats[patch].items():
            if 'roles' in stats:
                for role in stats['roles']:
                    role_counts[role] += stats['pick_rate']
                    total_picks += stats['pick_rate']
        
        # Normalize to percentages
        for role in role_counts:
            role_stats[patch][role] = (role_counts[role] / total_picks * 100) if total_picks > 0 else 0
    
    # Plot role distribution trends
    df_trends = pd.DataFrame(role_stats).T
    df_trends.plot(kind='area', stacked=True, ax=ax1)
    ax1.set_title('Role Distribution Trends Across Patches')
    ax1.set_xlabel('Patch')
    ax1.set_ylabel('Role Distribution (%)')
    ax1.legend(title='Roles', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot role synergy heatmap for the latest patch
    latest_patch = patches[-1]
    role_synergy = {r1: {r2: 0 for r2 in role_data['roles']} for r1 in role_data['roles']}
    
    # Calculate role synergies from win rates
    for hero1, stats1 in patch_stats[latest_patch].items():
        if 'synergy' in stats1:
            for hero2, winrate in stats1['synergy'].items():
                if hero2 in patch_stats[latest_patch]:
                    for role1 in stats1.get('roles', []):
                        for role2 in patch_stats[latest_patch][hero2].get('roles', []):
                            role_synergy[role1][role2] += winrate

    # Plot synergy heatmap
    df_synergy = pd.DataFrame(role_synergy)
    sns.heatmap(df_synergy, annot=True, cmap='RdYlBu', ax=ax2)
    ax2.set_title(f'Role Synergy Heatmap (Patch {latest_patch})')
    
    plt.tight_layout()
    return fig

def main():
    # Setup paths
    data_dir = Path("data/raw")
    model_path = Path("models/baseline/baseline.joblib")
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    model = joblib.load(model_path)
    loader = MLBBDataLoader(data_dir)
    loader.load_matches()
    
    # Get available heroes and compute features
    hero_features = loader.compute_hero_features()
    available_heroes = hero_features['heroes']
    
    # Initialize simulator
    simulator = MCTSDraftSimulator(
        model=model,
        data_loader=loader,
        available_heroes=available_heroes,
        iterations=1000
    )
    
    # 1. Example: Visualize draft states at different stages
    print("\n1. Analyzing draft progression...")
    stages = ['early_draft.json', 'mid_draft.json', 'late_draft.json']
    states = []
    win_probs = []
    
    for stage in stages:
        state = load_draft_state(f"examples/{stage}")
        states.append(state)
        features = loader.prepare_model_features(
            state.blue_picks, state.red_picks,
            state.blue_bans, state.red_bans
        )
        win_prob = model.predict_proba([features])[0][1]
        win_probs.append(win_prob)
        
        # Run MCTS and plot tree for each stage
        print(f"\nAnalyzing {stage}...")
        action, prob = simulator.get_best_action(state)
        print(f"Best action: {action} (win probability: {prob:.1%})")
        
        plot_mcts_tree(
            simulator.root,
            max_depth=2,
            save_path=output_dir / f"mcts_tree_{stage.split('.')[0]}.png",
            title=f"MCTS Tree - {stage.split('.')[0].replace('_', ' ').title()}"
        )
    
    # Plot draft sequence
    print("\n2. Plotting draft sequence...")
    plot_draft_sequence(
        states,
        win_probs,
        save_path=output_dir / "draft_sequence.png"
    )
    
    # 3. Win probability heatmap for late draft
    print("\n3. Generating win probability heatmap...")
    late_state = load_draft_state("examples/late_draft.json")
    action, _ = simulator.get_best_action(late_state)
    plot_win_probability_heatmap(
        simulator.root,
        depth=2,
        save_path=output_dir / "win_prob_heatmap.png"
    )
    
    # 4. Hero synergy and counter analysis
    print("\n4. Analyzing hero interactions...")
    
    # Synergy matrix
    print("- Plotting synergy matrix...")
    plot_synergy_matrix(
        hero_features,
        min_games=10,
        highlight_threshold=0.3,
        save_path=output_dir / "synergy_matrix.png"
    )
    
    # Counter matrix
    print("- Plotting counter matrix...")
    plot_counter_matrix(
        hero_features,
        min_games=10,
        highlight_threshold=0.3,
        save_path=output_dir / "counter_matrix.png"
    )
    
    # 5. Value distribution for final state
    print("\n5. Analyzing value distribution...")
    plot_value_distribution(
        simulator.root,
        save_path=output_dir / "value_distribution.png"
    )
    
    print("\nAll visualizations have been saved to examples/output/")
    print("Generated files:")
    print("- mcts_tree_early_draft.png")
    print("- mcts_tree_mid_draft.png")
    print("- mcts_tree_late_draft.png")
    print("- draft_sequence.png")
    print("- win_prob_heatmap.png")
    print("- synergy_matrix.png")
    print("- counter_matrix.png")
    print("- value_distribution.png")

if __name__ == "__main__":
    main()