"""
Data loader module for MLBB draft prediction.

This module handles loading and preprocessing match data from various sources,
computing hero-level features, and preparing data for model training.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, NamedTuple, Literal
import math
from enum import Enum
from datetime import datetime
import gzip
import bz2
import lzma

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Supported compression formats
CompressionType = Literal['gzip', 'bz2', 'xz', None]

def get_compression_opener(compression: CompressionType) -> callable:
    """Get the appropriate file opener for the compression type.
    
    Args:
        compression: Type of compression to use
        
    Returns:
        File opener function
        
    Raises:
        ValueError: If compression type is not supported
    """
    if compression == 'gzip':
        return gzip.open
    elif compression == 'bz2':
        return bz2.open
    elif compression == 'xz':
        return lzma.open
    elif compression is None:
        return open
    else:
        raise ValueError(f"Unsupported compression type: {compression}")

class EffectSize(Enum):
    """Effect size thresholds for Cohen's h statistic."""
    NEGLIGIBLE = 0.20  # Below this is considered negligible
    SMALL = 0.50      # Between NEGLIGIBLE and SMALL is a small effect
    MEDIUM = 0.80     # Between SMALL and MEDIUM is a medium effect
                      # Above MEDIUM is a large effect

class HeroBias(NamedTuple):
    """Container for hero side bias statistics."""
    hero: str
    effect_size: float
    category: str
    blue_rate: float
    red_rate: float
    blue_games: int
    red_games: int
    blue_ci: Tuple[float, float]
    red_ci: Tuple[float, float]

def get_z_score(ci_level: float = 0.95) -> float:
    """Calculate Z-score for a given confidence level.
    
    Args:
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Z-score corresponding to the confidence level
        
    Raises:
        ValueError: If ci_level is not between 0 and 1
    """
    if not 0 < ci_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    return stats.norm.ppf((1 + ci_level) / 2)

def wilson_interval(wins: int, n: int, ci_level: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for a proportion.
    
    Args:
        wins: Number of successes (wins)
        n: Total number of trials (games)
        ci_level: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower bound, upper bound)
        
    Note:
        Uses the Wilson score interval formula with specified confidence level
    """
    if n == 0:
        return (0.0, 0.0)
        
    z = get_z_score(ci_level)
    
    # Add z²/2 successes and z²/2 failures (continuity correction)
    p_hat = wins / n
    z2 = z * z
    
    # Calculate Wilson score interval
    denominator = 1 + z2/n
    center = (p_hat + z2/(2*n)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2/(4*n)) / n) / denominator
    
    return (
        max(0.0, center - spread),  # Clamp to [0,1]
        min(1.0, center + spread)
    )

def cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h effect size between two proportions.
    
    Args:
        p1: First proportion
        p2: Second proportion
        
    Returns:
        Cohen's h statistic
        
    Note:
        h = 2 * (arcsin(√p₁) - arcsin(√p₂))
        Interpretation guidelines:
        |h| < 0.20: negligible effect
        0.20 ≤ |h| < 0.50: small effect
        0.50 ≤ |h| < 0.80: medium effect
        |h| ≥ 0.80: large effect
    """
    # Handle edge cases to avoid math domain errors
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

def get_effect_size_category(h: float) -> str:
    """Get descriptive category for a Cohen's h value.
    
    Args:
        h: Cohen's h statistic (absolute value)
        
    Returns:
        String description of effect size category
    """
    h = abs(h)
    if h < EffectSize.NEGLIGIBLE.value:
        return "negligible"
    elif h < EffectSize.SMALL.value:
        return "small"
    elif h < EffectSize.MEDIUM.value:
        return "medium"
    else:
        return "large"

class MLBBDataLoader:
    """Load and preprocess MLBB match data for draft prediction."""
    
    REQUIRED_COLUMNS = {
        'match_id', 'patch_version', 'blue_picks', 'red_picks',
        'blue_bans', 'red_bans', 'winner'
    }
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing match data files
        """
        self.data_dir = Path(data_dir)
        self.matches = None
        self.hero_features = None
        self.heroes = []  # Initialize empty hero list
        self.hero_roles = self._load_hero_roles()
        
    def _load_hero_roles(self) -> Dict:
        """Load hero role mappings from JSON file."""
        role_file = self.data_dir.parent / 'static' / 'data' / 'hero_roles.json'
        with open(role_file) as f:
            return json.load(f)
        
    def load_matches(self, filename: str = "sample_matches.csv") -> pd.DataFrame:
        """Load match data from file.
        
        Args:
            filename: Name of file to load (supports CSV and JSON)
            
        Returns:
            DataFrame containing match data
            
        Raises:
            ValueError: If file format is unsupported or required columns are missing
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
            
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            with open(file_path) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Verify required columns
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert string representations of lists to actual lists
        list_cols = ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']
        for col in list_cols:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
        self.matches = df
        return df
        
    def compute_hero_features(self, patch_version: Optional[str] = None) -> Dict:
        """Compute hero-level features from match data.
        
        Args:
            patch_version: Optional patch version to filter matches
            
        Returns:
            Dictionary containing hero features:
                - heroes: List of all unique heroes
                - pick_rates: Dict mapping hero names to pick rates ∈ [0,1]
                  (normalized by 2*n_matches since each match has 10 picks)
                - ban_rates: Dict mapping hero names to ban rates ∈ [0,1]
                  (normalized by 2*n_matches since each match has 6 bans)
                - win_rates: Dict mapping hero names to win rates ∈ [0,1]
                  (wins divided by total games played)
                - side_win_rates: Dict mapping hero names to dict of blue/red rates ∈ [0,1]
                  (wins divided by games played on each side)
                - side_win_ci: Dict mapping hero names to dict of blue/red 95% CIs
                  (Wilson score intervals for win rates on each side)
                - side_effect_size: Dict mapping hero names to Cohen's h statistic
                  (standardized effect size of blue vs. red side win rate difference)
                - synergy_matrix: Matrix of hero synergy scores ∈ [-1,1]
                  (positive values indicate good synergy, normalized by n_matches)
                - counter_matrix: Matrix of hero counter scores ∈ [-1,1]
                  (positive values indicate hero i counters hero j, normalized by n_matches)
                
        Raises:
            ValueError: If no match data has been loaded, or if computed features
                      violate expected constraints:
                      - Rates not in [0,1]
                      - Matrices not in [-1,1]
                      - Matrix symmetry properties violated
                      - Sum constraints not met (pick_rates sum to 1.0,
                        ban_rates sum to 0.6, synergy effects sum to 0)
            
        Notes:
            - All rates and scores are normalized to be in fixed ranges for comparability
            - Synergy matrix is symmetric (synergy_matrix[i,j] = synergy_matrix[j,i])
            - Counter matrix is anti-symmetric (counter_matrix[i,j] = -counter_matrix[j,i])
            - For balanced matchups, synergy effects sum to approximately zero
            - Numerical comparisons use a tolerance of 1e-8 for floating-point arithmetic
        """
        if self.matches is None:
            raise ValueError("No match data loaded. Call load_matches() first.")
            
        # Filter by patch version if specified
        matches = self.matches
        if patch_version:
            matches = matches[matches['patch_version'] == patch_version]
            if len(matches) == 0:
                logger.warning(f"No matches found for patch {patch_version}")
                
        # Get unique heroes from all matches
        heroes = set()
        for col in ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']:
            for lst in matches[col]:
                heroes.update(lst)
        heroes = sorted(list(heroes))
        n_heroes = len(heroes)
        hero_to_idx = {hero: idx for idx, hero in enumerate(heroes)}
        
        # Initialize feature dictionaries and raw statistics
        pick_rates = {hero: 0.0 for hero in heroes}
        ban_rates = {hero: 0.0 for hero in heroes}
        win_rates = {hero: [0, 0] for hero in heroes}  # [wins, total]
        
        # Track raw counts for debugging
        raw_stats = {hero: {
            'picks': 0,
            'bans': 0,
            'wins': 0,
            'losses': 0,
            'blue_side': 0,
            'red_side': 0,
            'wins_blue': 0,
            'losses_blue': 0,
            'wins_red': 0,
            'losses_red': 0
        } for hero in heroes}
        
        # Initialize matrices
        synergy_matrix = np.zeros((n_heroes, n_heroes))
        counter_matrix = np.zeros((n_heroes, n_heroes))
        
        # Compute features from matches
        n_matches = len(matches)
        if n_matches == 0:
            self.hero_features = {
                'heroes': heroes,
                'pick_rates': pick_rates,
                'ban_rates': ban_rates,
                'win_rates': {h: 0.0 for h in heroes},
                'side_win_rates': {h: {'blue': 0.0, 'red': 0.0} for h in heroes},
                'side_win_ci': {h: {'blue': (0.0, 0.0), 'red': (0.0, 0.0)} for h in heroes},
                'side_effect_size': {h: 0.0 for h in heroes},
                'synergy_matrix': synergy_matrix,
                'counter_matrix': counter_matrix,
                'raw_stats': raw_stats
            }
            return self.hero_features
            
        for _, match in matches.iterrows():
            # Update pick rates and raw counts
            for hero in match['blue_picks']:
                pick_rates[hero] += 1
                raw_stats[hero]['picks'] += 1
                raw_stats[hero]['blue_side'] += 1
                if match['winner'] == 'blue':
                    raw_stats[hero]['wins'] += 1
                    raw_stats[hero]['wins_blue'] += 1
                else:
                    raw_stats[hero]['losses'] += 1
                    raw_stats[hero]['losses_blue'] += 1
                
            for hero in match['red_picks']:
                pick_rates[hero] += 1
                raw_stats[hero]['picks'] += 1
                raw_stats[hero]['red_side'] += 1
                if match['winner'] == 'red':
                    raw_stats[hero]['wins'] += 1
                    raw_stats[hero]['wins_red'] += 1
                else:
                    raw_stats[hero]['losses'] += 1
                    raw_stats[hero]['losses_red'] += 1
                
            # Update ban rates and raw counts
            for hero in match['blue_bans'] + match['red_bans']:
                ban_rates[hero] += 1
                raw_stats[hero]['bans'] += 1
                
            # Update win rates and raw counts
            winner = match['winner']
            winning_team = match['blue_picks'] if winner == 'blue' else match['red_picks']
            losing_team = match['red_picks'] if winner == 'blue' else match['blue_picks']
            
            for hero in winning_team:
                win_rates[hero][0] += 1  # wins
                win_rates[hero][1] += 1  # total
                raw_stats[hero]['wins'] += 1
                
            for hero in losing_team:
                win_rates[hero][1] += 1  # total
                raw_stats[hero]['losses'] += 1
                
            # Update synergy matrix
            for i, hero1 in enumerate(winning_team):
                idx1 = hero_to_idx[hero1]
                for hero2 in winning_team[i+1:]:
                    idx2 = hero_to_idx[hero2]
                    synergy_matrix[idx1, idx2] += 1
                    synergy_matrix[idx2, idx1] += 1
                    
            for i, hero1 in enumerate(losing_team):
                idx1 = hero_to_idx[hero1]
                for hero2 in losing_team[i+1:]:
                    idx2 = hero_to_idx[hero2]
                    synergy_matrix[idx1, idx2] -= 1
                    synergy_matrix[idx2, idx1] -= 1
                    
            # Update counter matrix
            for hero1 in winning_team:
                idx1 = hero_to_idx[hero1]
                for hero2 in losing_team:
                    idx2 = hero_to_idx[hero2]
                    counter_matrix[idx1, idx2] += 1
                    counter_matrix[idx2, idx1] -= 1
                    
        # Normalize features
        for hero in heroes:
            pick_rates[hero] /= (10 * n_matches)  # divide by total picks (10 picks per match)
            ban_rates[hero] /= (10 * n_matches)   # normalize ban rate to be comparable with pick rate
            wins, total = win_rates[hero]
            win_rates[hero] = wins / total if total > 0 else 0.0
            
        # Normalize matrices by number of matches to get average effects
        synergy_matrix /= n_matches
        counter_matrix /= n_matches
            
        # Compute side-specific win rates, CIs, and effect sizes
        side_win_rates = {}
        side_win_ci = {}
        side_effect_size = {}
        
        for hero in heroes:
            stats = raw_stats[hero]
            games_blue = stats['wins_blue'] + stats['losses_blue']
            games_red = stats['wins_red'] + stats['losses_red']
            
            # Compute point estimates
            blue_rate = stats['wins_blue'] / games_blue if games_blue > 0 else 0.0
            red_rate = stats['wins_red'] / games_red if games_red > 0 else 0.0
            
            side_win_rates[hero] = {
                'blue': blue_rate,
                'red': red_rate
            }
            
            # Compute Wilson score intervals
            side_win_ci[hero] = {
                'blue': wilson_interval(stats['wins_blue'], games_blue),
                'red': wilson_interval(stats['wins_red'], games_red)
            }
            
            # Compute Cohen's h effect size
            if games_blue > 0 and games_red > 0:
                h = cohens_h(blue_rate, red_rate)
                side_effect_size[hero] = h
            else:
                side_effect_size[hero] = 0.0
            
        # Store computed features
        self.hero_features = {
            'heroes': heroes,
            'pick_rates': pick_rates,
            'ban_rates': ban_rates,
            'win_rates': win_rates,
            'side_win_rates': side_win_rates,
            'side_win_ci': side_win_ci,
            'side_effect_size': side_effect_size,
            'synergy_matrix': synergy_matrix,
            'counter_matrix': counter_matrix,
            'raw_stats': raw_stats
        }
        
        # ─── Validation Checks ───────────────────────────────────────────
        tol = 1e-8  # Tolerance for floating-point comparisons
        
        # 1. Verify all rates are in [0,1]
        for name, rates in [('pick_rates', pick_rates),
                          ('ban_rates', ban_rates),
                          ('win_rates', win_rates)]:
            for hero, rate in rates.items():
                if not (0.0 <= rate <= 1.0):
                    stats = raw_stats[hero]
                    games_blue = stats['wins_blue'] + stats['losses_blue']
                    games_red = stats['wins_red'] + stats['losses_red']
                    error_msg = [
                        f"{name}[{hero}]={rate:.3f} out of bounds [0,1]",
                        f"Raw statistics for {hero}:",
                        f"- Picks: {stats['picks']} ({stats['blue_side']} blue, {stats['red_side']} red)",
                        f"- Bans: {stats['bans']}",
                        f"- Overall: {stats['wins']}/{stats['picks']} = {stats['wins']/stats['picks']:.3f}",
                        f"- Blue side: {stats['wins_blue']}/{games_blue} = {side_win_rates[hero]['blue']:.3f}",
                        f"- Red side: {stats['wins_red']}/{games_red} = {side_win_rates[hero]['red']:.3f}",
                        f"Number of matches processed: {n_matches}"
                    ]
                    raise ValueError("\n".join(error_msg))
        
        # 1c. Verify side-specific win rates and check for significant bias
        for hero, rates in side_win_rates.items():
            stats = raw_stats[hero]
            games_blue = stats['wins_blue'] + stats['losses_blue']
            games_red = stats['wins_red'] + stats['losses_red']
            
            # Verify rates are in [0,1]
            for side, rate in rates.items():
                if not (0.0 <= rate <= 1.0):
                    error_msg = [
                        f"side_win_rates[{hero}][{side}]={rate:.3f} out of bounds [0,1]",
                        f"Raw statistics for {hero} on {side} side:",
                        f"- Games: {games_blue if side == 'blue' else games_red}",
                        f"- Wins: {stats['wins_blue'] if side == 'blue' else stats['wins_red']}",
                        f"- Losses: {stats['losses_blue'] if side == 'blue' else stats['losses_red']}",
                        f"Overall win rate: {win_rates[hero]:.3f}"
                    ]
                    raise ValueError("\n".join(error_msg))
            
            # Check for statistically significant side bias
            if games_blue >= 10 and games_red >= 10:  # Require minimum sample size
                ci_blue = side_win_ci[hero]['blue']
                ci_red = side_win_ci[hero]['red']
                h = side_effect_size[hero]
                effect_cat = get_effect_size_category(h)
                
                # Check if CIs don't overlap or effect size is notable
                if (ci_blue[1] < ci_red[0] or ci_red[1] < ci_blue[0] or 
                    abs(h) >= EffectSize.SMALL.value):
                    logger.warning(
                        f"Side bias detected for {hero}:\n"
                        f"- Blue side: {rates['blue']:.3f} [{ci_blue[0]:.3f}, {ci_blue[1]:.3f}] "
                        f"({stats['wins_blue']}/{games_blue})\n"
                        f"- Red side: {rates['red']:.3f} [{ci_red[0]:.3f}, {ci_red[1]:.3f}] "
                        f"({stats['wins_red']}/{games_red})\n"
                        f"- Effect size: {h:.3f} ({effect_cat} effect)\n"
                        f"- Sample sizes: {games_blue} blue / {games_red} red games"
                    )
        
        # 1b. Verify sum constraints
        total_picks = sum(pick_rates.values())
        if abs(total_picks - 1.0) > tol:
            # Get pick distribution for debugging
            sorted_picks = sorted([(h, pick_rates[h], raw_stats[h]['picks']) 
                                 for h in heroes], 
                                key=lambda x: x[1], reverse=True)
            top_5 = sorted_picks[:5]
            error_msg = [
                f"Total pick_rates sum to {total_picks:.6f}, expected 1.0",
                f"This suggests an error in pick rate normalization.",
                f"Expected total picks = {10 * n_matches} (10 per match × {n_matches} matches)",
                "\nTop 5 most picked heroes:",
                *[f"- {h}: {rate:.3f} ({picks} picks)" for h, rate, picks in top_5]
            ]
            raise ValueError("\n".join(error_msg))
            
        total_bans = sum(ban_rates.values())
        if abs(total_bans - 0.6) > tol:
            # Get ban distribution for debugging
            sorted_bans = sorted([(h, ban_rates[h], raw_stats[h]['bans']) 
                                for h in heroes], 
                               key=lambda x: x[1], reverse=True)
            top_5 = sorted_bans[:5]
            error_msg = [
                f"Total ban_rates sum to {total_bans:.6f}, expected 0.6",
                f"This suggests an error in ban rate normalization.",
                f"Expected total bans = {6 * n_matches} (6 per match × {n_matches} matches)",
                "\nTop 5 most banned heroes:",
                *[f"- {h}: {rate:.3f} ({bans} bans)" for h, rate, bans in top_5]
            ]
            raise ValueError("\n".join(error_msg))
        
        # 2. Verify matrices are in [-1,1]
        if not (np.all(synergy_matrix >= -1) and np.all(synergy_matrix <= 1)):
            min_val = np.min(synergy_matrix)
            max_val = np.max(synergy_matrix)
            # Find extreme synergy pairs
            i_min, j_min = np.unravel_index(np.argmin(synergy_matrix), synergy_matrix.shape)
            i_max, j_max = np.unravel_index(np.argmax(synergy_matrix), synergy_matrix.shape)
            error_msg = [
                f"synergy_matrix entries out of [-1,1] bounds: [{min_val:.3f}, {max_val:.3f}]",
                f"This may indicate incorrect normalization by n_matches={n_matches}",
                "\nMost negative synergy:",
                f"- {heroes[i_min]} + {heroes[j_min]} = {synergy_matrix[i_min,j_min]:.3f}",
                "\nMost positive synergy:",
                f"- {heroes[i_max]} + {heroes[j_max]} = {synergy_matrix[i_max,j_max]:.3f}"
            ]
            raise ValueError("\n".join(error_msg))
            
        if not (np.all(counter_matrix >= -1) and np.all(counter_matrix <= 1)):
            min_val = np.min(counter_matrix)
            max_val = np.max(counter_matrix)
            # Find extreme counter pairs
            i_min, j_min = np.unravel_index(np.argmin(counter_matrix), counter_matrix.shape)
            i_max, j_max = np.unravel_index(np.argmax(counter_matrix), counter_matrix.shape)
            error_msg = [
                f"counter_matrix entries out of [-1,1] bounds: [{min_val:.3f}, {max_val:.3f}]",
                f"This may indicate incorrect normalization by n_matches={n_matches}",
                "\nStrongest counter relationship:",
                f"- {heroes[i_max]} counters {heroes[j_max]} ({counter_matrix[i_max,j_max]:.3f})",
                f"- {heroes[j_max]} counters {heroes[i_max]} ({counter_matrix[j_max,i_max]:.3f})"
            ]
            raise ValueError("\n".join(error_msg))
        
        # 3. Verify matrix symmetry properties
        if not np.allclose(synergy_matrix, synergy_matrix.T, atol=tol):
            max_diff = np.max(np.abs(synergy_matrix - synergy_matrix.T))
            i, j = np.unravel_index(np.argmax(np.abs(synergy_matrix - synergy_matrix.T)), synergy_matrix.shape)
            error_msg = [
                f"synergy_matrix asymmetry detected (max difference: {max_diff:.3e})",
                f"Largest asymmetry between heroes:",
                f"- {heroes[i]} + {heroes[j]}: {synergy_matrix[i,j]:.3f}",
                f"- {heroes[j]} + {heroes[i]}: {synergy_matrix[j,i]:.3f}",
                "\nRaw statistics:",
                f"- {heroes[i]}: {raw_stats[heroes[i]]['wins']}/{raw_stats[heroes[i]]['picks']} games",
                f"- {heroes[j]}: {raw_stats[heroes[j]]['wins']}/{raw_stats[heroes[j]]['picks']} games"
            ]
            raise ValueError("\n".join(error_msg))
            
        if not np.allclose(counter_matrix, -counter_matrix.T, atol=tol):
            max_diff = np.max(np.abs(counter_matrix + counter_matrix.T))
            i, j = np.unravel_index(np.argmax(np.abs(counter_matrix + counter_matrix.T)), counter_matrix.shape)
            error_msg = [
                f"counter_matrix symmetry error (max difference: {max_diff:.3e})",
                f"Largest error between heroes:",
                f"- {heroes[i]} vs {heroes[j]}: {counter_matrix[i,j]:.3f}",
                f"- {heroes[j]} vs {heroes[i]}: {counter_matrix[j,i]:.3f}",
                "\nRaw statistics:",
                f"- {heroes[i]}: {raw_stats[heroes[i]]['wins']}/{raw_stats[heroes[i]]['picks']} games",
                f"- {heroes[j]}: {raw_stats[heroes[j]]['wins']}/{raw_stats[heroes[j]]['picks']} games"
            ]
            raise ValueError("\n".join(error_msg))
        
        # 4. Synergy sum should be ~0 for balanced data
        total_synergy = np.sum(synergy_matrix)
        if abs(total_synergy) > tol:
            # Find heroes with most extreme synergy effects
            hero_synergies = np.sum(synergy_matrix, axis=1)
            top_3 = np.argsort(hero_synergies)[-3:]
            bot_3 = np.argsort(hero_synergies)[:3]
            error_msg = [
                f"Synergy matrix sum = {total_synergy:.3e}, expected ~0",
                f"This may indicate an imbalance in match outcomes.",
                "\nHeroes with highest total synergy:",
                *[f"- {heroes[i]}: {hero_synergies[i]:.3f} ({raw_stats[heroes[i]]['wins']}/{raw_stats[heroes[i]]['picks']} games)"
                  for i in reversed(top_3)],
                "\nHeroes with lowest total synergy:",
                *[f"- {heroes[i]}: {hero_synergies[i]:.3f} ({raw_stats[heroes[i]]['wins']}/{raw_stats[heroes[i]]['picks']} games)"
                  for i in bot_3]
            ]
            raise ValueError("\n".join(error_msg))
        # ─────────────────────────────────────────────────────────────────
        
        return self.hero_features
        
    def prepare_model_features(self, blue_picks: List[str], red_picks: List[str],
                             blue_bans: List[str], red_bans: List[str]) -> np.ndarray:
        """Prepare feature vector for model prediction.
        
        Args:
            blue_picks: List of heroes picked by blue team
            red_picks: List of heroes picked by red team
            blue_bans: List of heroes banned by blue team
            red_bans: List of heroes banned by red team
            
        Returns:
            Feature vector for model prediction
            
        Raises:
            ValueError: If hero features haven't been computed
        """
        if self.hero_features is None:
            raise ValueError("Hero features not computed. Call compute_hero_features() first.")
            
        heroes = self.hero_features['heroes']
        n_heroes = len(heroes)
        hero_to_idx = {hero: idx for idx, hero in enumerate(heroes)}
        
        # Initialize feature vector
        features = np.zeros(4 * n_heroes)  # picks and bans for both teams
        
        # Set pick features
        for hero in blue_picks:
            if hero in hero_to_idx:
                features[hero_to_idx[hero]] = 1
                
        for hero in red_picks:
            if hero in hero_to_idx:
                features[n_heroes + hero_to_idx[hero]] = 1
                
        # Set ban features
        for hero in blue_bans:
            if hero in hero_to_idx:
                features[2 * n_heroes + hero_to_idx[hero]] = 1
                
        for hero in red_bans:
            if hero in hero_to_idx:
                features[3 * n_heroes + hero_to_idx[hero]] = 1
                
        return features
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names corresponding to feature vector.
        
        Returns:
            List of feature names
            
        Raises:
            ValueError: If hero features haven't been computed
        """
        if self.hero_features is None:
            raise ValueError("Hero features not computed. Call compute_hero_features() first.")
            
        heroes = self.hero_features['heroes']
        feature_names = []
        
        for team in ['blue', 'red']:
            for action in ['pick', 'ban']:
                for hero in heroes:
                    feature_names.append(f"{team}_{action}_{hero}")
                    
        return feature_names
        
    def rank_side_bias(self, 
                      min_games: int = 10,
                      top_n: Optional[int] = None,
                      ascending: bool = False) -> List[HeroBias]:
        """Rank heroes by their side bias effect size.
        
        Args:
            min_games: Minimum number of games required on each side (default: 10)
            top_n: Optional number of heroes to return (default: all)
            ascending: Sort in ascending order of effect size (default: False)
            
        Returns:
            List of HeroBias tuples containing:
                - hero: Hero name
                - effect_size: Cohen's h statistic
                - category: Effect size category
                - blue_rate: Win rate on blue side
                - red_rate: Win rate on red side
                - blue_games: Number of games on blue side
                - red_games: Number of games on red side
                - blue_ci: Confidence interval for blue side win rate
                - red_ci: Confidence interval for red side win rate
            
        Raises:
            ValueError: If hero features haven't been computed
        """
        if self.hero_features is None:
            raise ValueError("Hero features not computed. Call compute_hero_features() first.")
            
        # Get required data from hero_features
        side_effect_size = self.hero_features['side_effect_size']
        side_win_rates = self.hero_features['side_win_rates']
        side_win_ci = self.hero_features['side_win_ci']
        raw_stats = self.hero_features['raw_stats']
        
        # Build list of hero bias statistics
        hero_biases = []
        for hero in self.hero_features['heroes']:
            stats = raw_stats[hero]
            games_blue = stats['wins_blue'] + stats['losses_blue']
            games_red = stats['wins_red'] + stats['losses_red']
            
            # Skip heroes with insufficient games
            if games_blue < min_games or games_red < min_games:
                continue
                
            hero_biases.append(HeroBias(
                hero=hero,
                effect_size=side_effect_size[hero],
                category=get_effect_size_category(side_effect_size[hero]),
                blue_rate=side_win_rates[hero]['blue'],
                red_rate=side_win_rates[hero]['red'],
                blue_games=games_blue,
                red_games=games_red,
                blue_ci=side_win_ci[hero]['blue'],
                red_ci=side_win_ci[hero]['red']
            ))
        
        # Sort by absolute effect size (or raw effect size if ascending)
        key_func = lambda x: x.effect_size if ascending else abs(x.effect_size)
        hero_biases.sort(key=key_func, reverse=not ascending)
        
        # Apply top_n filter if specified
        if top_n is not None:
            hero_biases = hero_biases[:top_n]
            
        return hero_biases
        
    def summarize_side_bias(self, 
                           min_games: int = 10,
                           min_effect: float = EffectSize.SMALL.value) -> str:
        """Generate a human-readable summary of side bias statistics.
        
        Args:
            min_games: Minimum number of games required on each side
            min_effect: Minimum effect size to include in summary
            
        Returns:
            Multi-line string containing bias summary
            
        Raises:
            ValueError: If hero features haven't been computed
        """
        if self.hero_features is None:
            raise ValueError("Hero features not computed. Call compute_hero_features() first.")
            
        # Get heroes ranked by effect size
        biases = self.rank_side_bias(min_games=min_games)
        
        # Filter for notable effects
        notable = [b for b in biases if abs(b.effect_size) >= min_effect]
        if not notable:
            return "No notable side biases detected."
            
        # Build summary
        lines = ["Side Bias Summary:"]
        lines.append("-" * 50)
        
        # Group by effect size category
        by_category = {
            "large": [],
            "medium": [],
            "small": []
        }
        
        for bias in notable:
            if bias.category in by_category:
                by_category[bias.category].append(bias)
                
        # Add each category to summary
        for category in ["large", "medium", "small"]:
            heroes = by_category[category]
            if heroes:
                lines.append(f"\n{category.title()} Effects:")
                for bias in heroes:
                    favored = "BLUE" if bias.effect_size > 0 else "RED"
                    lines.append(
                        f"  {bias.hero:15} h={bias.effect_size:6.3f} "
                        f"({favored} favored)\n"
                        f"    Blue: {bias.blue_rate:.3f} [{bias.blue_ci[0]:.3f}, "
                        f"{bias.blue_ci[1]:.3f}] ({bias.blue_games} games)\n"
                        f"    Red:  {bias.red_rate:.3f} [{bias.red_ci[0]:.3f}, "
                        f"{bias.red_ci[1]:.3f}] ({bias.red_games} games)"
                    )
                    
        return "\n".join(lines)
        
    def plot_side_bias(self,
                      min_games: int = 10,
                      top_n: int = 20,
                      figsize: Tuple[int, int] = (10, 8),
                      show_thresholds: bool = True,
                      ci_level: float = 0.95,
                      save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot a forest plot of hero side bias effect sizes with confidence intervals.
        
        Creates a horizontal forest plot showing Cohen's h effect sizes for the top-N
        heroes with the strongest side biases. Includes confidence intervals and visual
        indicators for effect size thresholds.
        
        Args:
            min_games: Minimum games required on each side (default: 10)
            top_n: Number of heroes to display (default: 20)
            figsize: Figure size as (width, height) tuple (default: (10, 8))
            show_thresholds: Whether to show effect size threshold lines (default: True)
            ci_level: Confidence level for intervals (default: 0.95)
            save_path: Optional path to save the plot (default: None, shows plot)
            
        Notes:
            - Positive values (right side) indicate blue-side advantage
            - Negative values (left side) indicate red-side advantage
            - Error bars show confidence intervals at specified level
            - Vertical lines (if shown) indicate small/medium/large effect thresholds
        """
        if self.hero_features is None:
            raise ValueError("Hero features not computed. Call compute_hero_features() first.")
            
        # Get top-N biased heroes
        biases = self.rank_side_bias(min_games=min_games, top_n=top_n)
        if not biases:
            logger.warning("No heroes found meeting the minimum games requirement.")
            return
            
        # Prepare data
        heroes = [b.hero for b in biases]
        effects = [b.effect_size for b in biases]
        
        # Compute CIs
        cis_low = []
        cis_high = []
        hover_text = []
        
        for b in biases:
            # Most conservative estimate of the difference
            ci_low = b.blue_ci[0] - b.red_ci[1]
            ci_high = b.blue_ci[1] - b.red_ci[0]
            cis_low.append(ci_low)
            cis_high.append(ci_high)
            
            # Create detailed hover text
            hover_text.append(
                f"Hero: {b.hero}<br>"
                f"Effect size (h): {b.effect_size:.3f}<br>"
                f"Category: {b.category}<br>"
                f"Blue side: {b.blue_rate:.3f} [{b.blue_ci[0]:.3f}, {b.blue_ci[1]:.3f}]<br>"
                f"Red side: {b.red_rate:.3f} [{b.red_ci[0]:.3f}, {b.red_ci[1]:.3f}]<br>"
                f"Sample sizes: {b.blue_games} blue / {b.red_games} red"
            )
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(heroes))
        
        # Plot CI error bars
        ax.hlines(y=y_pos, xmin=cis_low, xmax=cis_high,
                 color='gray', alpha=0.3, linewidth=2)
        
        # Plot effect sizes with different colors based on direction
        blue_mask = np.array(effects) > 0
        red_mask = ~blue_mask
        
        # Blue-favored heroes
        if any(blue_mask):
            ax.scatter(np.array(effects)[blue_mask], 
                      y_pos[blue_mask],
                      color='blue', label='Blue-favored', s=50)
            
        # Red-favored heroes
        if any(red_mask):
            ax.scatter(np.array(effects)[red_mask], 
                      y_pos[red_mask],
                      color='red', label='Red-favored', s=50)
            
        # Add effect size threshold lines
        if show_thresholds:
            for effect, style in [
                (EffectSize.NEGLIGIBLE.value, ':'),
                (EffectSize.SMALL.value, '--'),
                (EffectSize.MEDIUM.value, '-.')
            ]:
                ax.axvline(effect, color='gray', linestyle=style, alpha=0.5)
                ax.axvline(-effect, color='gray', linestyle=style, alpha=0.5)
                
        # Add zero line
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{h} ({g.blue_games + g.red_games} games)" 
                           for h, g in zip(heroes, biases)])
        
        ax.set_xlabel("Cohen's h (Blue vs. Red Side Effect Size)")
        ax.set_title("Hero Side Bias Analysis\n"
                    f"({int(ci_level*100)}% CIs, {min_games}+ games per side)")
        
        # Add legend for effect size thresholds
        if show_thresholds:
            threshold_lines = [
                plt.Line2D([0], [0], color='gray', linestyle=':', label='Negligible (h=0.2)'),
                plt.Line2D([0], [0], color='gray', linestyle='--', label='Small (h=0.5)'),
                plt.Line2D([0], [0], color='gray', linestyle='-.', label='Medium (h=0.8)')
            ]
            ax.legend(handles=threshold_lines +
                     ([plt.Line2D([0], [0], color='blue', marker='o', label='Blue-favored', linestyle='None')] if any(blue_mask) else []) +
                     ([plt.Line2D([0], [0], color='red', marker='o', label='Red-favored', linestyle='None')] if any(red_mask) else []),
                     loc='center left', bbox_to_anchor=(1, 0.5))
            
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
            
    def plotly_side_bias(self,
                        min_games: int = 10,
                        top_n: int = 20,
                        show_thresholds: bool = True,
                        ci_level: float = 0.95,
                        height: int = 600,
                        save_path: Optional[Union[str, Path]] = None) -> Optional[go.Figure]:
        """Create an interactive Plotly forest plot of hero side bias effect sizes.
        
        Args:
            min_games: Minimum games required on each side (default: 10)
            top_n: Number of heroes to display (default: 20)
            show_thresholds: Whether to show effect size threshold lines (default: True)
            ci_level: Confidence level for intervals (default: 0.95)
            height: Plot height in pixels (default: 600)
            save_path: Optional path to save as HTML (default: None)
            
        Returns:
            Plotly Figure object if Plotly is available, None otherwise
            
        Notes:
            Requires plotly package to be installed.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly is not installed. Run 'pip install plotly' to use interactive plots.")
            return None
            
        # Get hero data
        biases = self.rank_side_bias(min_games=min_games, top_n=top_n)
        if not biases:
            logger.warning("No heroes found meeting the minimum games requirement.")
            return None
            
        # Prepare data
        heroes = [b.hero for b in biases]
        effects = [b.effect_size for b in biases]
        
        # Compute CIs
        cis_low = []
        cis_high = []
        hover_text = []
        
        for b in biases:
            # Most conservative estimate of the difference
            ci_low = b.blue_ci[0] - b.red_ci[1]
            ci_high = b.blue_ci[1] - b.red_ci[0]
            cis_low.append(ci_low)
            cis_high.append(ci_high)
            
            # Create detailed hover text
            hover_text.append(
                f"Hero: {b.hero}<br>"
                f"Effect size (h): {b.effect_size:.3f}<br>"
                f"Category: {b.category}<br>"
                f"Blue side: {b.blue_rate:.3f} [{b.blue_ci[0]:.3f}, {b.blue_ci[1]:.3f}]<br>"
                f"Red side: {b.red_rate:.3f} [{b.red_ci[0]::.3f}, {b.red_ci[1]:.3f}]<br>"
                f"Sample sizes: {b.blue_games} blue / {b.red_games} red"
            )
            
        # Create figure
        fig = go.Figure()
        
        # Add CI error bars
        fig.add_trace(go.Scatter(
            x=effects,
            y=heroes,
            error_x=dict(
                type='data',
                symmetric=False,
                array=[h - l for h, l in zip(cis_high, effects)],
                arrayminus=[e - l for e, l in zip(effects, cis_low)],
                color='rgba(128, 128, 128, 0.3)',
                thickness=2
            ),
            mode='markers',
            marker=dict(
                color=['blue' if e > 0 else 'red' for e in effects],
                size=8
            ),
            name='Effect Size',
            hovertext=hover_text,
            hoverinfo='text'
        ))
        
        # Add threshold lines
        if show_thresholds:
            for effect, dash in [
                (EffectSize.NEGLIGIBLE.value, 'dot'),
                (EffectSize.SMALL.value, 'dash'),
                (EffectSize.MEDIUM.value, 'dashdot')
            ]:
                for sign in [-1, 1]:
                    fig.add_shape(
                        type='line',
                        x0=sign * effect,
                        x1=sign * effect,
                        y0=-1,
                        y1=len(heroes),
                        line=dict(
                            color='gray',
                            width=1,
                            dash=dash
                        )
                    )
                    
        # Add zero line
        fig.add_shape(
            type='line',
            x0=0, x1=0,
            y0=-1, y1=len(heroes),
            line=dict(color='black', width=1)
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Hero Side Bias Analysis<br>'
                     f'({int(ci_level*100)}% CIs, {min_games}+ games per side)',
                x=0.5
            ),
            xaxis_title="Cohen's h (Blue vs. Red Side Effect Size)",
            yaxis_title='Hero',
            height=height,
            showlegend=False,
            hovermode='closest'
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            
        return fig 

    def _prepare_bias_export_data(self,
                                min_games: int = 10,
                                top_n: Optional[int] = None,
                                include_metadata: bool = True,
                                extended_stats: bool = True) -> Tuple[List[Dict], Dict]:
        """Prepare side bias data for export.
        
        Args:
            min_games: Minimum games required on each side
            top_n: Optional number of heroes to include
            include_metadata: Whether to include export metadata
            extended_stats: Whether to include extended statistics in metadata
            
        Returns:
            Tuple of (bias_data, metadata)
            
        Raises:
            ValueError: If hero features haven't been computed
        """
        if self.hero_features is None:
            raise ValueError("Hero features not computed. Call compute_hero_features() first.")
            
        biases = self.rank_side_bias(min_games=min_games, top_n=top_n)
        if not biases:
            logger.warning("No bias data to export.")
            return [], {}
            
        # Prepare bias data
        data = []
        for b in biases:
            data.append({
                'hero': b.hero,
                'effect_size': b.effect_size,
                'category': b.category,
                'favored_side': 'blue' if b.effect_size > 0 else 'red',
                'blue_rate': b.blue_rate,
                'red_rate': b.red_rate,
                'blue_games': b.blue_games,
                'red_games': b.red_games,
                'total_games': b.blue_games + b.red_games,
                'blue_ci_low': b.blue_ci[0],
                'blue_ci_high': b.blue_ci[1],
                'red_ci_low': b.red_ci[0],
                'red_ci_high': b.red_ci[1],
                'win_rate_diff': b.blue_rate - b.red_rate
            })
            
        # Prepare metadata
        metadata = {}
        if include_metadata:
            # Basic metadata
            metadata.update({
                'export_date': datetime.now().isoformat(),
                'total_heroes': len(self.hero_features['heroes']),
                'heroes_analyzed': len(biases),
                'min_games_per_side': min_games,
                'effect_size_thresholds': {
                    'negligible': EffectSize.NEGLIGIBLE.value,
                    'small': EffectSize.SMALL.value,
                    'medium': EffectSize.MEDIUM.value
                }
            })
            
            # Side bias summary
            metadata['summary'] = {
                'blue_favored': sum(1 for b in biases if b.effect_size > 0),
                'red_favored': sum(1 for b in biases if b.effect_size < 0),
                'by_category': {
                    'large': sum(1 for b in biases if b.category == 'large'),
                    'medium': sum(1 for b in biases if b.category == 'medium'),
                    'small': sum(1 for b in biases if b.category == 'small'),
                    'negligible': sum(1 for b in biases if b.category == 'negligible')
                }
            }
            
            if extended_stats:
                # Effect size statistics
                effect_sizes = np.array([b.effect_size for b in biases])
                abs_effects = np.abs(effect_sizes)
                
                metadata['effect_size_stats'] = {
                    'mean': float(np.mean(effect_sizes)),
                    'median': float(np.median(effect_sizes)),
                    'std': float(np.std(effect_sizes)),
                    'abs_mean': float(np.mean(abs_effects)),
                    'abs_median': float(np.median(abs_effects)),
                    'percentiles': {
                        '10': float(np.percentile(abs_effects, 10)),
                        '25': float(np.percentile(abs_effects, 25)),
                        '75': float(np.percentile(abs_effects, 75)),
                        '90': float(np.percentile(abs_effects, 90))
                    }
                }
                
                # Win rate statistics
                blue_rates = np.array([b.blue_rate for b in biases])
                red_rates = np.array([b.red_rate for b in biases])
                rate_diffs = blue_rates - red_rates
                
                metadata['win_rate_stats'] = {
                    'blue_mean': float(np.mean(blue_rates)),
                    'red_mean': float(np.mean(red_rates)),
                    'diff_mean': float(np.mean(rate_diffs)),
                    'diff_std': float(np.std(rate_diffs))
                }
                
                # Sample size statistics
                blue_games = np.array([b.blue_games for b in biases])
                red_games = np.array([b.red_games for b in biases])
                total_games = blue_games + red_games
                
                metadata['sample_size_stats'] = {
                    'total_games': int(np.sum(total_games)),
                    'mean_games_per_hero': float(np.mean(total_games)),
                    'median_games_per_hero': float(np.median(total_games)),
                    'blue_total': int(np.sum(blue_games)),
                    'red_total': int(np.sum(red_games)),
                    'min_games': int(np.min(total_games)),
                    'max_games': int(np.max(total_games))
                }
                
                # Top heroes by absolute effect size
                top_k = min(5, len(biases))
                top_heroes = sorted(biases, key=lambda b: abs(b.effect_size), reverse=True)[:top_k]
                
                metadata['top_biased_heroes'] = [
                    {
                        'hero': b.hero,
                        'effect_size': b.effect_size,
                        'category': b.category,
                        'favored_side': 'blue' if b.effect_size > 0 else 'red',
                        'games': b.blue_games + b.red_games,
                        'blue_rate': b.blue_rate,
                        'red_rate': b.red_rate
                    }
                    for b in top_heroes
                ]
                
                # Statistical tests
                _, p_value = stats.normaltest(effect_sizes)
                metadata['statistical_tests'] = {
                    'normality_test': {
                        'test': 'D\'Agostino-Pearson',
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
                }
            
        return data, metadata
        
    def export_side_bias_csv(self,
                            path: Union[str, Path],
                            min_games: int = 10,
                            top_n: Optional[int] = None,
                            include_metadata: bool = True,
                            extended_stats: bool = True,
                            compression: CompressionType = None) -> None:
        """Export side bias ranking to a CSV file.
        
        Args:
            path: Path to save the CSV file
            min_games: Minimum games required on each side
            top_n: Optional number of heroes to include
            include_metadata: Whether to include metadata in a separate JSON file
            extended_stats: Whether to include extended statistics in metadata
            compression: Type of compression to use ('gzip', 'bz2', 'xz', or None)
        """
        path = Path(path)
        data, metadata = self._prepare_bias_export_data(
            min_games=min_games,
            top_n=top_n,
            include_metadata=include_metadata,
            extended_stats=extended_stats
        )
        
        if not data:
            return
            
        # Prepare paths and add compression extensions if needed
        if compression:
            if not str(path).endswith(f'.{compression}'):
                path = Path(f"{path}.{compression}")
            
        # Export main data
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, compression=compression)
        logger.info(f"Side bias data exported to CSV: {path}")
        
        # Export metadata if requested
        if include_metadata:
            meta_path = path.with_suffix('.meta.json')
            if compression:
                meta_path = Path(f"{meta_path}.{compression}")
                
            opener = get_compression_opener(compression)
            with opener(meta_path, 'wt', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata exported to: {meta_path}")
            
    def export_side_bias_json(self,
                             path: Union[str, Path],
                             min_games: int = 10,
                             top_n: Optional[int] = None,
                             include_metadata: bool = True,
                             extended_stats: bool = True,
                             indent: int = 2,
                             compression: CompressionType = None) -> None:
        """Export side bias ranking to a JSON file.
        
        Args:
            path: Path to save the JSON file
            min_games: Minimum games required on each side
            top_n: Optional number of heroes to include
            include_metadata: Whether to include metadata in the JSON
            extended_stats: Whether to include extended statistics in metadata
            indent: Number of spaces for JSON indentation
            compression: Type of compression to use ('gzip', 'bz2', 'xz', or None)
        """
        path = Path(path)
        data, metadata = self._prepare_bias_export_data(
            min_games=min_games,
            top_n=top_n,
            include_metadata=include_metadata,
            extended_stats=extended_stats
        )
        
        if not data:
            return
            
        # Add compression extension if needed
        if compression and not str(path).endswith(f'.{compression}'):
            path = Path(f"{path}.{compression}")
            
        # Combine data and metadata
        export_data = {
            'bias_data': data,
            'metadata': metadata if include_metadata else None
        }
        
        # Export to JSON with appropriate compression
        opener = get_compression_opener(compression)
        with opener(path, 'wt', encoding='utf-8') as f:
            json.dump(export_data, f, indent=indent)
        logger.info(f"Side bias data exported to JSON: {path}")
        
    def get_side_bias_dataframe(self,
                               min_games: int = 10,
                               top_n: Optional[int] = None,
                               to_csv: Optional[Union[str, Path]] = None,
                               compression: CompressionType = None) -> Optional[pd.DataFrame]:
        """Get side bias data as a pandas DataFrame.
        
        Args:
            min_games: Minimum games required on each side
            top_n: Optional number of heroes to include
            to_csv: Optional path to save DataFrame as CSV
            compression: Type of compression to use if saving
            
        Returns:
            DataFrame containing side bias data, or None if no data
        """
        data, _ = self._prepare_bias_export_data(
            min_games=min_games,
            top_n=top_n,
            include_metadata=False
        )
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # Optionally save to CSV
        if to_csv is not None:
            path = Path(to_csv)
            if compression and not str(path).endswith(f'.{compression}'):
                path = Path(f"{path}.{compression}")
            df.to_csv(path, index=False, compression=compression)
            logger.info(f"DataFrame exported to CSV: {path}")
            
        return df

    def compute_hero_stats_by_patch(self):
        """Compute hero features and statistics across patches."""
        hero_features = {
            'heroes': list(self.heroes),
            'patches': sorted(set(match['patch'] for match in self.matches)),
            'patch_stats': self.patch_stats(),
            'role_synergies': self._compute_role_synergies()
        }
        return hero_features

    def _compute_role_synergies(self):
        """Compute synergy matrix between different roles."""
        roles = ['Tank', 'Fighter', 'Assassin', 'Marksman', 'Mage', 'Support']
        synergies = {role: {other: 0 for other in roles} for role in roles}
        counts = {role: {other: 0 for other in roles} for role in roles}
        
        for match in self.matches:
            for team in ['blue_team', 'red_team']:
                team_heroes = match[team]
                team_roles = [self.hero_roles.get(hero, 'Unknown') for hero in team_heroes]
                team_won = (match['winner'] == 'blue') if team == 'blue_team' else (match['winner'] == 'red')
                
                for i, role1 in enumerate(team_roles):
                    for role2 in team_roles[i+1:]:
                        if role1 in roles and role2 in roles:
                            counts[role1][role2] += 1
                            counts[role2][role1] += 1
                            if team_won:
                                synergies[role1][role2] += 1
                                synergies[role2][role1] += 1
        
        # Convert to win rates
        for role1 in roles:
            for role2 in roles:
                if counts[role1][role2] > 0:
                    synergies[role1][role2] = synergies[role1][role2] / counts[role1][role2]
                
        return synergies

    def compute_temporal_stats(self, 
                             window_size: str = '7D',
                             min_games: int = 10) -> Dict[str, pd.DataFrame]:
        """Compute rolling statistics for heroes over time.
        
        Args:
            window_size: Size of rolling window (e.g. '7D' for 7 days)
            min_games: Minimum games required in window for stats
            
        Returns:
            Dictionary containing DataFrames with temporal stats:
            - pick_trends: Daily pick rates
            - ban_trends: Daily ban rates
            - win_trends: Daily win rates
            - confidence_intervals: Upper/lower bounds for win rates
        """
        if 'timestamp' not in self.matches.columns:
            raise ValueError("Match data must include timestamp column")
            
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.matches['timestamp']):
            self.matches['timestamp'] = pd.to_datetime(self.matches['timestamp'])
            
        # Sort by timestamp
        matches = self.matches.sort_values('timestamp')
        
        # Initialize DataFrames for each metric
        heroes = sorted(list(set(
            hero for picks in matches[['blue_picks', 'red_picks']].values.flatten()
            for hero in picks
        )))
        
        dates = pd.date_range(
            start=matches['timestamp'].min(),
            end=matches['timestamp'].max(),
            freq='D'
        )
        
        # Initialize empty DataFrames
        pick_trends = pd.DataFrame(index=dates, columns=heroes)
        ban_trends = pd.DataFrame(index=dates, columns=heroes)
        win_trends = pd.DataFrame(index=dates, columns=heroes)
        ci_upper = pd.DataFrame(index=dates, columns=heroes)
        ci_lower = pd.DataFrame(index=dates, columns=heroes)
        
        # Compute rolling statistics
        for date in dates:
            window_end = date
            window_start = date - pd.Timedelta(window_size)
            window_matches = matches[
                (matches['timestamp'] > window_start) &
                (matches['timestamp'] <= window_end)
            ]
            
            if len(window_matches) < min_games:
                continue
                
            for hero in heroes:
                # Calculate pick rate
                picks = sum(
                    1 for picks in window_matches['blue_picks'].values
                    for h in picks if h == hero
                ) + sum(
                    1 for picks in window_matches['red_picks'].values
                    for h in picks if h == hero
                )
                pick_rate = picks / (len(window_matches) * 10)  # 10 picks per match
                pick_trends.loc[date, hero] = pick_rate
                
                # Calculate ban rate
                bans = sum(
                    1 for bans in window_matches['blue_bans'].values
                    for h in bans if h == hero
                ) + sum(
                    1 for bans in window_matches['red_bans'].values
                    for h in bans if h == hero
                )
                ban_rate = bans / (len(window_matches) * 6)  # 6 bans per match
                ban_trends.loc[date, hero] = ban_rate
                
                # Calculate win rate and confidence intervals
                hero_matches = window_matches[
                    window_matches.apply(
                        lambda row: hero in row['blue_picks'] or hero in row['red_picks'],
                        axis=1
                    )
                ]
                if len(hero_matches) < min_games:
                    continue
                    
                wins = sum(
                    1 for _, match in hero_matches.iterrows()
                    if (hero in match['blue_picks'] and match['winner'] == 'blue') or
                       (hero in match['red_picks'] and match['winner'] == 'red')
                )
                
                win_rate = wins / len(hero_matches)
                win_trends.loc[date, hero] = win_rate
                
                # Calculate Wilson confidence intervals
                ci_lower.loc[date, hero], ci_upper.loc[date, hero] = wilson_interval(
                    wins, len(hero_matches)
                )
        
        return {
            'pick_trends': pick_trends.fillna(method='ffill').fillna(0),
            'ban_trends': ban_trends.fillna(method='ffill').fillna(0),
            'win_trends': win_trends.fillna(method='ffill').fillna(0.5),
            'confidence_intervals': {
                'upper': ci_upper.fillna(method='ffill').fillna(0.5),
                'lower': ci_lower.fillna(method='ffill').fillna(0.5)
            }
        }

    def plot_hero_trends(self,
                        hero: str,
                        stats: Dict[str, pd.DataFrame],
                        metric: str = 'win_rate',
                        show_ci: bool = True) -> None:
        """Plot temporal trends for a specific hero.
        
        Args:
            hero: Name of hero to plot
            stats: Dictionary of temporal stats from compute_temporal_stats()
            metric: Which metric to plot ('win_rate', 'pick_rate', or 'ban_rate')
            show_ci: Whether to show confidence intervals (win rate only)
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plotting")
            
        if metric not in ['win_rate', 'pick_rate', 'ban_rate']:
            raise ValueError("Invalid metric. Must be win_rate, pick_rate, or ban_rate")
            
        df = None
        title = f"{hero} "
        yaxis_title = ""
        
        if metric == 'win_rate':
            df = stats['win_trends'][hero]
            title += "Win Rate Over Time"
            yaxis_title = "Win Rate"
        elif metric == 'pick_rate':
            df = stats['pick_trends'][hero]
            title += "Pick Rate Over Time"
            yaxis_title = "Pick Rate"
        else:  # ban_rate
            df = stats['ban_trends'][hero]
            title += "Ban Rate Over Time"
            yaxis_title = "Ban Rate"
            
        fig = go.Figure()
        
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.values,
            mode='lines',
            name=metric.replace('_', ' ').title(),
            line=dict(width=2)
        ))
        
        # Add confidence intervals for win rate
        if metric == 'win_rate' and show_ci:
            fig.add_trace(go.Scatter(
                x=stats['confidence_intervals']['upper'][hero].index,
                y=stats['confidence_intervals']['upper'][hero].values,
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=stats['confidence_intervals']['lower'][hero].index,
                y=stats['confidence_intervals']['lower'][hero].values,
                mode='lines',
                name='Lower CI',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(68, 68, 68, 0.2)',
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=yaxis_title,
            yaxis=dict(tickformat='.0%'),
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()

    def compute_role_time_series(match_data: List[Dict], patches: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """
        Compute role-based statistics across patches.
        
        Args:
            match_data: List of match data dictionaries
            patches: List of patch versions to analyze
        
        Returns:
            Dictionary of role stats over time with structure:
            {role: {'win_rate': [...], 'pick_rate': [...], 'ban_rate': [...]}}
        """
        role_stats = {}
        
        for patch in patches:
            patch_matches = [m for m in match_data if m['patch'] == patch]
            
            for role in ROLES:
                if role not in role_stats:
                    role_stats[role] = {
                        'win_rate': [],
                        'pick_rate': [],
                        'ban_rate': []
                    }
                
                # Calculate statistics for this role in this patch
                role_picks = sum(1 for m in patch_matches for h in m['picks'] if h['role'] == role)
                role_bans = sum(1 for m in patch_matches for h in m['bans'] if h['role'] == role)
                role_wins = sum(1 for m in patch_matches for h in m['picks'] 
                              if h['role'] == role and h['team'] == m['winning_team'])
                
                total_matches = len(patch_matches)
                total_picks = sum(len(m['picks']) for m in patch_matches)
                total_bans = sum(len(m['bans']) for m in patch_matches)
                
                role_stats[role]['win_rate'].append(role_wins / role_picks if role_picks > 0 else 0)
                role_stats[role]['pick_rate'].append(role_picks / total_picks)
                role_stats[role]['ban_rate'].append(role_bans / total_bans)
        
        return role_stats

    def compute_role_vs_role_matrix(match_data: List[Dict], patch: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute win rates between different roles.
        
        Args:
            match_data: List of match data dictionaries
            patch: Optional patch version to filter matches
        
        Returns:
            Dictionary of {attacker_role: {defender_role: win_rate}}
        """
        if patch:
            match_data = [m for m in match_data if m['patch'] == patch]
        
        role_matchups = {role: {def_role: [] for def_role in ROLES} for role in ROLES}
        
        for match in match_data:
            team1_roles = [h['role'] for h in match['picks'] if h['team'] == 1]
            team2_roles = [h['role'] for h in match['picks'] if h['team'] == 2]
            winning_team = match['winning_team']
            
            # Record outcomes for each role matchup
            for atk_role in team1_roles:
                for def_role in team2_roles:
                    role_matchups[atk_role][def_role].append(1 if winning_team == 1 else 0)
                    role_matchups[def_role][atk_role].append(1 if winning_team == 2 else 0)
        
        # Convert lists to average win rates
        return {
            atk: {
                def_: sum(outcomes)/len(outcomes) if outcomes else 0.5
                for def_, outcomes in defenders.items()
            }
            for atk, defenders in role_matchups.items()
        }

    def plot_role_trends(self, trends: Dict, metric: str = 'meta_score',
                        plot_library: str = 'plotly') -> Union[plt.Figure, 'go.Figure']:
        """Plot role-based trends over time.
        
        Args:
            trends: Dictionary from compute_role_trends()
            metric: Which metric to plot ('pick_rates', 'ban_rates', 'win_rates', or 'meta_score')
            plot_library: 'plotly' or 'matplotlib'
            
        Returns:
            Plot figure object
        """
        if metric not in trends:
            raise ValueError(f"Invalid metric: {metric}")
            
        if plot_library == 'plotly' and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            plot_library = 'matplotlib'
            
        if plot_library == 'plotly':
            fig = go.Figure()
            for role in trends[metric]:
                fig.add_trace(go.Scatter(
                    x=trends['patches'],
                    y=trends[metric][role],
                    name=role,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title=f'MLBB Role {metric.replace("_", " ").title()} Trends',
                xaxis_title='Patch Version',
                yaxis_title=metric.replace('_', ' ').title(),
                hovermode='x unified'
            )
            return fig
            
        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            for role in trends[metric]:
                ax.plot(trends['patches'], trends[metric][role], 
                       marker='o', label=role)
                
            ax.set_title(f'MLBB Role {metric.replace("_", " ").title()} Trends')
            ax.set_xlabel('Patch Version')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
            
    def export_role_trends(self, trends: Dict, output_path: Union[str, Path],
                          compression: CompressionType = None) -> None:
        """Export role trends data to JSON file.
        
        Args:
            trends: Dictionary from compute_role_trends()
            output_path: Path to save the JSON file
            compression: Optional compression format to use
        """
        output_path = Path(output_path)
        opener = get_compression_opener(compression)
        
        with opener(output_path, 'wt') as f:
            json.dump({
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'data_version': '1.0'
                },
                'trends': trends
            }, f, indent=2)