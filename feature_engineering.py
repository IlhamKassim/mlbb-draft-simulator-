"""
Feature engineering module for MLBB draft prediction.

This module handles the conversion of draft data into model features.
"""

import logging
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class DraftFeatureExtractor:
    """Extract features from draft data for model training."""
    
    def __init__(self):
        self.hero_encoder = LabelEncoder()
        self.n_heroes = 0
        self.hero_to_idx = {}
        
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the feature extractor on training data.
        
        Args:
            df: DataFrame with columns blue_picks, red_picks, blue_bans, red_bans
        """
        # Get unique heroes from all picks and bans
        heroes = set()
        for col in ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']:
            # Handle string lists by evaluating them
            if df[col].dtype == 'object':
                heroes.update(*df[col].apply(eval).tolist())
            else:
                heroes.update(*df[col].tolist())
                
        heroes = sorted(list(heroes))
        self.hero_encoder.fit(heroes)
        self.n_heroes = len(heroes)
        self.hero_to_idx = {h: i for i, h in enumerate(heroes)}
        
        logger.info(f"Fitted feature extractor with {self.n_heroes} unique heroes")
        
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform draft data into model features.
        
        Args:
            df: DataFrame with columns blue_picks, red_picks, blue_bans, red_bans
            
        Returns:
            X: Feature matrix
            feature_names: List of feature names
        """
        if not self.n_heroes:
            raise ValueError("Feature extractor must be fitted first")
            
        # Initialize feature matrix
        n_samples = len(df)
        n_features = self.n_heroes * 4  # picks and bans for both teams
        X = np.zeros((n_samples, n_features))
        
        # Generate feature names
        feature_names = []
        for team in ['blue', 'red']:
            for action in ['pick', 'ban']:
                for hero in self.hero_encoder.classes_:
                    feature_names.append(f"{team}_{action}_{hero}")
        
        # Fill feature matrix
        for i, row in df.iterrows():
            # Handle string lists by evaluating them if needed
            blue_picks = eval(row['blue_picks']) if isinstance(row['blue_picks'], str) else row['blue_picks']
            red_picks = eval(row['red_picks']) if isinstance(row['red_picks'], str) else row['red_picks']
            blue_bans = eval(row['blue_bans']) if isinstance(row['blue_bans'], str) else row['blue_bans']
            red_bans = eval(row['red_bans']) if isinstance(row['red_bans'], str) else row['red_bans']
            
            # Set pick features
            for hero in blue_picks:
                idx = self.hero_to_idx[hero]
                X[i, idx] = 1
                
            for hero in red_picks:
                idx = self.hero_to_idx[hero] + self.n_heroes
                X[i, idx] = 1
                
            # Set ban features
            for hero in blue_bans:
                idx = self.hero_to_idx[hero] + 2 * self.n_heroes
                X[i, idx] = 1
                
            for hero in red_bans:
                idx = self.hero_to_idx[hero] + 3 * self.n_heroes
                X[i, idx] = 1
                
        return X, feature_names
        
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance scores from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature importance scores")
            
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)) 