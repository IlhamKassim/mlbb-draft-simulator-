#!/usr/bin/env python3
"""
Custom training script for MLBB draft win probability model.
This script handles the training process with proper data validation.

Usage:
    python train_custom_model.py
"""

import os
import sys
import logging
import ast
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_list_string(s):
    """Safely parse a string representation of a list."""
    if isinstance(s, list):
        return s  # Already a list
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse: {s}, error: {e}")
        return []

class SimpleFeatureExtractor:
    """A simpler feature extractor for MLBB draft data."""
    
    def __init__(self):
        self.heroes = []
        self.hero_to_idx = {}
        self.n_heroes = 0
        
    def fit(self, df):
        """Fit the feature extractor by identifying unique heroes."""
        heroes = set()
        
        # Extract all heroes from the picks and bans columns
        for col in ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']:
            for hero_list in df[col].tolist():
                heroes.update(hero_list)
        
        self.heroes = sorted(list(heroes))
        self.hero_to_idx = {hero: idx for idx, hero in enumerate(self.heroes)}
        self.n_heroes = len(self.heroes)
        logger.info(f"Found {self.n_heroes} unique heroes in the dataset")
        
    def transform(self, df):
        """Transform draft data into feature vectors."""
        # Feature matrix dimensions: [n_samples, n_heroes * 4]
        # 4 sections: blue picks, red picks, blue bans, red bans
        n_samples = len(df)
        n_features = self.n_heroes * 4
        X = np.zeros((n_samples, n_features))
        
        # Generate feature names
        feature_names = []
        for team in ['blue', 'red']:
            for action in ['pick', 'ban']:
                for hero in self.heroes:
                    feature_names.append(f"{team}_{action}_{hero}")
        
        # Fill the feature matrix
        for i, row in df.iterrows():
            # Blue picks
            for hero in row['blue_picks']:
                if hero in self.hero_to_idx:
                    idx = self.hero_to_idx[hero]
                    X[i, idx] = 1
            
            # Red picks
            for hero in row['red_picks']:
                if hero in self.hero_to_idx:
                    idx = self.n_heroes + self.hero_to_idx[hero]
                    X[i, idx] = 1
            
            # Blue bans
            for hero in row['blue_bans']:
                if hero in self.hero_to_idx:
                    idx = 2 * self.n_heroes + self.hero_to_idx[hero]
                    X[i, idx] = 1
            
            # Red bans
            for hero in row['red_bans']:
                if hero in self.hero_to_idx:
                    idx = 3 * self.n_heroes + self.hero_to_idx[hero]
                    X[i, idx] = 1
        
        return X, feature_names
        
    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from a trained model."""
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
        
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

def train_model():
    """Train and save the draft prediction model."""
    # Define file paths
    data_file = Path("/Users/hamboii/mlbb_counter_system/data/raw/sample_matches.csv")
    output_dir = Path("/Users/hamboii/mlbb_counter_system/models/baseline")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} matches from {data_file}")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return False
        
    # Clean and validate data
    logger.info("Validating and preparing data...")
    
    # Validate winner column
    df["winner"] = df["winner"].str.strip().str.lower()
    if not df["winner"].isin({"blue", "red"}).all():
        invalid_values = df[~df["winner"].isin({"blue", "red"})]["winner"].unique()
        logger.error(f"Invalid values in 'winner' column: {invalid_values}")
        return False
        
    # Parse list columns
    list_cols = ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']
    for col in list_cols:
        df[col] = df[col].apply(parse_list_string)
        
    # Extract features using our simple extractor
    logger.info("Extracting features...")
    feature_extractor = SimpleFeatureExtractor()
    feature_extractor.fit(df)
    
    X, feature_names = feature_extractor.transform(df)
    y = (df["winner"] == "blue").astype(int)
    
    logger.info(f"Features extracted. Shape: {X.shape}")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    logger.info("Training model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test) if len(X_test) > 0 else "N/A"
    logger.info(f"Model trained. Training accuracy: {train_score:.4f}, Test accuracy: {test_score}")
    
    # Save model and feature extractor
    model_path = output_dir / "baseline.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    extractor_path = output_dir / "feature_extractor.joblib"
    joblib.dump(feature_extractor, extractor_path)
    logger.info(f"Feature extractor saved to {extractor_path}")
    
    # Save feature names
    feature_names_path = output_dir / "feature_names.joblib"
    joblib.dump(feature_names, feature_names_path)
    logger.info(f"Feature names saved to {feature_names_path}")
    
    return True

if __name__ == "__main__":
    success = train_model()
    if success:
        logger.info("✅ Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training failed.")
        sys.exit(1)