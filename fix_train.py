#!/usr/bin/env python3
"""
fix_train.py – Modified training script for MLBB draft win‑probability model with more forgiving data validation

Usage:
    python fix_train.py
"""

import sys
import logging
from pathlib import Path
import ast

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

from feature_engineering import DraftFeatureExtractor

# ─── Config & Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

def parse_list_string(s):
    """Safely parse a string representation of a list."""
    if isinstance(s, list):
        return s  # Already a list
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        logger.warning(f"Failed to parse: {s}")
        return []

def fix_train_model():
    """Modified training script that's more forgiving with input data format"""
    # Paths
    data_file = Path("/Users/hamboii/mlbb_counter_system/data/raw/sample_matches.csv")
    output_dir = Path("/Users/hamboii/mlbb_counter_system/models/baseline")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data with more forgiving validation
    try:
        df = pd.read_csv(data_file)
        logger.info(f"📊 Loaded {len(df)} matches from {data_file}")
    except Exception as e:
        logger.error(f"❌ Failed to read CSV file '{data_file}': {e}")
        return
    
    # Clean up the winner column to ensure it only contains 'blue' or 'red'
    df["winner"] = df["winner"].str.strip().str.lower()
    
    # Print what the winner column contains for debugging
    unique_winners = df["winner"].unique()
    logger.info(f"Unique winner values: {unique_winners}")
    
    if not df["winner"].isin({"blue", "red"}).all():
        logger.error("❌ Column 'winner' contains invalid values (expected 'blue' or 'red')")
        invalid_winners = df[~df["winner"].isin({"blue", "red"})]["winner"].unique()
        logger.error(f"Invalid winners: {invalid_winners}")
        return
    
    # Parse list strings in the data
    list_cols = ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']
    for col in list_cols:
        df[col] = df[col].apply(parse_list_string)
    
    # Feature engineering
    logger.info("⚙️ Extracting features...")
    feature_extractor = DraftFeatureExtractor()
    feature_extractor.fit(df)
    
    X, feature_names = feature_extractor.transform(df)
    y = (df["winner"] == "blue").astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"📈 Training set: {len(X_train)} samples")
    logger.info(f"🔍 Test set: {len(X_test)} samples")
    
    # Train model
    logger.info("🏋️ Training model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model and feature extractor
    model_path = output_dir / "baseline.joblib"
    joblib.dump(model, model_path)
    logger.info(f"💾 Model saved to {model_path}")
    
    extractor_path = output_dir / "feature_extractor.joblib"
    joblib.dump(feature_extractor, extractor_path)
    logger.info(f"💾 Feature extractor saved to {extractor_path}")
    
    # Save feature_names for later use
    feature_names_path = output_dir / "feature_names.joblib"
    joblib.dump(feature_names, feature_names_path)
    logger.info(f"💾 Feature names saved to {feature_names_path}")
    
    logger.info("✅ Training completed successfully!")

if __name__ == "__main__":
    fix_train_model()