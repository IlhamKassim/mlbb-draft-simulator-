#!/usr/bin/env python3
"""
train.py – Train MLBB draft win‑probability model

Setup:
    pip install -r requirements.txt
    # Ensure uvicorn is installed for your API server:
    pip install uvicorn[standard]

Usage:
    # Train the model (it will pick up the first CSV in data/raw/)
    python train.py --data data/raw --output models

    # Once your API is implemented (api/main.py), run:
    python -m uvicorn api.main:app --reload
"""

import argparse
import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from feature_engineering import DraftFeatureExtractor

# ─── Config & Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"match_id", "blue_picks", "red_picks", "blue_bans", "red_bans", "winner"}


# ─── Data Loading & Validation ─────────────────────────────────────────────────
def load_matches(csv_path: Path) -> pd.DataFrame:
    """Load and validate matches CSV."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loaded {len(df)} matches from {csv_path}")
    except Exception as e:
        logger.error(f"❌ Failed to read CSV file '{csv_path}': {e}")
        sys.exit(1)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"❌ CSV is missing required columns: {missing}")
        sys.exit(1)

    if df.empty:
        logger.error(f"❌ Loaded DataFrame is empty from '{csv_path}'")
        sys.exit(1)

    # Ensure winner values are valid
    if not df["winner"].isin({"blue", "red"}).all():
        logger.error("❌ Column 'winner' contains invalid values (expected 'blue' or 'red')")
        sys.exit(1)

    # Validate team sizes
    for col in ['blue_picks', 'red_picks']:
        try:
            pick_lengths = df[col].apply(eval).apply(len)
            if not (pick_lengths == 5).all():
                logger.error(f"❌ {col} must contain exactly 5 heroes per team")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error parsing {col}: {e}")
            sys.exit(1)

    for col in ['blue_bans', 'red_bans']:
        try:
            ban_lengths = df[col].apply(eval).apply(len)
            if not (ban_lengths == 3).all():
                logger.error(f"❌ {col} must contain exactly 3 bans per team")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error parsing {col}: {e}")
            sys.exit(1)

    return df


# ─── Model Training ────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, output_dir: Path, test_size: float = 0.2,
               random_state: int = 42) -> None:
    """Train and save the draft prediction model.
    
    Args:
        df: DataFrame with match data
        output_dir: Directory to save model artifacts
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    # Feature engineering
    logger.info("⚙️  Extracting features...")
    feature_extractor = DraftFeatureExtractor()
    feature_extractor.fit(df)
    
    X, feature_names = feature_extractor.transform(df)
    y = (df["winner"] == "blue").astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"📈 Training set: {len(X_train)} samples")
    logger.info(f"🔍 Test set: {len(X_test)} samples")
    
    # Train model
    logger.info("🏋️  Training model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"📊 Train accuracy: {train_score:.3f}")
    logger.info(f"📊 Test accuracy: {test_score:.3f}")
    
    y_pred = model.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Get feature importance
    importance_dict = feature_extractor.get_feature_importance(model, feature_names)
    top_features = dict(list(importance_dict.items())[:10])
    logger.info("🔝 Top 10 important features:")
    for feature, importance in top_features.items():
        logger.info(f"  • {feature}: {importance:.3f}")
    
    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "baseline_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"💾 Model saved to {model_path}")
    
    extractor_path = output_dir / "feature_extractor.joblib"
    joblib.dump(feature_extractor, extractor_path)
    logger.info(f"💾 Feature extractor saved to {extractor_path}")
    
    metadata = {
        "n_samples": len(df),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "top_features": top_features
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"📝 Model metadata saved to {metadata_path}")


# ─── Main Training Flow ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train MLBB draft win‑probability model")
    parser.add_argument(
        "--data", "-d", required=True,
        help="Path to CSV file or directory containing CSV files"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Directory where trained model will be saved"
    )
    parser.add_argument(
        "--test-size", "-t", type=float, default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", "-r", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Resolve data path (file vs. directory)
    data_path = Path(args.data)
    if data_path.is_dir():
        csv_list = list(data_path.glob("*.csv"))
        if not csv_list:
            logger.error(f"❌ No CSV files found in directory '{data_path}'")
            sys.exit(1)
        csv_file = csv_list[0]
        logger.info(f"🔍 Found CSV: {csv_file}")
    elif data_path.is_file():
        csv_file = data_path
    else:
        logger.error(f"❌ Data path '{data_path}' does not exist")
        sys.exit(1)

    # Load & validate data
    df = load_matches(csv_file)
    
    # Train model
    train_model(
        df=df,
        output_dir=Path(args.output),
        test_size=args.test_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main() 