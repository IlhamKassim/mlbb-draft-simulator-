import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from data_loader import MLBBDataLoader

def main(data_dir, output_dir):
    """
    Train a baseline model using match data.

    :param data_dir: Directory containing raw match data.
    :param output_dir: Directory to save the trained model.
    """
    loader = MLBBDataLoader(data_dir)
    data = loader.load_matches('sample_matches.csv')

    # Prepare features and labels
    features = []
    labels = []
    for _, row in data.iterrows():
        feature_vector = loader.prepare_model_features(
            row['blue_picks'], row['red_picks'], row['blue_bans'], row['red_bans']
        )
        features.append(feature_vector)
        labels.append(1 if row['winner'] == 'blue' else 0)

    # Train model
    model = GradientBoostingClassifier()
    model.fit(features, labels)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'baseline.joblib'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline MLBB model.")
    parser.add_argument('--data', required=True, help="Path to the data directory.")
    parser.add_argument('--output', required=True, help="Path to the output directory.")
    args = parser.parse_args()

    main(args.data, args.output)