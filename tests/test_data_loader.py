"""Tests for the data loader module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_loader import MLBBDataLoader

@pytest.fixture
def sample_data(tmp_path):
    """Create sample match data for testing."""
    data = [
        {
            "match_id": "1",
            "patch_version": "1.0.0",
            "blue_picks": ["Gusion", "Tigreal", "Layla", "Eudora", "Alucard"],
            "red_picks": ["Franco", "Miya", "Zilong", "Nana", "Balmond"],
            "blue_bans": ["Fanny", "Ling", "Lancelot"],
            "red_bans": ["Hayabusa", "Hanzo", "Helcurt"],
            "winner": "blue"
        },
        {
            "match_id": "2",
            "patch_version": "1.0.0",
            "blue_picks": ["Franco", "Miya", "Zilong", "Nana", "Balmond"],
            "red_picks": ["Gusion", "Tigreal", "Layla", "Eudora", "Alucard"],
            "blue_bans": ["Hayabusa", "Hanzo", "Helcurt"],
            "red_bans": ["Fanny", "Ling", "Lancelot"],
            "winner": "red"
        }
    ]
    
    # Save as CSV
    df = pd.DataFrame(data)
    csv_path = tmp_path / "matches.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = tmp_path / "matches.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
        
    return tmp_path

def test_load_matches_csv(sample_data):
    """Test loading matches from CSV file."""
    loader = MLBBDataLoader(sample_data)
    df = loader.load_matches("matches.csv")
    
    assert len(df) == 2
    assert list(df.columns) == ["match_id", "patch_version", "blue_picks", "red_picks",
                               "blue_bans", "red_bans", "winner"]
    assert isinstance(df.blue_picks[0], list)
    assert len(df.blue_picks[0]) == 5

def test_load_matches_json(sample_data):
    """Test loading matches from JSON file."""
    loader = MLBBDataLoader(sample_data)
    df = loader.load_matches("matches.json")
    
    assert len(df) == 2
    assert list(df.columns) == ["match_id", "patch_version", "blue_picks", "red_picks",
                               "blue_bans", "red_bans", "winner"]
    assert isinstance(df.blue_picks[0], list)
    assert len(df.blue_picks[0]) == 5

def test_compute_hero_features(sample_data):
    """Test computing hero features."""
    loader = MLBBDataLoader(sample_data)
    loader.load_matches("matches.csv")
    features = loader.compute_hero_features()
    
    # Check all required features are present
    assert set(features.keys()) == {"pick_rates", "ban_rates", "win_rates",
                                  "synergy_matrix", "counter_matrix", "heroes"}
    
    # Check dimensions
    n_heroes = len(features["heroes"])
    assert features["synergy_matrix"].shape == (n_heroes, n_heroes)
    assert features["counter_matrix"].shape == (n_heroes, n_heroes)
    
    # Check rates sum to expected values
    assert sum(features["pick_rates"].values()) == 1.0  # Each match has 10 picks
    assert sum(features["ban_rates"].values()) == 0.6   # Each match has 6 bans

def test_prepare_model_features(sample_data):
    """Test preparing features for model prediction."""
    loader = MLBBDataLoader(sample_data)
    loader.load_matches("matches.csv")
    loader.compute_hero_features()
    
    features = loader.prepare_model_features(
        blue_picks=["Gusion", "Tigreal", "Layla"],
        red_picks=["Franco", "Miya", "Zilong"],
        blue_bans=["Fanny", "Ling"],
        red_bans=["Hayabusa", "Hanzo"]
    )
    
    # Check feature vector dimensions
    n_heroes = len(loader.hero_features["heroes"])
    expected_length = 4 * n_heroes + 6  # 4 one-hot vectors + 6 team features
    assert len(features) == expected_length
    
    # Check feature names match
    feature_names = loader.get_feature_names()
    assert len(feature_names) == expected_length

def test_invalid_file_format(sample_data):
    """Test loading file with unsupported format."""
    loader = MLBBDataLoader(sample_data)
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_matches("matches.txt")

def test_missing_columns(tmp_path):
    """Test loading file with missing required columns."""
    # Create data with missing columns
    data = pd.DataFrame({
        "match_id": ["1"],
        "blue_picks": [["Gusion", "Tigreal"]]
    })
    
    csv_path = tmp_path / "matches.csv"
    data.to_csv(csv_path, index=False)
    
    loader = MLBBDataLoader(tmp_path)
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.load_matches("matches.csv")

def test_features_before_loading():
    """Test computing features before loading data."""
    loader = MLBBDataLoader("dummy_path")
    with pytest.raises(ValueError, match="No match data loaded"):
        loader.compute_hero_features()

def test_model_features_before_computing():
    """Test preparing model features before computing hero features."""
    loader = MLBBDataLoader("dummy_path")
    with pytest.raises(ValueError, match="Hero features not computed"):
        loader.prepare_model_features([], [], [], [])

@pytest.fixture
def data_loader():
    """Create a data loader instance for testing."""
    data_dir = Path("data/raw")
    return MLBBDataLoader(data_dir)

def test_load_matches(data_loader):
    """Test loading match data from CSV file."""
    data_loader.load_matches("sample_matches.csv")
    assert isinstance(data_loader.matches, pd.DataFrame)
    assert len(data_loader.matches) > 0
    assert all(col in data_loader.matches.columns for col in [
        'match_id', 'patch_version', 'blue_picks', 'red_picks',
        'blue_bans', 'red_bans', 'winner'
    ])
    
    # Test that lists are properly loaded
    assert isinstance(data_loader.matches.loc[0, 'blue_picks'], list)
    assert isinstance(data_loader.matches.loc[0, 'red_picks'], list)
    assert isinstance(data_loader.matches.loc[0, 'blue_bans'], list)
    assert isinstance(data_loader.matches.loc[0, 'red_bans'], list)

def test_compute_hero_features(data_loader):
    """Test computing hero-level features."""
    data_loader.load_matches("sample_matches.csv")
    features = data_loader.compute_hero_features()
    
    assert isinstance(features, dict)
    assert 'pick_rates' in features
    assert 'ban_rates' in features
    assert 'win_rates' in features
    assert 'synergy_matrix' in features
    assert 'counter_matrix' in features
    
    # Test feature values
    assert all(0 <= rate <= 1 for rate in features['pick_rates'].values())
    assert all(0 <= rate <= 1 for rate in features['ban_rates'].values())
    assert all(0 <= rate <= 1 for rate in features['win_rates'].values())

def test_prepare_model_features(data_loader):
    """Test preparing features for model prediction."""
    data_loader.load_matches("sample_matches.csv")
    data_loader.compute_hero_features()
    
    # Test with sample draft
    blue_team = {
        'picks': ['Gusion', 'Tigreal', 'Layla'],
        'bans': ['Fanny', 'Ling']
    }
    red_team = {
        'picks': ['Franco', 'Miya', 'Zilong'],
        'bans': ['Hayabusa', 'Hanzo']
    }
    
    features = data_loader.prepare_model_features(blue_team, red_team)
    feature_names = data_loader.get_feature_names()
    
    assert isinstance(features, np.ndarray)
    assert len(features) == len(feature_names)
    assert all(isinstance(name, str) for name in feature_names)

def test_invalid_file_format(data_loader):
    """Test handling of invalid file formats."""
    with pytest.raises(ValueError):
        data_loader.load_matches("invalid.txt")

def test_missing_required_columns(tmp_path):
    """Test handling of missing required columns."""
    # Create a temporary CSV with missing columns
    invalid_df = pd.DataFrame({
        'match_id': [1],
        'blue_picks': [['Hero1', 'Hero2']],
        # Missing other required columns
    })
    temp_file = tmp_path / "invalid.csv"
    invalid_df.to_csv(temp_file, index=False)
    
    data_loader = MLBBDataLoader(tmp_path)
    with pytest.raises(ValueError):
        data_loader.load_matches("invalid.csv")

def test_load_matches(clean_data_loader, sample_matches):
    """Test loading match data from CSV file."""
    df = clean_data_loader.load_matches()
    
    # Check if all required columns are present
    required_cols = ['match_id', 'patch_version', 'blue_picks', 'red_picks', 
                    'blue_bans', 'red_bans', 'winner']
    assert all(col in df.columns for col in required_cols)
    
    # Check if the data matches our sample
    pd.testing.assert_frame_equal(df, sample_matches)
    
    # Verify list columns are actually lists
    list_cols = ['blue_picks', 'red_picks', 'blue_bans', 'red_bans']
    for col in list_cols:
        assert all(isinstance(x, list) for x in df[col])

def test_compute_hero_features(clean_data_loader):
    """Test computation of hero-level features."""
    clean_data_loader.load_matches()
    features = clean_data_loader.compute_hero_features()
    
    # Check if features dictionary contains expected keys
    expected_keys = ['pick_rates', 'ban_rates', 'win_rates', 
                    'synergy_matrix', 'counter_matrix']
    assert all(key in features for key in expected_keys)
    
    # Verify rates are between 0 and 1
    for rate_key in ['pick_rates', 'ban_rates', 'win_rates']:
        rates = features[rate_key]
        assert all(0 <= rate <= 1 for rate in rates.values())
    
    # Check matrix dimensions
    n_heroes = len(features['pick_rates'])
    assert features['synergy_matrix'].shape == (n_heroes, n_heroes)
    assert features['counter_matrix'].shape == (n_heroes, n_heroes)

def test_prepare_model_features(clean_data_loader):
    """Test preparation of features for model prediction."""
    clean_data_loader.load_matches()
    clean_data_loader.compute_hero_features()
    
    # Sample draft data
    draft = {
        'blue_picks': ['Gusion', 'Tigreal', 'Layla'],
        'red_picks': ['Franco', 'Miya', 'Zilong'],
        'blue_bans': ['Fanny', 'Ling'],
        'red_bans': ['Hayabusa', 'Hanzo']
    }
    
    features = clean_data_loader.prepare_model_features(
        draft['blue_picks'], draft['red_picks'],
        draft['blue_bans'], draft['red_bans']
    )
    
    # Check if features is a numpy array
    assert isinstance(features, np.ndarray)
    
    # Get feature names and verify they match the feature vector length
    feature_names = clean_data_loader.get_feature_names()
    assert len(feature_names) == len(features)

def test_invalid_file_format(clean_data_loader, test_data_dir):
    """Test handling of invalid file format."""
    # Create an invalid file
    invalid_path = test_data_dir / "invalid.txt"
    invalid_path.write_text("invalid data")
    
    with pytest.raises(ValueError, match="Unsupported file format"):
        clean_data_loader.load_matches("invalid.txt")

def test_missing_required_columns(clean_data_loader, test_data_dir):
    """Test handling of missing required columns."""
    # Create a CSV file with missing columns
    invalid_df = pd.DataFrame({
        'match_id': range(1, 4),
        'winner': ['blue', 'red', 'blue']
    })
    invalid_path = test_data_dir / "missing_columns.csv"
    invalid_df.to_csv(invalid_path, index=False)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_data_loader.load_matches("missing_columns.csv")

def test_filter_by_patch(clean_data_loader):
    """Test filtering matches by patch version."""
    clean_data_loader.load_matches()
    features = clean_data_loader.compute_hero_features(patch_version="1.0.0")
    
    # All matches in our sample are from patch 1.0.0
    assert len(features['pick_rates']) > 0
    
    # Test with non-existent patch
    features = clean_data_loader.compute_hero_features(patch_version="999.0.0")
    assert all(rate == 0 for rate in features['pick_rates'].values())

def test_matrix_normalization(clean_data_loader):
    """Test that matrices are properly normalized after feature computation."""
    clean_data_loader.load_matches()
    features = clean_data_loader.compute_hero_features()
    
    # Check synergy matrix bounds
    assert np.all(features['synergy_matrix'] >= -1)
    assert np.all(features['synergy_matrix'] <= 1)
    
    # Check counter matrix bounds
    assert np.all(features['counter_matrix'] >= -1)
    assert np.all(features['counter_matrix'] <= 1)
    
    # Verify synergy matrix symmetry
    assert np.allclose(features['synergy_matrix'], features['synergy_matrix'].T)
    
    # Check that counter matrix represents zero-sum relationships
    # (if hero A counters hero B by x, then hero B counters hero A by -x)
    assert np.allclose(features['counter_matrix'], -features['counter_matrix'].T)
    
    # For matches with equal wins/losses, synergy effects should sum to zero
    # (approximately, due to floating point)
    assert abs(np.sum(features['synergy_matrix'])) < 1e-10

def test_safe_list_parsing(tmp_path):
    """Test safe parsing of string representations of lists."""
    # Create test data with various malformed inputs
    data = pd.DataFrame({
        'match_id': range(1, 5),
        'patch_version': ['1.0.0'] * 4,
        'blue_picks': [
            "['Gusion', 'Tigreal']",  # Valid list
            "__import__('os').system('echo hack')",  # Malicious code
            "not a list",  # Invalid format
            "{'set', 'not', 'list'}"  # Wrong collection type
        ],
        'red_picks': [["Franco", "Miya"]] * 4,  # Already lists
        'blue_bans': [["Fanny"]] * 4,
        'red_bans': [["Hanzo"]] * 4,
        'winner': ['blue'] * 4
    })
    
    # Save test data
    test_file = tmp_path / "malicious.csv"
    data.to_csv(test_file, index=False)
    
    loader = MLBBDataLoader(tmp_path)
    
    # Should raise ValueError for malicious/malformed inputs
    with pytest.raises(ValueError):
        loader.load_matches("malicious.csv")
        
def test_list_parsing_edge_cases(tmp_path):
    """Test edge cases in list parsing."""
    # Create test data with edge cases
    data = pd.DataFrame({
        'match_id': range(1, 4),
        'patch_version': ['1.0.0'] * 3,
        'blue_picks': [
            "[]",  # Empty list
            "[None, None]",  # List with None
            "[1, 2, 3]"  # List of non-strings
        ],
        'red_picks': [["Franco", "Miya"]] * 3,
        'blue_bans': [["Fanny"]] * 3,
        'red_bans': [["Hanzo"]] * 3,
        'winner': ['blue'] * 3
    })
    
    # Save test data
    test_file = tmp_path / "edge_cases.csv"
    data.to_csv(test_file, index=False)
    
    loader = MLBBDataLoader(tmp_path)
    
    # Should handle edge cases appropriately
    with pytest.raises(ValueError):
        loader.load_matches("edge_cases.csv")

def test_feature_value_ranges(clean_data_loader):
    """Test that all computed features are within expected ranges."""
    clean_data_loader.load_matches()
    features = clean_data_loader.compute_hero_features()
    
    # Check pick rates (should sum to 1.0 per match since normalized by 2*n_matches)
    assert all(0 <= rate <= 1 for rate in features['pick_rates'].values())
    assert abs(sum(features['pick_rates'].values()) - 1.0) < 1e-10
    
    # Check ban rates (should sum to 0.6 per match)
    assert all(0 <= rate <= 1 for rate in features['ban_rates'].values())
    assert abs(sum(features['ban_rates'].values()) - 0.6) < 1e-10
    
    # Check win rates
    assert all(0 <= rate <= 1 for rate in features['win_rates'].values())
    
    # Verify that heroes appearing in both teams have ~50% win rate
    # (in sample data where each matchup appears twice with opposite winners)
    common_heroes = set(features['pick_rates'].keys())
    for hero in common_heroes:
        if features['pick_rates'][hero] > 0:
            assert abs(features['win_rates'][hero] - 0.5) < 1e-10 