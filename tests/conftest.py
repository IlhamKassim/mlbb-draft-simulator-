import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def sample_matches():
    """Create sample match data for testing."""
    return pd.DataFrame({
        'match_id': range(1, 4),
        'patch_version': ['1.0.0'] * 3,
        'blue_picks': [
            ['Gusion', 'Tigreal', 'Layla'],
            ['Franco', 'Miya', 'Zilong'],
            ['Alucard', 'Estes', 'Nana']
        ],
        'red_picks': [
            ['Franco', 'Miya', 'Zilong'],
            ['Gusion', 'Tigreal', 'Layla'],
            ['Fanny', 'Angela', 'Chou']
        ],
        'blue_bans': [
            ['Fanny', 'Ling'],
            ['Hayabusa', 'Hanzo'],
            ['Gusion', 'Kadita']
        ],
        'red_bans': [
            ['Hayabusa', 'Hanzo'],
            ['Fanny', 'Ling'],
            ['Selena', 'Harith']
        ],
        'winner': ['blue', 'red', 'blue']
    })

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory, sample_matches):
    """Create a temporary directory with test data."""
    test_dir = tmp_path_factory.mktemp("test_data")
    sample_matches.to_csv(test_dir / "sample_matches.csv", index=False)
    return test_dir

@pytest.fixture(scope="function")
def clean_data_loader(test_data_dir):
    """Create a fresh data loader instance for each test."""
    from data_loader import MLBBDataLoader
    return MLBBDataLoader(test_data_dir) 