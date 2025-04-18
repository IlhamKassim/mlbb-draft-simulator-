import os
import pytest
import tempfile
from mlbb import create_app
from mlbb.utils.data_loader import DataLoader

@pytest.fixture
def app():
    """Create and configure a test Flask application instance."""
    app = create_app({
        'TESTING': True,
        'DATA_PATH': os.path.join(os.path.dirname(__file__), 'test_data')
    })
    return app

@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()

@pytest.fixture
def mock_data_path(monkeypatch):
    """Configure the test data path."""
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    def mock_get_data_path(*args, **kwargs):
        return test_data_path
    monkeypatch.setattr(DataLoader, 'get_data_path', mock_get_data_path)
    return test_data_path

@pytest.fixture(autouse=True)
def app_context(app):
    """Create an application context for tests."""
    with app.app_context():
        yield

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data files here if needed
        yield temp_dir

@pytest.fixture(autouse=True)
def mock_data_path(monkeypatch, test_data_dir):
    """Override the data directory path for testing."""
    monkeypatch.setenv('MLBB_DATA_DIR', test_data_dir)

def pytest_configure(config):
    """Configure pytest for our test suite."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    ) 