import pytest
import json
from urllib.parse import quote
from mlbb import create_app

@pytest.fixture
def app():
    """Create and configure a test Flask application."""
    app = create_app({
        'TESTING': True,
        'DEBUG': False
    })
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

def test_draft_page(client):
    """Test the draft simulator page loads correctly."""
    response = client.get('/draft/')
    assert response.status_code == 200

def test_hero_search_no_params(client):
    """Test hero search endpoint with no parameters."""
    response = client.get('/draft/search')
    assert response.status_code == 200
    data = response.get_json()
    assert 'results' in data

def test_hero_search_with_invalid_role(client):
    """Test hero search endpoint with invalid role."""
    response = client.get('/draft/search?role=invalid')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'Invalid role'

def test_hero_search_with_valid_role(client):
    """Test hero search endpoint with valid role."""
    response = client.get('/draft/search?role=tank')
    assert response.status_code == 200
    data = response.get_json()
    assert 'results' in data

def test_shareable_link_roundtrip(client):
    """Test that draft state can be shared via URL and restored."""
    # Simulate a draft state payload
    state = {
        "blueBans": ["Esmeralda"],
        "redPicks": ["Lancelot"],
        "currentPhase": "pick",
        "currentTeam": "blue",
        "currentIndex": 0
    }
    
    # Generate URL hash
    encoded = quote(json.dumps(state))
    response = client.get(f'/draft#{encoded}')
    assert response.status_code == 200
    
    # Confirm draft slots restored in HTML
    assert b'Esmeralda' in response.data
    assert b'Lancelot' in response.data

def test_hero_search(client):
    """Test the hero search endpoint."""
    response = client.get('/draft/search?q=esme&role=mage')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'results' in data
    assert any('Esmeralda' in hero['name'] for hero in data['results'])

def test_invalid_role_filter(client):
    """Test that invalid role filter returns appropriate error."""
    response = client.get('/draft/search?role=invalid')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data 