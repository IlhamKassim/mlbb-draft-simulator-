import pytest
from flask import url_for

def test_search_by_name(client, mock_data_path):
    """Test searching heroes by name."""
    response = client.get('/search?q=Fighter')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['name'] == 'Test Fighter'

def test_search_by_role(client, mock_data_path):
    """Test filtering heroes by role."""
    response = client.get('/search?role=Tank')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['name'] == 'Test Tank'

def test_search_by_name_and_role(client, mock_data_path):
    """Test searching heroes by both name and role."""
    response = client.get('/search?q=Test&role=Mage')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['name'] == 'Test Mage'
    assert 'Mage' in data[0]['roles']

def test_search_no_results(client, mock_data_path):
    """Test search with no matching results."""
    response = client.get('/search?q=NonexistentHero')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 0

def test_search_invalid_role(client, mock_data_path):
    """Test search with invalid role."""
    response = client.get('/search?role=InvalidRole')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 0

def test_search_case_insensitive(client, mock_data_path):
    """Test that search is case insensitive."""
    response = client.get('/search?q=test')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 3  # Should find all test heroes

def test_search_partial_match(client, mock_data_path):
    """Test searching with partial name match."""
    response = client.get('/search?q=Fight')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['name'] == 'Test Fighter' 