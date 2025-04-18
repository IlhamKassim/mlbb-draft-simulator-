"""
Draft simulator blueprint for handling draft mode functionality.
"""
from flask import (
    Blueprint, render_template, jsonify, request, 
    current_app, abort
)
from typing import Dict, Any, Tuple, Union
from werkzeug.exceptions import HTTPException
from requests.exceptions import RequestException, ConnectionError, Timeout
from mlbb.utils.data_loader import DataLoader, DataLoadError
from mlbb.utils.errors import ValidationError
from mlbb_api import get_all_heroes, get_hero_details
import os
import json

draft_bp = Blueprint('draft', __name__, url_prefix='/draft')

# Error handlers
@draft_bp.errorhandler(ValidationError)
def handle_validation_error(error: ValidationError) -> Tuple[Dict[str, Any], int]:
    """Handle validation errors in the draft blueprint."""
    current_app.logger.warning(f'Draft validation error: {error}')
    return jsonify({
        'error': str(error),
        'status': 400,
        'type': 'validation_error',
        'details': getattr(error, 'details', None)
    }), 400

@draft_bp.errorhandler(DataLoadError)
def handle_data_load_error(error: DataLoadError) -> Tuple[Dict[str, Any], int]:
    """Handle data loading errors in the draft blueprint."""
    current_app.logger.error(f'Draft data load error: {error}')
    return jsonify({
        'error': 'Failed to load game data',
        'status': 500,
        'type': 'data_load_error',
        'retry_after': 30
    }), 500

@draft_bp.errorhandler(ConnectionError)
def handle_connection_error(error: ConnectionError) -> Tuple[Union[str, Dict[str, Any]], int]:
    """Handle connection errors in the draft blueprint."""
    current_app.logger.error(f'Connection error in draft: {error}')
    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            'error': 'Connection failed',
            'status': 503,
            'type': 'connection_error',
            'retry_after': 15,
            'details': 'The service is temporarily unavailable. Please check your connection and try again.'
        }), 503
    return render_template('errors/connection_error.html', 
                         error=error,
                         retry_after=15), 503

@draft_bp.errorhandler(Timeout)
def handle_timeout_error(error: Timeout) -> Tuple[Union[str, Dict[str, Any]], int]:
    """Handle timeout errors in the draft blueprint."""
    current_app.logger.error(f'Timeout error in draft: {error}')
    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            'error': 'Request timed out',
            'status': 504,
            'type': 'timeout_error',
            'retry_after': 5
        }), 504
    return render_template('errors/timeout_error.html', 
                         error=error,
                         retry_after=5), 504

@draft_bp.route('/')
def draft_page():
    """Render the draft simulator page."""
    try:
        # Load hero mappings data
        mappings_path = os.path.join(current_app.root_path, 'static', 'data', 'hero_mappings.json')
        if not os.path.exists(mappings_path):
            current_app.logger.error(f"Hero mappings file not found at {mappings_path}")
            return render_template('errors/error.html',
                                error="Configuration Error",
                                details="Hero mappings file is missing"), 500

        with open(mappings_path) as f:
            hero_roles = json.load(f)

        # Load hero list from test data
        test_data_path = os.path.join(current_app.root_path, 'static', 'data', 'hero_data.json')
        if not os.path.exists(test_data_path):
            current_app.logger.error(f"Hero data file not found at {test_data_path}")
            return render_template('errors/error.html',
                                error="Configuration Error",
                                details="Hero data file is missing"), 500

        with open(test_data_path) as f:
            hero_data = json.load(f)
            heroes = [hero['name'] for hero in hero_data['heroes']]

        return render_template('draft.html',
                            hero_roles=hero_roles,
                            hero_list=heroes)
    except Exception as e:
        current_app.logger.error(f"Error loading draft page: {str(e)}")
        return render_template('errors/error.html',
                             error="Failed to load hero data",
                             details=str(e)), 500

@draft_bp.route('/api/hero_details/<hero>')
def hero_details(hero):
    """Get detailed information about a specific hero."""
    try:
        details = get_hero_details(hero)
        return jsonify(details)
    except Exception as e:
        current_app.logger.error(f"Error fetching hero details for {hero}: {str(e)}")
        return jsonify({'error': f'Failed to fetch details for {hero}'}), 500

@draft_bp.route('/heroes')
def get_heroes():
    """Get all heroes with their data."""
    try:
        heroes = get_all_heroes()
        return jsonify(heroes)
    except Exception as e:
        current_app.logger.error(f"Error fetching heroes: {str(e)}")
        return jsonify({'error': 'Failed to fetch hero data'}), 500

@draft_bp.route('/heroes/by-role/<role>')
def get_heroes_by_role(role):
    """Get heroes filtered by role."""
    try:
        with open(os.path.join(current_app.root_path, 'static/data/hero_roles.json')) as f:
            hero_roles = json.load(f)
        if role not in hero_roles['roles']:
            return jsonify({'error': 'Invalid role'}), 400
        
        heroes = get_all_heroes()
        filtered_heroes = [
            hero for hero in heroes
            if any(r == role for r in get_hero_details(hero).get('roles', []))
        ]
        return jsonify(filtered_heroes)
    except Exception as e:
        current_app.logger.error(f"Error fetching heroes by role: {str(e)}")
        return jsonify({'error': 'Failed to fetch heroes by role'}), 500

@draft_bp.route('/search')
def search_heroes():
    """Search heroes by name and/or role."""
    query = request.args.get('q', '')
    role = request.args.get('role', '')
    
    if role and role not in ['tank', 'fighter', 'assassin', 'mage', 'marksman', 'support']:
        return jsonify({'error': 'Invalid role'}), 400
    
    # TODO: Implement actual hero search logic
    results = []  # This would be populated from your database/data source
    return jsonify({'results': results})

@draft_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'error': str(error)}), 400

@draft_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@draft_bp.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500 