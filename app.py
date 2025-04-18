"""
MLBB Draft Simulator - Main Application Factory

This module initializes the Flask application using the factory pattern,
registers blueprints, and sets up logging and error handling.
"""
import logging
import os
from typing import Optional, Type

from flask import Flask, render_template, request, jsonify, send_from_directory, session, url_for
import json
from datetime import datetime
import random
from logging.handlers import RotatingFileHandler

from mlbb.blueprints.draft import draft_bp
from mlbb.blueprints.analysis import bp as analysis_bp
from mlbb.config import Config, Development, Production
from mlbb.utils.data_loader import DataLoader
from mlbb.utils.errors import register_error_handlers

def setup_logging(app: Flask) -> None:
    """Configure logging for the application.
    
    Args:
        app: Flask application instance.
    """
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    file_handler = RotatingFileHandler(
        'logs/mlbb.log',
        maxBytes=10240,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('MLBB Draft Simulator startup')

def create_app(config_object: Optional[Type[Config]] = None) -> Flask:
    """Create and configure the Flask application.
    
    Args:
        config_object: Configuration class to use. Defaults to Development.
        
    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__, static_url_path='/static', static_folder='static')
    
    # Load configuration
    if config_object is None:
        config_object = Development if app.debug else Production
    app.config.from_object(config_object)
    
    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Initialize logging
    setup_logging(app)
    
    # Register blueprints
    app.register_blueprint(draft_bp, url_prefix='/draft')
    app.register_blueprint(analysis_bp, url_prefix='/analysis')
    
    # Setup error handlers
    register_error_handlers(app)
    
    # Initialize data loader
    DataLoader.initialize(app)
    
    # Version static assets
    @app.context_processor
    def inject_version():
        return dict(version=app.config['VERSION'])
    
    app.secret_key = 'your_secret_key_here'  # Required for session management

    # Explicitly serve PNG files with correct MIME type
    @app.route('/static/img/items/<path:filename>')
    def serve_item_image(filename):
        return send_from_directory(os.path.join(app.static_folder, 'img', 'items'),
                                 filename,
                                 mimetype='image/png')

    # Add logging for static file requests
    @app.after_request
    def after_request(response):
        if response.status_code == 404 and request.path.startswith('/static/'):
            print(f"404 for static file: {request.path}")
            print(f"Looking in: {os.path.join(app.static_folder, request.path.replace('/static/', ''))}")
        return response

    # Add after the item image route
    @app.route('/static/img/heroes/<path:filename>')
    def serve_hero_image(filename):
        return send_from_directory(os.path.join(app.static_folder, 'img', 'heroes'),
                                 filename,
                                 mimetype='image/png')

    # Add custom error handler for 500 errors
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Server Error: {error}')
        return jsonify({
            'error': 'Internal server error',
            'message': str(error)
        }), 500

    @app.errorhandler(404)
    def not_found_error(error):
        app.logger.error(f'Not Found: {error}')
        return jsonify({
            'error': 'Not found',
            'message': str(error)
        }), 404

    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 