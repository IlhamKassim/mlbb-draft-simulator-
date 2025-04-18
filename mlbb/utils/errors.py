"""
Error handling utilities for the MLBB Draft Simulator.

This module provides custom error handlers and error pages for various
HTTP status codes and application exceptions.
"""
from typing import Tuple, Union
from flask import Flask, render_template, jsonify, request

def register_error_handlers(app: Flask) -> None:
    """Register custom error handlers with the Flask application.
    
    Args:
        app: Flask application instance.
    """
    
    def is_api_request() -> bool:
        """Check if the current request is an API request."""
        return request.path.startswith('/api/') or request.headers.get('Accept') == 'application/json'
    
    @app.errorhandler(404)
    def not_found_error(error) -> Tuple[Union[str, dict], int]:
        """Handle 404 Not Found errors.
        
        Returns JSON for API requests, renders error page for browser requests.
        """
        app.logger.info(f'404 Error: {request.url}')
        if is_api_request():
            return jsonify({
                'error': 'Resource not found',
                'status': 404
            }), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error) -> Tuple[Union[str, dict], int]:
        """Handle 500 Internal Server errors.
        
        Logs the error and returns appropriate response format.
        """
        app.logger.error(f'Server Error: {error}')
        app.logger.exception(error)
        
        if is_api_request():
            return jsonify({
                'error': 'Internal server error',
                'status': 500
            }), 500
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error) -> Tuple[Union[str, dict], int]:
        """Handle 403 Forbidden errors."""
        app.logger.info(f'403 Error: {request.url}')
        if is_api_request():
            return jsonify({
                'error': 'Forbidden',
                'status': 403
            }), 403
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(400)
    def bad_request_error(error) -> Tuple[Union[str, dict], int]:
        """Handle 400 Bad Request errors."""
        app.logger.info(f'400 Error: {request.url} - {error}')
        if is_api_request():
            return jsonify({
                'error': str(error),
                'status': 400
            }), 400
        return render_template('errors/400.html', error=error), 400

class DataLoadError(Exception):
    """Exception raised when loading game data fails."""
    pass

class ValidationError(Exception):
    """Exception raised when request validation fails."""
    pass 