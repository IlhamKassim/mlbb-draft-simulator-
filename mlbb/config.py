"""
Configuration classes for the MLBB Draft Simulator.

This module defines different configuration classes for development
and production environments.
"""
import os
from typing import Dict, Any

class Config:
    """Base configuration class with common settings."""
    
    # Application version
    VERSION: str = '1.0.0'
    
    # Security
    SECRET_KEY: str = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    
    # Data paths
    DATA_DIR: str = 'static/data'
    HERO_DATA_FILE: str = 'hero_descriptions.json'
    ITEM_BUILDS_FILE: str = 'hero_item_builds.json'
    PRO_STATS_FILE: str = 'pro_stats.json'
    
    # Cache settings
    CACHE_TYPE: str = 'simple'
    CACHE_DEFAULT_TIMEOUT: int = 300
    
    # Static asset settings
    STATIC_FOLDER: str = 'static'
    
    @staticmethod
    def init_app(app: Any) -> None:
        """Initialize application with this configuration.
        
        Args:
            app: Flask application instance.
        """
        pass

class Development(Config):
    """Development configuration."""
    
    DEBUG: bool = True
    
    # Development-specific settings
    TEMPLATES_AUTO_RELOAD: bool = True
    SEND_FILE_MAX_AGE_DEFAULT: int = 0
    
    @staticmethod
    def init_app(app: Any) -> None:
        """Initialize development-specific settings."""
        Config.init_app(app)
        
        # Enable debug toolbar in development
        try:
            from flask_debugtoolbar import DebugToolbarExtension
            DebugToolbarExtension(app)
        except ImportError:
            pass

class Production(Config):
    """Production configuration."""
    
    DEBUG: bool = False
    
    # Override with environment variables in production
    SECRET_KEY: str = os.environ.get('SECRET_KEY') or 'production-key-required'
    
    # Production cache settings
    CACHE_TYPE: str = 'redis'
    CACHE_REDIS_URL: str = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Static asset settings
    STATIC_FOLDER: str = '/var/www/mlbb/static'
    
    @staticmethod
    def init_app(app: Any) -> None:
        """Initialize production-specific settings."""
        Config.init_app(app)
        
        # Configure production logging
        import logging
        from logging.handlers import SMTPHandler
        
        # Email error logs to admins
        mail_handler = SMTPHandler(
            mailhost=('localhost', 25),
            fromaddr='no-reply@mlbb-draft.com',
            toaddrs=['admin@mlbb-draft.com'],
            subject='MLBB Draft Application Error'
        )
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler) 