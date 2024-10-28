# src/utils/__init__.py
# Empty file to make utils a package

# src/utils/logger.py
import logging
import sys
from datetime import datetime

class AppLogger:
    """Comprehensive terminal logger"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._setup_logger()

    def _setup_logger(self):
        """Setup detailed logging configuration"""
        self.logger = logging.getLogger('SignalAnalysisApp')
        self.logger.setLevel(logging.DEBUG)

        # Create console handler with detailed formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Create a detailed format for terminal output
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """Get the configured logger"""
        return self.logger

# Create logger instance
logger_instance = AppLogger()
app_logger = logger_instance.get_logger()

# Export the logger
__all__ = ['app_logger']