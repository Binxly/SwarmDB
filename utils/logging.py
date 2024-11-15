import logging
import sys
from typing import Optional
from config.settings import settings

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create a logger instance with consistent formatting and level."""
    logger = logging.getLogger(name or __name__)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(settings.log_format))
        logger.addHandler(handler)
        
        logger.setLevel(settings.log_level)
        logger.propagate = False
    
    return logger
