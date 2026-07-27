"""
Logger utility optimized for Edge devices.
Implements log rotation to ensure bounded disk usage.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_LOG_FILE, DEFAULT_LOG_MAX_BYTES, DEFAULT_LOG_BACKUP_COUNT

def get_logger(
    name: str, 
    log_level: int = logging.INFO, 
    log_file: Optional[str] = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT
) -> logging.Logger:
    """
    Configures and returns a logger instance.
    
    Features:
    - Bounded disk footprint using RotatingFileHandler.
    - Console output (stdout) for easy debugging, systemd, or Docker ingestion.
    - Prevents duplicate handlers if called multiple times for the same module.
    
    Args:
        name: Name of the logger (usually __name__).
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Path to the log file. If None, only console logging is used.
        max_bytes: Maximum size of a single log file before rotation.
        backup_count: Number of rotated backup files to keep.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Return immediately if the logger is already configured
    # This prevents duplicate log entries if get_logger is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)
    
    # Standardized edge format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler with Rotation
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console only if file system access fails
            logger.warning(f"Failed to setup file logging at {log_file}: {e}")

    return logger