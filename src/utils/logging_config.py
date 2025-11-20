"""Structured logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        json_format: Whether to use JSON formatting (useful for log aggregation)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("slm_train_deploy")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    if json_format:
        # JSON format for structured logging (e.g., Azure Monitor)
        formatter = logging.Formatter(
            '{"time":"%(asctime)s",'
            '"level":"%(levelname)s",'
            '"name":"%(name)s",'
            '"message":"%(message)s",'
            '"module":"%(module)s",'
            '"function":"%(funcName)s",'
            '"line":%(lineno)d}'
        )
    else:
        # Human-readable format
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"slm_train_deploy.{name}")
