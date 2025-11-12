"""
Logging utilities for experiments.
No emojis, no decorative output - plain informative text only.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger that writes to both file and console.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Name of log file (if None, uses name + timestamp)
        level: Logging level
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    # Create formatter - plain, informative
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_file = f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def create_experiment_dir(base_dir: Path, experiment_name: str) -> Path:
    """
    Create a timestamped directory for an experiment run.

    Args:
        base_dir: Base directory for all experiments
        experiment_name: Name of the experiment

    Returns:
        Path to the created experiment directory
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = base_dir / experiment_name / f"run_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
