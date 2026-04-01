"""Centralized logging setup for the pipeline."""

import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir: str = "logs", phase: str = "pipeline") -> logging.Logger:
    """Set up file + console logging. Returns the root logger."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{phase}_{timestamp}.log")

    # Root logger
    logger = logging.getLogger("context_engineering")
    logger.setLevel(logging.DEBUG)

    # File handler — DEBUG level (everything)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — INFO level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger
