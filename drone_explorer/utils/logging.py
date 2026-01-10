import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = "drone_logger",
    log_file: str = "rl-run.log",
    level: int = logging.INFO,
    console: bool = True,
    file_mode: str = "a",
):
    """
    Sets up a logger that can log to console and/or file.

    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, file logging is disabled.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to log to stdout.
        file_mode: Mode to open the log file ('w' for overwrite, 'a' for append).
    Returns:
        logging.Logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logging

    formatter = logging.Formatter(
        fmt="{asctime} - {levelname} - {message}",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Clear previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
        fh = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
