import logging
import os
from src.config import LOG_FILE, LOG_DIR

# Make sure the logs folder actually exists before we try to write to it
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a given name.
    Every file (data_loader, train, api, etc.) will call this
    function to get its own logger, but all logs go to the same
    file and the same console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid attaching duplicate handlers if this function
    # gets called more than once for the same logger name
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger