import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOGS_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Define custom log format: include filename and function name
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s] - %(message)s'

def get_logger(name):
    """
    Function to initialize logger in different scripts.
    This logger automatically logs: filename and function name.
    Logs go to both terminal (stdout) and file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger already configured
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOGS_FILE)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        # Stream handler (terminal)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        # Add both handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
