import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOGS_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Define custom log format: include filename and function name
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s] - %(message)s'

logging.basicConfig(
    filename=LOGS_FILE,
    format=LOG_FORMAT,
    level=logging.INFO
)

def get_logger(name):
    """
    Function to initialize logger in different scripts.
    This logger automatically logs: filename and function name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger