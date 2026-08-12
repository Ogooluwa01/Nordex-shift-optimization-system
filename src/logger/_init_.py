import logging
import os
from logging.handlers import RotatingFileHandler

# constants
LOG_DIR = "logs"
LOG_FILE = "app.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# ENSURING THAT THE LOG DIRECTORY ACTUALLY EXISTS
log_path_dir = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(log_path_dir, exist_ok=True)
log_file_path = os.path.join(log_path_dir, LOG_FILE)

def configure_logger():
    """
    Configure the logger to log messages to a file with rotation.
    """

    logger = logging.getLogger()

    # prevent duplicate log entries
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s -%(levelname)s - %(message)s]"
    )

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
