import logging
import os

def setup_logger(log_file='training.log', log_level=logging.INFO):
    """Set up the logger for the training process."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create a file handler
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file_handler = logging.FileHandler(os.path.join('logs', log_file))
    file_handler.setLevel(log_level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_metrics(metrics):
    """Log training metrics."""
    logger = logging.getLogger()
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

def log_error(error_message):
    """Log error messages."""
    logger = logging.getLogger()
    logger.error(error_message)