import logging
import os

def get_logger():

    logger = logging.getLogger("fraud_detection")

    # Prevent duplicate logs
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create logs folder if not exists
    os.makedirs("logs", exist_ok=True)

    # File handler
    file_handler = logging.FileHandler("logs/app.log")

    # Console handler (prints logs in terminal)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger