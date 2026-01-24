"""App logger module."""

import logging

# Configure the root logger with both file and console handlers
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)

# Get the root logger
logger = logging.getLogger()
