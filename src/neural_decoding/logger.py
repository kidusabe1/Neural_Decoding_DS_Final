import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "neural_decoding",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
        import logging

        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    # Example usage:
    # logger = setup_logger()
    # logger.info("Pipeline started.")
    # logger.warning("This is a warning.")
    # logger.error("An error occurred.")
    # logger.debug("Debugging info.")


def get_logger(name: str = "neural_decoding") -> logging.Logger:
    pass
