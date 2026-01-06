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
    pass


def get_logger(name: str = "neural_decoding") -> logging.Logger:
    pass
