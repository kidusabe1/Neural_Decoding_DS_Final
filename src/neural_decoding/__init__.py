"""
Neural Decoding Pipeline

A machine learning pipeline for decoding neural activity.
"""

from neural_decoding.config import DecodingConfig
from neural_decoding.logger import setup_logger

__version__ = "0.1.0"
__author__ = "Kidus Abebe"

__all__ = [
    "DecodingConfig",
    "setup_logger",
]
