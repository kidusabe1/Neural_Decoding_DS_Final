"""Neural Decoding Pipeline - A machine learning pipeline for decoding neural activity."""

from __future__ import annotations

from neural_decoding.config import DecodingConfig, DEFAULT_CONFIG, Paths
from neural_decoding.logger import logger

__version__ = "0.1.0"
__author__ = "Kidus Abebe"

__all__ = [
    "DecodingConfig",
    "DEFAULT_CONFIG",
    "Paths",
    "logger",
]

