"""
Neural Decoding Pipeline
A machine learning pipeline for decoding neural activity.
"""
from .config import DEFAULT_CONFIG
from .data.loader import load_dataset
from .evaluation.metrics import get_R2, get_rho, evaluate_decoder
from .models.wiener import WienerFilterDecoder, WienerCascadeDecoder
from .models.kalman import KalmanFilterDecoder


from neural_decoding.config import DecodingConfig
from neural_decoding.logger import setup_logger

__version__ = "0.1.0"
__author__ = "Kidus Abebe"

__all__ = [
    "DecodingConfig",
    "setup_logger",
]
