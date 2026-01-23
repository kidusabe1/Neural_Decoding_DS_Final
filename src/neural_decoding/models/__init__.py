"""
Neural decoding models module.
"""

from neural_decoding.models.base import BaseDecoder
from neural_decoding.models.wiener import WienerFilterDecoder, WienerCascadeDecoder
from neural_decoding.models.kalman import KalmanFilterDecoder

# Optional neural-net decoders (TensorFlow required)
try:
    from neural_decoding.models.neural_nets import DenseNNDecoder, LSTMDecoder
except Exception:  # TensorFlow not installed or other import issue
    DenseNNDecoder = None
    LSTMDecoder = None

__all__ = [
    "BaseDecoder",
    "WienerFilterDecoder",
    "WienerCascadeDecoder",
    "KalmanFilterDecoder",
    "DenseNNDecoder",
    "LSTMDecoder",
]
