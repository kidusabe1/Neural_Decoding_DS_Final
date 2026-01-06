"""
Neural decoding models module.
"""

from neural_decoding.models.base import BaseDecoder
from neural_decoding.models.wiener import WienerFilterDecoder, WienerCascadeDecoder
from neural_decoding.models.kalman import KalmanFilterDecoder
from neural_decoding.models.svr import SVRDecoder
from neural_decoding.models.xgboost_decoder import XGBoostDecoder
from neural_decoding.models.neural_nets import (
    DenseNNDecoder,
    SimpleRNNDecoder,
    GRUDecoder,
    LSTMDecoder,
)

__all__ = [
    "BaseDecoder",
    "WienerFilterDecoder",
    "WienerCascadeDecoder",
    "KalmanFilterDecoder",
    "SVRDecoder",
    "XGBoostDecoder",
    "DenseNNDecoder",
    "SimpleRNNDecoder",
    "GRUDecoder",
    "LSTMDecoder",
]
