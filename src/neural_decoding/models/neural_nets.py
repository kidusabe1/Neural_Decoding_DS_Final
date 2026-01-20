from typing import Optional

import numpy as np

from neural_decoding.models.base import BaseDecoder


class DenseNNDecoder(BaseDecoder):
    def __init__(self, units: int = 400, dropout_rate: float = 0.25, num_epochs: int = 10, batch_size: int = 128, verbose: int = 1):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DenseNNDecoder":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class LSTMDecoder(BaseDecoder):
    def __init__(self, units: int = 400, dropout_rate: float = 0.25, num_epochs: int = 10, batch_size: int = 128, verbose: int = 1):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMDecoder":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
