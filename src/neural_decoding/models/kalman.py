from typing import Optional

import numpy as np

from neural_decoding.models.base import BaseDecoder


class KalmanFilterDecoder(BaseDecoder):
    def __init__(self, noise_scale_c: float = 1.0):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KalmanFilterDecoder":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
