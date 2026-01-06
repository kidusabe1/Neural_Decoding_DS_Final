from typing import Optional

import numpy as np

from neural_decoding.models.base import BaseDecoder


class WienerFilterDecoder(BaseDecoder):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WienerFilterDecoder":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class WienerCascadeDecoder(BaseDecoder):
    def __init__(self, degree: int = 3):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WienerCascadeDecoder":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
