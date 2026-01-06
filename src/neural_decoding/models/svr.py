from typing import Optional

import numpy as np

from neural_decoding.models.base import BaseDecoder


class SVRDecoder(BaseDecoder):
    def __init__(self, regularization_c: float = 1.0, max_iterations: int = 1000, kernel: str = "rbf"):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRDecoder":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
