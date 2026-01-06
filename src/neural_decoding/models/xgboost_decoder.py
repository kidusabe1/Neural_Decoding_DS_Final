from typing import Optional

import numpy as np

from neural_decoding.models.base import BaseDecoder


class XGBoostDecoder(BaseDecoder):
    def __init__(self, max_depth: int = 3, n_estimators: int = 100, learning_rate: float = 0.3, use_gpu: bool = False):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostDecoder":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
