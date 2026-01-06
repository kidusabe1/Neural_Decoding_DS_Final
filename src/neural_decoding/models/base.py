from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDecoder(ABC):
    def __init__(self, name: str = "BaseDecoder"):
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDecoder":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def _validate_input(self, X: np.ndarray) -> None:
        pass

    def _check_is_fitted(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
