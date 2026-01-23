from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDecoder(ABC):
    def __init__(self, name: str = "BaseDecoder"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDecoder":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def _validate_input(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a numpy array.")
        if X.ndim < 2:
            raise ValueError("Input X must be at least 2D array.")

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted", False):
            raise RuntimeError(f"{self.__class__.__name__} instance is not fitted yet.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
