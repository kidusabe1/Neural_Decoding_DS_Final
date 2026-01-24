"""Base class for all decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDecoder(ABC):
    """Abstract base class for all neural decoders.

    Attributes:
        name: Name of the decoder.
        is_fitted: Whether the decoder has been fitted.
    """

    def __init__(self, name: str = "BaseDecoder"):
        """Initialize base decoder.

        Args:
            name: Name of the decoder.
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseDecoder:
        """Fit the decoder to training data.

        Args:
            X: Input features (time x features).
            y: Target values (time x outputs).

        Returns:
            Self instance.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for new data.

        Args:
            X: Input features (time x features).

        Returns:
            Predicted outputs (time x outputs).
        """
        ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the decoder and predict on the same data.

        Args:
            X: Input features.
            y: Target values.

        Returns:
            Predicted outputs.
        """
        self.fit(X, y)
        return self.predict(X)

    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input data format.

        Args:
            X: Input array to validate.

        Raises:
            ValueError: If input is not a numpy array or has wrong dimensions.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a numpy array.")
        if X.ndim < 2:
            raise ValueError("Input X must be at least 2D array.")

    def _check_is_fitted(self) -> None:
        """Check if model is fitted.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not getattr(self, "is_fitted", False):
            raise RuntimeError(f"{self.__class__.__name__} instance is not fitted yet.")

    def __repr__(self) -> str:
        """Return string representation of the decoder."""
        return f"{self.__class__.__name__}()"
