"""Wiener filter based decoders."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn import linear_model

from neural_decoding.models.base import BaseDecoder


class WienerFilterDecoder(BaseDecoder):
    """Linear regression decoder (Wiener filter) matching KordingLab implementation.

    Attributes:
        fit_intercept: Whether to calculate the intercept.
        model: Trained sklearn LinearRegression model.
    """

    def __init__(self, fit_intercept: bool = True):
        """Initialize Wiener Filter Decoder.

        Args:
            fit_intercept: Whether to fit an intercept term.
        """
        super().__init__(name="WienerFilter")
        self.fit_intercept = fit_intercept
        self.model: Optional[linear_model.LinearRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> WienerFilterDecoder:
        """Fit the Wiener filter.

        Args:
            X: Input features.
            y: Target outputs.

        Returns:
            Self instance.
        """
        self.model = linear_model.LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the Wiener filter.

        Args:
            X: Input features.

        Returns:
            Predicted outputs.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X)

