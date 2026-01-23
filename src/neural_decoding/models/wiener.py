from typing import Optional

import numpy as np
from sklearn import linear_model

from neural_decoding.models.base import BaseDecoder


class WienerFilterDecoder(BaseDecoder):
    """Linear regression decoder (Wiener filter) matching KordingLab implementation."""

    def __init__(self, fit_intercept: bool = True):
        super().__init__(name="WienerFilter")
        self.fit_intercept = fit_intercept
        self.model: Optional[linear_model.LinearRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WienerFilterDecoder":
        self.model = linear_model.LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X)


class WienerCascadeDecoder(BaseDecoder):
    """Linear + static nonlinearity (polynomial) decoder matching KordingLab implementation."""

    def __init__(self, degree: int = 3):
        super().__init__(name="WienerCascade")
        self.degree = degree
        self.models = None  # list of (linear_model, poly_coef)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WienerCascadeDecoder":
        num_outputs = y.shape[1]
        models = []
        for i in range(num_outputs):
            regr = linear_model.LinearRegression()
            regr.fit(X, y[:, i])
            y_lin = regr.predict(X)
            p = np.polyfit(y_lin, y[:, i], self.degree)
            models.append((regr, p))
        self.models = models
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.models is None:
            raise RuntimeError("Model not fitted.")
        num_outputs = len(self.models)
        y_pred = np.empty((X.shape[0], num_outputs))
        for i, (regr, p) in enumerate(self.models):
            y_lin = regr.predict(X)
            y_pred[:, i] = np.polyval(p, y_lin)
        return y_pred
