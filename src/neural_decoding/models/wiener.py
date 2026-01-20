from typing import Optional

import numpy as np

from neural_decoding.models.base import BaseDecoder



from sklearn.linear_model import LinearRegression
import numpy.polynomial.polynomial as poly

class WienerCascadeDecoder(BaseDecoder):
    def __init__(self, degree: int = 3):
        self.degree = degree
        self.linear = LinearRegression()
        self.poly_coef = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WienerCascadeDecoder":
        y_lin = self.linear.fit(X, y).predict(X)
        # Fit polynomial to each output dimension
        self.poly_coef = []
        for i in range(y.shape[1]):
            coef = poly.Polynomial.fit(y_lin[:, i], y[:, i], self.degree).convert().coef
            self.poly_coef.append(coef)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        y_lin = self.linear.predict(X)
        y_pred = np.zeros_like(y_lin)
        for i, coef in enumerate(self.poly_coef):
            y_pred[:, i] = poly.polyval(y_lin[:, i], coef)
        return y_pred