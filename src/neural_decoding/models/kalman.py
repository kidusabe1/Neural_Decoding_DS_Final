from typing import Optional

import numpy as np
from numpy.linalg import inv

from neural_decoding.models.base import BaseDecoder


class KalmanFilterDecoder(BaseDecoder):
    """Kalman filter decoder matching the KordingLab implementation (Wu et al. 2003 style)."""

    def __init__(self, noise_scale_c: float = 1.0):
        super().__init__(name="KalmanFilter")
        self.C_scale = noise_scale_c
        self.model = None  # tuple (A, W, H, Q)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KalmanFilterDecoder":
        # X: neural data (time x neurons); y: outputs (time x kinematic dims)
        X_mat = np.matrix(y.T)
        Z_mat = np.matrix(X.T)

        num_time = X_mat.shape[1]
        X1 = X_mat[:, 0 : num_time - 1]
        X2 = X_mat[:, 1:]

        # Transition matrix A and its covariance W
        A = X2 * X1.T * inv(X1 * X1.T)
        W = (X2 - A * X1) * (X2 - A * X1).T / (num_time - 1) / self.C_scale

        # Measurement matrix H and its covariance Q
        H = Z_mat * X_mat.T * inv(X_mat * X_mat.T)
        Q = (Z_mat - H * X_mat) * (Z_mat - H * X_mat).T / num_time

        self.model = (A, W, H, Q)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, y_init: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        A, W, H, Q = self.model
        X_true = np.matrix(X.T)

        num_states = A.shape[0]
        num_timesteps = X_true.shape[1]
        states = np.empty((num_states, num_timesteps))
        P_m = np.matrix(np.zeros((num_states, num_states)))
        P = np.matrix(np.zeros((num_states, num_states)))

        if y_init is not None:
            state = np.matrix(np.asarray(y_init).reshape(-1, 1))
        else:
            state = np.matrix(np.zeros((num_states, 1)))
        states[:, 0] = np.squeeze(state)

        for t in range(X_true.shape[1] - 1):
            P_m = A * P * A.T + W
            state_m = A * state
            K = P_m * H.T * inv(H * P_m * H.T + Q)
            P = (np.matrix(np.eye(num_states)) - K * H) * P_m
            state = state_m + K * (X_true[:, t + 1] - H * state_m)
            states[:, t + 1] = np.squeeze(state)

        return np.asarray(states.T)
