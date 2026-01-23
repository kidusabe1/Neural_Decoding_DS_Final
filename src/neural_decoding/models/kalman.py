"""Kalman filter implementation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.linalg import inv

from neural_decoding.models.base import BaseDecoder


class KalmanFilterDecoder(BaseDecoder):
    """Kalman filter decoder matching the KordingLab implementation (Wu et al. 2003 style).

    Attributes:
        C_scale (float): Scaling factor for noise covariance.
        model (Optional[Tuple]): Tuple containing (A, W, H, Q) matrices.
    """

    def __init__(self, noise_scale_c: float = 1.0):
        """Initialize Kalman Filter Decoder.

        Args:
            noise_scale_c: Scaling factor for noise covariance.
        """
        super().__init__(name="KalmanFilter")
        self.C_scale = noise_scale_c
        self.model: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None  # tuple (A, W, H, Q)

    def fit(self, X: np.ndarray, y: np.ndarray) -> KalmanFilterDecoder:
        """Fit the Kalman Filter model.

        Args:
            X: Neural data (time x neurons).
            y: Outputs (time x kinematic dims).

        Returns:
            Self instance.
        """
        # X: neural data (time x neurons); y: outputs (time x kinematic dims)
        X_mat = y.T
        Z_mat = X.T

        num_time = X_mat.shape[1]
        X1 = X_mat[:, 0 : num_time - 1]
        X2 = X_mat[:, 1:]

        # Transition matrix A and its covariance W
        A = X2 @ X1.T @ inv(X1 @ X1.T)
        W = (X2 - A @ X1) @ (X2 - A @ X1).T / (num_time - 1) / self.C_scale

        # Measurement matrix H and its covariance Q
        H = Z_mat @ X_mat.T @ inv(X_mat @ X_mat.T)
        Q = (Z_mat - H @ X_mat) @ (Z_mat - H @ X_mat).T / num_time

        self.model = (A, W, H, Q)
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, y_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict using the fitted Kalman Filter.

        Args:
            X: Neural data (time x neurons).
            y_init: Optional initial state (kinematics).

        Returns:
            Predicted outputs (time x kinematics).
        
        Raises:
            RuntimeError: If model is not fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        A, W, H, Q = self.model
        X_true = X.T

        num_states = A.shape[0]
        num_timesteps = X_true.shape[1]
        states = np.empty((num_states, num_timesteps))
        P_m = np.zeros((num_states, num_states))
        P = np.zeros((num_states, num_states))

        if y_init is not None:
            state = np.asarray(y_init).reshape(-1, 1)
        else:
            state = np.zeros((num_states, 1))
        states[:, 0] = state.flatten()

        for t in range(X_true.shape[1] - 1):
            P_m = A @ P @ A.T + W
            state_m = A @ state
            K = P_m @ H.T @ inv(H @ P_m @ H.T + Q)
            P = (np.eye(num_states) - K @ H) @ P_m

            # Ensure observation is a column vector
            observation = X_true[:, t + 1].reshape(-1, 1)
            state = state_m + K @ (observation - H @ state_m)
            states[:, t + 1] = state.flatten()

        return np.asarray(states.T)
