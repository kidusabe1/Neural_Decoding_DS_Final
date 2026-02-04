"""Kalman filter implementation.

Edited by Gerrik 24/1/2025 - Added type hints and error handling for unfitted model.
"""
try:
    from __future__ import annotations
    from typing import Optional, Tuple
except ImportError:
    raise ImportError(
        "Your Python version does not support type hints. Please upgrade to Python 3.7 or higher."
    )

import numpy as np
from numpy.linalg import inv

from neural_decoding.models.base import BaseDecoder


class KalmanFilterDecoder(BaseDecoder):
    """Kalman filter decoder matching the KordingLab implementation (Wu et al. 2003 style).

    Attributes:
        C_scale (float): Scaling factor for noise covariance.
        model (Optional[Tuple]): Tuple containing (A, W, H, Q) matrices.
        A is coefficent matrix for state transition between time steps of cursor pos.
        W is noise covariance for cursor position.
        H is coefficent matrix for mapping neural data to cursor pos.
        Q is noise covariance for neural data.
    """

    def __init__(self, noise_scale_c: float = 1.0) -> None:
        """Initialize Kalman Filter Decoder.

        Args:
            noise_scale_c: Scaling factor for noise covariance.

        Returns:
            KalmanFilterDecoder.
        """
        super().__init__(name="KalmanFilter")
        self.C_scale = noise_scale_c
        self.model: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
            None  # tuple (A, W, H, Q) Preallocate space for model.
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> KalmanFilterDecoder:
        """Fit the Kalman Filter model.
        Args:
            X: Neural data (time x neurons).
            y: Outputs (time x kinematic dims).

        Returns:
            Returns the kalman filter, fitted to the data.
            Sets the A, W, H, Q matrices in self.model.
        """
        # X: neural data (time x neurons); y: outputs (time x kinematic dims)
        X_mat = y.T
        Z_mat = X.T

        # In paper this should be C, the number
        # of columns or neurons or signal sources effecting state.
        num_time = X_mat.shape[1]
        # Align sequences: X1 (t-1), X2 (t) with equal length
        X1 = X_mat[:, :-1]
        X2 = X_mat[:, 1:]

        # Transition matrix A and noise term for position W.
        # X2@X1.T is covariance between neurons between time steps. A
        # (X1@X1.T) is covariance of neurons at time step t-1.
        # inv(X1@X1.T) is inverse of that covariance, and may not work if
        # matrix is singular. Measures conditional dependence between time samples.
        # If a matrix is singular, that means signal sources are linearly dependent.

        A = (
            X2 @ X1.T @ inv(X1 @ X1.T)
        )  # X2X1^T (X1X1^T)^-1. X2X1^T is the covariance between neurons between time steps.
        W = (X2 - A @ X1) @ (X2 - A @ X1).T / (num_time - 1) / self.C_scale

        # Measurement matrix H and noise matrix Q.
        H = Z_mat @ X_mat.T @ inv(X_mat @ X_mat.T)
        Q = (Z_mat - H @ X_mat) @ (Z_mat - H @ X_mat).T / num_time

        self.model = (A, W, H, Q)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, y_init: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict using the fitted Kalman Filter.

        Args:
            X: Neural data (time x neurons).
            y_init: Optional initial state (kinematics).

        Returns:
            Predicted outputs size(time x kinematics).

        Raises:
            RuntimeError: If model is not fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Please call 'fit' before 'predict'.")

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
