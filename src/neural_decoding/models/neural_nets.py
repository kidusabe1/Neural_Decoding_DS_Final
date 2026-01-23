"""Neural network based decoders."""

from __future__ import annotations

from typing import Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from neural_decoding.models.base import BaseDecoder


class DenseNNDecoder(BaseDecoder):
    """Dense Neural Network decoder.

    Attributes:
        units: Number of hidden units.
        dropout_rate: Dropout rate.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        verbose: Verbosity level for training.
        model: Trained Keras model.
    """

    def __init__(
        self,
        units: int = 400,
        dropout_rate: float = 0.25,
        num_epochs: int = 10,
        batch_size: int = 128,
        verbose: int = 1,
    ):
        """Initialize Dense Neural Network Decoder.

        Args:
            units: Number of neurons in hidden layers.
            dropout_rate: Fraction of units to drop.
            num_epochs: Number of epochs to train.
            batch_size: Batch size.
            verbose: Verbosity mode (0=silent, 1=progress bar).
        """
        super().__init__(name="DenseNN")
        self.units = units
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model: Optional[keras.Model] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> DenseNNDecoder:
        """Fit the dense neural network.

        Args:
            X: Input features.
            y: Target outputs.

        Returns:
            Self instance.
        """
        input_shape = (X.shape[1],)
        n_outputs = y.shape[1]
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Dense(self.units, activation="relu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.units, activation="relu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(n_outputs),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X,
            y,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self.model = model
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model.

        Args:
            X: Input features.

        Returns:
            Predicted outputs.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X, verbose=0)


class LSTMDecoder(BaseDecoder):
    """LSTM based decoder.

    Attributes:
        units: Number of LSTM units.
        dropout_rate: Dropout rate.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        verbose: Verbosity level for training.
        model: Trained Keras model.
    """

    def __init__(
        self,
        units: int = 400,
        dropout_rate: float = 0.25,
        num_epochs: int = 10,
        batch_size: int = 128,
        verbose: int = 1,
    ):
        """Initialize LSTM Decoder.

        Args:
            units: Number of LSTM units.
            dropout_rate: Dropout rate.
            num_epochs: Number of epochs to train.
            batch_size: Batch size.
            verbose: Verbosity mode.
        """
        super().__init__(name="LSTM")
        self.units = units
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model: Optional[keras.Model] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSTMDecoder:
        """Fit the LSTM model.

        Args:
            X: Input features (reshaped automatically if 2D).
            y: Target outputs.

        Returns:
            Self instance.
        """
        # Reshape X to (samples, timesteps, features) if needed
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        input_shape = (X.shape[1], X.shape[2])
        n_outputs = y.shape[1]
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(self.units, dropout=self.dropout_rate),
                layers.Dense(n_outputs),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X,
            y,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self.model = model
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained LSTM.

        Args:
            X: Input features.

        Returns:
            Predicted outputs.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X, verbose=0)
