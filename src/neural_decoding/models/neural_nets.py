from typing import Optional

import numpy as np

from neural_decoding.models.base import BaseDecoder



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DenseNNDecoder(BaseDecoder):
    def __init__(self, units: int = 400, dropout_rate: float = 0.25, num_epochs: int = 10, batch_size: int = 128, verbose: int = 1):
        super().__init__(name="DenseNN")
        self.units = units
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DenseNNDecoder":
        input_shape = (X.shape[1],)
        n_outputs = y.shape[1]
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(self.units, activation="relu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.units, activation="relu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(n_outputs)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=self.num_epochs, batch_size=self.batch_size, verbose=self.verbose)
        self.model = model
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X, verbose=0)


class LSTMDecoder(BaseDecoder):
    def __init__(self, units: int = 400, dropout_rate: float = 0.25, num_epochs: int = 10, batch_size: int = 128, verbose: int = 1):
        super().__init__(name="LSTM")
        self.units = units
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMDecoder":
        # Reshape X to (samples, timesteps, features) if needed
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        input_shape = (X.shape[1], X.shape[2])
        n_outputs = y.shape[1]
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(self.units, dropout=self.dropout_rate),
            layers.Dense(n_outputs)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=self.num_epochs, batch_size=self.batch_size, verbose=self.verbose)
        self.model = model
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X, verbose=0)
