"""Tests for decoder models."""

import numpy as np
import pytest

from neural_decoding.models.wiener import WienerFilterDecoder
from neural_decoding.models.kalman import KalmanFilterDecoder


@pytest.fixture
def dummy_data():
    """Create dummy training data."""
    X = np.random.rand(100, 10)  # 100 samples, 10 neurons
    y = np.random.rand(100, 2)   # 100 samples, 2 kinematic vars
    return X, y


def test_wiener_filter_fit_predict(dummy_data):
    """Test Wiener Filter fitting and prediction."""
    X, y = dummy_data
    model = WienerFilterDecoder()
    
    # Check default state
    assert not model.is_fitted
    
    # Fit
    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None
    
    # Predict
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_kalman_filter_fit_predict(dummy_data):
    """Test Kalman Filter fitting and prediction."""
    X, y = dummy_data
    model = KalmanFilterDecoder()
    
    # Fit
    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None
    
    # Predict
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_model_not_fitted_error():
    """Test error when predicting without fitting."""
    model = WienerFilterDecoder()
    X = np.random.rand(10, 5)
    
    with pytest.raises(RuntimeError):
        model.predict(X)
