from typing import Tuple, Dict

import numpy as np

from bayes_opt import BayesianOptimization

from neural_decoding.evaluation.metrics import get_R2
from neural_decoding.models import (
    WienerFilterDecoder,
    WienerCascadeDecoder,
    KalmanFilterDecoder,
    DenseNNDecoder,
    LSTMDecoder,
)


def _stack_train_valid(X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_full = np.vstack([X_train, X_valid])
    y_full = np.vstack([y_train, y_valid])
    return X_full, y_full


def optimize_wiener_filter(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    init_points: int = 5,
    n_iter: int = 5,
    kappa: float = 2.5,
) -> Tuple[WienerFilterDecoder, Dict]:
    """Bayesian optimization over a simple hyperparameter for the Wiener filter.

    We optimize whether to use an intercept term.
    """

    def wf_evaluate(fit_intercept: float) -> float:
        use_intercept = bool(round(fit_intercept))
        model = WienerFilterDecoder(fit_intercept=use_intercept)
        model.fit(X_train, y_train)
        y_valid_pred = model.predict(X_valid)
        return float(np.mean(get_R2(y_valid, y_valid_pred)))

    wf_bo = BayesianOptimization(wf_evaluate, {"fit_intercept": (0.0, 1.0)}, verbose=0)
    wf_bo.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa)

    best_params = wf_bo.res["max"]["max_params"]
    use_intercept = bool(round(best_params["fit_intercept"]))

    X_full, y_full = _stack_train_valid(X_train, X_valid, y_train, y_valid)
    best_model = WienerFilterDecoder(fit_intercept=use_intercept)
    best_model.fit(X_full, y_full)
    return best_model, best_params


def optimize_wiener_cascade(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    init_points: int = 5,
    n_iter: int = 5,
    kappa: float = 10.0,
) -> Tuple[WienerCascadeDecoder, Dict]:
    """Bayesian optimization over polynomial degree for Wiener cascade."""

    def wc_evaluate(degree: float) -> float:
        deg_int = int(degree)
        model = WienerCascadeDecoder(degree=deg_int)
        model.fit(X_train, y_train)
        y_valid_pred = model.predict(X_valid)
        return float(np.mean(get_R2(y_valid, y_valid_pred)))

    wc_bo = BayesianOptimization(wc_evaluate, {"degree": (1.0, 6.99)}, verbose=0)
    wc_bo.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa)

    best_params = wc_bo.res["max"]["max_params"]
    best_degree = int(best_params["degree"])

    X_full, y_full = _stack_train_valid(X_train, X_valid, y_train, y_valid)
    best_model = WienerCascadeDecoder(degree=best_degree)
    best_model.fit(X_full, y_full)
    return best_model, best_params


def optimize_kalman_filter(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    init_points: int = 5,
    n_iter: int = 5,
    kappa: float = 2.5,
) -> Tuple[KalmanFilterDecoder, Dict]:
    """Bayesian optimization over process noise scale for Kalman filter."""

    def kf_evaluate(noise_scale_c: float) -> float:
        model = KalmanFilterDecoder(noise_scale_c=noise_scale_c)
        model.fit(X_train, y_train)
        y_valid_pred = model.predict(X_valid)
        return float(np.mean(get_R2(y_valid, y_valid_pred)))

    kf_bo = BayesianOptimization(kf_evaluate, {"noise_scale_c": (0.1, 10.0)}, verbose=0)
    kf_bo.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa)

    best_params = kf_bo.res["max"]["max_params"]
    best_c = float(best_params["noise_scale_c"])

    X_full, y_full = _stack_train_valid(X_train, X_valid, y_train, y_valid)
    best_model = KalmanFilterDecoder(noise_scale_c=best_c)
    best_model.fit(X_full, y_full)
    return best_model, best_params


def optimize_dense_nn(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    init_points: int = 20,
    n_iter: int = 20,
    kappa: float = 10.0,
) -> Tuple[DenseNNDecoder, Dict]:
    """Bayesian optimization for DenseNNDecoder hyperparameters.

    Optimizes: num_units, dropout_rate, num_epochs.
    """

    def dnn_evaluate(num_units: float, frac_dropout: float, n_epochs: float) -> float:
        units = int(num_units)
        dropout = float(frac_dropout)
        epochs = int(n_epochs)
        model = DenseNNDecoder(units=units, dropout_rate=dropout, num_epochs=epochs)
        model.fit(X_train, y_train)
        y_valid_pred = model.predict(X_valid)
        return float(np.mean(get_R2(y_valid, y_valid_pred)))

    dnn_bo = BayesianOptimization(
        dnn_evaluate,
        {"num_units": (50, 600), "frac_dropout": (0.0, 0.5), "n_epochs": (2, 21)},
    )
    dnn_bo.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa)

    best_params = dnn_bo.res["max"]["max_params"]
    units = int(best_params["num_units"])
    dropout = float(best_params["frac_dropout"])
    epochs = int(best_params["n_epochs"])

    X_full, y_full = _stack_train_valid(X_train, X_valid, y_train, y_valid)
    best_model = DenseNNDecoder(units=units, dropout_rate=dropout, num_epochs=epochs)
    best_model.fit(X_full, y_full)
    return best_model, best_params


def optimize_lstm(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    init_points: int = 20,
    n_iter: int = 20,
    kappa: float = 10.0,
) -> Tuple[LSTMDecoder, Dict]:
    """Bayesian optimization for LSTMDecoder hyperparameters.

    Optimizes: num_units, dropout_rate, num_epochs.
    """

    def lstm_evaluate(num_units: float, frac_dropout: float, n_epochs: float) -> float:
        units = int(num_units)
        dropout = float(frac_dropout)
        epochs = int(n_epochs)
        model = LSTMDecoder(units=units, dropout_rate=dropout, num_epochs=epochs)
        model.fit(X_train, y_train)
        y_valid_pred = model.predict(X_valid)
        return float(np.mean(get_R2(y_valid, y_valid_pred)))

    lstm_bo = BayesianOptimization(
        lstm_evaluate,
        {"num_units": (50, 600), "frac_dropout": (0.0, 0.5), "n_epochs": (2, 21)},
    )
    lstm_bo.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa)

    best_params = lstm_bo.res["max"]["max_params"]
    units = int(best_params["num_units"])
    dropout = float(best_params["frac_dropout"])
    epochs = int(best_params["n_epochs"])

    X_full, y_full = _stack_train_valid(X_train, X_valid, y_train, y_valid)
    best_model = LSTMDecoder(units=units, dropout_rate=dropout, num_epochs=epochs)
    best_model.fit(X_full, y_full)
    return best_model, best_params
