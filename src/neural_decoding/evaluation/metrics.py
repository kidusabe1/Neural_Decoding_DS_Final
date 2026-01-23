from typing import Union

import numpy as np


def get_R2(y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
    """R-squared (coefficient of determination) per output."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    n_outputs = y_true.shape[1]
    r2 = np.zeros(n_outputs)
    for i in range(n_outputs):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2[i] = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return r2 if n_outputs > 1 else r2[0]


def get_rho(y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
    """Pearson correlation per output."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    n_outputs = y_true.shape[1]
    rho = np.zeros(n_outputs)
    for i in range(n_outputs):
        rho[i] = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
    return rho if n_outputs > 1 else rho[0]


def evaluate_decoder(y_true: np.ndarray, y_pred: np.ndarray, decoder_name: str = "Decoder") -> dict:
    """Convenience wrapper returning r2 and rho (plus rmse-like metric for completeness)."""
    r2 = get_R2(y_true, y_pred)
    rho = get_rho(y_true, y_pred)
    rmse = np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))
    return {"decoder": decoder_name, "r2": r2, "pearson_correlation": rho, "rmse": rmse}

# Backward-compatible aliases
get_r2_score = get_R2
get_pearson_correlation = get_rho
get_rmse = lambda y_true, y_pred: np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))
