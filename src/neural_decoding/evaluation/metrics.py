from typing import Union, List

import numpy as np


def get_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
    pass


def get_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
    pass


def get_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
    pass


def evaluate_decoder(y_true: np.ndarray, y_pred: np.ndarray, decoder_name: str = "Decoder") -> dict:
    pass
