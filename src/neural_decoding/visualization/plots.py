from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: Optional[List[str]] = None,
    title: str = "Predicted vs Actual",
    figsize: tuple = (12, 4),
    time_indices: Optional[np.ndarray] = None,
) -> plt.Figure:
    pass


def plot_decoder_comparison(results: Dict[str, Dict], metric: str = "r2", figsize: tuple = (10, 6)) -> plt.Figure:
    pass


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_idx: int = 0,
    title: str = "Predicted vs Actual",
    figsize: tuple = (6, 6),
) -> plt.Figure:
    pass


def save_figure(fig: plt.Figure, filepath: Path, dpi: int = 150, formats: List[str] = None) -> None:
    pass
