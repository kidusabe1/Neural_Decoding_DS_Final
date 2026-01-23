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
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    fig, axes = plt.subplots(n_outputs, 1, figsize=figsize, sharex=True)
    if n_outputs == 1:
        axes = [axes]
    if time_indices is None:
        time_indices = np.arange(y_true.shape[0])
    for i in range(n_outputs):
        ax = axes[i]
        ax.plot(time_indices, y_true[:, i] if n_outputs > 1 else y_true, label="True")
        ax.plot(time_indices, y_pred[:, i] if n_outputs > 1 else y_pred, label="Predicted")
        name = output_names[i] if output_names and i < len(output_names) else f"Output {i+1}"
        ax.set_title(f"{name}")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_decoder_comparison(results: Dict[str, Dict], metric: str = "r2", figsize: tuple = (10, 6)) -> plt.Figure:
    decoders = list(results.keys())
    metric_vals = [np.mean(results[d][metric]) if isinstance(results[d][metric], (np.ndarray, list)) else results[d][metric] for d in decoders]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(decoders, metric_vals, color="skyblue")
    ax.set_ylabel(metric)
    ax.set_title(f"Decoder Comparison: {metric}")
    for i, v in enumerate(metric_vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_idx: int = 0,
    title: str = "Predicted vs Actual",
    figsize: tuple = (6, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    y_true_dim = y_true[:, output_idx] if y_true.ndim > 1 else y_true
    y_pred_dim = y_pred[:, output_idx] if y_pred.ndim > 1 else y_pred
    ax.scatter(y_true_dim, y_pred_dim, alpha=0.5)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title + f" (Output {output_idx+1})")
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, "r--", alpha=0.7)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filepath: Path, dpi: int = 150, formats: List[str] = None) -> None:
    if formats is None:
        formats = ["png"]
    for fmt in formats:
        out_path = filepath.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, format=fmt)
        print(f"Saved figure to {out_path}")
