from typing import Tuple, Optional

import numpy as np


def bin_spikes(spike_times: np.ndarray, bin_size: float, start_time: float, end_time: float) -> np.ndarray:
    """
    Bin spike times into firing rates per neuron per time bin.
    Returns array of shape (n_time_bins, n_neurons)
    """
    n_neurons = spike_times.shape[0]
    n_bins = int(np.ceil((end_time - start_time) / bin_size))
    binned = np.zeros((n_bins, n_neurons))
    bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
    for i in range(n_neurons):
        spikes = spike_times[i]
        counts, _ = np.histogram(spikes, bins=bin_edges)
        binned[:, i] = counts / bin_size
    return binned


def bin_output(output_signal: np.ndarray, output_times: np.ndarray, bin_size: float, start_time: float, end_time: float, downsample_factor: int = 1) -> np.ndarray:
    """
    Bin output signal to match neural data bins.
    Returns binned output of shape (n_time_bins, n_features)
    """
    n_bins = int(np.ceil((end_time - start_time) / bin_size))
    binned = []
    bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
    for i in range(n_bins):
        mask = (output_times >= bin_edges[i]) & (output_times < bin_edges[i+1])
        if np.any(mask):
            binned.append(np.mean(output_signal[mask], axis=0))
        else:
            binned.append(np.zeros(output_signal.shape[1]))
    binned = np.stack(binned)
    if downsample_factor > 1:
        binned = binned[::downsample_factor]
    return binned


def get_spikes_with_history(neural_data: np.ndarray, bins_before: int, bins_after: int, bins_current: int = 1) -> np.ndarray:
    """
    Create covariate matrix with spike history from multiple time bins.
    Returns 3D array (n_samples, n_bins_total, n_neurons)
    """
    n_time, n_neurons = neural_data.shape
    n_bins = bins_before + bins_after + bins_current
    X = np.zeros((n_time - bins_before - bins_after, n_bins, n_neurons))
    for t in range(bins_before, n_time - bins_after):
        idx = 0
        for b in range(-bins_before, bins_current + bins_after):
            X[t - bins_before, idx, :] = neural_data[t + b, :]
            idx += 1
    return X


def flatten_spike_history(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D spike history matrix to 2D for non-recurrent models.
    (n_samples, n_bins, n_neurons) -> (n_samples, n_bins * n_neurons)
    """
    return X.reshape(X.shape[0], -1)


def prepare_train_test_split(
    neural_data: np.ndarray,
    outputs: np.ndarray,
    bins_before: int,
    bins_after: int,
    bins_current: int = 1,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    Prepare data for training by creating spike history and splitting.
    Returns X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
    y = outputs[bins_before:outputs.shape[0]-bins_after]
    # Flatten for non-recurrent models
    X_flat = flatten_spike_history(X)
    # Split into train/val/test
    n = X_flat.shape[0]
    np.random.seed(random_seed)
    idx = np.random.permutation(n)
    n_test = int(test_size * n)
    n_valid = int(validation_size * n)
    n_train = n - n_test - n_valid
    X_train = X_flat[idx[:n_train]]
    y_train = y[idx[:n_train]]
    X_valid = X_flat[idx[n_train:n_train+n_valid]]
    y_valid = y[idx[n_train:n_train+n_valid]]
    X_test = X_flat[idx[n_train+n_valid:]]
    y_test = y[idx[n_train+n_valid:]]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def zscore_normalize(data: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize data. Returns normalized data, mean, std.
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    std[std == 0] = 1.0
    normed = (data - mean) / std
    return normed, mean, std
