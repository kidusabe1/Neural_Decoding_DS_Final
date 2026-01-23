from typing import Tuple, Optional

import numpy as np


def bin_spikes(spike_times, bin_size, start_time, end_time):
    """
    Function that puts spikes into bins
    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for putting spikes in bins
    wdw_end: number (any format)
        the end time for putting spikes in bins
    Returns
    -------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    """
    edges = np.arange(start_time, end_time, bin_size)  # Get edges of time bins
    num_bins = edges.shape[0] - 1  # Number of bins
    num_neurons = len(spike_times)  # Number of neurons
    neural_data = np.empty([num_bins, num_neurons])  # Initialize array for binned neural data
    # Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        spikes_1d = np.array(spike_times[i]).flatten()
        neural_data[:, i] = np.histogram(spikes_1d, edges)[0]
    return neural_data


def bin_output(outputs, output_times, bin_size, start_time, end_time, downsample_factor=1):
    outputs = np.atleast_2d(outputs)
    if outputs.shape[0] == 1 and outputs.shape[1] > 1:
        outputs = outputs.T
    """
    Function that puts outputs into bins
    Parameters
    ----------
    outputs: matrix of size "number of times the output was recorded" x "number of features in the output"
        each entry in the matrix is the value of the output feature
    output_times: a vector of size "number of times the output was recorded"
        each entry has the time the output was recorded
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for binning the outputs
    wdw_end: number (any format)
        the end time for binning the outputs
    downsample_factor: integer, optional, default=1
        how much to downsample the outputs prior to binning
        larger values will increase speed, but decrease precision
    Returns
    -------
    outputs_binned: matrix of size "number of time bins" x "number of features in the output"
        the average value of each output feature in every time bin
    """
    # Downsample output
    if downsample_factor != 1:
        downsample_idxs = np.arange(0, output_times.shape[0], downsample_factor)
        outputs = outputs[downsample_idxs, :]
        output_times = output_times[downsample_idxs]
    edges = np.arange(start_time, end_time, bin_size)  # Get edges of time bins
    num_bins = edges.shape[0] - 1  # Number of bins
    output_dim = outputs.shape[1]  # Number of output features
    outputs_binned = np.empty([num_bins, output_dim])  # Initialize matrix of binned outputs
    # Loop through bins, and get the mean outputs in those bins
    for i in range(num_bins):
        idxs = np.where((np.squeeze(output_times) >= edges[i]) & (np.squeeze(output_times) < edges[i + 1]))[0]
        for j in range(output_dim):
            if idxs.size == 0:
                outputs_binned[i, j] = np.nan
            else:
                outputs_binned[i, j] = np.mean(outputs[idxs, j])
    return outputs_binned


def get_spikes_with_history(neural_data: np.ndarray, bins_before: int, bins_after: int, bins_current: int = 1) -> np.ndarray:
    """
    Function that creates the covariate matrix of neural activity
    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding
    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """
    num_examples = neural_data.shape[0]  # Number of total time bins we have neural data for
    num_neurons = neural_data.shape[1]  # Number of neurons
    surrounding_bins = bins_before + bins_after + bins_current  # Number of surrounding time bins used for prediction
    X = np.empty([num_examples, surrounding_bins, num_neurons])  # Initialize covariate matrix with NaNs
    X[:] = np.nan
    start_idx = 0
    for i in range(num_examples - bins_before - bins_after):
        end_idx = start_idx + surrounding_bins
        X[i + bins_before, :, :] = neural_data[start_idx:end_idx, :]
        start_idx = start_idx + 1
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
    end_idx = X.shape[0] - bins_after if bins_after > 0 else X.shape[0]
    X = X[bins_before:end_idx]
    end_y_idx = outputs.shape[0] - bins_after if bins_after > 0 else outputs.shape[0]
    y = outputs[bins_before:end_y_idx]
    # Flatten for non-recurrent models
    X_flat = flatten_spike_history(X)
    # Drop any samples with NaNs (e.g., empty output bins)
    valid_mask = (~np.isnan(X_flat).any(axis=1)) & (~np.isnan(y).any(axis=1))
    X_flat = X_flat[valid_mask]
    y = y[valid_mask]
    n = X_flat.shape[0]
    if n == 0:
        raise ValueError("No valid samples available after removing NaNs. Check binning parameters.")
    # Split into train/val/test
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
