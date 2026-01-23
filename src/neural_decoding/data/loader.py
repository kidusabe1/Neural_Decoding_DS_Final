"""Data loading utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import io


def load_pickle_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load neural data and outputs from a pickle file.

    Args:
        file_path: Path to the pickle file.

    Returns:
        Tuple containing neural data and outputs.
    """
    with open(file_path, "rb") as f:
        neural_data, outputs = pickle.load(f)
    return neural_data, outputs


def load_h5_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load neural data from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        Tuple containing neural data and outputs.

    Raises:
        NotImplementedError: As HDF5 loading is not yet implemented.
    """
    raise NotImplementedError("HDF5 loading not implemented.")


def load_dataset(file_path: Path) -> Dict[str, np.ndarray]:
    """Load Matlab .mat data and return a dict with spike_times, outputs, output_times.

    Args:
        file_path: Path to the .mat file.

    Returns:
        Dictionary containing 'spike_times', 'outputs', and 'output_times'.

    Raises:
        KeyError: If required keys are missing from the .mat file.
    """
    data = io.loadmat(file_path)
    # Extract spike_times as a list of 1D arrays (one per neuron)
    if "spike_times" in data:
        spike_times_raw = data["spike_times"]  # shape (n_neurons, 1), dtype=object
        spike_times = [
            np.array(spike_times_raw[i, 0]).flatten()
            for i in range(spike_times_raw.shape[0])
        ]
    else:
        raise KeyError("No 'spike_times' key found in the .mat file.")

    # Extract outputs and output_times
    if "vels" in data and "vel_times" in data:
        outputs = data["vels"]
        output_times = data["vel_times"].flatten()
    elif "pos" in data and "pos_times" in data:
        outputs = data["pos"]
        output_times = data["pos_times"].flatten()
    else:
        raise KeyError("No valid outputs and output_times found in the .mat file.")

    return {
        "spike_times": spike_times,
        "outputs": outputs,
        "output_times": output_times,
    }
