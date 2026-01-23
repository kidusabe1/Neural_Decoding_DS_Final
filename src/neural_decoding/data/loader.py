from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from scipy import io



def load_pickle_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load neural data and outputs from a pickle file.
    Returns tuple (neural_data, outputs)
    """
    with open(file_path, 'rb') as f:
        neural_data, outputs = pickle.load(f)
    return neural_data, outputs



def load_h5_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("HDF5 loading not implemented.")




def load_dataset(file_path: Path) -> dict:
    """
    Load Matlab .mat data and return a dict with spike_times, vels, vel_times.
    """
    data = io.loadmat(file_path)
    # Extract spike_times as a list of 1D arrays (one per neuron)
    if 'spike_times' in data:
        spike_times_raw = data['spike_times']  # shape (n_neurons, 1), dtype=object
        spike_times = [np.array(spike_times_raw[i, 0]).flatten() for i in range(spike_times_raw.shape[0])]
    else:
        raise KeyError("No 'spike_times' key found in the .mat file.")

    # Extract outputs and output_times
    if 'vels' in data and 'vel_times' in data:
        outputs = data['vels']
        output_times = data['vel_times'].flatten()
    elif 'pos' in data and 'pos_times' in data:
        outputs = data['pos']
        output_times = data['pos_times'].flatten()
    else:
        raise KeyError("No valid outputs and output_times found in the .mat file.")

    return {
        'spike_times': spike_times,
        'outputs': outputs,
        'output_times': output_times,
    }
