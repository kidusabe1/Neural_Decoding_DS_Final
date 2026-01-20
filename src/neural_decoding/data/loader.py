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
    return {
        'spike_times': data['spike_times'],
        'vels': data['vels'],
        'vel_times': data['vel_times']
    }
