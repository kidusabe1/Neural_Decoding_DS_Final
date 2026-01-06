from typing import Tuple, Optional

import numpy as np


def bin_spikes(spike_times: np.ndarray, bin_size: float, start_time: float, end_time: float) -> np.ndarray:
    pass


def bin_output(output_signal: np.ndarray, output_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    pass


def get_spikes_with_history(neural_data: np.ndarray, bins_before: int, bins_after: int, bins_current: int = 1) -> np.ndarray:
    pass


def flatten_spike_history(X: np.ndarray) -> np.ndarray:
    pass


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
    pass


def zscore_normalize(data: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass
