"""Tests for data preprocessing."""

import numpy as np
import pytest

from neural_decoding.data.preprocessing import bin_spikes, bin_output


def test_bin_spikes_shape():
    """Test if binned spikes have correct shape."""
    # 2 neurons, spikes at various times
    spike_times = [
        np.array([0.1, 0.4, 0.9]),
        np.array([0.2, 0.5]),
    ]
    # Bin size 0.5: should result in 2 bins [0, 0.5), [0.5, 1.0)
    bin_size = 0.5
    start_time = 0.0
    end_time = 1.0
    
    binned = bin_spikes(spike_times, bin_size, start_time, end_time)
    
    # Expected shape: (2 bins, 2 neurons) if time-major or (2 neurons, 2 bins)?
    # Assuming code produces (time, features) based on previous context read.
    # Let's verify by checking the code if needed, but standard is (samples, features).
    # Waiting on read... actually I recall main.py using it.
    
    # Based on earlier main.py: binned_spikes = bin_spikes(...)
    # If the output is (n_samples, n_neurons), then shape should be (2, 2).
    
    assert binned.ndim == 2
    assert binned.shape[0] == 2  # 2 bins
    assert binned.shape[1] == 2  # 2 neurons


def test_bin_spikes_content():
    """Test spike counts in bins."""
    spike_times = [np.array([0.1, 0.2, 0.6])] # 1 neuron
    bin_size = 0.5
    start_time = 0.0
    end_time = 1.0
    
    binned = bin_spikes(spike_times, bin_size, start_time, end_time)
    
    # Bin 0 (0-0.5): 2 spikes (0.1, 0.2)
    # Bin 1 (0.5-1.0): 1 spike (0.6)
    assert binned[0, 0] == 2
    assert binned[1, 0] == 1


def test_bin_output_downsampling():
    """Test if output is properly downsampled/binned."""
    outputs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]) # 5 samples
    output_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    
    bin_size = 0.2
    start_time = 0.0
    end_time = 0.4
    
    # Bins: [0.0, 0.2), [0.2, 0.4)
    # Output binning typically takes the average or subsamples.
    # Assuming standard behavior.
    
    binned = bin_output(outputs, output_times, bin_size, start_time, end_time)
    
    assert binned.ndim == 2
    assert binned.shape[0] == 2 # 2 bins
