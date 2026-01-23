"""Tests for config module."""

from __future__ import annotations

from pathlib import Path

from neural_decoding.config import Paths, DecodingConfig, DataConfig


def test_paths_from_here() -> None:
    """Test Paths initialization."""
    paths = Paths.from_here()
    assert isinstance(paths.project_root, Path)
    assert isinstance(paths.data_raw, Path)
    assert isinstance(paths.data_processed, Path)
    assert isinstance(paths.models, Path)
    assert isinstance(paths.reports, Path)


def test_decoding_config_defaults() -> None:
    """Test DecodingConfig with default values."""
    config = DecodingConfig()
    assert config.data.bin_size == 0.05
    assert config.data.test_size == 0.2
    assert config.wiener.degree == 3
    assert config.kalman.noise_scale_c == 1.0


def test_get_decoder_config() -> None:
    """Test get_decoder_config method."""
    config = DecodingConfig()
    
    wiener_cfg = config.get_decoder_config("wiener_filter")
    assert wiener_cfg.degree == 3
    
    kalman_cfg = config.get_decoder_config("kalman")
    assert kalman_cfg.noise_scale_c == 1.0
    
    nn_cfg = config.get_decoder_config("dense_nn")
    assert nn_cfg.units == 400


def test_get_decoder_config_invalid() -> None:
    """Test get_decoder_config with invalid decoder name."""
    config = DecodingConfig()
    try:
        config.get_decoder_config("invalid_decoder")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown decoder" in str(e)
