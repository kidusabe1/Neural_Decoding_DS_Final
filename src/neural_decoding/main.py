"""Neural Decoding Pipeline - main entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np

# Data and preprocessing
from neural_decoding.data.loader import load_dataset
from neural_decoding.data.preprocessing import (
    bin_spikes,
    bin_output,
    prepare_train_test_split,
)

# Model imports
from neural_decoding.models.base import BaseDecoder
from neural_decoding.models.wiener import (
    WienerFilterDecoder,
    WienerCascadeDecoder,
)
from neural_decoding.models.kalman import KalmanFilterDecoder

# Optional neural-net decoders (TensorFlow)
try:
    from neural_decoding.models.neural_nets import DenseNNDecoder, LSTMDecoder
except (ImportError, ModuleNotFoundError):
    DenseNNDecoder = None
    LSTMDecoder = None

# Evaluation
from neural_decoding.evaluation.metrics import evaluate_decoder
from neural_decoding.visualization.plots import plot_predictions, save_figure
from neural_decoding.logger import logger

# Constants for default values
DEFAULT_BIN_SIZE = 0.05
DEFAULT_TEST_SIZE = 0.2
DEFAULT_START_TIME = 0.0
DEFAULT_BINS_BEFORE = 0
DEFAULT_BINS_AFTER = 0
DEFAULT_BINS_CURRENT = 1
DEFAULT_UNITS = 400
DEFAULT_DROPOUT = 0.25
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_VERBOSE = 1
DEFAULT_NOISE_SCALE = 1.0
DEFAULT_DEGREE = 3
DEFAULT_OUTPUT_DIR = "./reports/figures"


def run_data_loading(
    data_path: Path,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Load neural and output data from file.

    Args:
        data_path: Path to the data file.

    Returns:
        Tuple containing neural data (spikes) and tuple of (outputs, output_times).
    """
    logger.info("Loading data from %s", data_path)
    data = load_dataset(data_path)
    neural_data = data["spike_times"]
    outputs = data["outputs"]
    output_times = data["output_times"]
    logger.info("Data loaded successfully")
    return neural_data, (outputs, output_times)


def run_preprocessing(
    neural_data: np.ndarray,
    outputs: Tuple[np.ndarray, np.ndarray],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess neural and output data.

    Args:
        neural_data: Spike data.
        outputs: Tuple of (output_values, output_times).
        config: Configuration dictionary.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info("Preprocessing data...")
    bin_size = config.get("bin_size", DEFAULT_BIN_SIZE)
    start_time = config.get("start_time", DEFAULT_START_TIME)
    end_time = config.get("end_time", None)

    if end_time is None:
        # Find the latest spike time across all neurons
        end_time = max(
            (np.max(neuron) if len(neuron) > 0 else 0 for neuron in neural_data)
        )

    logger.debug("Binning spikes with bin_size=%.3f", bin_size)
    binned_spikes = bin_spikes(neural_data, bin_size, start_time, end_time)
    outputs_arr, output_times = outputs
    binned_outputs = bin_output(
        outputs_arr, output_times, bin_size, start_time, end_time
    )

    # Set default values for bins_before and bins_after
    bins_before = config.get("bins_before", DEFAULT_BINS_BEFORE)
    bins_after = config.get("bins_after", DEFAULT_BINS_AFTER)
    bins_current = config.get("bins_current", DEFAULT_BINS_CURRENT)

    logger.debug(
        "Creating train/test split with bins_before=%d, bins_after=%d, bins_current=%d",
        bins_before,
        bins_after,
        bins_current,
    )
    # Call prepare_train_test_split with all required arguments
    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_train_test_split(
        binned_spikes,
        binned_outputs,
        bins_before,
        bins_after,
        bins_current,
        test_size=config.get("test_size", DEFAULT_TEST_SIZE),
    )

    logger.info(
        "Preprocessing complete: X_train shape=%s, X_test shape=%s",
        X_train.shape,
        X_test.shape,
    )
    return X_train, X_test, y_train, y_test


def _create_decoder(decoder_name: str, config: Dict[str, Any]) -> BaseDecoder:
    """Factory function to create decoder instances.

    Args:
        decoder_name: Name of the decoder.
        config: Configuration dictionary.

    Returns:
        Instantiated decoder.

    Raises:
        ImportError: If dependencies are missing.
        ValueError: If decoder name is unknown.
    """
    name = decoder_name.lower()

    if name in ["wiener", "wiener_filter"]:
        return WienerFilterDecoder()

    if name in ["wiener_cascade", "wiener_cascade_decoder", "wc"]:
        degree = config.get("degree", DEFAULT_DEGREE)
        return WienerCascadeDecoder(degree=degree)

    if name == "kalman":
        noise_scale = config.get("noise_scale_c", DEFAULT_NOISE_SCALE)
        return KalmanFilterDecoder(noise_scale_c=noise_scale)

    # Neural network decoders
    units = config.get("units", DEFAULT_UNITS)
    dropout = config.get("dropout_rate", DEFAULT_DROPOUT)
    epochs = config.get("num_epochs", DEFAULT_EPOCHS)
    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
    verbose = config.get("verbose", DEFAULT_VERBOSE)

    if name == "dense_nn":
        if DenseNNDecoder is None:
            raise ImportError("DenseNNDecoder unavailable.")
        return DenseNNDecoder(
            units=units,
            dropout_rate=dropout,
            num_epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    if name == "lstm":
        if LSTMDecoder is None:
            raise ImportError("LSTMDecoder unavailable.")
        return LSTMDecoder(
            units=config.get("units", 128),  # LSTM might want fewer units by default?
            dropout_rate=dropout,
            num_epochs=config.get("num_epochs", 50),
            batch_size=batch_size,
            verbose=verbose,
        )

    raise ValueError(f"Unknown decoder: {decoder_name}")


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    decoder_name: str,
    config: Dict[str, Any],
) -> BaseDecoder:
    """Train a decoder on the training data.

    Args:
        X_train: Training features.
        y_train: Training targets.
        decoder_name: Name of the decoder to use.
        config: Configuration dictionary.

    Returns:
        Trained decoder instance.

    Raises:
        ImportError: If required libraries are missing.
        ValueError: If decoder name is unknown.
    """
    logger.info("Training decoder: %s", decoder_name)
    decoder = _create_decoder(decoder_name, config)

    logger.debug("Fitting decoder on training data")
    decoder.fit(X_train, y_train)
    logger.info("Training complete")
    return decoder


def run_evaluation(
    decoder: BaseDecoder, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate decoder on test data.

    Args:
        decoder: Trained decoder instance.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        Tuple of (results_dict, predictions).
    """
    logger.info("Evaluating decoder...")
    y_pred = decoder.predict(X_test)
    results = evaluate_decoder(y_test, y_pred, decoder_name=decoder.name)
    logger.info("Evaluation Results: %s", results)
    return results, y_pred


def run_visualization(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    decoder_name: str,
) -> None:
    """Generate and save visualization of predictions.

    Args:
        y_test: True values.
        y_pred: Predicted values.
        output_dir: Directory to save figures.
        decoder_name: Name of the decoder.
    """
    logger.info("Visualizing results...")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_predictions(
        y_test,
        y_pred,
        title=f"{decoder_name} Predictions vs True Output",
        figsize=(10, 4),
    )
    fig_path = output_dir / f"{decoder_name}_pred_vs_true.png"
    save_figure(fig, fig_path)
    logger.info("Saved figure: %s", fig_path)


def main(
    data_path: Optional[Path] = None,
    decoder_name: str = "wiener_filter",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Run the neural decoding pipeline.

    Args:
        data_path: Path to the data file.
        decoder_name: Name of the decoder to use.
        config: Configuration dictionary.

    Returns:
        Dictionary of evaluation results.

    Raises:
        ValueError: If data_path is not provided.
    """
    if config is None:
        config = {}

    if data_path is None:
        logger.error("Please provide a data path.")
        raise ValueError("data_path is required")

    try:
        neural_data, outputs = run_data_loading(data_path)
        X_train, X_test, y_train, y_test = run_preprocessing(
            neural_data, outputs, config
        )
        decoder = run_training(X_train, y_train, decoder_name, config)
        results, y_pred = run_evaluation(decoder, X_test, y_test)
        output_dir = Path(config.get("output_dir", "./reports/figures"))
        run_visualization(y_test, y_pred, output_dir, decoder_name)
        logger.info("Pipeline completed successfully")
        return results
    except Exception as e:
        logger.error("Pipeline failed with error: %s", str(e), exc_info=True)
        raise


def parse_arguments() -> Tuple[Path, str, Dict[str, Any]]:
    """Parse command line arguments.

    Returns:
        Tuple of (data_path, decoder_name, config_dict).
    """
    parser = argparse.ArgumentParser(
        description="Neural Decoding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neural_decoding.main --data_path data/raw/m1_data_raw.mat --decoder wiener_filter
  python -m neural_decoding.main --data_path data/raw/hc_data_raw.mat --decoder kalman --bin_size 0.1
        """,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data file (.mat format)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="wiener_filter",
        choices=[
            "wiener_filter",
            "wiener_cascade",
            "kalman",
            "dense_nn",
            "lstm",
        ],
        help="Decoder type to use (default: wiener_filter)",
    )
    parser.add_argument(
        "--bin_size",
        type=float,
        default=DEFAULT_BIN_SIZE,
        help=f"Bin size for spike binning in seconds (default: {DEFAULT_BIN_SIZE})",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Test set proportion (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bayes_opt",
        action="store_true",
        help="Use Bayesian optimization for decoder hyperparameters",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    config = {
        "bin_size": args.bin_size,
        "test_size": args.test_size,
        "output_dir": args.output_dir,
        "bayes_opt": args.bayes_opt,
    }

    return Path(args.data_path), args.decoder, config


if __name__ == "__main__":
    data_path, decoder_name, config = parse_arguments()
    main(data_path, decoder_name, config)
