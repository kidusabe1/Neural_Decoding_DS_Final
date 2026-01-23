import argparse
from pathlib import Path
import numpy as np
from typing import Optional, List

# Data and preprocessing
from neural_decoding.data import load_dataset, bin_spikes, bin_output, prepare_train_test_split

# Model imports
from neural_decoding.models import (
    WienerFilterDecoder,
    WienerCascadeDecoder,
    KalmanFilterDecoder,
)

# Optional neural-net decoders (TensorFlow)
try:
    from neural_decoding.models import DenseNNDecoder, LSTMDecoder
except Exception:
    DenseNNDecoder = None
    LSTMDecoder = None

# Evaluation
from neural_decoding.evaluation.metrics import evaluate_decoder

# Visualization
import os


def run_data_loading(data_path: Path):
    print(f"Loading data from {data_path}")
    data = load_dataset(data_path)
    neural_data = data["spike_times"]
    outputs = data["outputs"]
    output_times = data["output_times"]
    return neural_data, (outputs, output_times)


def run_preprocessing(neural_data, outputs, config):
    print("Preprocessing data...")
    bin_size = config.get("bin_size", 0.05)
    start_time = config.get("start_time", 0.0)
    end_time = config.get("end_time", None)
    if end_time is None:
        # Find the latest spike time across all neurons
        end_time = max((np.max(neuron) if len(neuron) > 0 else 0 for neuron in neural_data))
    binned_spikes = bin_spikes(neural_data, bin_size, start_time, end_time)
    outputs_arr, output_times = outputs
    binned_outputs = bin_output(outputs_arr, output_times, bin_size, start_time, end_time)
    # Set default values for bins_before and bins_after
    bins_before = config.get("bins_before", 0)
    bins_after = config.get("bins_after", 0)
    bins_current = config.get("bins_current", 1)
    # Call prepare_train_test_split with all required arguments
    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_train_test_split(
        binned_spikes, binned_outputs, bins_before, bins_after, bins_current, test_size=config.get("test_size", 0.2)
    )
    # For now, ignore validation set in return (or update as needed)
    return X_train, X_test, y_train, y_test


def run_training(X_train, y_train, decoder_name: str, config):
    print(f"Training decoder: {decoder_name}")
    name = decoder_name.lower()
    if name in ["wiener", "wiener_filter"]:
        decoder = WienerFilterDecoder()
    elif name in ["wiener_cascade", "wiener_cascade_decoder", "wc"]:
        decoder = WienerCascadeDecoder(degree=config.get("degree", 3))
    elif name == "kalman":
        decoder = KalmanFilterDecoder(noise_scale_c=config.get("noise_scale_c", 1.0))
    elif name == "dense_nn":
        if DenseNNDecoder is None:
            raise ImportError("DenseNNDecoder unavailable (TensorFlow not installed).")
        decoder = DenseNNDecoder(
            units=config.get("units", 400),
            dropout_rate=config.get("dropout_rate", 0.25),
            num_epochs=config.get("num_epochs", 10),
            batch_size=config.get("batch_size", 128),
            verbose=config.get("verbose", 1),
        )
    elif name == "lstm":
        if LSTMDecoder is None:
            raise ImportError("LSTMDecoder unavailable (TensorFlow not installed).")
        decoder = LSTMDecoder(
            units=config.get("units", 128),
            dropout_rate=config.get("dropout_rate", 0.25),
            num_epochs=config.get("num_epochs", 50),
            batch_size=config.get("batch_size", 128),
            verbose=config.get("verbose", 1),
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}")
    decoder.fit(X_train, y_train)
    return decoder


def run_evaluation(decoder, X_test, y_test):
    print("Evaluating decoder...")
    y_pred = decoder.predict(X_test)
    results = evaluate_decoder(y_test, y_pred, decoder_name=decoder.name)
    print("Evaluation Results:", results)
    return results, y_pred


def run_visualization(y_test, y_pred, output_dir: Path, decoder_name: str):
    print("Visualizing results...")
    from neural_decoding.visualization.plots import plot_predictions, save_figure
    os.makedirs(output_dir, exist_ok=True)
    fig = plot_predictions(y_test, y_pred, title=f"{decoder_name} Predictions vs True Output", figsize=(10, 4))
    fig_path = Path(output_dir) / f"{decoder_name}_pred_vs_true.png"
    save_figure(fig, fig_path)


def main(data_path: Optional[Path] = None, decoder_name: str = "wiener_filter", config=None):
    if config is None:
        config = {}
    if data_path is None:
        print("Please provide a data path.")
        return
    neural_data, outputs = run_data_loading(data_path)
    X_train, X_test, y_train, y_test = run_preprocessing(neural_data, outputs, config)
    decoder = run_training(X_train, y_train, decoder_name, config)
    results, y_pred = run_evaluation(decoder, X_test, y_test)
    output_dir = Path(config.get("output_dir", "./reports/figures"))
    run_visualization(y_test, y_pred, output_dir, decoder_name)
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural Decoding Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--decoder", type=str, default="wiener_filter", help="Decoder name (wiener_filter or dense_nn)")
    parser.add_argument("--bin_size", type=float, default=0.05, help="Bin size for spike binning")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--output_dir", type=str, default="./reports/figures", help="Directory to save figures")
    parser.add_argument("--bayes_opt", action="store_true", help="Use Bayesian optimization for decoder hyperparameters")
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
