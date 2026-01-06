import argparse
from pathlib import Path
from typing import Optional, List


def run_data_loading(data_path: Path):
    pass


def run_preprocessing(neural_data, outputs, config):
    pass


def run_training(X_train, y_train, decoder_name: str, config):
    pass


def run_evaluation(decoder, X_test, y_test):
    pass


def run_visualization(y_test, y_pred, output_dir: Path, decoder_name: str):
    pass


def main(data_path: Optional[Path] = None, decoder_name: str = "wiener_filter", config=None):
    pass


def parse_arguments():
    pass


if __name__ == "__main__":
    pass
