"""CLI entry point for neural_decoding package."""

from __future__ import annotations

from neural_decoding.main import main, parse_arguments

if __name__ == "__main__":
    data_path, decoder_name, config = parse_arguments()
    main(data_path, decoder_name, config)
