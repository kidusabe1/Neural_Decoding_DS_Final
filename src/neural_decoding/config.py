from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    pass


@dataclass
class WienerConfig:
    pass


@dataclass
class KalmanConfig:
    pass


@dataclass
class SVRConfig:
    pass


@dataclass
class XGBoostConfig:
    pass


@dataclass
class NeuralNetConfig:
    pass


@dataclass
class DecodingConfig:
    def get_decoder_config(self, decoder_name: str):
        pass


DEFAULT_CONFIG = DecodingConfig()
