"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Paths:
    """Standard project paths."""

    project_root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path

    @staticmethod
    def from_here() -> Paths:
        """Initialize paths relative to config.py location."""
        root = Path(__file__).resolve().parents[2]
        return Paths(
            project_root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            models=root / "models",
            reports=root / "reports",
        )


@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    bin_size: float = 0.05
    test_size: float = 0.2
    start_time: float = 0.0
    end_time: Optional[float] = None
    bins_before: int = 0
    bins_after: int = 0
    bins_current: int = 1


@dataclass
class WienerConfig:
    """Wiener filter decoder configuration."""

    degree: int = 3


@dataclass
class KalmanConfig:
    """Kalman filter decoder configuration."""

    noise_scale_c: float = 1.0


@dataclass
class SVRConfig:
    """Support Vector Regression configuration."""

    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"


@dataclass
class XGBoostConfig:
    """XGBoost decoder configuration."""

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1


@dataclass
class NeuralNetConfig:
    """Neural network decoder configuration."""

    units: int = 400
    dropout_rate: float = 0.25
    num_epochs: int = 10
    batch_size: int = 128
    verbose: int = 1


@dataclass
class DecodingConfig:
    """Main decoding pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    wiener: WienerConfig = field(default_factory=WienerConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    svr: SVRConfig = field(default_factory=SVRConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    neural_net: NeuralNetConfig = field(default_factory=NeuralNetConfig)

    def get_decoder_config(self, decoder_name: str):
        """Get configuration for a specific decoder."""
        name = decoder_name.lower()
        if name in ["wiener", "wiener_filter"]:
            return self.wiener
        elif name == "kalman":
            return self.kalman
        elif name == "svr":
            return self.svr
        elif name == "xgboost":
            return self.xgboost
        elif name in ["dense_nn", "lstm"]:
            return self.neural_net
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")


DEFAULT_CONFIG = DecodingConfig()
