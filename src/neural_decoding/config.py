"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class Paths:
    """Standard project paths.

    Attributes:
        project_root: Root directory of the project.
        data_raw: Directory for raw data.
        data_processed: Directory for processed data.
        models: Directory for saved models.
        reports: Directory for reports and figures.
    """

    project_root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path

    @staticmethod
    def from_here() -> Paths:
        """Initialize paths relative to config.py location.

        Returns:
            Paths object with absolute paths.
        """
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
    """Data preprocessing configuration.

    Attributes:
        bin_size: Time bin size in seconds.
        test_size: Proportion of data to use for testing.
        start_time: Start time for data selection.
        end_time: End time for data selection.
        bins_before: Number of bins before current to include.
        bins_after: Number of bins after current to include.
        bins_current: Number of current bins to include (usually 1).
    """

    bin_size: float = 0.05
    test_size: float = 0.2
    start_time: float = 0.0
    end_time: Optional[float] = None
    bins_before: int = 0
    bins_after: int = 0
    bins_current: int = 1


@dataclass
class WienerConfig:
    """Wiener filter decoder configuration.

    Attributes:
        degree: Degree of polynomial for cascade decoder.
    """

    degree: int = 3


@dataclass
class KalmanConfig:
    """Kalman filter decoder configuration.

    Attributes:
        noise_scale_c: Scaling factor for noise covariance.
    """

    noise_scale_c: float = 1.0


@dataclass
class NeuralNetConfig:
    """Neural network decoder configuration.

    Attributes:
        units: Number of hidden units.
        dropout_rate: Dropout rate.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        verbose: Verbosity level.
    """

    units: int = 400
    dropout_rate: float = 0.25
    num_epochs: int = 10
    batch_size: int = 128
    verbose: int = 1


@dataclass
class DecodingConfig:
    """Main decoding pipeline configuration.

    Attributes:
        data: Data configuration.
        wiener: Wiener filter configuration.
        kalman: Kalman filter configuration.
        svr: SVR configuration.
        xgboost: XGBoost configuration.
        neural_net: Neural network configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    wiener: WienerConfig = field(default_factory=WienerConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    neural_net: NeuralNetConfig = field(default_factory=NeuralNetConfig)

    def get_decoder_config(
        self, decoder_name: str
    ) -> Union[WienerConfig, KalmanConfig, NeuralNetConfig]:
        """Get configuration for a specific decoder.

        Args:
            decoder_name: Name of the decoder.

        Returns:
            Configuration object for the requested decoder.

        Raises:
            ValueError: If decoder name is unknown.
        """
        name = decoder_name.lower()
        if name in ["wiener", "wiener_filter"]:
            return self.wiener
        elif name == "kalman":
            return self.kalman
        elif name in ["dense_nn", "lstm"]:
            return self.neural_net
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")


DEFAULT_CONFIG = DecodingConfig()
