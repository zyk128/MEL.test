"""Global configuration for EV charging load analysis and future FL experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Data and feature-engineering settings."""

    cities: List[str] = field(default_factory=lambda: ["SZH", "AMS", "JHB", "LOA", "MEL", "SPO"])
    use_remove_zero: bool = True
    time_col: str = "Unnamed: 0"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seq_len: int = 48
    pred_len: int = 24
    top_k_stations: int = 20
    vmd_K: int = 6
    vmd_alpha: int = 2000


@dataclass
class ModelConfig:
    """TCN-LSTM model settings."""

    tcn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    fc_hidden: int = 64
    input_dim: int = 1


@dataclass
class FedConfig:
    """Federated learning settings."""

    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    aggregation: str = "fedprox"
    fedprox_mu: float = 0.01
    n_clusters: int = 3
    min_clients_per_round: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    seed: int = 42
    device: str = "cuda"
    data_dir: str = str(DATA_DIR)
    output_dir: str = str(OUTPUT_DIR)
