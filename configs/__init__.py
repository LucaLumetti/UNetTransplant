from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Union

import toml


# Define dataclasses with a leading underscore
@dataclass
class _DataConfig:
    NAME: str = "ComposedDataset"
    DATASET_NAMES: List[str] = field(default_factory=lambda: [])
    DATASET_CLASSES: List[str] = field(default_factory=lambda: ["all"])
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 2
    DATA_PREPROCESSED_PATH: Path = Path(
        "/work/grana_maxillo/UNetMergingData/preprocessed_data"
    )
    DATA_RAW_PATH: Path = Path("/work/grana_maxillo/UNetMergingData/raw_data")
    PATCH_SIZE: List[int] = field(default_factory=lambda: [96, 96, 96])
    PATCH_OVERLAP: float = 0.0
    OUTPUT_DIR: Path = Path("")


@dataclass
class _BackboneConfig:
    NAME: str = "ResidualUNet3D"
    COMPILE: bool = True
    IN_CHANNELS: int = 1
    PRETRAIN_CHECKPOINTS: Optional[Path] = None
    N_EPOCHS_FREEZE: int = 5
    DROPOUT_PROB: float = 0.1
    TASK_VECTOR_INIT: Optional[Path] = None


@dataclass
class _HeadsConfig:
    NAME: str = "TaskHeads"
    COMPILE: bool = True
    PRETRAIN_CHECKPOINTS: Optional[Path] = None
    IN_CHANNELS: int = 64
    LOAD_FROM_CHECKPOINTS: bool = False


@dataclass
class _OptimizerConfig:
    NAME: str = "SGD"
    LR: float = 0.1
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0001
    BACKBONE_LR: float = 0.001
    HEAD_LR: float = 0.1


@dataclass
class _LossConfig:
    NAME: str = "DiceBCELoss"


@dataclass
class _TrainConfig:
    EPOCHS: int = 200
    RESUME: Optional[Path] = None
    SAVE_EVERY: int = 10


# Global variable to store the instances
_instances = {}


def load_config(file_path: Path):
    """Load the configuration from a TOML file and create dataclass instances."""
    global _instances
    config_data = toml.load(file_path)
    instances = {}
    dataclasses_to_load = [
        _DataConfig,
        _BackboneConfig,
        _HeadsConfig,
        _OptimizerConfig,
        _LossConfig,
        _TrainConfig,
    ]

    for cls in dataclasses_to_load:
        instance = cls()  # Create a default instance
        cls_name = cls.__name__[1:]  # Remove the leading underscore
        if cls_name in config_data:
            for field in fields(cls):
                if field.name not in config_data[cls_name]:
                    continue
                value = config_data[cls_name][field.name]
                if field.type == Path:
                    value = Path(value)
                setattr(instance, field.name, value)
        instances[cls_name] = instance

    _instances = instances


def initialize_config(config_path: Union[Path, str]):
    """Initialize the configuration by loading a specific TOML file."""
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    load_config(config_path)


def get_config(config_name: str):
    """Retrieve a specific configuration instance."""
    if not _instances:
        raise RuntimeError(
            "Configuration has not been initialized. Call `initialize_config` first."
        )
    if config_name not in _instances:
        raise ValueError(f"Configuration '{config_name}' not found.")
    return _instances[config_name]


def generate_config_json():
    """Generate a JSON representation of the configuration."""
    config = {}
    for name, instance in _instances.items():
        config[name] = instance.__dict__
    return config


def __getattr__(name: str):
    if name in globals().keys():
        return globals()[name]
    """Retrieve configuration instance by name without needing to call it as a function."""
    config_name = f"{name}"
    return get_config(config_name)
