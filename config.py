from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    NAME = "ComposedDataset"
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    DATA_PATH = Path("/work/grana_maxillo/UNetMergingData/preprocessed_data")


@dataclass
class ModelConfig:
    NAME = "UNet3D"
    COMPILE = True
    IN_CHANNELS = 1
    OUT_CHANNELS = 82


@dataclass
class OptimizerConfig:
    NAME = "SGD"
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001


@dataclass
class LossConfig:
    NAME = "DiceBCELoss"


@dataclass
class TrainConfig:
    EPOCHS = 20
