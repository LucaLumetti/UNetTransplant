from pathlib import Path

from preprocessing.dataset_class_mapping import DATASET_MAPPING_LABELS


DATASETS_INPUT_PATH = Path("/work/grana_maxillo/UNetMerging/data")
DATASETS_OUTPUT_PATH = Path("/work/grana_maxillo/UNetMerging/preprocessed_data")

DATA_PROB_DROP_EMPTY_PATCH = 0.05
DATA_NUM_PATCHES_PER_SUBJECT = 10
DATALOADER_NUM_WORKERS = 1
BATCH_SIZE = 1

TRAIN_EPOCHS = 10
TRAIN_LOSS_NAME = "dice_bce"
TRAIN_MODEL_NAME = "unet3d"
TRAIN_OPTIM_NAME = "sgd"

OPTIM_LR = 0.1
OPTIM_MOMENTUM = 0.9
OPTIM_WEIGHT_DECAY = 0.0001

MODEL_IN_CHANNELS = 1
MODEL_OUT_CHANNELS = max([max(v.values()) for v in DATASET_MAPPING_LABELS.values()])