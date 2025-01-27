import argparse
from pathlib import Path

import wandb

import configs
from experiments import ExperimentFactory
from models.modelFactory import ModelFactory
from taskvectors.TaskVector import TaskVector


def main():
    task_vector = TaskVector(
        checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_grid_search_configs/taskvector_tf2_Lower_Jawbone_BLR_0_WD_0.1.toml/epoch0005_2025-01-23 15:49:33.364213.pth"
    )
    backbone = ModelFactory.create_from_checkpoint(configs.BackboneConfig)
    pass


if __name__ == "__main__":
    configs.initialize_config(
        "/work/grana_maxillo/UNetMerging/grid_search_configs/taskvector_tf2_Lower_Jawbone_BLR_0.1_WD_0.1.toml"
    )
    main()
