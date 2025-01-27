import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm

import configs
from datasets import DatasetFactory
from experiments import BaseExperiment, FinetuneExperiment
from losses import LossFactory
from metrics.Metrics import Metrics
from models import ModelFactory
from models.taskheads import TaskHeads
from models.TaskVectorModel import TaskVectorModel
from models.unet3d.unet3d import UNet3D
from optimizers import OptimizerFactory
from preprocessing.preprocessor import Preprocessor

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskVectorExperiment(FinetuneExperiment):
    def __init__(
        self,
    ):
        super(TaskVectorExperiment, self).__init__()
        self.backbone = TaskVectorModel(self.backbone)
        self.backbone = self.backbone.cuda()
        self.metrics = Metrics()

        # need to specify again the parameters to optimize
        self.parameters_to_optimize = [
            {
                "params": self.backbone.parameters(),
                "lr": configs.OptimizerConfig.BACKBONE_LR,
                # "momentum": configs.OptimizerConfig.MOMENTUM,
                "weight_decay": configs.OptimizerConfig.WEIGHT_DECAY,
            },
            {
                "params": self.heads.parameters(),
                "lr": configs.OptimizerConfig.HEAD_LR,
                # "momentum": configs.OptimizerConfig.MOMENTUM,
                # "weight_decay": configs.OptimizerConfig.WEIGHT_DECAY,
            },
        ]
        self.optimizer = OptimizerFactory.create(self.parameters_to_optimize)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=configs.TrainConfig.EPOCHS
        )

    @torch.no_grad()
    def compute_task_vector_norm(
        self,
    ):
        task_vector_params = [p for p in self.backbone.params if p.requires_grad]
        flatten_params = torch.cat([p.flatten() for p in list(task_vector_params)])
        norm = torch.norm(flatten_params)
        return norm

    def evaluate(self):
        super().evaluate()
        print(f"Task vector norm: {self.compute_task_vector_norm()}")

    def save(self, epoch):
        now = datetime.now()
        if not os.path.exists(f"checkpoints/{wandb.run.name}"):
            os.makedirs(f"checkpoints/{wandb.run.name}")
        checkpoint_filename = (
            f"checkpoints/{wandb.run.name}/epoch{epoch:04}_{now}_task_vector.pth"
        )
        torch.save(
            {
                "params_state_dict": self.backbone.params.state_dict(),
                "heads_state_dict": self.heads.state_dict(),
            },
            checkpoint_filename,
        )
        print(f"Checkpoint saved at {checkpoint_filename}")
