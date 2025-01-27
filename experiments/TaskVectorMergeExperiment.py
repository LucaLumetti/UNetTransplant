import os
from datetime import datetime

import torch
import wandb

import configs
from experiments import BaseExperiment
from metrics.Metrics import Metrics
from models.TaskVectorModel import TaskVectorModel

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskVectorMergeExperiment(BaseExperiment):
    def __init__(
        self,
    ):
        super(TaskVectorMergeExperiment, self).__init__()
        self.backbone = TaskVectorModel(self.backbone)
        self.backbone = self.backbone.cuda()
        self.metrics = Metrics()

        assert (
            configs.TASK_VECTORS_TO_MERGE
        ), "Missing TASK_VECTORS_TO_MERGE in config. Needed for this experiment!"

    def train(
        self,
    ):
        raise Exception("Cannot train Merged Task Vector (yet?)")

    def evaluate(self):
        super().evaluate()
        print(f"Task vector norm: {self.compute_task_vector_norm()}")

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
