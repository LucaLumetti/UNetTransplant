from typing import List

import torch

from taskvectors.TaskVector import TaskVector


class MergedModels(torch.nn.Module):
    def __init__(
        self, pretrained_checkpoints: dict, task_checkpoint: dict, task_name: str
    ):
        self.pretrained_checkpoints = pretrained_checkpoints
        self.task_vector = TaskVector(pretrained_checkpoints, task_checkpoint)
        self.task_name = task_name

    def forward(self, task_name: str, x: torch.Tensor):
        task_vector = self.task_vectors[task_name]
        return task_vector(x)
