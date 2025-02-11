from typing import Callable, List, Optional

import torch
import torch.nn as nn

import configs
from losses import LossFactory
from preprocessing.dataset_class_mapping import DATASET_ORIGINAL_LABELS
from task.Task import Task


class TaskHeads(nn.Module):
    def __init__(self, input_channels: int, tasks: List[Task]):
        super(TaskHeads, self).__init__()
        self.tasks = tasks
        self.loss_fn = LossFactory.create()
        self.task_heads = nn.ModuleDict(
            {
                str(task.dataset_idx): nn.Conv3d(
                    input_channels, task.num_output_channels, kernel_size=1
                )
                for task in tasks
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        task_idx: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        return self.train_forward(x, task_idx, y)
        if self.training:
            return self.train_forward(x, task_idx, y)
        else:
            return self.eval_forward(x, task_idx)

    # TODO: maybe remove y dependency and compute loss in the parent. Mask would still be used. Maybe a problem for grouped_y
    def train_forward(
        self,
        x: torch.Tensor,
        task_idx: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        # group x based on the task_idx
        grouped_x = {
            task.dataset_idx: x[task_idx == task.dataset_idx] for task in self.tasks
        }
        grouped_x = {k: v for k, v in grouped_x.items() if v.shape[0] > 0}

        if y is not None:
            # group y based on the task_idx
            grouped_y = {
                task.dataset_idx: y[task_idx == task.dataset_idx] for task in self.tasks
            }
            grouped_y = {k: v for k, v in grouped_y.items() if v.shape[0] > 0}

        predictions = []
        losses = []
        loss_weights = []
        for i, task in enumerate(self.tasks):
            i = task.dataset_idx
            num_classes = task.num_output_channels
            if i not in grouped_x.keys():
                continue
            task_x = grouped_x[i]
            task_y = grouped_y[i] if y is not None else None
            task_head = self.task_heads[str(i)]

            prediction = task_head(task_x)
            if task_y is not None:
                loss = self.loss_fn(prediction, task_y)
                losses.append(loss)
                loss_weights.append(num_classes)
            predictions.append(prediction)

        # predictions = torch.cat(predictions)
        if len(losses) == 0:
            losses = torch.tensor(0)
        else:
            # losses = torch.stack(losses).mean()
            losses = torch.stack(losses)
            loss_weights = torch.tensor(loss_weights, device=losses.device).float()
            losses = (losses * loss_weights).sum() / loss_weights.sum()
        return predictions, losses

    def eval_forward(self, x: torch.Tensor, task_idx: torch.Tensor):
        raise NotImplementedError("Eval forward not implemented yet.")
