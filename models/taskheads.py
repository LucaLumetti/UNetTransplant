from typing import Callable, List

import torch
import torch.nn as nn

import configs
from losses import LossFactory
from preprocessing.dataset_class_mapping import DATASET_ORIGINAL_LABELS


class Task:
    def __init__(self, dataset_name, dataset_idx, num_output_channels):
        self.dataset_idx = dataset_idx
        self.num_output_channels = num_output_channels
        self.dataset_name = dataset_name
        self.mask = torch.ones((1, num_output_channels, 1, 1, 1), dtype=torch.bool)

        if (
            configs.DataConfig.INCLUDE_ONLY_CLASSES is not None
            and configs.DataConfig.INCLUDE_ONLY_CLASSES != []
        ):
            for class_idx, class_name in DATASET_ORIGINAL_LABELS[dataset_name].items():
                if class_name not in configs.DataConfig.INCLUDE_ONLY_CLASSES:
                    self.mask[:, int(class_idx) - 1, :, :, :] = 0


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

    # x and y have shape (B, C, H, W, D)
    # task_idx have shape (B)
    def forward(self, x: torch.Tensor, y: torch.Tensor, task_idx: torch.Tensor):
        # group x based on the task_idx
        grouped_x = {
            task.dataset_idx: x[task_idx == task.dataset_idx] for task in self.tasks
        }
        # group y based on the task_idx
        grouped_y = {
            task.dataset_idx: y[task_idx == task.dataset_idx] for task in self.tasks
        }
        predictions = []
        losses = []
        for i, task in enumerate(self.tasks):
            i = task.dataset_idx
            prediction = self.task_heads[str(i)](grouped_x[i])
            mask = task.mask.to(x.device)
            loss = self.loss_fn(prediction, grouped_y[i], mask=mask)
            losses.append(loss)
            predictions.append(prediction * mask)

        return torch.cat(predictions), torch.stack(losses).mean()


if __name__ == "__main__":
    # example usage
    task1 = Task("dataset1", 0, 1)
    task2 = Task("dataset2", 1, 3)
    tasks = [task1, task2]
    task_heads = TaskHeads(64, tasks, nn.CrossEntropyLoss())
    x = torch.rand(8, 64, 16, 16, 16)
    y = torch.randint(0, 3, (8, 1, 16, 16, 16))
    task_idx = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1])
    prediction, loss = task_heads(x, y, task_idx)
    print(prediction.shape, loss)
