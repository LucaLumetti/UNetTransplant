import numpy as np
import numpy.typing as npt
import torch
from torchmetrics.functional import dice

import configs
from preprocessing.dataset_class_mapping import DATASET_ORIGINAL_LABELS
from task.task import Task


class Metrics:
    def __init__(self, task: Task):
        self.num_classes = task.num_output_channels

    def compute(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, average: str = "macro"
    ) -> dict[str, torch.Tensor]:
        metrics = {
            "dice": dice(
                y_pred,
                y_true,
                average=average,
                ignore_index=0,
                num_classes=self.num_classes,
            ),
        }
        for key in metrics:
            metrics[key] = torch.nan_to_num(metrics[key])
        return metrics
