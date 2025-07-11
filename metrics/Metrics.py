import torch
from torchmetrics.functional import dice

from task.Task import Task


class Metrics:
    def __init__(self, task: Task):
        self.num_classes = task.num_output_channels + 1

    def compute(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        average: str = "macro",
        keep_nan: bool = False,
    ) -> dict[str, torch.Tensor]:
        metrics = {
            "dice": dice(
                y_pred,
                y_true,
                average=average,
                ignore_index=0,
                num_classes=self.num_classes,
                multiclass=True,
            ),
        }
        if keep_nan:
            return metrics
        for key in metrics:
            metrics[key] = metrics[key][torch.isfinite(metrics[key])]
        return metrics
