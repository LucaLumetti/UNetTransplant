import numpy as np
import numpy.typing as npt
import torch
from torchmetrics.functional import dice


class Metrics:
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        metrics = {
            "dice": dice(
                y_pred, y_true, average="macro", ignore_index=0, num_classes=2
            ),
        }
        for key in metrics:
            metrics[key] = np.nan_to_num(metrics[key])
        return metrics
