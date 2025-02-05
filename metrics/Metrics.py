import numpy as np
import numpy.typing as npt
import torch
from torchmetrics.functional import dice

import configs
from preprocessing.dataset_class_mapping import DATASET_ORIGINAL_LABELS


class Metrics:
    def __init__(self):
        num_selected_classes = (
            len(configs.DataConfig.INCLUDE_ONLY_CLASSES)
            if configs.DataConfig.INCLUDE_ONLY_CLASSES
            else 0
        )
        if num_selected_classes == 0:
            num_selected_classes = sum(
                [
                    len(DATASET_ORIGINAL_LABELS[dataset_name].keys())
                    for dataset_name in configs.DataConfig.DATASET_NAMES
                ]
            )
        self.num_classes = num_selected_classes + 1

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        metrics = {
            "dice": dice(
                y_pred,
                y_true,
                average="macro",
                ignore_index=0,
                num_classes=self.num_classes,
            ),
        }
        for key in metrics:
            metrics[key] = np.nan_to_num(metrics[key])
        return metrics
