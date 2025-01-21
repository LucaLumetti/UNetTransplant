import os
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import wandb

from metrics.Metrics import Metrics


class BaseExperiment:
    def __init__(self):
        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler._LRScheduler
        self.metrics: Metrics = Metrics()

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save(self, epoch):
        now = datetime.now()
        if not os.path.exists(f"checkpoints/{wandb.run.name}"):
            os.makedirs(f"checkpoints/{wandb.run.name}")
        checkpoint_filename = f"checkpoints/{wandb.run.name}/epoch{epoch:04}_{now}.pth"
        torch.save(
            {
                "epoch": epoch,
                "backbone_state_dict": self.backbone.state_dict(),
                "heads_state_dict": self.heads.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # "scheduler_state_dict": self.scheduler.state_dict(),
            },
            checkpoint_filename,
        )
        print(f"Checkpoint saved at {checkpoint_filename}")

    def load(
        self,
        checkpoint_path: Path,
        load_optimizer: bool = False,
        load_scheduler=False,
        load_heads=False,
    ) -> None:
        checkpoint = torch.load(checkpoint_path)
        # self.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        self.legacy_model_load(checkpoint["model_state_dict"])

        if load_heads:
            self.heads.load_state_dict(checkpoint["heads_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Checkpoint loaded from {checkpoint_path}")

    def legacy_model_load(self, backbone_state_dict: dict):
        # remove _orig_mod from keys
        str_to_remove = "_orig_mod."
        backbone_state_dict = {
            k.replace(str_to_remove, ""): v for k, v in backbone_state_dict.items()
        }

        # remove the last 1x1 conv layer
        keys = list(backbone_state_dict.keys())
        for key in keys:
            if "final_conv" in key:
                del backbone_state_dict[key]

        self.backbone.load_state_dict(backbone_state_dict)

    def debug_batch(self, x, y, pred):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        pred = np.concatenate(
            [
                np.zeros(
                    (pred.shape[0], 1, pred.shape[2], pred.shape[3], pred.shape[4])
                ),
                pred,
            ],
            axis=1,
        )
        pred = pred.argmax(axis=1)
        pred = pred.reshape(y.shape)
        print(f"x: {x.shape}, y: {y.shape}, mask: {pred.shape}")
        np.save("debug/image.npy", x)
        np.save("debug/label.npy", y)
        np.save("debug/pred.npy", pred)

    def compute_metrics(
        self,
        y_true: Union[torch.Tensor, npt.NDArray],
        y_pred: Union[torch.Tensor, npt.NDArray],
    ) -> dict:
        if "metrics" not in self.__dict__:
            self.metrics = Metrics()
        # if torch tensor, convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().detach().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().detach().numpy()

        background = np.zeros(
            (y_pred.shape[0], 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4])
        )
        y_pred = np.concatenate([background, y_pred], axis=1)

        y_pred = y_pred.argmax(axis=1, keepdims=True)

        metrics = self.metrics.compute(y_true, y_pred)
        return metrics
