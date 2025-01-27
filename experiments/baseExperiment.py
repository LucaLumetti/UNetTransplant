import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import wandb
from tqdm import tqdm

import configs
from metrics.Metrics import Metrics
from preprocessing.preprocessor import Preprocessor


class BaseExperiment:
    def __init__(self):
        self.backbone: torch.nn.Module
        self.heads: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any
        self.metrics: Metrics = Metrics()

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
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
        current_state_dict = self.backbone.state_dict()

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

        # self.backbone.load_state_dict(backbone_state_dict)

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

    def evaluate(
        self,
    ):
        self.backbone.eval()
        self.heads.eval()

        with torch.no_grad():
            losses = []
            metrics = defaultdict(list)

            for i, sample in tqdm(
                enumerate(self.val_loader),
                desc=f"Val",
                total=len(self.val_loader),
            ):
                image = sample[0]
                label = sample[1]
                dataset_indices = sample[2]

                pred = self.predict(image, dataset_idx=dataset_indices)

                pred = pred.to("cpu")
                label = label.to("cpu")
                metric_values = self.metrics.compute(pred, label.squeeze())
                for key, value in metric_values.items():
                    metrics[key].append(value)

            dict_to_log = {
                # "Val/Loss": sum(losses) / len(losses),
            }
            for key, value in metrics.items():
                dict_to_log[f"Val/{key}"] = sum(value) / len(value)
            wandb.log(dict_to_log)

    def predict(
        self,
        image_array: Union[torch.Tensor, npt.NDArray],
        dataset_idx: Optional[int] = None,
    ):
        self.backbone.eval()
        self.heads.eval()

        if isinstance(image_array, torch.Tensor):
            if image_array.device != "cpu":
                image_array = image_array.detach().cpu()

            image_array = image_array.numpy()

        image_array = image_array.squeeze()

        image_array, _ = Preprocessor.pad_to_patchable_shape(image_array, None)

        if image_array.ndim == 3:
            image_array = image_array[np.newaxis, ...]

        patches = Preprocessor.extract_patches(image_array, None, patch_overlap=0.2)

        num_classes_to_predict = len(configs.DataConfig.DATASET_NAMES)
        spatial_shape = image_array.shape[-3:]
        num_pred_classes = sum(t.num_output_channels for t in self.heads.tasks)
        pred = torch.zeros((num_pred_classes, *spatial_shape), device="cuda")

        for image_patch, _, coords in patches:
            image_patch = torch.from_numpy(image_patch).unsqueeze(0).to("cuda")
            backbone_pred = self.backbone(image_patch)
            heads_pred, loss = self.heads(backbone_pred, dataset_idx)

            heads_pred = torch.concatenate(heads_pred).squeeze(0)

            pred[
                :,
                coords[0] : heads_pred.shape[-3] + coords[0],
                coords[1] : heads_pred.shape[-2] + coords[1],
                coords[2] : heads_pred.shape[-1] + coords[2],
            ] = heads_pred.detach()

            del image_patch, backbone_pred, heads_pred

        background_channel = torch.zeros((1, *spatial_shape), device="cuda")

        pred = torch.concatenate([background_channel, pred], axis=0)

        del background_channel, patches

        pred = pred.argmax(axis=0)

        return pred
