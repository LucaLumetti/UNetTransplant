import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torchio as tio
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
        self.starting_epoch = 0

    def train(self):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        return self.functional_predict(self.backbone, self.heads, *args, **kwargs)

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
        self.legacy_model_load(checkpoint["backbone_state_dict"])

        if load_heads:
            self.heads.load_state_dict(checkpoint["heads_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint:
            self.starting_epoch = checkpoint["epoch"]
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

    def debug_batch(self, x, y, pred_logits, folder_name="batch"):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        pred_logits = pred_logits.cpu().detach().numpy()
        pred_logits = np.concatenate(
            [
                np.zeros(
                    (
                        pred_logits.shape[0],
                        1,
                        pred_logits.shape[2],
                        pred_logits.shape[3],
                        pred_logits.shape[4],
                    )
                ),
                pred_logits,
            ],
            axis=1,
        )
        pred_label = pred_logits.argmax(axis=1)
        pred_label = pred_label.reshape(-1, 96, 96, 96).astype(np.uint8)
        print(f"x: {x.shape}, y: {y.shape}, mask: {pred_label.shape}")
        os.makedirs(f"debug/{folder_name}", exist_ok=True)
        np.save(f"debug/{folder_name}/image.npy", x)
        np.save(f"debug/{folder_name}/label.npy", y)
        np.save(f"debug/{folder_name}/pred_logits.npy", pred_logits)
        np.save(f"debug/{folder_name}/pred_label.npy", pred_label)

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

            for i, subject in tqdm(
                enumerate(self.val_dataset.dataset),
                desc=f"Val",
                total=len(self.val_dataset.dataset),
            ):
                pred = self.predict(subject)

                # pred = pred.to("cpu")
                # label = label.to("cpu")
                # pred = pred[
                #     : original_shape[-3], : original_shape[-2], : original_shape[-1]
                # ]
                try:
                    label = subject["labels"][tio.DATA].squeeze()
                    np.save(f"debug/label_{i}.npy", label.detach().cpu().numpy())
                    np.save(f"debug/pred_{i}.npy", pred.detach().cpu().numpy())
                    np.save(
                        f"debug/image_{i}.npy",
                        subject["images"][tio.DATA].detach().cpu().numpy(),
                    )
                    metric_values = self.metrics.compute(pred, label)
                except Exception as e:
                    print("Failed to compute metrics: ", e)
                    print("pred shape: ", pred.shape)
                    print("label shape: ", label.shape)
                    print("pred unique: ", np.unique(pred))
                    print("label unique: ", np.unique(label))
                    continue
                for key, value in metric_values.items():
                    metrics[key].append(value)

            dict_to_log = {
                # "Val/Loss": sum(losses) / len(losses),
            }
            for key, value in metrics.items():
                dict_to_log[f"Val/{key}"] = sum(value) / len(value)
            wandb.log(dict_to_log)

    @staticmethod
    def functional_predict(
        backbone: torch.nn.Module,
        heads: torch.nn.Module,
        subject: tio.Subject,
        dataset_idx: Optional[int] = None,
    ):
        backbone.eval()
        heads.eval()

        # TODO: fix this squeeze() unsqueeze() hell
        patch_sampler = tio.inference.GridSampler(
            subject,
            patch_size=(96, 96, 96),
            patch_overlap=20,
        )

        pred_aggregator = tio.inference.GridAggregator(patch_sampler)

        spatial_shape = subject["images"][tio.DATA].shape[-3:]
        num_pred_classes = sum(t.num_output_channels for t in heads.tasks)
        pred = torch.zeros((num_pred_classes, *spatial_shape), device="cuda")

        for patch in patch_sampler:
            # unsqueeze to add batch dimension
            image_patch = patch["images"][tio.DATA].unsqueeze(0).to("cuda")
            dataset_idx = patch["dataset_idx"].unsqueeze(0)
            location = patch[tio.LOCATION].unsqueeze(0)

            backbone_pred = backbone(image_patch)
            heads_pred, loss = heads(backbone_pred, dataset_idx)

            heads_pred = torch.concatenate(heads_pred)

            pred_aggregator.add_batch(heads_pred, location)

        pred = pred_aggregator.get_output_tensor()
        background_channel = torch.zeros((1, *spatial_shape), device=pred.device)

        pred = torch.concatenate([background_channel, pred], axis=0)

        pred = pred.argmax(axis=0)

        return pred
