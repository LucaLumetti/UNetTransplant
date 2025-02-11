import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torchio as tio
import wandb
from tqdm import tqdm

import configs
from datasets.PatchDataset import PatchDataset
from metrics.Metrics import Metrics
from models.modelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from preprocessing.preprocessor import Preprocessor
from task.Task import Task


class BaseExperiment:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.starting_epoch = 0

        self.tasks = self.setup_tasks()

        self.train_dataset, self.val_dataset = self.setup_datasets(tasks=self.tasks)
        self.train_loader = self.train_dataset.get_dataloader()

        self.backbone = self.setup_backbone()
        self.heads = self.setup_heads()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        self.params_to_clip = list(self.backbone.parameters()) + list(
            self.heads.parameters()
        )
        self.params_to_clip = [p for p in self.params_to_clip if p.requires_grad]

        # TODO: assuming 1 task for now
        self.metrics: Metrics = Metrics(task=self.tasks[0])

    def setup_tasks(
        self,
    ) -> List[Task]:
        if len(configs.DataConfig.DATASET_NAMES) != 1:
            raise ValueError(
                f"Only one dataset is supported for now. Please change {configs.DataConfig.DATASET_NAMES=}"
            )
        dataset_name = configs.DataConfig.DATASET_NAMES[0]
        task = Task(
            dataset_name=dataset_name,
            label_ranges=configs.DataConfig.DATASET_CLASSES,
        )
        return [task]

    def setup_datasets(self, tasks: List[Task]):
        # TODO: DatasetFactory
        train_dataset = PatchDataset(split="train", task=tasks[0])
        val_dataset = PatchDataset(split="val", task=tasks[0])
        return train_dataset, val_dataset

    def setup_backbone(self):
        backbone = ModelFactory.create_backbone(configs.BackboneConfig)
        return backbone.cuda()

    def setup_heads(self):
        assert self.tasks is not None, "Tasks must be setup before heads!"
        model = ModelFactory.create_heads(configs.HeadsConfig, tasks=self.tasks)
        return model

    def setup_optimizer(
        self,
    ) -> torch.optim.Optimizer:
        parameters_to_optimize = [
            {
                "params": self.backbone.parameters(),
                "lr": configs.OptimizerConfig.BACKBONE_LR,
                "weight_decay": configs.OptimizerConfig.WEIGHT_DECAY,
            },
            {
                "params": self.heads.parameters(),
                "lr": configs.OptimizerConfig.HEAD_LR,
            },
        ]
        optim = OptimizerFactory.create(parameters_to_optimize)

        return optim

    def setup_scheduler(
        self,
    ):
        if self.optimizer is None:
            raise ValueError(
                "Cannot setup scheduler without optimizer. Call setup_optimizer() first."
            )

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=configs.TrainConfig.EPOCHS
        )

    def train(self):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        return self.functional_predict(self.backbone, self.heads, *args, **kwargs)

    def get_checkpoint_path(
        self,
    ) -> Path:
        path = Path(f"checkpoints/{self.experiment_name}")
        if not path.exists():
            path.mkdir(parents=True)
        return path

    def save(self, epoch):
        checkpoint_path = self.get_checkpoint_path()
        checkpoint_filename = checkpoint_path / f"epoch{epoch:04}.pth"
        torch.save(
            {
                "epoch": epoch,
                "backbone_state_dict": self.backbone.state_dict(),
                "heads_state_dict": self.heads.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            checkpoint_filename,
        )
        print(f"Checkpoint saved at {checkpoint_filename}")

    def load(
        self,
        checkpoint_path: Path,
        load_backbone: bool = False,
        load_optimizer: bool = False,
        load_scheduler=False,
        load_heads=False,
        load_epoch=False,
    ) -> None:
        checkpoint = torch.load(checkpoint_path)
        if not any([load_backbone, load_optimizer, load_scheduler, load_heads]):
            raise ValueError(
                "At least one of load_backbone, load_optimizer, load_scheduler, load_heads must be True, otherwise you would not load anything and calling load is useless."
            )

        if load_backbone:
            self.legacy_model_load(checkpoint["backbone_state_dict"])
        if load_heads:
            self.heads.load_state_dict(checkpoint["heads_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint and load_epoch:
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
        del x, y, pred_logits, pred_label

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

    @torch.no_grad()
    def evaluate(
        self,
    ):
        self.backbone.eval()
        self.heads.eval()

        metrics = defaultdict(list)
        data = self.val_dataset.dataset  # TODO: dataloader
        total_length = len(data)

        for i, subject in enumerate(tqdm(data, desc="Val")):
            pred = self.predict(subject)

            try:
                label = subject["labels"][tio.DATA].squeeze()
                metric_values = self.metrics.compute(pred, label)
            except Exception as e:
                print(f"Error computing metrics in subject {i}: {e}")
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
        patch_overlap: int = 0,
    ):
        backbone.eval()
        heads.eval()

        # TODO: fix this squeeze() unsqueeze() hell
        patch_sampler = tio.inference.GridSampler(
            subject,
            patch_size=(96, 96, 96),
            patch_overlap=patch_overlap,
        )

        pred_aggregator = tio.inference.GridAggregator(
            patch_sampler, overlap_mode="hann"
        )

        spatial_shape = subject["images"][tio.DATA].shape[-3:]

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
        values, indices = pred.max(dim=0)
        indices += 1
        indices[values < 0] = 0  # TODO: check that this should not be 0.5
        return indices

    def log_train_status(self, loss: Union[List[float], float], epoch):
        if isinstance(loss, list):
            average_loss = sum(loss) / len(loss)
            max_loss = max(loss)
            min_loss = min(loss)
        else:
            average_loss = loss
            max_loss = loss
            min_loss = loss

        wandb.log(
            {
                "Train/Average Loss": average_loss,
                "Train/Max Loss": max_loss,
                "Train/Min Loss": min_loss,
                "Epoch": epoch,
            }
        )

    def get_image_label_idx(self, sample, device: Optional[str] = None):
        image = sample["images"][tio.DATA]
        label = sample["labels"][tio.DATA]
        dataset_indices = sample["dataset_idx"]
        image, label = image.cuda(), label.cuda()

        return image, label, dataset_indices
