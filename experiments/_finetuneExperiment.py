import gc
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torchio as tio
import wandb
from tqdm import tqdm

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments import BaseExperiment
from losses import LossFactory
from models import ModelFactory
from models.taskheads import TaskHeads
from models.unet3d.unet3d import UNet3D
from optimizers import OptimizerFactory
from preprocessing.preprocessor import Preprocessor

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FinetuneExperiment(BaseExperiment):
    def __init__(
        self,
    ):
        # self.train_dataset = DatasetFactory.create(split="train")
        # self.val_dataset = DatasetFactory.create(split="val")
        self.train_dataset = PatchDataset(split="train", dataset_name="ToothFairy2")
        self.val_dataset = PatchDataset(split="val", dataset_name="ToothFairy2")

        # TODO: is there the correct place?
        self.train_loader = self.train_dataset.get_dataloader()
        # self.val_loader = self.val_dataset.get_dataloader(batch_size=1, num_workers=0)

        self.backbone = self.setup_backbone()
        self.heads = self.setup_heads()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

    def setup_backbone(self):
        self.backbone = ModelFactory.create(configs.BackboneConfig)
        if configs.BackboneConfig.PRETRAIN_CHECKPOINTS is not None:
            self.load(
                checkpoint_path=configs.BackboneConfig.PRETRAIN_CHECKPOINTS,
            )

        return self.backbone

    def setup_heads(self):
        model = ModelFactory.create(
            configs.HeadsConfig, tasks=self.train_dataset.get_tasks()
        )
        return model

    def setup_optimizer(
        self,
    ) -> torch.optim.Optimizer:
        parameters_to_optimize = [
            {
                "params": self.backbone.parameters(),
                "lr": configs.OptimizerConfig.BACKBONE_LR,
                # "momentum": configs.OptimizerConfig.MOMENTUM,
                "weight_decay": configs.OptimizerConfig.WEIGHT_DECAY,
            },
            {
                "params": self.heads.parameters(),
                "lr": configs.OptimizerConfig.HEAD_LR,
                # "momentum": configs.OptimizerConfig.MOMENTUM,
                # "weight_decay": configs.OptimizerConfig.WEIGHT_DECAY,
            },
        ]
        self.params_to_clip = list(self.backbone.parameters()) + list(
            self.heads.parameters()
        )
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
        self.backbone.train()
        self.heads.train()

        for epoch in range(configs.TrainConfig.EPOCHS):
            for i, sample in tqdm(
                enumerate(self.train_loader),
                desc=f"Epoch {epoch}",
                total=len(self.train_loader),
            ):
                image, label, dataset_indices = self.get_image_label_idx(
                    sample, device="cuda"
                )

                backbone_pred = self.backbone(image)
                if configs.BackboneConfig.N_EPOCHS_FREEZE > epoch:
                    backbone_pred = backbone_pred.detach()

                heads_pred, loss = self.heads(backbone_pred, dataset_indices, label)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.params_to_clip, 1.0)
                self.optimizer.step()

                wandb.log(
                    {
                        "Train/Loss": loss.item(),
                        "Train/lr_backbone": self.optimizer.param_groups[0]["lr"],
                        "Train/lr_heads": self.optimizer.param_groups[1]["lr"],
                        # "Train/wd_0": self.optimizer.param_groups[0]["weight_decay"],
                        # "Train/wd_1": self.optimizer.param_groups[1]["weight_decay"],
                        "train/Delta_Tetha": self.compute_task_vector_norm(),
                    }
                )

                del image, label, dataset_indices, backbone_pred, heads_pred, loss
                del sample
                gc.collect()

            self.scheduler.step()

            if epoch % configs.TrainConfig.SAVE_EVERY == 0:
                self.evaluate()
                self.backbone.train()
                self.heads.train()
                self.save(epoch)
