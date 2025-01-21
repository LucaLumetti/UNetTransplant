from collections import defaultdict
from datetime import datetime

import torch
import wandb
from tqdm import tqdm

import configs
from datasets import DatasetFactory
from experiments import BaseExperiment
from losses import LossFactory
from models import ModelFactory
from models.taskheads import TaskHeads
from models.unet3d.unet3d import UNet3D
from optimizers import OptimizerFactory

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FinetuneExperiment(BaseExperiment):
    def __init__(
        self,
    ):
        super(BaseExperiment, self).__init__()

        self.train_dataset = DatasetFactory.create(split="train")
        self.val_dataset = DatasetFactory.create(split="val")

        # TODO: is there the correct place?
        self.train_loader = self.train_dataset.get_dataloader()
        self.val_loader = self.val_dataset.get_dataloader()

        self.backbone: UNet3D = ModelFactory.create(configs.BackboneConfig)
        self.backbone = self.backbone.cuda()

        self.heads: TaskHeads = ModelFactory.create(
            configs.HeadsConfig, tasks=self.train_dataset.get_tasks()
        )
        self.heads = self.heads.cuda()

        # wandb.watch(self.model, log="all", log_freq=100)

        self.parameters_to_optimize = [
            {"params": self.backbone.parameters(), "lr": 0.001},
            {"params": self.heads.parameters(), "lr": 0.1},
        ]
        self.optimizer = OptimizerFactory.create(self.parameters_to_optimize)
        # self.metrics = Metrics(self.config.model['n_classes'])

    def train(self):
        self.backbone.train()
        self.heads.train()

        for epoch in range(configs.TrainConfig.EPOCHS):
            for i, sample in tqdm(
                enumerate(self.train_loader),
                desc=f"Epoch {epoch}",
                total=len(self.train_loader),
            ):
                image = sample[0].to(device)
                label = sample[1].to(device)
                dataset_indices = sample[2].to(device)

                backbone_pred = self.backbone(image)
                heads_pred, loss = self.heads(backbone_pred, label, dataset_indices)

                self.optimizer.zero_grad()
                loss.backward()

                parameters = list(self.backbone.parameters()) + list(
                    self.heads.parameters()
                )
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                self.optimizer.step()

                wandb.log(
                    {
                        "Train/Loss": loss.item(),
                        "Train/lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # torch.cuda.empty_cache()

            if epoch % 10 == 0:
                # self.test(phase='Val')
                # self.model.train()
                self.save(epoch)

            # torch.cuda.empty_cache()

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
                image = sample[0].to(device)
                label = sample[1].to(device)
                dataset_indices = sample[2].to(device)

                backbone_pred = self.backbone(image)
                heads_pred, loss = self.heads(backbone_pred, label, dataset_indices)

                losses.append(loss.item())
                metrics_values = self.compute_metrics(label, heads_pred)
                for key, value in metrics_values.items():
                    metrics[key].append(value)

            dict_to_log = {
                "Val/Loss": sum(losses) / len(losses),
            }
            for key, value in metrics.items():
                dict_to_log[f"Val/{key}"] = sum(value) / len(value)
            wandb.log(dict_to_log)
