from datetime import datetime
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

import configs
import datasets
from datasets import DatasetFactory
from experiments import BaseExperiment
from losses import LossFactory
from models import ModelFactory
from models.taskheads import TaskHeads
from models.unet3d.unet3d import UNet3D
from optimizers import OptimizerFactory

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainExperiment(BaseExperiment):
    def __init__(
        self,
    ):
        super(BaseExperiment, self).__init__()

        self.train_dataset = DatasetFactory.create(split="train")
        self.val_dataset = DatasetFactory.create(split="val")

        # TODO: is there the correct place?
        self.train_loader = self.train_dataset.get_dataloader()
        self.val_loader = self.val_dataset.get_dataloader(batch_size=1, num_workers=0)

        self.backbone = self.setup_backbone()
        self.heads = self.setup_heads()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        if configs.TrainConfig.RESUME:
            self.load(
                Path(configs.TrainConfig.RESUME),
                load_optimizer=True,
                load_scheduler=True,
                load_heads=True,
            )

    def setup_backbone(self):
        model = ModelFactory.create(configs.BackboneConfig)
        return model

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

    def save(self, epoch):
        now = datetime.now()
        checkpoint_filename = f"checkpoint_{epoch}_{now}.pth"
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

    def train(self):
        self.backbone.train()
        self.heads.train()
        self.params = list(self.backbone.parameters()) + list(self.heads.parameters())

        for epoch in range(self.starting_epoch, configs.TrainConfig.EPOCHS):
            losses = []
            for i, sample in tqdm(
                enumerate(self.train_loader),
                desc=f"Epoch {epoch}",
                total=len(self.train_loader),
            ):
                image = sample[0].to(device)
                label = sample[1].to(device)
                dataset_indices = sample[2].to(device)

                if label.max() == 0:
                    continue

                backbone_pred = self.backbone(image)
                heads_pred, loss = self.heads(backbone_pred, dataset_indices, label)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.optimizer.step()

                losses.append(loss.item())

            mean_loss = sum(losses) / len(losses)
            tqdm.write(f"Loss: {mean_loss}")
            wandb.log(
                {
                    "Train/Loss": mean_loss,
                    "Train/lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            # torch.cuda.empty_cache()
            self.scheduler.step()
            if epoch % 25 == 0:
                # self.evaluate()
                # self.backbone.train()
                # self.heads.train()
                self.save(epoch)
