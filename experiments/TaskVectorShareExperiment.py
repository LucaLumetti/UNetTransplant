import gc
import os
from datetime import datetime
from pathlib import Path
from typing import List, Union

import torch
import wandb
from tqdm import tqdm

import configs
from experiments.BaseExperiment import BaseExperiment
from models.modelFactory import ModelFactory
from models.TaskVectorModel import TaskVectorModel

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskVectorShareExperiment(BaseExperiment):
    def __init__(self, experiment_name: str):
        super(TaskVectorShareExperiment, self).__init__(experiment_name=experiment_name)

    def setup_backbone(self):
        # self.backbone = super(TaskVectorShareExperiment, self).setup_backbone()

        self.load(
            checkpoint_path=configs.BackboneConfig.PRETRAIN_CHECKPOINTS,
            load_backbone=True,
        )

        return self.backbone

    def setup_heads(self):
        assert self.tasks is not None, "Tasks must be setup before heads!"
        model = ModelFactory.create_heads(configs.HeadsConfig, tasks=self.tasks)
        if configs.HeadsConfig.LOAD_FROM_CHECKPOINTS:
            self.heads = model
            self.load(
                checkpoint_path=configs.BackboneConfig.PRETRAIN_CHECKPOINTS,
                load_heads=True,
            )
        return model

    @torch.no_grad()
    def compute_task_vector_norm(
        self,
    ):
        task_vector_params = [p for p in self.backbone.params if p.requires_grad]
        flatten_params = torch.cat([p.flatten() for p in list(task_vector_params)])
        norm = torch.norm(flatten_params)
        return norm

    def save(self, epoch):
        now = datetime.now()
        if not os.path.exists(
            f"{configs.DataConfig.OUTPUT_DIR}checkpoints/{wandb.run.name}"
        ):
            os.makedirs(f"{configs.DataConfig.OUTPUT_DIR}checkpoints/{wandb.run.name}")
        checkpoint_filename = f"{configs.DataConfig.OUTPUT_DIR}checkpoints/{wandb.run.name}/epoch{epoch:04}_{now}_task_vector.pth"
        torch.save(
            {
                "pretrain_state_dict": self.backbone.params0.state_dict(),
                "delta_state_dict": self.backbone.params.state_dict(),
                "heads_state_dict": self.heads.state_dict(),
                "task": self.tasks,
            },
            checkpoint_filename,
        )
        print(f"Checkpoint saved at {checkpoint_filename}")

    def train(self):
        self.backbone.train()
        self.heads.train()

        for epoch in range(configs.TrainConfig.EPOCHS):
            losses = []
            for sample in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                total=len(self.train_loader),
            ):
                image, label, dataset_indices = self.get_image_label_idx(sample)

                if configs.BackboneConfig.N_EPOCHS_FREEZE > epoch:
                    with torch.no_grad():
                        backbone_pred = self.backbone(image)
                else:
                    backbone_pred = self.backbone(image)

                _, loss = self.heads(backbone_pred, dataset_indices, label)

                self.optimizer.zero_grad()
                loss.backward()

                losses.append(loss.item())

                torch.nn.utils.clip_grad_norm_(self.params_to_clip, 1.0)
                self.optimizer.step()

                gc.collect()

            self.log_train_status(
                loss=losses,
                epoch=epoch,
            )

            # self.scheduler.step()

            if epoch % configs.TrainConfig.SAVE_EVERY == 0:
                self.evaluate()
                self.backbone.train()
                self.heads.train()
                if epoch > 0:
                    self.save(epoch)

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
        if any([load_epoch, load_optimizer, load_scheduler]):
            raise ValueError(
                "Cannot load epoch, optimizer, scheduler, in this Experiment."
            )
        assert (
            load_backbone or load_heads
        ), "Backbone must be loaded in this Experiment."
        if load_backbone:
            delta_zero = checkpoint["pretrain_state_dict"]
            delta_task = checkpoint["delta_state_dict"]
            combined = {k: delta_zero[k] + delta_task[k] for k in delta_zero.keys()}

            backbone = ModelFactory.create_backbone(configs.BackboneConfig)
            backbone = TaskVectorModel(backbone)
            backbone.params0 = torch.nn.ParameterList(combined.values())
            self.backbone = backbone
        elif load_heads:
            self.heads.load_state_dict(checkpoint["heads_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def log_train_status(self, loss: Union[List[float], float], epoch: int):
        if isinstance(loss, list):
            average_loss = sum(loss) / len(loss) if len(loss) > 0 else -1
            max_loss = max(loss) if len(loss) > 0 else -1
            min_loss = min(loss) if len(loss) > 0 else -1
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
                "Train/lr_backbone": self.optimizer.param_groups[0]["lr"],
                "Train/lr_heads": self.optimizer.param_groups[1]["lr"],
                "Train/Delta_Theta": self.compute_task_vector_norm(),
            }
        )
