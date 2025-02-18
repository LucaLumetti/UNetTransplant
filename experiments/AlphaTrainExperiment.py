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
from models.TaskVectorAlpha import TaskVectorAlpha
from taskvectors.TaskVector import TaskVector

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlphaTrainExperiment(BaseExperiment):
    def __init__(self, experiment_name: str):
        task_vector_paths = [
            "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ek90lzx4_taskvector_tf_mandible/epoch0020_2025-02-10 14:24:21.502936_task_vector.pth",
            "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ff3xd0ge_taskvector_tf_pharynx/epoch0020_2025-02-10 11:57:53.601929_task_vector.pth",
            "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_37pgtnb8_taskvector_tf_lriac/epoch0020_2025-02-10 14:07:23.136610_task_vector.pth",
        ]
        self.task_vectors = [TaskVector(checkpoints=path) for path in task_vector_paths]
        self.combined_task_vector = (
            self.task_vectors[0] + self.task_vectors[1] + self.task_vectors[2]
        )

        super(AlphaTrainExperiment, self).__init__(experiment_name=experiment_name)

        self.backbone = TaskVectorAlpha(
            task_vectors=self.task_vectors,
        )

        self.heads = self.combined_task_vector.get_heads_from_params(
            tasks=[self.combined_task_vector.task]
        )

    def setup_tasks(self):
        combined_task = self.combined_task_vector.task
        return [combined_task]

    def setup_backbone(self):
        self.backbone = super(AlphaTrainExperiment, self).setup_backbone()
        return self.backbone

    def setup_heads(self):
        self.heads = ModelFactory.create_heads(configs.HeadsConfig, tasks=self.tasks)
        return self.heads

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
                image, label, dataset_indices = self.get_image_label_idx(
                    sample, device="cuda"
                )

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
        load_backbone: bool = True,
        load_optimizer: bool = False,
        load_scheduler=False,
        load_heads=False,
        load_epoch=False,
    ) -> None:
        assert load_backbone is True, "Backbone must be loaded"
        assert (
            sum(
                [
                    load_optimizer,
                    load_scheduler,
                    load_heads,
                    load_epoch,
                ]
            )
            == 0
        ), "Only backbone can be loaded"
        super(AlphaTrainExperiment, self).load(
            checkpoint_path=checkpoint_path,
            load_backbone=True,
        )

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
