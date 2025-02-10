import gc
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

import configs
from experiments import BaseExperiment

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainExperiment(BaseExperiment):
    def __init__(self, experiment_name: str):
        super(PretrainExperiment, self).__init__(experiment_name=experiment_name)
        if configs.TrainConfig.RESUME:
            self.load(
                Path(configs.TrainConfig.RESUME),
                load_optimizer=True,
                load_scheduler=True,
                load_heads=True,
            )

    def save(self, epoch):
        now = datetime.now()
        saving_path = self.get_checkpoint_path()
        checkpoint_filename = saving_path / f"checkpoint_{epoch}_{now}.pth"
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

        for epoch in range(self.starting_epoch, configs.TrainConfig.EPOCHS):
            losses = []
            for i, sample in enumerate(
                tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch}",
                    total=len(self.train_loader),
                )
            ):
                image, label, dataset_indices = self.get_image_label_idx(sample)

                backbone_pred = self.backbone(image)
                heads_pred, loss = self.heads(backbone_pred, dataset_indices, label)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.params_to_clip, 1.0)
                self.optimizer.step()

                losses.append(loss.item())
                gc.collect()

            self.log_train_status(losses, epoch)

            # torch.cuda.empty_cache()
            self.scheduler.step()
            if epoch % configs.TrainConfig.SAVE_EVERY == 0:
                self.save(epoch)
