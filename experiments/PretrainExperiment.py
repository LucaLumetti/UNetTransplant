import gc
from datetime import datetime
from pathlib import Path

import torch
import torchio as tio
import wandb
from tqdm import tqdm

import configs
from experiments import BaseExperiment

# TODO: support DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_grads(module: torch.nn.Module, allow_none_grads=False):
    grads = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            if param.grad is None and allow_none_grads:
                grads.append(torch.zeros_like(param).flatten())
            else:
                assert param.grad is not None, f"Gradient of {name} is None"
                grads.append(param.grad.flatten())
    return torch.concat(grads)


def zero_grads(module: torch.nn.Module):
    for name, param in module.named_parameters():
        if param.requires_grad:
            param.grad = None


def perturb_model(model, sigma):
    for nn, m in model.named_parameters():
        if m.dim() > 1:
            perturbation = []
            for j in range(m.shape[0]):
                pert = torch.randn(m.shape[1:]).cuda()
                pert *= (torch.norm(m[j]) / (torch.norm(pert) + 1e-10)) * sigma
                perturbation.append(pert)
            perturbation = torch.stack(perturbation, dim=0)
            m += perturbation


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
                self.evaluate()
                if epoch > 0:
                    self.save(epoch)

    def compute_fisher(self):
        self.backbone.eval()
        self.heads.eval()
        data = self.val_dataset.dataset  # TODO: dataloader
        fish = torch.zeros_like(get_module_grads(self.backbone, allow_none_grads=True))
        total_patch_forwarded = 0
        for i, subject in enumerate(tqdm(data, desc="Val")):  # type: ignore
            grid_sampler = tio.GridSampler(
                subject, patch_size=(96, 96, 96), patch_overlap=0
            )
            for patch in grid_sampler:
                zero_grads(self.backbone)
                total_patch_forwarded += 1
                image, label, dataset_indices = self.get_image_label_idx(patch)
                image, label, dataset_indices = (
                    image.unsqueeze(0),
                    label.unsqueeze(0),
                    dataset_indices.unsqueeze(0),
                )
                backbone_pred = self.backbone(image)
                _, loss = self.heads(backbone_pred, dataset_indices, label)
                loss.backward()
                exp_cond_prob = 1  # Chissa se è giusta
                fish += exp_cond_prob * get_module_grads(self.backbone) ** 2
        fish /= total_patch_forwarded
        fisher = fish.sum()
        return fisher.item()

    def evaluate(
        self,
    ):
        fisher_flatness = self.compute_fisher()
        dict_to_log = {
            "Val/Fisher_Flatness": fisher_flatness,
        }
        wandb.log(dict_to_log)
        print(dict_to_log)
        self.backbone.train()
        self.heads.train()
