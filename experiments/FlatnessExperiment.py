from pathlib import Path
import configs
from experiments.BaseExperiment import BaseExperiment
from tqdm import tqdm
import torchio as tio
import torch

from metrics.Metrics import Metrics

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

class FlatnessExperiment(BaseExperiment):
    def __init__(self, experiment_name: str):
        super(FlatnessExperiment, self).__init__(experiment_name=experiment_name)
        if configs.TrainConfig.RESUME:
            self.load(
                Path(configs.TrainConfig.RESUME),
                load_backbone=True,
                load_heads=True,
            )

    def load(
        self,
        checkpoint_path: Path,
        load_backbone: bool = True,
        load_heads=True,
        load_optimizer: bool = False,
        load_scheduler: bool = False,
        load_epoch: bool = False,
    ) -> None:
        checkpoint = torch.load(checkpoint_path)
        backbone_state_dict = checkpoint["backbone_state_dict"]
        heads_state_dict = checkpoint["heads_state_dict"]

        new_heads_state_dict = {}
        for key, value in heads_state_dict.items():
            new_key = key.replace("4", "8")
            new_heads_state_dict[new_key] = value
        
        if load_backbone:
            self.legacy_model_load(backbone_state_dict)
        if load_heads:
            self.heads.load_state_dict(new_heads_state_dict)

    def evaluate(self):
        #fisher = self.compute_fisher()
        loss_original = self.compute_loss()
        sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
        print(f"ORIGINAL LOSS: {loss_original}")
        for sigma in sigmas:
            ploss = self.compute_perturbed_loss(sigma=sigma)
            print(f"PERTUBATION LOSS SIGMA {sigma}: {ploss}")
        #print(f"FISHER:{fisher} -- PERTUBATION LOSS: {ploss} -- ORIGINAL LOSS: {loss_original}")
    
    @torch.no_grad()
    def compute_perturbed_loss(self, sigma = 0.1):
        self.backbone.eval()
        self.heads.eval()
        data = self.val_dataset.dataset  # TODO: dataloader
        total_length = len(data)
        total_patch_forwarded = 0
        losses = []
        perturb_model(self.backbone, sigma)
        for i, subject in enumerate(tqdm(data, desc="Val")):
            grid_sampler = tio.GridSampler(subject, patch_size=(96, 96, 96), patch_overlap=0)
            patch_losses = []
            for patch in grid_sampler:
                zero_grads(self.backbone)
                total_patch_forwarded += 1
                image, label, dataset_indices = self.get_image_label_idx(patch)
                image, label, dataset_indices = image.unsqueeze(0), label.unsqueeze(0), dataset_indices.unsqueeze(0)
                backbone_pred = self.backbone(image)
                heads_pred, loss = self.heads(backbone_pred, dataset_indices, label)
                patch_losses.append(loss)
            losses.append(torch.stack(patch_losses).mean())
        losses = torch.mean(torch.stack(losses).mean())
        return losses

    @torch.no_grad()
    def compute_loss(self):
        self.backbone.eval()
        self.heads.eval()
        data = self.val_dataset.dataset  # TODO: dataloader
        total_length = len(data)
        total_patch_forwarded = 0
        losses = []
        for i, subject in enumerate(tqdm(data, desc="Val")):
            grid_sampler = tio.GridSampler(subject, patch_size=(96, 96, 96), patch_overlap=0)
            patch_losses = []
            for patch in grid_sampler:
                zero_grads(self.backbone)
                total_patch_forwarded += 1
                image, label, dataset_indices = self.get_image_label_idx(patch)
                image, label, dataset_indices = image.unsqueeze(0), label.unsqueeze(0), dataset_indices.unsqueeze(0)
                backbone_pred = self.backbone(image)
                heads_pred, loss = self.heads(backbone_pred, dataset_indices, label)
                patch_losses.append(loss)
            losses.append(torch.stack(patch_losses).mean())
        losses = torch.mean(torch.stack(losses).mean())
        return losses

    def compute_fisher(self):
        self.backbone.eval()
        self.heads.eval()
        data = self.val_dataset.dataset  # TODO: dataloader
        total_length = len(data)
        fish = torch.zeros_like(get_module_grads(self.backbone, allow_none_grads=True))

        total_patch_forwarded = 0
        for i, subject in enumerate(tqdm(data, desc="Val")):
            grid_sampler = tio.GridSampler(subject, patch_size=(96, 96, 96), patch_overlap=0)
            for patch in grid_sampler:
                zero_grads(self.backbone)
                total_patch_forwarded += 1
                image, label, dataset_indices = self.get_image_label_idx(patch)
                image, label, dataset_indices = image.unsqueeze(0), label.unsqueeze(0), dataset_indices.unsqueeze(0)
                backbone_pred = self.backbone(image)
                heads_pred, loss = self.heads(backbone_pred, dataset_indices, label)
                loss.backward()
                exp_cond_prob = 1 # Chissa se è giusta
                fish += exp_cond_prob * get_module_grads(self.backbone) ** 2

        fish /= total_patch_forwarded
        print(fish.sum())
        return fish.sum()

    def train(self,):
        pass
