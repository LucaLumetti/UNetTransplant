import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim.optimizer import Optimizer

from config import OptimizerConfig


class OptimizerFactory:
    @staticmethod
    def create(model: nn.Module) -> Optimizer:
        name = OptimizerConfig.NAME
        if name in torch.optim.__dict__:
            optimizer_class = getattr(torch.optim, name)
        else:
            raise Exception(f"Optimizer {name} not implemented")

        try:
            optim = optimizer_class(
                model.parameters(),
                lr=OptimizerConfig.LEARNING_RATE,
                momentum=OptimizerConfig.MOMENTUM,
                weight_decay=OptimizerConfig.WEIGHT_DECAY,
            )
        except TypeError as e:
            raise TypeError(f"Could not instantiate {optimizer_class}: {e}")

        return optim
