from functools import partial
import torch

from config import OPTIM_LR, OPTIM_MOMENTUM, OPTIM_WEIGHT_DECAY


class OptimizerFactory:
    def __init__(self, optimizer_name: str):
        self.optimizer_name = optimizer_name.lower()

    def create(self):
        if self.optimizer_name == "adam":
            return torch.optim.Adam
        elif self.optimizer_name == "sgd":
            return partial(
                torch.optim.SGD,
                lr=OPTIM_LR,
                momentum=OPTIM_MOMENTUM,
                weight_decay=OPTIM_WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer_name}")