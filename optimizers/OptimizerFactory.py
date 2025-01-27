import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim.optimizer import Optimizer

import configs


class OptimizerFactory:
    @staticmethod
    def create(parameters) -> Optimizer:
        name = configs.OptimizerConfig.NAME
        if name in torch.optim.__dict__:
            optimizer_class = getattr(torch.optim, name)
        else:
            raise Exception(f"Optimizer {name} not implemented")

        try:
            optim = optimizer_class(
                parameters,
                # momentum=configs.OptimizerConfig.MOMENTUM,
                # weight_decay=configs.OptimizerConfig.WEIGHT_DECAY,
            )
        except TypeError as e:
            raise TypeError(f"Could not instantiate {optimizer_class}: {e}")

        return optim
