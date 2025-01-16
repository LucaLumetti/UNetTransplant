from functools import partial

import torch

from training.losses.CrossEntropyLoss import RobustCrossEntropyLoss
from training.losses.DiceCELoss import DC_and_BCE_loss
from training.losses.soft_dice_loss import MemoryEfficientSoftDiceLoss


class LossFactory:
    def __init__(self, loss_name: str):
        self.loss_name = loss_name

    def create(self):
        if self.loss_name == "ce":
            return RobustCrossEntropyLoss()
        elif self.loss_name == "dice":
            return MemoryEfficientSoftDiceLoss()
        elif self.loss_name == "dice_bce":
            return partial(DC_and_BCE_loss, bce_kwargs={}, soft_dice_kwargs={})
        else:
            raise ValueError(f"Loss {self.loss_name} not supported.")
