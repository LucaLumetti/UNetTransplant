import torch
from torch import nn

from losses.DiceLoss import DiceLoss


class DiceBCELoss(nn.Module):
    def __init__(
        self,
    ):
        super(DiceBCELoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceLoss(activation=torch.sigmoid, batch_dice=False)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        mask = target != -1

        dice_loss = self.dice(net_output, target, mask=mask)
        target = target.float()

        bce_loss = (self.bce(net_output, target) * mask).sum() / torch.clip(
            mask.sum(), min=1e-8
        )
        print(f"DICE: {dice_loss.item()}, BCE: {bce_loss.item()}")

        result = dice_loss + bce_loss
        return result
