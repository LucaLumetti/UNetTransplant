from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from losses.DiceLoss import DiceLoss


class DiceCELoss(nn.Module):
    def __init__(
        self,
    ):
        super(DiceCELoss, self).__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice = DiceLoss(activation=torch.sigmoid, batch_dice=False)

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if mask is not None:
            mask = mask.flatten()
        y_shape_background = list(net_output.shape)
        y_shape_background[1] += 1

        target_onehot = torch.zeros(
            y_shape_background, device=net_output.device, dtype=torch.bool
        )
        target = target.long()
        target[target > 42] = 0
        print(f"{target.min()=}, {target.max()=}")
        target_onehot.scatter_(1, target, 1)
        target_onehot = target_onehot[:, 1:]

        dice_loss = self.dice(net_output, target_onehot, mask=mask)
        # target_onehot = target_onehot.float()

        # if mask is not None:
        #     net_output = net_output[:, mask, :, :, :]
        #     target_onehot = target_onehot[:, mask, :, :, :]
        ce_loss = self.ce(net_output, target.squeeze(1) - 1)

        result = dice_loss + ce_loss

        tqdm.write(
            f"Dice: {dice_loss.item():.2f},\tCE: {ce_loss.item():.2f},\tTotal: {result.item():.2f}"
        )
        return result
