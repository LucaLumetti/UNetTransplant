import torch
from torch import nn
from tqdm import tqdm

from losses.DiceLoss import DiceLoss


class DiceBCELoss(nn.Module):
    def __init__(
        self,
    ):
        super(DiceBCELoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(activation=torch.sigmoid, batch_dice=False)

    def forward(
        self, net_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ):
        mask = mask.flatten()
        y_shape_background = list(net_output.shape)
        y_shape_background[1] += 1

        target_onehot = torch.zeros(
            y_shape_background, device=net_output.device, dtype=torch.bool
        )
        target_onehot.scatter_(1, target.long(), 1)
        target_onehot = target_onehot[:, 1:]

        dice_loss = self.dice(net_output, target_onehot, mask=mask)
        target_onehot = target_onehot.float()

        masked_net_output = net_output[:, mask, :, :, :]
        masked_target = target_onehot[:, mask, :, :, :]
        bce_loss = self.bce(masked_net_output, masked_target)

        result = dice_loss + bce_loss

        tqdm.write(
            f"Dice: {dice_loss.item():.2f},\tBCE: {bce_loss.item():.2f},\tTotal: {result.item():.2f}"
        )
        return result
