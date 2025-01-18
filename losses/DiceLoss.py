from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(
        self,
        activation: Optional[Callable] = nn.Sigmoid(),
        batch_dice: bool = False,
        smooth: float = 1.0,
    ):
        super(DiceLoss, self).__init__()

        self.batch_dice = batch_dice
        self.nonlinearity = activation
        self.smooth = smooth

    def forward(self, x, y, mask: Optional[torch.Tensor] = None):
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

        # TODO: i can create just a vector here!
        if mask is None:
            mask = y != -1

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        # with torch.no_grad():
        y_onehot = y * mask
        x = x * mask
        sum_gt = y_onehot.sum(axes)

        intersect = (x * y_onehot).sum(axes)
        sum_pred = x.sum(axes)

        if self.batch_dice:
            # if self.ddp:
            #     intersect = AllGatherGrad.apply(intersect).sum(0)
            #     sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            #     sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (sum_gt + sum_pred + self.smooth)

        dc = 1 - dc
        dc_ = dc.sum() / (dc > 0).sum()
        return dc_
