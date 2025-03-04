from typing import Callable, Optional

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

    def forward(self, x, y, mask=None):
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)

        if mask is not None:
            mask = mask.flatten()
            y = y[:, mask, :, :, :]
            x = x[:, mask, :, :, :]

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))
        sum_gt = y.sum(axes)

        intersect = (x * y).sum(axes)
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
        return dc.mean()
