from torch import nn


def init_weight_he(model: nn.Module, neg_slope=1e-2):
    for module in model.modules():
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
