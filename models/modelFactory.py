import torch
from torch import nn

import models
from config import ModelConfig


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


class ModelFactory:
    @staticmethod
    def create() -> torch.nn.Module:
        name = ModelConfig.NAME
        if name in models.__dict__:
            model_class = getattr(models, name)
        else:
            raise Exception(f"Model {name} not found")

        try:
            model = model_class(ModelConfig.IN_CHANNELS, ModelConfig.OUT_CHANNELS)
        except TypeError as e:
            raise TypeError(f"Could not instantiate {model_class}: {e}")

        cuda_capability = torch.cuda.get_device_capability(0)
        if cuda_capability[0] >= 7 and ModelConfig.COMPILE:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            model = torch.compile(model=model)

        init_weight_he(model)

        return model
