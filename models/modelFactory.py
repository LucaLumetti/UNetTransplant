from typing import List, Optional, Tuple

import torch
from torch import nn

import configs
import models
from models.unet3d.unet3d import ResidualUNet3D
from task.Task import Task


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
    def create_backbone(model_config: configs._BackboneConfig) -> torch.nn.Module:
        name = model_config.NAME
        assert (
            name == "ResidualUNet3D"
        ), f"Backbone {name} not supported, only ResidualUNet3D is supported"

        try:
            model = ResidualUNet3D(
                in_channels=model_config.IN_CHANNELS,
                dropout_prob=model_config.DROPOUT_PROB,
            )
        except TypeError as e:
            raise TypeError(f"Could not instantiate model: {e}")
        init_weight_he(model)
        model = model.cuda()
        # TODO: load checkpoint?
        return model

    @staticmethod
    def create_heads(
        heads_config: configs._HeadsConfig, tasks: List[Task]
    ) -> torch.nn.Module:
        name = heads_config.NAME
        assert (
            name == "TaskHeads"
        ), f"Heads {name} not supported, only TaskHeads is supported"

        try:
            model = models.TaskHeads(
                input_channels=heads_config.IN_CHANNELS,
                tasks=tasks,
            )
        except TypeError as e:
            raise TypeError(f"Could not instantiate model: {e}")
        init_weight_he(model)
        model = model.cuda()
        return model

    @staticmethod
    def create(model_config, tasks: Optional[List[Task]] = None) -> torch.nn.Module:
        name = model_config.NAME
        if name in models.__dict__:
            model_class = getattr(models, name)
        else:
            raise Exception(f"Model {name} not found")

        try:
            if tasks is not None:
                model = model_class(model_config.IN_CHANNELS, tasks)
            else:
                model = model_class(
                    model_config.IN_CHANNELS,
                )
        except TypeError as e:
            raise TypeError(f"Could not instantiate {model_class}: {e}")

        cuda_capability = torch.cuda.get_device_capability(0)
        # if cuda_capability[0] >= 7 and model_config.COMPILE:
        #     model = torch.compile(model=model)

        init_weight_he(model)
        model = model.cuda()

        return model

    @staticmethod
    def create_from_checkpoint(
        checkpoint_path: str,
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        backbone = ModelFactory.create(configs.BackboneConfig)
        checkpoint = torch.load(checkpoint_path)

        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        head = nn.Conv3d(32, 4, kernel_size=1)
        head_state_dict = checkpoint["heads_state_dict"]
        head_state_dict = {
            k.replace("task_heads.7.", ""): v for k, v in head_state_dict.items()
        }
        head.load_state_dict(head_state_dict)
        return backbone, head
