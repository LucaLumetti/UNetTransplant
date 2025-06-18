from torch import nn


def get_number_of_learnable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def number_of_features_per_level(init_channel_number: int, num_levels: int) -> list:
    return [init_channel_number * 2**k for k in range(num_levels)]
