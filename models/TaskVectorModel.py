import copy

import torch
import torch.nn as nn


def make_functional(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values


class TaskVectorModel(nn.Module):
    """Train only the delta of a given model, i.e., train the task vector, that, applied to the initial model, gives the final model."""

    def __init__(self, model: nn.Module) -> None:
        """Initializes the linearized model."""
        super().__init__()

        self.func0, params0 = make_functional(
            model.eval(), disable_autograd_tracking=True
        )

        # params is a copy of params but initialized as zero
        params = copy.deepcopy(params0)
        for p in params:
            p.data.zero_()

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        for p in self.params0:
            p.requires_grad = False

        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        params_and_tv = [p + tv for p, tv in zip(self.params0, self.params)]
        return self.func0(params_and_tv, x)
