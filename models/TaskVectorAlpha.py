import copy
from typing import List

import torch
import torch.nn as nn

import configs
from models.modelFactory import ModelFactory
from taskvectors.TaskVector import TaskVector


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


class TaskVectorAlpha(nn.Module):
    def __init__(self, task_vectors: List[TaskVector]) -> None:
        super().__init__()

        backbone = task_vectors[
            0
        ].get_base_backbone()  # should be the same for all, TODO: assert this

        self.func0, params0 = make_functional(
            backbone.eval(), disable_autograd_tracking=True
        )
        self.params0 = nn.ParameterList(params0)

        self.task_vector_params = [tv.params for tv in task_vectors]

        self._model_name = backbone.__class__.__name__

        for p in self.params0:
            p.requires_grad = False

        for tv_params in self.task_vector_params:
            for p in tv_params:
                p.requires_grad = False

        # create n alphas that will be learned
        self.alphas = [
            nn.Parameter(torch.tensor(0.0)) for _ in range(len(task_vectors))
        ]

    def __call__(self, x) -> torch.Tensor:
        weighted_combined_tv = self.params0
        for alpha, tv_params in zip(self.alphas, self.task_vector_params):
            weighted_combined_tv = [
                p + alpha * tv_p for p, tv_p in zip(weighted_combined_tv, tv_params)
            ]
        return self.func0(weighted_combined_tv, x)
