from collections import OrderedDict
from typing import List

import torch

from taskvectors.TaskVector import TaskVector


def vector_to_state_dict(vector, state_dict):
    """
    Convert a flattened parameter vector to a PyTorch state dict.

    Args:
        vector (torch.Tensor): The flattened parameter vector.
        state_dict (dict): The original state dict with parameter shapes.

    Returns:
        dict: The new state dict with parameters in the correct shape.
    """
    new_state_dict = state_dict.copy()
    sorted_dict = OrderedDict(sorted(new_state_dict.items(), key=lambda x: int(x[0])))
    torch.nn.utils.vector_to_parameters(vector, sorted_dict.values())
    return sorted_dict


def state_dict_to_vector(state_dict):
    """
    Convert a PyTorch state dict to a flattened parameter vector.

    Args:
        state_dict (dict): The state dict to convert.

    Returns:
        torch.Tensor: The flattened parameter vector.
    """
    sorted_dict = OrderedDict(sorted(state_dict.items()))
    return torch.nn.utils.parameters_to_vector(sorted_dict.values())


class TiesMerging:
    def __init__(
        self,
        task_vector: TaskVector,
    ):
        self.task_vector = task_vector

    def topk(self, params, k=0.2):
        assert 0 < k < 1, "k must be between 0 and 1"

        k = int(k * params.size(1))

        topk = params.abs().kthvalue(k, dim=1, keepdim=True)
        mask = params.abs() < topk.values

        params[mask] = 0
        return params

    def elect(self, params):
        sign_to_mult = torch.sign(torch.sum(params, dim=0))
        majority = torch.sign(torch.sum(params))
        sign_to_mult[sign_to_mult == 0] = majority
        return sign_to_mult

    def merge(self, params, signs):
        rows_to_keep = torch.where(signs.unsqueeze(0) > 0, params > 0, params < 0)
        selected_entries = params * rows_to_keep
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
        return disjoint_aggs

    def __call__(
        self,
    ):
        flat_params = torch.stack(
            [
                torch.concatenate([p.view(-1) for p in params], dim=0)
                for params in self.task_vector.params
            ]
        )
        flat_params = self.topk(flat_params.clone())
        sign = self.elect(flat_params.clone())
        merged_params = self.merge(flat_params.clone(), sign)
        state_dict = vector_to_state_dict(
            merged_params, self.task_vector.params[0].state_dict()
        )
        tv = TaskVector(
            "Ties merged",
            params=[torch.nn.ParameterList(state_dict.values())],
            alphas=[1],
        )
        tv._head_params = self.task_vector._head_params
        return tv
