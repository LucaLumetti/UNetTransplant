from collections import OrderedDict
from copy import deepcopy
from typing import Optional, cast
from task.Task import Task

import torch
from taskvectors.TaskVector import TaskVector


def vector_to_state_dict(vector, state_dict):
    new_state_dict = state_dict.copy()
    sorted_dict = OrderedDict(sorted(new_state_dict.items(), key=lambda x: int(x[0])))
    torch.nn.utils.vector_to_parameters(vector, sorted_dict.values())
    return sorted_dict


def state_dict_to_vector(state_dict):
    sorted_dict = OrderedDict(sorted(state_dict.items()))
    return torch.nn.utils.parameters_to_vector(sorted_dict.values())


def ties_merge(params1, params2):
    merged_params = []
    
    for p1, p2 in zip(params1, params2):
        p1_flat = p1.view(-1)
        p2_flat = p2.view(-1)
        
        if p1_flat.numel() > p2_flat.numel():
            p2_flat = torch.cat([p2_flat, torch.zeros(p1_flat.numel() - p2_flat.numel(), device=p1.device)])
        elif p2_flat.numel() > p1_flat.numel():
            p1_flat = torch.cat([p1_flat, torch.zeros(p2_flat.numel() - p1_flat.numel(), device=p2.device)])
        
        combined = torch.stack([p1_flat, p2_flat], dim=0)
        
        # filtering out the smallest values
        k = int(0.2 * combined.size(1))
        threshold = combined.abs().kthvalue(k, dim=1, keepdim=True).values
        mask = combined.abs() < threshold
        combined[mask] = 0
        
        # sign
        sign_to_mult = torch.sign(combined.sum(dim=0))
        majority_sign = torch.sign(combined.sum())
        sign_to_mult[sign_to_mult == 0] = majority_sign
        
        # electing 
        valid_rows = torch.where(sign_to_mult.unsqueeze(0) > 0, combined > 0, combined < 0)
        selected_entries = combined * valid_rows
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        merged_param = selected_entries.sum(dim=0) / torch.clamp(non_zero_counts, min=1)
        
        merged_params.append(merged_param.view_as(p1))
    
    return torch.nn.ParameterList([torch.nn.Parameter(p) for p in merged_params])


class TaskVectorTies(TaskVector):
    def __init__(
        self,
        task: Optional[Task] = None,
        checkpoints: Optional[str] = None,
        params: Optional[torch.nn.ParameterList] = None,
    ):
        super().__init__(task=task, checkpoints=checkpoints, params=params)

    def __iadd__(self, other: "TaskVector"):
        self.task = self.task + other.task
        self.params = ties_merge(self.params, other.params)
        self._head_params += other._head_params
        return self
