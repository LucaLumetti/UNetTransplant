from collections import OrderedDict, defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import torch

import configs
from models.modelFactory import ModelFactory
from models.taskheads import TaskHeads
from models.TaskVectorModel import TaskVectorModel


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


class TaskVector:
    def __init__(
        self,
        task_name: Optional[str] = None,
        checkpoints: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        params: Optional[List[torch.nn.ParameterList]] = None,
        alphas: List[float] = [1],
        use_norm: bool = False,
    ):
        self.task_name = task_name
        assert (
            checkpoints is not None or model is not None or params is not None
        ), "Either checkpoints or model or param list must be provided."
        assert (
            sum([checkpoints is not None, model is not None, params is not None]) == 1
        ), "Only one of checkpoints, model or params must be provided."

        if checkpoints is not None:
            params = self.get_params_from_checkpoints(checkpoints)
        elif model is not None:
            params = self.create_params_from_model(model)
        elif params is not None:
            params = params
        else:
            raise ValueError("No params provided.")

        if not isinstance(params, List):
            params = [params]

        if not isinstance(alphas, List):
            alphas = [alphas]

        self._params = params
        self._alphas = alphas
        self.use_norm = use_norm

        assert len(self._params) == len(
            self._alphas
        ), "Number of params and alphas must be the same."

    @property
    def params(self):
        combined_params = []
        total_alpha = sum(self._alphas)
        alphas = self._alphas
        if total_alpha != 0 and self.use_norm:
            alphas = [a / total_alpha for a in alphas]
        for a, p in zip(alphas, self._params):
            weighted_params = [a * param for param in p]
            combined_params.append(weighted_params)

        result_params = [sum(p) for p in zip(*combined_params)]
        return torch.nn.ParameterList(result_params)

    def get_params_from_checkpoints(self, checkpoints: str):
        ckpt = torch.load(checkpoints)
        # TODO: fix params->delta and params0->pretrain
        delta_params = ckpt["delta_state_dict"]
        self._head_params = [ckpt["heads_state_dict"]]
        self._backbone_params = ckpt["pretrain_state_dict"]
        param_list = torch.nn.ParameterList(delta_params.values())

        return param_list

    def create_params_from_model(self, model: torch.nn.Module):
        pass

    def __add__(self, other):
        both_names = f"{self.task_name} and {other.task_name}"
        both_params = self._params + other._params
        both_alphas = self._alphas + other._alphas
        tv = TaskVector(task_name=both_names, params=both_params, alphas=both_alphas)
        tv._head_params = self._head_params + other._head_params
        tv._backbone_params = self._backbone_params
        return tv

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return TaskVector(
            task_name=self.task_name,
            params=self._params,
            alphas=[-a for a in self._alphas],
        )

    def get_backbone_and_heads(self, tasks):
        backbone = ModelFactory.create(configs.BackboneConfig)
        # params0_state_dict = self._backbone_params
        # params0_state_dict = {
        #     k.replace("_orig_mod.", ""): v
        #     for k, v in params0_state_dict.items()
        #     if "final_conv" not in k
        # }

        # backbone.load_state_dict(params0_state_dict)
        backbone = TaskVectorModel(backbone)
        backbone.params0 = torch.nn.ParameterList(self._backbone_params.values())
        backbone.params = self.params

        heads = self.get_heads_from_params(tasks)

        return backbone, heads

    def get_heads_from_params(self, tasks):
        heads_state_dict = defaultdict(list)
        for head_params in self._head_params:
            for name, params in head_params.items():
                heads_state_dict[name].append(params)

        for name, params in heads_state_dict.items():
            heads_state_dict[name] = torch.concat(params, dim=0)

        num_different_heads = len(heads_state_dict.keys()) / 2
        assert num_different_heads == 1, "Still TODO: multiple heads"

        taskhead = TaskHeads(configs.HeadsConfig.IN_CHANNELS, tasks)
        taskhead.load_state_dict(heads_state_dict)

        return taskhead

    def create_params_histogram(self):
        params = self.params
        flattened_params = torch.cat([p.flatten() for p in params])
        plt.hist(flattened_params.cpu().detach().numpy(), bins=100)
        plt.savefig(f"debug/params_histogram_{self.task_name}.png")
        plt.close()
