from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import List, Optional, cast

import matplotlib.pyplot as plt
import torch

import configs
from models.modelFactory import ModelFactory
from models.taskheads import TaskHeads
from models.TaskVectorModel import TaskVectorModel
from task.Task import Task


class TaskVector:
    def __init__(
        self,
        task: Optional[Task] = None,
        checkpoints: Optional[str] = None,
        params: Optional[torch.nn.ParameterList] = None,
        # use_norm: bool = False,
    ):
        assert (
            checkpoints is not None or params is not None
        ), "Either checkpoints or param list must be provided."
        assert (
            sum([checkpoints is not None, params is not None]) == 1
        ), "Only one of checkpoints or params must be provided."
        assert not (
            task is None and checkpoints is None
        ), "Task can be inferred from checkpoints, otherwise it must be provided."
        assert not (
            task is not None and checkpoints is not None
        ), "If checkpoints are provided, task should not be provided, as it will be inferred from the checkpoints."

        if checkpoints is not None:
            params = self.get_params_from_checkpoints(checkpoints)
        elif params is not None:
            params = params
        else:
            raise ValueError("No params provided.")
        if checkpoints is None and task is not None:
            self.task = cast(Task, task)
        self.params = cast(torch.nn.ParameterList, params)

    def get_params_from_checkpoints(self, checkpoints: str):
        ckpt = torch.load(checkpoints)

        delta_params = ckpt["delta_state_dict"]
        self._head_params = [ckpt["heads_state_dict"]]
        self._backbone_params = ckpt["pretrain_state_dict"]
        tasks = ckpt["task"]
        assert (
            len(tasks) == 1
        ), "Still TODO: you tried to load a checkpoint with multiple tasks, but only one is supported for now."
        self.task = cast(Task, ckpt["task"][0])
        param_list = torch.nn.ParameterList(delta_params.values())

        return param_list

    def create_params_from_model(self, model: torch.nn.Module):
        pass

    def __add__(self, other: "TaskVector"):
        tv = deepcopy(self)
        tv += other
        return tv

    def __radd__(self, other: "TaskVector"):
        return self.__add__(other)

    def __iadd__(self, other: "TaskVector"):
        self.task = self.task + other.task
        self.params = torch.nn.ParameterList(
            [a + b for a, b in zip(self.params, other.params)]
        )
        self._head_params += other._head_params
        return self

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__add__(other)

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __neg__(self):
        return self * -1

    def __imul__(self, other: float):
        self.params = torch.nn.ParameterList([p * other for p in self.params])
        return self

    def __mul__(self, other: float):
        tv = deepcopy(self)
        tv *= other
        return tv

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def get_backbone_and_heads(self, tasks):
        backbone = ModelFactory.create_backbone(configs.BackboneConfig)
        backbone = TaskVectorModel(backbone)
        backbone.params0 = torch.nn.ParameterList(self._backbone_params.values())
        backbone.params = self.params

        heads = self.get_heads_from_params(tasks)

        return backbone, heads

    def get_base_backbone(self):
        backbone = ModelFactory.create_backbone(configs.BackboneConfig)
        backbone = TaskVectorModel(backbone)
        backbone.params0 = torch.nn.ParameterList(self._backbone_params.values())

        return backbone

    def get_heads_from_params(self, tasks):
        heads_state_dict = defaultdict(list)
        for head_params in self._head_params:
            for name, params in head_params.items():
                heads_state_dict[name].append(params)

        for name, params in heads_state_dict.items():
            heads_state_dict[name] = torch.concat(params, dim=0)

        num_different_heads = len(heads_state_dict.keys()) / 2
        assert num_different_heads == 1, "Still TODO: multiple heads"

        taskhead = ModelFactory.create_heads(configs.HeadsConfig, tasks)
        taskhead.load_state_dict(heads_state_dict)

        return taskhead

    def create_params_histogram(self):
        params = self.params
        flattened_params = torch.cat([p.flatten() for p in params])
        plt.hist(flattened_params.cpu().detach().numpy(), bins=100)
        plt.savefig(
            f"{configs.DataConfig.OUTPUT_DIR}debug/params_histogram_{self.task.task_name}.png"
        )
        plt.close()
