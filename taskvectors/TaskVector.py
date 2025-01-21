from typing import Optional

import torch


class TaskVector:
    def __init__(
        self,
        pretrained_checkpoints: Optional[dict] = None,
        finetuned_checkpoints: Optional[dict] = None,
        task_vector: Optional[dict] = None,
    ):
        if pretrained_checkpoints is not None and finetuned_checkpoints is not None:
            assert (
                task_vector is None
            ), "Task vector should be None if both checkpoints are provided"
            self.task_vector = {
                key: finetuned_checkpoints - pretrained_checkpoints
                for key in finetuned_checkpoints.keys()
            }
        elif task_vector is not None:
            assert (
                pretrained_checkpoints is None and finetuned_checkpoints is None
            ), "Both checkpoints should be None if task vector is provided"
            self.task_vector = task_vector
        else:
            raise ValueError(
                "Either task vector or both checkpoints should be provided"
            )

    def __add__(self, other: "TaskVector"):
        assert (
            self.task_vector.keys() == other.task_vector.keys()
        ), "Task vectors should have the same keys"

        return TaskVector(
            task_vector={
                key: self.task_vector[key] + other.task_vector[key]
                for key in self.task_vector.keys()
            }
        )

    def get_parameters(
        self,
    ):
        return self.task_vector
