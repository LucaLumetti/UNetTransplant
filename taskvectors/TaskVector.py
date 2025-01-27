from typing import Optional

import torch


class TaskVector:
    def __init__(
        self,
        checkpoints: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        assert (
            checkpoints is not None or model is not None
        ), "Either checkpoints or model must be provided."
        assert (
            checkpoints is None or model is None
        ), "Only one of checkpoints or model must be provided."

        if checkpoints is not None:
            self.params = self.get_params_from_checkpoints(checkpoints)
        else:
            self.model = model

    def get_params_from_checkpoints(self, checkpoints: str):
        ckpt = torch.load(checkpoints)
        backbone_params = ckpt["backbone_state_dict"]
        param_list = torch.nn.ParameterList(backbone_params.values())

        return param_list

    def create_params_from_model(self, model: torch.nn.Module):
        pass


if __name__ == "__main__":
    task_vector = TaskVector(
        checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_grid_search_configs/taskvector_tf2_Lower_Jawbone_BLR_0_WD_0.1.toml/epoch0005_2025-01-23 15:49:33.364213.pth"
    )
    print(task_vector.model)
