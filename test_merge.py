import os
from pathlib import Path

import numpy as np
import torch
import torchio as tio

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments.BaseExperiment import BaseExperiment
from metrics.Metrics import Metrics
from models.modelFactory import ModelFactory
from models.taskheads import Task
from taskvectors.TaskVector import TaskVector
from taskvectors.TiesMerging import TiesMerging


def main():
    tv1 = TaskVector(
        checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ek90lzx4_taskvector_tf_mandible/epoch0010_2025-02-10 05:59:06.254827_task_vector.pth"
    )
    tv2 = TaskVector(
        checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_37pgtnb8_taskvector_tf_lriac/epoch0010_2025-02-10 04:59:47.758080_task_vector.pth"
    )
    tv3 = TaskVector(
        checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ff3xd0ge_taskvector_tf_pharynx/epoch0020_2025-02-10 11:57:53.601929_task_vector.pth"
    )
    combined = tv1 + 0.5 * tv2 + 0.5 * tv3

    for task_vector in [combined]:
        task = task_vector.task
        task_vector.create_params_histogram()
        backbone, heads = task_vector.get_backbone_and_heads(tasks=[task])
        backbone.eval()
        heads.eval()
        backbone, heads = backbone.cuda(), heads.cuda()

        dataset = PatchDataset(split="val", task=task)
        subject = dataset.dataset[0]

        print(f"Predicting {task_vector.task.task_name}")
        with torch.no_grad():
            out = BaseExperiment.functional_predict(
                backbone=backbone, heads=heads, subject=subject
            )
        x = subject["images"][tio.DATA]
        y = subject["labels"][tio.DATA]
        metrics = Metrics(task=task_vector.task).compute(out, y, average="none")
        print(f'Dice: {metrics["dice"]}')

        output_path = Path(
            f"debug/merge/{task_vector.task.task_name}_{metrics['dice']}"
        )
        os.makedirs(output_path, exist_ok=True)
        np.save(output_path / "image.npy", x[0])
        np.save(output_path / "label.npy", y[0])
        np.save(output_path / "pred.npy", out.cpu().detach().numpy().astype(np.uint8))
    # x = subject['images'][tio.DATA]
    # y = subject['labels'][tio.DATA]
    # metrics = Metrics().compute(out, y)
    # print(f'Dice: {metrics["dice"]}')

    # np.save("debug/merge_image.npy", x[0])
    # np.save("debug/merge_label.npy", y[0])
    # np.save("debug/merge_pred.npy", out.cpu().detach().numpy().astype(np.uint8))

    # print("Done")


if __name__ == "__main__":
    # if hostname is ailb-login-02 disable cuda cudnn benchmark
    if os.uname().nodename == "ailb-login-02":
        torch.backends.cudnn.enabled = False
        # set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    configs.initialize_config("/work/grana_maxillo/UNetMerging/configs/merge_test.toml")
    main()
