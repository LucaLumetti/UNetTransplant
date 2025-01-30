import os

import numpy as np
import torch

import configs
from datasets import DatasetFactory
from experiments.baseExperiment import BaseExperiment
from models.modelFactory import ModelFactory
from taskvectors.TaskVector import TaskVector
from taskvectors.TiesMerging import TiesMerging


def main():
    task_vector_lowerjaw = TaskVector(
        task_name="Lower Jaw",
        checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_mandible.toml/epoch0050_2025-01-28 00:37:57.256466_task_vector.pth",
        alphas=[1],
    )
    task_vector_lowerjaw.create_params_histogram()

    task_vector_pharynx = TaskVector(
        task_name="Pharynx",
        checkpoints="/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_pharynx.toml/epoch0050_2025-01-28 00:10:50.931316_task_vector.pth",
        alphas=[1],
    )
    task_vector_pharynx.create_params_histogram()

    task_vector_combined = task_vector_pharynx + task_vector_lowerjaw

    ties_vector_merged = TiesMerging(task_vector_combined)()

    print(task_vector_lowerjaw.params[0])
    print(task_vector_pharynx.params[0])
    print(ties_vector_merged.params[0])

    dataset = DatasetFactory.create(split="val")

    backbone, heads = ties_vector_merged.get_backbone_and_heads(
        tasks=dataset.get_tasks()
    )
    backbone.eval()
    heads.eval()

    x, y, i = dataset[0]
    i = torch.tensor([i]).cuda()
    sample_input = torch.from_numpy(x).unsqueeze(0).cuda()
    backbone, heads = backbone.cuda(), heads.cuda()

    out = BaseExperiment.functional_predict(
        backbone=backbone, heads=heads, image_array=x, dataset_idx=i
    )
    np.save("debug/merge_image.npy", x[0])
    np.save("debug/merge_label.npy", y[0])
    np.save("debug/merge_pred.npy", out.cpu().detach().numpy())
    pass


if __name__ == "__main__":
    configs.initialize_config("/work/grana_maxillo/UNetMerging/configs/merge_test.toml")
    main()
