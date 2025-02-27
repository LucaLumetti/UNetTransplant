import os
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from matplotlib import pyplot as plt
from tqdm import tqdm

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments.BaseExperiment import BaseExperiment
from experiments.FlatnessExperiment import FlatnessExperiment
from metrics.Metrics import Metrics
from models.modelFactory import ModelFactory
from models.taskheads import Task
from taskvectors.TaskVector import TaskVector
from taskvectors.TiesMerging import TiesMerging


def main(tv1_path, tv2_path):
    tv1 = TaskVector(checkpoints=tv1_path)
    tv2 = TaskVector(checkpoints=tv2_path)

    dataset = PatchDataset(split="val", task=(tv1 + tv2).task)

    range_a1 = np.linspace(0, 1.5, 16)
    range_a2 = np.linspace(0, 1.5, 16)
    grid_search_dices = np.zeros((2, len(range_a1), len(range_a2)))

    for idx1, a1 in enumerate(range_a1):
        for idx2, a2 in enumerate(range_a2):
            task_vector = tv1 * a1 + tv2 * a2
            task = task_vector.task

            backbone, heads = task_vector.get_backbone_and_heads(tasks=[task])
            backbone.eval()
            heads.eval()
            backbone, heads = backbone.cuda(), heads.cuda()

            print(f"Predicting {task_vector.task.task_name}")
            dices = []
            for subject in dataset.dataset:
                with torch.no_grad():
                    out = BaseExperiment.functional_predict(
                        backbone=backbone, heads=heads, subject=subject
                    )
                # x = subject["images"][tio.DATA]
                y = subject["labels"][tio.DATA]
                metrics = Metrics(task=task_vector.task).compute(out, y, average="none")  # type: ignore
                dices.append(metrics["dice"])
            grid_search_dices[:, idx1, idx2] = np.stack(dices).mean(0)
            print(f"a1: {a1}, a2: {a2}, Dice: {np.mean(dices)}")

    for name, idx in zip(["Task1_Naive", "Task2_Naive"], [0, 1]):
        plot(name, range_a1, range_a2, grid_search_dices[idx])
    plot("Merged_Naive", range_a1, range_a2, grid_search_dices.mean(axis=0))


def plot(name, range_a1, range_a2, grid_search_dices):
    # Flip the matrix vertically
    flipped_grid = np.flipud(grid_search_dices)

    # Plot the heatmap
    plt.imshow(flipped_grid, cmap="viridis", vmin=0, vmax=1)

    # Set x and y ticks
    plt.xticks(np.arange(len(range_a1)), np.round(range_a1, 2), rotation=90)
    plt.yticks(
        np.arange(len(range_a2)), np.round(range_a2[::-1], 2)
    )  # Reverse y-axis ticks

    plt.xlabel("Alpha 1")
    plt.ylabel("Alpha 2")
    plt.colorbar()

    # Find min and max values and their locations
    min_val, max_val = np.min(grid_search_dices), np.max(grid_search_dices)
    min_idx = np.argwhere(grid_search_dices == min_val)
    max_idx = np.argwhere(grid_search_dices == max_val)

    # Function to determine text color based on background brightness
    def get_text_color(value):
        return "white" if value < 0.5 else "black"

    # Add text annotations for min and max values
    for i, j in min_idx:
        plt.text(
            j,
            len(range_a2) - 1 - i,
            f"{min_val:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color=get_text_color(min_val),
        )

    for i, j in max_idx:
        plt.text(
            j,
            len(range_a2) - 1 - i,
            f"{max_val:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color=get_text_color(max_val),
        )

    # Save figure
    plt.savefig(f"debug/merge/{name}.png")
    plt.close()

    # Save the grid as a .npy file
    np.save(f"debug/merge/{name}.npy", grid_search_dices)

    print(f"Mean Dice: {np.mean(dice)}")


if __name__ == "__main__":
    # if hostname is ailb-login-02 disable cuda cudnn benchmark
    if os.uname().nodename == "ailb-login-02":
        torch.backends.cudnn.enabled = False
        # set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    configs.initialize_config("/work/grana_maxillo/UNetMerging/configs/merge_test.toml")

    # tv1_path = "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_jm51zjx0_taskvector_tf_mandible/epoch0010_2025-02-12 00:44:59.964354_task_vector.pth"
    # tv2_path = "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_4vdgsin0_taskvector_tf_lriac/epoch0010_2025-02-11 22:01:59.450714_task_vector.pth"

    tv1_path = "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_ek90lzx4_taskvector_tf_mandible/epoch0010_2025-02-10 05:59:06.254827_task_vector.pth"
    tv2_path = "/work/grana_maxillo/UNetMerging/checkpoints/TaskVectorTrainExperiment_37pgtnb8_taskvector_tf_lriac/epoch0010_2025-02-10 04:59:47.758080_task_vector.pth"

    main(tv1_path, tv2_path)
