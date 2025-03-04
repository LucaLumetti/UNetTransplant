import os
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from matplotlib import pyplot as plt
from tqdm import tqdm

import configs
from datasets.PatchDataset import PatchDataset
from experiments.BaseExperiment import BaseExperiment
from metrics.Metrics import Metrics
from taskvectors.TaskVector import TaskVector
from taskvectors.TaskVectorTies import TaskVectorTies


def main(tv1_path, tv2_path, merge_class):
    assert merge_class in [
        "TaskVector",
        "TaskVectorTies",
    ], f"Invalid merge class: {merge_class}"
    TaskVectorClass = TaskVector if merge_class == "TaskVector" else TaskVectorTies
    tv1 = TaskVectorClass(checkpoints=tv1_path)
    tv2 = TaskVectorClass(checkpoints=tv2_path)

    dataset = PatchDataset(split="val", task=(tv1 + tv2).task)
    dataset = [subject for subject in dataset.dataset]

    range_a1 = np.linspace(0, 2, 21)
    range_a2 = np.linspace(0, 2, 21)
    grid_search_dices = torch.zeros((2, len(range_a1), len(range_a2)))

    tqdm_bar = tqdm(total=len(range_a1) * len(range_a2))

    for idx1, a1 in enumerate(range_a1):
        for idx2, a2 in enumerate(range_a2):
            tqdm_bar.update(1)
            task_vector = tv1 * a1 + tv2 * a2
            task = task_vector.task

            backbone, heads = task_vector.get_backbone_and_heads(tasks=[task])
            backbone.eval()
            heads.eval()
            backbone, heads = backbone.cuda(), heads.cuda()

            # print(f"Predicting {task_vector.task.task_name}")
            dices = []
            for subject in dataset:
                with torch.no_grad():
                    out = BaseExperiment.functional_predict(
                        backbone=backbone, heads=heads, subject=subject
                    )
                # x = subject["images"][tio.DATA]
                y = subject["labels"][tio.DATA]
                metrics = Metrics(task=task_vector.task).compute(
                    out, y, average="none", keep_nan=True  # type: ignore
                )
                dices.append(metrics["dice"])
            stacked_dices = torch.stack(dices)
            grid_search_dices[:, idx1, idx2] = torch.nanmean(stacked_dices, dim=0)[1:]

    tv1_task_name = tv1.task.task_name or "Task1"
    tv2_task_name = tv2.task.task_name or "Task2"
    for name, idx in zip([tv1_task_name, tv2_task_name], [0, 1]):
        plot(
            name,
            range_a1,
            range_a2,
            grid_search_dices[idx],
            tv1_task_name=tv1_task_name,
            tv2_task_name=tv2_task_name,
        )
    plot(
        f"{tv1_task_name}+{tv2_task_name}",
        range_a1,
        range_a2,
        torch.nanmean(grid_search_dices, dim=0),
        tv1_task_name=tv1_task_name,
        tv2_task_name=tv2_task_name,
    )


def plot(
    name,
    range_a1,
    range_a2,
    grid_search_dices,
    tv1_task_name="Task1",
    tv2_task_name="Task2",
):
    grid_search_dices = grid_search_dices.numpy()
    # Flip the matrix vertically
    flipped_grid = np.flipud(grid_search_dices)

    # Plot the heatmap
    plt.imshow(flipped_grid, cmap="viridis", vmin=0, vmax=1)

    # Set x and y ticks
    plt.xticks(np.arange(len(range_a1)), np.round(range_a1, 2), rotation=90)
    plt.yticks(
        np.arange(len(range_a2)), np.round(range_a2[::-1], 2)
    )  # Reverse y-axis ticks

    plt.xlabel(f"Alpha 1 - {tv1_task_name}")
    plt.ylabel(f"Alpha 2 - {tv2_task_name}")
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
    plt.savefig(f"{configs.DataConfig.OUTPUT_DIR}{name}.png")
    plt.close()

    # Save the grid as a .npy file
    np.save(f"{configs.DataConfig.OUTPUT_DIR}{name}.npy", grid_search_dices)


def override_config(self, overrides: list):
    for override in overrides:
        override = override.replace(" =", "=")
        override = override.replace("= ", "=")
        key, value = override.split("=")

        config_class = getattr(configs, key.split(".")[0])
        config_key = key.split(".")[1]
        type_of_value = type(getattr(config_class, config_key))
        if type_of_value == list:
            value = eval(value)
        else:
            value = type_of_value(value)
        setattr(config_class, config_key, value)
        print(f"Overriding {config_class.__class__.__name__}.{key} with {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tv1_checkpoint", type=str, required=True)
    parser.add_argument("--tv2_checkpoint", type=str, required=True)
    parser.add_argument(
        "--merge_class", type=str, required=False, default="TaskVectorTies"
    )
    args = parser.parse_args()

    if os.uname().nodename == "ailb-login-02":
        torch.backends.cudnn.enabled = False
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    configs.initialize_config("configs/leonardo/merge.toml")

    if args.tv1_checkpoint[0] != "/":
        args.tv1_checkpoint = Path(configs.DataConfig.OUTPUT_DIR) / args.tv1_checkpoint
    if args.tv2_checkpoint[0] != "/":
        args.tv2_checkpoint = Path(configs.DataConfig.OUTPUT_DIR) / args.tv2_checkpoint

    tv1_task_name = Path(args.tv1_checkpoint).stem.split("__")[1].split("_")[0]
    tv1_pretrain_kind = Path(args.tv1_checkpoint).stem.split("__")[1].split("_")[1]

    tv2_task_name = Path(args.tv2_checkpoint).stem.split("__")[1].split("_")[0]
    tv2_pretrain_kind = Path(args.tv2_checkpoint).stem.split("__")[1].split("_")[1]

    tv1_checkpoint_path = (
        Path(args.tv1_checkpoint)
        / [x for x in os.listdir(args.tv1_checkpoint) if "0010" in x][0]
    )
    tv2_checkpoint_path = (
        Path(args.tv2_checkpoint)
        / [x for x in os.listdir(args.tv2_checkpoint) if "0010" in x][0]
    )

    assert tv1_pretrain_kind == tv2_pretrain_kind, "Pretrain kind must be the same"

    configs.DataConfig.OUTPUT_DIR = f"{configs.DataConfig.OUTPUT_DIR}merge/{tv1_task_name}+{tv2_task_name}_{tv2_pretrain_kind}_{args.merge_class}/"
    Path(configs.DataConfig.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(
        f"Task 1:\n\tName: {tv1_task_name}\n\tKind: {tv1_pretrain_kind}\n\tCheckpoint: {tv1_checkpoint_path}"
    )
    print(
        f"Task 2:\n\tName: {tv2_task_name}\n\tKind: {tv2_pretrain_kind}\n\tCheckpoint: {tv2_checkpoint_path}"
    )
    main(tv1_checkpoint_path, tv2_checkpoint_path, args.merge_class)
