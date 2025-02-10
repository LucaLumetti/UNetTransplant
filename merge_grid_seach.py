import os
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from matplotlib import pyplot as plt
from tqdm import tqdm

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments.baseExperiment import BaseExperiment
from metrics.Metrics import Metrics
from models.modelFactory import ModelFactory
from models.taskheads import Task
from taskvectors.TaskVector import TaskVector
from taskvectors.TiesMerging import TiesMerging

trained_taskvectors = {
    "Lower Jawbone": "/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_mandible.toml/epoch0010_2025-02-06 21:14:31.176747_task_vector.pth",
    "Upper Jawbone": "/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_skull.toml/epoch0010_2025-02-06 22:35:05.818140_task_vector.pth",
    "IAC": "/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_lriac.toml/epoch0010_2025-02-06 22:31:12.042787_task_vector.pth",
    "Pharynx": "/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_pharynx.toml/epoch0010_2025-02-06 22:29:27.968639_task_vector.pth",
    # 'Teeth': '/work/grana_maxillo/UNetMerging/checkpoints/TaskVector_/work/grana_maxillo/UNetMerging/configs/taskvector_tf_teeth.toml/epoch0010_2025-02-07 06:23:03.002797_task_vector.pth',
}

taskvector_tasks = {
    "Lower Jawbone": Task("ToothFairy2", 8, 1, ["Lower Jawbone"]),
    "Upper Jawbone": Task("ToothFairy2", 8, 1, ["Upper Jawbone"]),
    "IAC": Task(
        "ToothFairy2",
        8,
        2,
        ["Left Inferior Alveolar Canal", "Right Inferior Alveolar Canal"],
    ),
    "Pharynx": Task("ToothFairy2", 8, 1, ["Pharynx"]),
    "Teeth": Task(
        "ToothFairy2",
        8,
        32,
        [
            "Upper Left Central Incisor",
            "Upper Left Lateral Incisor",
            "Upper Left Canine",
            "Upper Left First Premolar",
            "Upper Left Second Premolar",
            "Upper Left First Molar",
            "Upper Left Second Molar",
            "Upper Left Third Molar (Wisdom Tooth)",
            "Upper Right Central Incisor",
            "Upper Right Lateral Incisor",
            "Upper Right Canine",
            "Upper Right First Premolar",
            "Upper Right Second Premolar",
            "Upper Right First Molar",
            "Upper Right Second Molar",
            "Upper Right Third Molar (Wisdom Tooth)",
            "Lower Left Central Incisor",
            "Lower Left Lateral Incisor",
            "Lower Left Canine",
            "Lower Left First Premolar",
            "Lower Left Second Premolar",
            "Lower Left First Molar",
            "Lower Left Second Molar",
            "Lower Left Third Molar (Wisdom Tooth)",
            "Lower Right Central Incisor",
            "Lower Right Lateral Incisor",
            "Lower Right Canine",
            "Lower Right First Premolar",
            "Lower Right Second Premolar",
            "Lower Right First Molar",
            "Lower Right Second Molar",
            "Lower Right Third Molar (Wisdom Tooth)",
        ],
    ),
}


class TaskVectorTask:
    def __init__(self, task_vector: TaskVector, task: Task):
        self.task_vector = task_vector
        self.task = task

    def __add__(self, other):
        assert self.task.dataset_name == other.task.dataset_name
        assert self.task.dataset_idx == other.task.dataset_idx
        new_tv = self.task_vector + other.task_vector
        new_ioc = self.task.include_only_classes + other.task.include_only_classes
        new_task = Task(
            self.task.dataset_name,
            self.task.dataset_idx,
            self.task.num_output_channels + other.task.num_output_channels,
            new_ioc,
        )
        return TaskVectorTask(new_tv, new_task)


def main():
    taskvectors_and_tasks = []

    for task_name, task_vector_path in trained_taskvectors.items():
        task_vector = TaskVector(
            task_name=task_name,
            checkpoints=task_vector_path,
            alphas=[1],
        )
        task = taskvector_tasks[task_name]
        taskvectors_and_tasks.append(TaskVectorTask(task_vector, task))

    for idx1, tvt1 in enumerate(tqdm(taskvectors_and_tasks)):
        for idx2, tvt2 in enumerate(taskvectors_and_tasks):
            if idx1 >= idx2:
                continue
            alpha1 = np.linspace(0, 2, 21)
            alpha2 = np.linspace(0, 2, 21)

            combined = tvt1 + tvt2
            task_vector = combined.task_vector
            task = combined.task
            task_vector.create_params_histogram()

            configs.DataConfig.INCLUDE_ONLY_CLASSES = task.include_only_classes
            dataset = PatchDataset(split="val", dataset_name="ToothFairy2")
            subject = dataset.dataset[0]

            grid_search_dices = np.zeros((len(alpha1), len(alpha2)))
            for idx_a1, a1 in enumerate(alpha1):
                for idx_a2, a2 in enumerate(alpha2):
                    task_vector._alphas = [a1, a2]
                    backbone, heads = task_vector.get_backbone_and_heads(tasks=[task])
                    backbone.eval()
                    heads.eval()
                    backbone, heads = backbone.cuda(), heads.cuda()

                    with torch.no_grad():
                        out = BaseExperiment.functional_predict(
                            backbone=backbone, heads=heads, subject=subject
                        )
                    x = subject["images"][tio.DATA]
                    y = subject["labels"][tio.DATA]
                    metrics = Metrics().compute(out, y, average="macro")
                    grid_search_dices[idx_a1, idx_a2] = metrics["dice"]
                    tqdm.write(
                        f'{task_vector.task_name} - {a1} - {a2}. Dice: {metrics["dice"]}'
                    )

                    output_path = Path(
                        f"debug/merge/{task_vector.task_name}_{metrics['dice']}_a1_{a1}_a2_{a2}"
                    )
                    # os.makedirs(output_path, exist_ok=True)
                    # np.save(output_path / "image.npy", x[0])
                    # np.save(output_path / "label.npy", y[0])
                    # np.save(output_path / "pred.npy", out.cpu().detach().numpy().astype(np.uint8))
                # create heatmap from grid_search_dices.
                # grid_search_dices is a NxN matrix where N is the number of alphas and the value is between 0 and 1. -1 means that
                # the dice was not computed and should be dark, other values should have a gradient colormap.
            plt.imshow(grid_search_dices, cmap="viridis")
            # add labels and ticks
            plt.xticks(np.arange(len(alpha1)), alpha1.tolist())
            plt.yticks(np.arange(len(alpha2)), alpha2.tolist())
            plt.xlabel("Alpha 1")
            plt.ylabel("Alpha 2")
            plt.colorbar()
            plt.savefig(f"debug/merge/{task_vector.task_name}_grid_search.png")
            plt.close()


if __name__ == "__main__":
    configs.initialize_config("/work/grana_maxillo/UNetMerging/configs/merge_test.toml")
    main()
