import os
from itertools import product
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchio as tio
from matplotlib import pyplot as plt
from tqdm import tqdm

import configs
from datasets import DatasetFactory
from datasets.PatchDataset import PatchDataset
from experiments.BaseExperiment import BaseExperiment
from metrics.Metrics import Metrics
from models.modelFactory import ModelFactory
from models.taskheads import Task
from taskvectors.TaskVector import TaskVector
from taskvectors.TaskVectorTies import TaskVectorTies


def main(tv_paths, merge_class):
    assert merge_class in [
        "TaskVector",
        "TaskVectorTies",
    ], f"Invalid merge class: {merge_class}"
    TaskVectorClass = TaskVector if merge_class == "TaskVector" else TaskVectorTies

    task_vectors = [
        TaskVectorClass(checkpoints=tv)
        for tv in tqdm(tv_paths, desc="Loading task vectors")
    ]

    initial_task_vector = task_vectors[0]
    combined_task_vector = initial_task_vector.__add_many__(task_vectors[1:])

    incremental_dices = []

    dataset = PatchDataset(split="train", task=combined_task_vector.task)
    dataset = [subject for subject in dataset.dataset]
    task = combined_task_vector.task
    backbone, heads = combined_task_vector.get_backbone_and_heads(tasks=[task])
    backbone.eval()
    heads.eval()
    backbone, heads = backbone.cuda(), heads.cuda()

    dices = []
    for subject in tqdm(dataset):
        if 5 not in subject["labels"][tio.DATA].unique():
            continue
        with torch.no_grad():
            out = BaseExperiment.functional_predict(
                backbone=backbone, heads=heads, subject=subject
            )
        x = subject["images"][tio.DATA]
        y = subject["labels"][tio.DATA]
        metrics = Metrics(task=combined_task_vector.task).compute(
            out, y, average="none", keep_nan=True  # type: ignore
        )

        np.save("debug/x.npy", x.numpy())
        np.save("debug/y.npy", y.numpy())
        np.save("debug/out.npy", out.numpy().astype(np.uint8))

        print(metrics["dice"])
        dices.append(metrics["dice"])
    dices = np.array(dices)[:, 1:]
    print(np.nanmean(dices, axis=0))
    plot(dices)


def plot(dices):
    # TODO
    pass


def override_config(configs, overrides: list):
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


def get_task_vectors_paths():
    task_vectors_classes = [
        ("Mandible", ["1"]),
        ("Pharynx", ["7"]),
        ("Canals", ["3-4"]),
        ("Teeth", ["11-18,21-28,31-38,41-48"]),
        ("Implants", ["8,9,10"]),
        # ("Skull", ["2"]),
    ]
    # task_vectors_classes = [
    #     ("Spleen", ["1"]),
    #     ("Kidney", ["2,3"]),
    #     # ("Esophagus", ["5"]),
    #     ("Liver", ["6"]),
    #     ("Stomach", ["7"]),
    #     # ("Pancreas", ["11"]),
    # ]
    task_vectors_classes = [x[0] for x in task_vectors_classes]

    checkpoints_path = (
        # Path("/leonardo_scratch/large/userexternal/llumetti/output_UNetMergingcheckpoints")
        Path("/work/grana_maxillo/UNetMergingOutput")
    )
    checkpoints_path = [
        x
        for x in checkpoints_path.glob("*")
        if x.is_dir() and "+" not in x.name and "TaskVectorTrainExperiment" in x.name
    ]

    grouped_checkpoints = {}
    for checkpoint_path in checkpoints_path:
        tv_task_name = checkpoint_path.name.split("__", 1)[1].split("_", 1)[0]
        tv_pretrain_kind = checkpoint_path.name.split("__", 1)[1].split("_", 1)[1]
        if "+" in tv_task_name:
            continue
        if tv_task_name not in task_vectors_classes:
            continue
        if tv_pretrain_kind not in grouped_checkpoints:
            grouped_checkpoints[tv_pretrain_kind] = {}
        grouped_checkpoints[tv_pretrain_kind][tv_task_name] = checkpoint_path

    sorted_checkpoints = {}
    for pretrain_kind, tv_checkpoints in grouped_checkpoints.items():
        sorted_checkpoints[pretrain_kind] = []
        for task_name, checkpoint_path in sorted(
            tv_checkpoints.items(), key=lambda item: task_vectors_classes.index(item[0])
        ):
            sorted_checkpoints[pretrain_kind].append(checkpoint_path)

    return sorted_checkpoints


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_class", type=str, required=False, default="TaskVector")
    args = parser.parse_args()

    if os.uname().nodename == "ailb-login-02":
        torch.backends.cudnn.enabled = False
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    configs.initialize_config("configs/ailb/merge.toml")

    task_vector_paths = get_task_vectors_paths()
    for k, v in task_vector_paths.items():
        v = [x / [x for x in os.listdir(x) if "0030" in x][0] for x in v]
        task_vector_paths[k] = v

    main(task_vector_paths["Stable1_yg0hh"], args.merge_class)
