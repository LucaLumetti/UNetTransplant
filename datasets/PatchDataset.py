import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import torch
import torchio as tio

import configs
from custom_types import Split
from datasets.LoadableDataset import LoadableDataset
from models.taskheads import Task
from preprocessing.dataset_class_mapping import DATASET_IDX, DATASET_ORIGINAL_LABELS


class PatchDataset:
    def __init__(self, split: Split, dataset_name: str):
        class_to_new_label_mapping = self.get_label_remap_dict(dataset_name)
        preprocessing = [
            tio.RemapLabels(class_to_new_label_mapping),
            tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5)),
        ]
        augmentations = []
        if split == "train":
            augmentations = [
                tio.RandomNoise(p=0.3),
                tio.RandomGamma(p=0.2),
                # tio.RandomMotion(p=0.05),
                # tio.RandomGhosting(p=0.05),
                # tio.RandomAffine(p=0.1),
            ]

        transforms = tio.Compose(preprocessing + augmentations)

        self.dataset_name = dataset_name
        self.split = split

        self.dataset_json = self.load_dataset_json()  # unused ?
        self.subjects = self.load_subjects()
        self.num_output_channels = (
            len(DATASET_ORIGINAL_LABELS[self.dataset_name].keys()) + 1
        )
        if configs.DataConfig.INCLUDE_ONLY_CLASSES is not None:
            self.num_output_channels = len(configs.DataConfig.INCLUDE_ONLY_CLASSES) + 1

        self.dataset = tio.SubjectsDataset(self.subjects, transform=transforms)

        print(f"Loaded {len(self.subjects)} subjects from dataset {self.dataset_name}")

    def get_label_remap_dict(self, dataset_name):
        class_names_to_predict = configs.DataConfig.INCLUDE_ONLY_CLASSES
        if class_names_to_predict is None or len(class_names_to_predict) == 0:
            class_to_predict = range(len(DATASET_ORIGINAL_LABELS[dataset_name].keys()))
        else:
            class_to_predict = []
            for class_idx, class_name in DATASET_ORIGINAL_LABELS[dataset_name].items():
                if class_name in class_names_to_predict:
                    class_to_predict.append(int(class_idx))

        # class_to_new_label_mapping = {
        #     class_idx: idx for idx, class_idx in enumerate(class_to_predict, start=1)
        # }
        max_label = max(int(k) for k in DATASET_ORIGINAL_LABELS[dataset_name].keys())
        class_to_new_label_mapping = {key: 0 for key in range(1, max_label + 1)}
        for idx, class_idx in enumerate(class_to_predict, start=1):
            class_to_new_label_mapping[int(class_idx)] = idx
        return class_to_new_label_mapping

    def load_dataset_json(
        self,
    ) -> dict:
        dataset_json_path = (
            configs.DataConfig.DATA_RAW_PATH / self.dataset_name / "dataset.json"
        )
        if not dataset_json_path.exists():
            raise ValueError(f"Dataset {self.dataset_name} not found.")
        with open(dataset_json_path, "r") as f:
            dataset_info = json.load(f)

        # optional default fields
        if "file_ending" not in dataset_info:
            dataset_info["file_ending"] = ".nii.gz"

        return dataset_info

    def load_subjects(
        self,
    ) -> List:
        dataset_path = configs.DataConfig.DATA_RAW_PATH / self.dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset {self.dataset_name} not preprocessed yet.")

        suffix_to_search = f"*{self.dataset_json['file_ending']}"

        subjects = []

        images_path = list((dataset_path / "imagesTr").glob(suffix_to_search))
        labels_path = list((dataset_path / "labelsTr").glob(suffix_to_search))

        amount_of_val_volumes = 5
        amount_of_train_volumes = len(images_path) - amount_of_val_volumes

        images_path = sorted(images_path)
        labels_path = sorted(labels_path)

        if "train" == self.split:
            images_path = images_path[:amount_of_train_volumes]
            labels_path = labels_path[:amount_of_train_volumes]
        elif "val" == self.split:
            images_path = images_path[amount_of_train_volumes:]
            labels_path = labels_path[amount_of_train_volumes:]

        for images_path, labels_path in zip(images_path, labels_path):
            subjects.append(
                tio.Subject(
                    images=tio.ScalarImage(path=images_path),
                    labels=tio.LabelMap(path=labels_path),
                    dataset_idx=torch.tensor(DATASET_IDX[self.dataset_name]),
                )
            )

        return subjects

    def get_tasks(
        self,
    ):
        num_output_channels = configs.DataConfig.INCLUDE_ONLY_CLASSES
        if num_output_channels is None or len(num_output_channels) == 0:
            num_output_channels = len(DATASET_ORIGINAL_LABELS[self.dataset_name].keys())
        else:
            num_output_channels = len(num_output_channels)
        task = Task(
            dataset_name=self.dataset_name,
            dataset_idx=DATASET_IDX[self.dataset_name],
            # num_output_channels=len(DATASET_ORIGINAL_LABELS[self.dataset_name].keys()),
            num_output_channels=num_output_channels,
        )
        return [task]

    def get_dataloader(
        self,
        batch_size: int = None,
        num_workers: int = None,
    ) -> torch.utils.data.DataLoader:
        if self.split == "train":
            return self.get_train_dataloader(batch_size, num_workers)
        elif self.split == "val":
            return self.get_val_dataloader(batch_size, num_workers)

    def get_train_dataloader(
        self,
        batch_size: int = None,
        num_workers: int = None,
    ) -> torch.utils.data.DataLoader:
        if batch_size is None:
            batch_size = configs.DataConfig.BATCH_SIZE
        if num_workers is None:
            num_workers = configs.DataConfig.NUM_WORKERS

        label_probs = {}
        for i in range(1, self.num_output_channels):
            label_probs[i] = 1
        label_probs[0] = sum(label_probs.values()) / 3  # background 25%

        patch_sampler = tio.data.LabelSampler(
            patch_size=(96, 96, 96),
            label_probabilities=label_probs,
        )
        patches_queue = tio.Queue(
            subjects_dataset=self.dataset,
            max_length=15,
            samples_per_volume=3,
            sampler=patch_sampler,
            num_workers=num_workers,
        )

        patches_loader = tio.SubjectsLoader(
            patches_queue,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        return patches_loader

    def get_val_dataloader(
        self,
        batch_size: int = None,
        num_workers: int = None,
    ) -> torch.utils.data.DataLoader:
        raise NotImplementedError("Val dataloader not implemented yet.")
        if batch_size is None:
            batch_size = 1
        if num_workers is None:
            num_workers = 1

        # patch_sampler = tio.data.GridSampler(
        #     self,
        #     patch_size=(96, 96, 96),
        #     patch_overlap=0,
        # )
        return []
