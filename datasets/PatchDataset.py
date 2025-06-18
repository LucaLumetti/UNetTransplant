import json
from typing import List, Optional, Tuple, cast

import numpy as np
import torch
import torchio as tio
from sympy import Idx

import configs
from custom_types import Split
from preprocessing.dataset_class_mapping import DATASET_IDX, DATASET_ORIGINAL_LABELS
from task.Task import Task


class AssertShapeEqualsOrGreater(tio.Transform):
    def __init__(
        self,
        target_shape: Tuple[int, int, int],
        padding_mode: str = "constant",
        padding_value: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_shape = np.array(target_shape)
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in subject.get_images(intensity_only=False):
            current_shape = np.array(image.spatial_shape)
            padding_needed = np.maximum(self.target_shape - current_shape, 0)

            pad_params: Tuple[int, int, int, int, int, int] = (
                padding_needed[0] // 2,
                padding_needed[0] - padding_needed[0] // 2,
                padding_needed[1] // 2,
                padding_needed[1] - padding_needed[1] // 2,
                padding_needed[2] // 2,
                padding_needed[2] - padding_needed[2] // 2,
            )

            image.set_data(
                tio.transforms.Pad(pad_params, padding_mode=self.padding_value)(
                    image
                ).data
            )
        return subject


class PatchDataset:
    def __init__(self, split: Split, task: Task):
        self.task = task
        self.dataset_name = task.dataset_name
        labels_to_remap = {}
        for idx, key in enumerate(DATASET_ORIGINAL_LABELS[self.dataset_name].keys()):
            key = int(key)
            label_to_assign = 0
            if key in task.labels_to_predict:
                label_to_assign = task.labels_to_predict[key]
            labels_to_remap[int(key)] = label_to_assign

        preprocessing = [
            tio.RemapLabels(labels_to_remap),
            AssertShapeEqualsOrGreater((96, 96, 96)),
            tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5)),
        ]
        augmentations = []
        if split == "train":
            augmentations = [
                tio.RandomNoise(p=0.3),
                tio.RandomGamma(p=0.2),
                tio.RandomAffine(p=0.1),
            ]

        transforms = tio.Compose(preprocessing + augmentations)

        self.split = split

        self.dataset_json = self.load_dataset_json()  # TODO: unused ?
        self.subjects = self.load_subjects()
        self.num_output_channels = task.num_output_channels

        self.dataset = tio.SubjectsDataset(self.subjects, transform=transforms)

        print(f"Loaded {len(self.subjects)} subjects from dataset {self.dataset_name}")

    def load_dataset_json(
        self,
    ) -> dict:
        dataset_json_path = (
            configs.DataConfig.DATA_RAW_PATH / self.dataset_name / "dataset.json"
        )
        if not dataset_json_path.exists():
            raise ValueError(
                f"Dataset {self.dataset_name} not found. Cannot load {dataset_json_path}"
            )
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
                    filename=images_path.name,
                )
            )

        return subjects

    def get_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        if self.split == "train":
            return self.get_train_dataloader(batch_size, num_workers)
        elif self.split == "val":
            return self.get_val_dataloader(batch_size, num_workers)
        else:
            raise ValueError(f"Split {self.split} not recognized.")

    def get_train_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        if batch_size is None:
            batch_size = configs.DataConfig.BATCH_SIZE
        if num_workers is None:
            num_workers = configs.DataConfig.NUM_WORKERS
            num_workers = cast(int, num_workers)

        label_probs = {}
        for i in range(1, self.num_output_channels):
            label_probs[i] = 1
        label_probs[0] = 0.1

        patch_sampler = tio.data.LabelSampler(
            patch_size=(96, 96, 96),
            label_probabilities=label_probs,
        )
        patches_queue = tio.Queue(
            subjects_dataset=self.dataset,
            max_length=300,
            samples_per_volume=100,
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

    # TODO: implement
    def get_val_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        raise NotImplementedError("Val dataloader not implemented yet.")
