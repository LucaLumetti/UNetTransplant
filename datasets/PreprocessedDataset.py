import json
import math
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import torch

import configs
from custom_types import Split
from datasets.LoadableDataset import LoadableDataset
from models.taskheads import Task
from preprocessing.dataset_class_mapping import DATASET_IDX, DATASET_ORIGINAL_LABELS


class PreprocessedDataset(LoadableDataset):
    def __init__(self, split: Split, dataset_name: str):
        super().__init__()

        self.dataset_name = dataset_name
        if isinstance(split, str):
            split = [split]
        self.split = split

        self.dataset_json = self.load_dataset_json()
        self.subjects = self.load_subjects()

        print(f"Loaded {len(self.subjects)} subjects from dataset {self.dataset_name}")

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

    def load_subjects(self) -> List:
        if "train" in self.split or "val" in self.split:
            return self.load_trainval_subjects()
        elif "test" in self.split:
            return self.load_test_subjects()
        else:
            raise ValueError(f"Split {self.split} not supported.")

    def load_trainval_subjects(
        self,
    ) -> List:
        preprocessed_dataset_path = (
            configs.DataConfig.DATA_PREPROCESSED_PATH / self.dataset_name
        )
        if not preprocessed_dataset_path.exists():
            raise ValueError(f"Dataset {self.dataset_name} not preprocessed yet.")

        all_volumes_dir = list(preprocessed_dataset_path.glob("*"))
        all_volumes_dir = [volume for volume in all_volumes_dir if volume.is_dir()]

        # hacky way to split the dataset into train/val. For test a different approach should be used TODO
        # TODO: support multiple splits in the same dataset i.e., ['train', 'val']
        all_volumes_dir = sorted(all_volumes_dir)
        amount_of_train_volumes = math.floor(0.9 * len(all_volumes_dir))
        if "train" in self.split:
            all_volumes_dir = all_volumes_dir[:amount_of_train_volumes]
        elif "val" in self.split:
            all_volumes_dir = all_volumes_dir[amount_of_train_volumes:]
        else:
            raise ValueError(f"Split {self.split} not supported.")

        subjects = []
        for volume_dir in all_volumes_dir:
            images_path = list(
                (preprocessed_dataset_path / volume_dir / "images").glob("*.npy")
            )
            labels_path = list(
                (preprocessed_dataset_path / volume_dir / "labels").glob("*.npy")
            )
            if len(images_path) == 0 or len(labels_path) == 0:
                # print(f"Volume {volume_dir} has no images, skipping.")
                continue
            subjects.append({"images": images_path, "labels": labels_path})
        return subjects

    def load_test_subjects(
        self,
    ) -> List:
        # differently from trainval, test load whole volumes and preprocess them. Caller will have to split them into patches
        dataset_path = configs.DataConfig.DATA_RAW_PATH / self.dataset_name
        if (
            not (dataset_path / "imagesTs").exists()
            or not (dataset_path / "labelsTs").exists()
        ):
            raise ValueError(f"Dataset {self.dataset_name} not found.")

        images_dir_path = dataset_path / "imagesTs"
        labels_dir_path = dataset_path / "labelsTs"
        images_path = list(images_dir_path.glob(f"*{self.dataset_info['file_ending']}"))

        for image_path in images_path:
            label_path = labels_dir_path / image_path.name
            if not label_path.exists():
                raise ValueError(
                    f"Label {label_path} not found for image {images_path}."
                )

        subjects = []
        for image_path in images_path:
            subjects.append(
                {"images": image_path, "labels": labels_dir_path / image_path.name}
            )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        images_paths = subject["images"]
        labels_paths = subject["labels"]

        random_patch_idx = np.random.randint(0, len(images_paths))
        try:
            image = np.load(images_paths[random_patch_idx])
            label = np.load(labels_paths[random_patch_idx])
        except Exception as e:
            print(f"Failed to load image {images_paths[random_patch_idx]}")
            import shutil

            patient_folder = Path(images_paths[random_patch_idx]).parent.parent
            if patient_folder.exists():
                shutil.rmtree(patient_folder)
            return self.__getitem__(idx + 1)
        # image = np.memmap(images_paths[random_patch_idx], shape=(1, 96, 96, 96), dtype="float32", mode="r")
        # label = np.memmap(labels_paths[random_patch_idx], shape=(62, 96, 96, 96), dtype="float32", mode="r")

        return image, label, DATASET_IDX[self.dataset_name]

    def get_tasks(
        self,
    ):
        task = Task(
            dataset_name=self.dataset_name,
            dataset_idx=DATASET_IDX[self.dataset_name],
            num_output_channels=len(DATASET_ORIGINAL_LABELS[self.dataset_name].keys()),
        )
        return [task]
