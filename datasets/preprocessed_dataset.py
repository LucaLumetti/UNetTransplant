from typing import List

import numpy as np
import torch

from config import DATASETS_OUTPUT_PATH


class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.subjects = self.load_subjects()
        super().__init__()

    def load_subjects(
        self,
    ) -> List:
        preprocessed_dataset_path = DATASETS_OUTPUT_PATH / self.dataset_name
        if not preprocessed_dataset_path.exists():
            raise ValueError(f"Dataset {self.dataset_name} not preprocessed yet.")

        all_volumes_dir = list(preprocessed_dataset_path.glob("*"))
        all_volumes_dir = [volume for volume in all_volumes_dir if volume.is_dir()]
        subjects = []
        for volume_dir in all_volumes_dir:
            images_path = (preprocessed_dataset_path / volume_dir / "images").glob(
                "*.npy"
            )
            labels_path = (preprocessed_dataset_path / volume_dir / "labels").glob(
                "*.npy"
            )
            subjects.append({"images": list(images_path), "labels": list(labels_path)})
        return subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        images_paths = subject["images"]
        labels_paths = subject["labels"]

        random_patch_idx = np.random.randint(0, len(images_paths))

        image = np.load(images_paths[random_patch_idx])
        label = np.load(labels_paths[random_patch_idx])

        return image, label
