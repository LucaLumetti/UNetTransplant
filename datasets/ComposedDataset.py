import math
from typing import List

import torch

import configs
from custom_types import Split
from datasets.LoadableDataset import LoadableDataset
from datasets.PreprocessedDataset import PreprocessedDataset


class ComposedDataset(LoadableDataset):
    def __init__(
        self,
        split: Split,
    ):
        if isinstance(split, str):
            split = [split]
        self.split = split
        dataset_names = configs.DataConfig.DATASET_NAMES
        self.combined_dataset = []
        self.weights = torch.tensor([1.0 for _ in dataset_names])

        for dataset_name in dataset_names:
            dataset = PreprocessedDataset(split=split, dataset_name=dataset_name)
            self.combined_dataset.append(dataset)
            self.weights[len(self.combined_dataset) - 1] = math.sqrt(len(dataset))
        super().__init__()

    def __len__(self):
        return sum([len(dataset) for dataset in self.combined_dataset])

    def __getitem__(self, idx):
        for dataset in self.combined_dataset:
            if idx < len(dataset):
                image, label, dataset_idx = dataset[idx]
                # image, label = self.augmenter(image, label)
                return image, label, dataset_idx
            idx -= len(dataset)
        raise ValueError(f"Index {idx} out of range.")

    def get_tasks(
        self,
    ):
        tasks = []
        for dataset in self.combined_dataset:
            tasks += dataset.get_tasks()
        return tasks

    # def __getitem__(self, idx):
    #     dataset_idx = torch.multinomial(self.weights, 1).item()
    #     dataset = self.combined_dataset[dataset_idx]
    #     sample = dataset[idx % len(dataset)]
    #     image, label = sample["image"], sample["label"]
    #     image, label = self.augmenter(image, label)
    #     return image, label
