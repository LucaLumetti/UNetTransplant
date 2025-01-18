import math
from typing import List

import torch

from datasets.LoadableDataset import LoadableDataset
from datasets.preprocessedDataset import PreprocessedDataset


class ComposedDataset(LoadableDataset):
    def __init__(
        self,
    ):
        # dataset_names = ["AMOS", "SegThor", "BTCV_Abdomen", "BTCV_Cervix", "ZhimingCui", "AbdomenCT-1K", "Skull", "TotalSegmentator"]
        # TODO: get them dynamically by reading the main folder
        dataset_names = [
            "AbdomenCT-1K",
            "BTCV_Abdomen",
            "SegThor",
            "ZhimingCui",
            "Skull",
            "BTCV_Cervix",
        ]
        self.combined_dataset = []
        self.weights = torch.tensor([1.0 for _ in dataset_names])

        # self.augmenter = Augmenter() # TODO
        for dataset_name in dataset_names:
            dataset = PreprocessedDataset(dataset_name=dataset_name)
            self.combined_dataset.append(dataset)
            self.weights[len(self.combined_dataset) - 1] = math.sqrt(len(dataset))
        super().__init__()

    def __len__(self):
        return sum([len(dataset) for dataset in self.combined_dataset])

    def __getitem__(self, idx):
        for dataset in self.combined_dataset:
            if idx < len(dataset):
                image, label = dataset[idx]
                # image, label = self.augmenter(image, label)
                return image, label
            idx -= len(dataset)
        raise ValueError(f"Index {idx} out of range.")

    # def __getitem__(self, idx):
    #     dataset_idx = torch.multinomial(self.weights, 1).item()
    #     dataset = self.combined_dataset[dataset_idx]
    #     sample = dataset[idx % len(dataset)]
    #     image, label = sample["image"], sample["label"]
    #     image, label = self.augmenter(image, label)
    #     return image, label
