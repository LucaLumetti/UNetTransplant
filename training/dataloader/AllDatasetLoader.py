from typing import List
import torch

from datasets.preprocessed_dataset import PreprocessedDataset

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,):
        print(f"Fix the combined dataset, you need to make a weighted sampling")
        dataset_names = ['AMOS', 'SegThor']
        self.combined_dataset = []
        for dataset_name in dataset_names:
            dataset = PreprocessedDataset(dataset_name=dataset_name)
            self.combined_dataset.append(dataset)
        super().__init__()

    def __len__(self):
        return sum([len(dataset) for dataset in self.combined_dataset])
    
    def __getitem__(self, idx):
        for dataset in self.combined_dataset:
            if idx < len(dataset):
                image, label = dataset[idx]
                image, label = self.augmenter(image, label)
                return image, label
            idx -= len(dataset)
        raise ValueError(f"Index {idx} out of range.")