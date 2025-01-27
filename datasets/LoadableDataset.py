from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

import configs


class LoadableDataset(Dataset):
    def get_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> DataLoader:
        sampler = None  # TODO add support for DDP
        batch_size = batch_size or configs.DataConfig.BATCH_SIZE
        num_workers = num_workers or configs.DataConfig.NUM_WORKERS
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=sampler is None,
            sampler=sampler,
            prefetch_factor=2 if "train" in self.split else 1,
        )

        return dataloader
