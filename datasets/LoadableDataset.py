import torch
from torch.utils.data import DataLoader, Dataset

from config import DataConfig


class LoadableDataset(Dataset):
    def get_dataloader(
        self,
    ) -> DataLoader:
        sampler = None  # TODO add support for DDP
        dataloader = DataLoader(
            self,
            batch_size=DataConfig.BATCH_SIZE,
            num_workers=DataConfig.NUM_WORKERS,
            pin_memory=False,
            shuffle=sampler is None,
            sampler=sampler,
        )

        return dataloader
