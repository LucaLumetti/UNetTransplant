import torch
from torch.utils.data import DataLoader, Dataset

import configs


class LoadableDataset(Dataset):
    def get_dataloader(
        self,
    ) -> DataLoader:
        sampler = None  # TODO add support for DDP
        dataloader = DataLoader(
            self,
            batch_size=configs.DataConfig.BATCH_SIZE,
            num_workers=configs.DataConfig.NUM_WORKERS,
            pin_memory=False,
            shuffle=sampler is None,
            sampler=sampler,
        )

        return dataloader
