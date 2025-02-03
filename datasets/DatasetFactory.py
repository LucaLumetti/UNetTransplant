import inspect
from typing import Literal, Optional, Union

import configs
import datasets
from custom_types import Split
from datasets import ComposedDataset
from datasets.LoadableDataset import LoadableDataset
from datasets.PatchDataset import PatchDataset


class DatasetFactory:
    @staticmethod
    def create(split: Split) -> Union[LoadableDataset, ComposedDataset, PatchDataset]:
        name = configs.DataConfig.NAME
        if name in datasets.__dict__:
            model_class = getattr(datasets, name)
        else:
            raise Exception(f"Dataset {name} not found.")

        try:
            # check if model_class requires a name argument
            signature = inspect.signature(model_class)
            if "dataset_name" in signature.parameters:
                model = model_class(
                    split=split, dataset_name=configs.DataConfig.DATASET_NAME
                )
            else:
                model = model_class(split=split)
        except TypeError as e:
            raise TypeError(f"Could not instantiate {model_class}: {e}")
        return model
