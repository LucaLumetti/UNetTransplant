import datasets
from config import DataConfig
from datasets.LoadableDataset import LoadableDataset


class DatasetFactory:
    @staticmethod
    def create() -> LoadableDataset:
        name = DataConfig.NAME
        if name in datasets.__dict__:
            model_class = getattr(datasets, name)
        else:
            raise Exception(f"Dataset {name} not found.")

        try:
            model = model_class()
        except TypeError as e:
            raise TypeError(f"Could not instantiate {model_class}: {e}")
        return model
