import json
from pathlib import Path
from typing import Optional

class RawDataset:
    def __init__(self, dataset_json: Path, *, num_samples: Optional[int] = None):
        with open(dataset_json, "r") as f:
            data = json.load(f)
            self.name: str = data["name"]
            self.labels: dict = data["labels"]
            self.num_training_samples: int = data["numTraining"]
            self.train_samples: list = data["training"]
            self.folder = dataset_json.parent
            self.imagesTr = dataset_json.parent / "imagesTr"
            self.labelsTr = dataset_json.parent / "labelsTr"

        if self.train_samples is None or len(self.train_samples) == 0:
             # read from folder imagesTr
            files = sorted(list(self.imagesTr.iterdir()))
            self.train_samples = [{"image": str(file), "label": str(self.labelsTr / file.name)} for file in files]

        if num_samples is not None:
            self.train_samples = self.train_samples[:num_samples]