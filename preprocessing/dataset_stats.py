import json
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

from preprocessing.raw_dataset import RawDataset


class DatasetStats:
    """
    For a given dataset, this class should store incrementally some statistics
    and later be able to return them aggregated."""

    def __init__(self, dataset: RawDataset):
        self.dataset = dataset
        self._shapes = []
        self._spacings = []
        self._max_values = []
        self._min_values = []

        self.iterate_samples()

    @property
    def median_shape(self) -> tuple:
        median_shape = [None, None, None]
        for i in range(3):
            median_shape[i] = sum([shape[i] for shape in self._shapes]) / len(
                self._shapes
            )
        return median_shape

    @property
    def median_spacing(self) -> tuple:
        median_spacing = [None, None, None]
        for i in range(3):
            median_spacing[i] = sum([spacing[i] for spacing in self._spacings]) / len(
                self._spacings
            )
        return median_spacing

    @property
    def max_shape(self) -> tuple:
        max_shape = [None, None, None]
        for i in range(3):
            max_shape[i] = max([shape[i] for shape in self._shapes])
        return max_shape

    @property
    def min_shape(self) -> tuple:
        min_shape = [None, None, None]
        for i in range(3):
            min_shape[i] = min([shape[i] for shape in self._shapes])
        return min_shape

    @property
    def max_spacing(self) -> tuple:
        max_spacing = [None, None, None]
        for i in range(3):
            max_spacing[i] = max([spacing[i] for spacing in self._spacings])
        return max_spacing

    @property
    def min_spacing(self) -> tuple:
        min_spacing = [None, None, None]
        for i in range(3):
            min_spacing[i] = min([spacing[i] for spacing in self._spacings])
        return min_spacing

    @property
    def max_value(self) -> int:
        return max(self._max_values)

    @property
    def min_value(self) -> int:
        return min(self._min_values)

    def iterate_samples(self, n: int = None):
        num_train_samples = len(list(self.dataset.imagesTr.glob("*")))
        assert (
            num_train_samples == self.dataset.num_training_samples
        ), f"Number of samples in imagesTr ({num_train_samples}) does not match numTraining ({self.dataset.num_training_samples})"

        sample: str
        for sample in tqdm(
            self.dataset.train_samples, desc=f"Processing stats for {self.dataset.name}"
        ):
            image: sitk.Image = sitk.ReadImage(self.dataset.folder / sample["image"])
            label: sitk.Image = sitk.ReadImage(self.dataset.folder / sample["label"])
            assert (
                image.GetSize() == label.GetSize()
            ), f"Image and label sizes do not match for sample {sample}"
            self.add_sample(image)

    def add_sample(self, sample: sitk.Image):
        image_array = sitk.GetArrayFromImage(sample)
        self._shapes.append(sample.GetSize())
        self._spacings.append(sample.GetSpacing())
        self._max_values.append(image_array.max())
        self._min_values.append(image_array.min())

    def __str__(self):
        str_repr = "Dataset {}:\n\tAvg. shape: {}\n\tMax shape: {}\n\tMin shape: {}\n\tAvg. spacing: {}\n\tMax spacing: {}\n\tMin spacing: {}\n\tMax values: {}\n\tMin values: {}"
        return str_repr.format(
            self.dataset.name,
            self.median_shape,
            self.max_shape,
            self.min_shape,
            self.median_spacing,
            self.max_spacing,
            self.min_spacing,
            self.max_value,
            self.min_value,
        )
