import multiprocessing
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from tqdm import tqdm

from preprocessing.dataset_class_mapping import DATASET_MAPPING_LABELS
from preprocessing.dataset_stats import DatasetStats
from preprocessing.raw_dataset import RawDataset


class Preprocessor:
    def __init__(self, dataset: RawDataset):
        self.dataset = dataset
        # self.stats = stats

    @staticmethod
    def extract_patches(
        image: npt.NDArray,
        label: npt.NDArray,
        patch_size: Union[int, Tuple[int, int, int]] = (96, 96, 96),
        patch_overlap: Union[float, Tuple[float, float, float]] = (
            0.0,
            0.0,
            0.0,
        ),
    ):
        if label is not None and image.shape[-3:] != label.shape[-3:]:
            print(f"Image and label shape mismatch: {image.shape} != {label.shape}")
            raise ValueError("Image and label shape mismatch.")

        if not all([image.shape[i + 1] >= patch_size[i] for i in range(3)]):
            raise ValueError("Patch size is larger than image size.")

        patches = []

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        if isinstance(patch_overlap, float):
            patch_overlap = (patch_overlap, patch_overlap, patch_overlap)

        stride = [int(patch_size[i] * (1 - patch_overlap[i])) for i in range(3)]
        for z in range(0, image.shape[-3] - patch_size[0] + 1, stride[0]):
            for y in range(0, image.shape[-2] - patch_size[1] + 1, stride[1]):
                for x in range(0, image.shape[-1] - patch_size[2] + 1, stride[2]):
                    patch_image = image[
                        :,
                        z : z + patch_size[0],
                        y : y + patch_size[1],
                        x : x + patch_size[2],
                    ]
                    if label is not None:
                        patch_label = label[
                            :,
                            z : z + patch_size[0],
                            y : y + patch_size[1],
                            x : x + patch_size[2],
                        ]
                    else:
                        patch_label = None
                    patches.append((patch_image, patch_label, (z, y, x)))

        return patches

    @staticmethod
    def preprocess(
        image: sitk.Image,
        label: sitk.Image,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        # image = self.resample(image, target_spacing, interpolator=sitk.sitkLinear)
        # label = self.resample(
        #     label, target_spacing, interpolator=sitk.sitkNearestNeighbor
        # )
        image_array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
        label_array = sitk.GetArrayFromImage(label).astype(np.int8, copy=False)

        image_array = Preprocessor.clip_percentile(image_array)
        image_array = Preprocessor.normalize(image_array)
        return image_array, label_array

    @staticmethod
    def resample(
        image: sitk.Image,
        target_spacing: Tuple[float, float, float],
        interpolator: int,
    ):
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(
            [
                int(image.GetSize()[i] * image.GetSpacing()[i] / target_spacing[i])
                for i in range(3)
            ]
        )
        return resampler.Execute(image)

    @staticmethod
    def clip_percentile(image_array: npt.NDArray, mask: Optional[npt.NDArray] = None):
        """
        Clip the intensities of the image to the 0.5 and 99.5 percentile of the intensities of the training set.
        """
        lower_bound = np.percentile(image_array, 0.5)
        upper_bound = np.percentile(image_array, 99.5)
        image_array = np.clip(image_array, lower_bound, upper_bound)
        return image_array

    @staticmethod
    def normalize(image_array: npt.NDArray, mask: Optional[npt.NDArray] = None):
        """
        Normalize the intensities of the image to have zero mean and unit variance.
        """
        mean = np.mean(image_array)
        std = np.std(image_array)
        image_array = (image_array - mean) / std
        return image_array

    def run(
        self,
        output_folder: Path,
        keep_empty_prob: float = 0.05,
    ):
        for sample in tqdm(
            self.dataset.train_samples, desc=f"Preprocessing {self.dataset.name}"
        ):
            self.process_sample(sample, output_folder, keep_empty_prob)

    def run_parallel(
        self,
        output_folder: Path,
        keep_empty_prob: float = 0.05,
    ):
        num_cpus = 4
        print("Using", num_cpus, "CPUs")
        with multiprocessing.Pool(num_cpus) as pool:
            pool.starmap(
                self.process_sample,
                [
                    (sample, output_folder, keep_empty_prob)
                    for sample in self.dataset.train_samples
                ],
            )

    def process_sample(self, sample: dict, output_folder: Path, keep_empty_prob: float):
        image = sample["image"]
        label = sample["label"]

        # check if the image and label are alredy preprocessed
        image_name = image.split("/")[-1].split(".")[0]

        image_folder = output_folder / self.dataset.name / image_name / "images"
        label_folder = output_folder / self.dataset.name / image_name / "labels"

        create_only_full = False
        already_preproc = (output_folder / self.dataset.name / image_name).exists()
        if already_preproc:
            # check that a *_full.npy file exists
            if (image_folder / f"{image_name}_full.npy").exists() and (
                label_folder / f"{image_name}_full.npy"
            ).exists():
                tqdm.write(f"Already preprocessed {sample['image']}")
                return
            else:
                create_only_full = True

        # print(f"Processing {image_name}")
        try:
            image: sitk.Image = sitk.ReadImage(self.dataset.folder / sample["image"])
            label: sitk.Image = sitk.ReadImage(self.dataset.folder / sample["label"])
        except RuntimeError as e:
            print(
                f"Could not read image {sample['image']} or label {sample['label']}\n{e}"
            )
            return

        image_array, label_array = self.preprocess(image, label)

        if self.dataset.name == "ZhimingCui":
            label_array = (label_array > 0).astype(label_array.dtype)

        if self.dataset.name == "ToothFairy2":
            label_array = self.convert_toothfairy2_labels(label_array)

        image_name = sample["image"].split("/")[-1].split(".")[0]
        dataset_name = self.dataset.name

        image_folder = output_folder / dataset_name / image_name / "images"
        label_folder = output_folder / dataset_name / image_name / "labels"

        image_array, label_array = self.pad_to_patchable_shape(image_array, label_array)
        # label_array = self.make_one_hot(label_array)

        if image_array.ndim == 3:
            image_array = image_array[np.newaxis, ...]

        if label_array.ndim == 3:
            label_array = label_array[np.newaxis, ...]

        if not image_folder.exists():
            image_folder.mkdir(parents=True)
        if not label_folder.exists():
            label_folder.mkdir(parents=True)

        np.save(image_folder / f"{image_name}_full.npy", image_array)
        np.save(label_folder / f"{image_name}_full.npy", label_array)

        if create_only_full:
            return

        try:
            patches = Preprocessor.extract_patches(image_array, label_array)
        except ValueError:
            print(f"Image too small? {image_array.shape}")
            return

        for i, (image_patch, label_patch, coords) in tqdm(
            enumerate(patches), desc=f"Saving patches for {image_name}"
        ):
            if np.random.rand() > keep_empty_prob and np.max(label_patch) == 0:
                continue
            np.save(image_folder / f"{image_name}_{i}.npy", image_patch)
            np.save(label_folder / f"{image_name}_{i}.npy", label_patch)

    # def make_one_hot(self, label_array: npt.NDArray):
    #     one_hot = np.zeros(
    #         (MODEL_OUT_CHANNELS, *label_array.shape), dtype=label_array.dtype
    #     )
    #     mapping = DATASET_MAPPING_LABELS[self.dataset.name]
    #     for original, mapped in mapping.items():
    #         one_hot[mapped - 1] = label_array == original

    #     # set to -1 if the classes is not related to the dataset
    #     for i in range(MODEL_OUT_CHANNELS):
    #         if i + 1 not in mapping.values():
    #             one_hot[i] = -1

    #     return one_hot

    # TODO: make this function accept a single npt.NDArray and just call it two times in the parent fn
    @staticmethod
    def pad_to_patchable_shape(
        image_array: npt.NDArray, label_array: Optional[npt.NDArray] = None
    ):
        for i in range(3):
            if image_array.shape[3 - i - 1] < 96:
                pad = 96 - image_array.shape[3 - 1 - i]

                label_padding_at_dim_i = ((0, 0),) * 3
                label_padding_at_dim_i = list(label_padding_at_dim_i)
                label_padding_at_dim_i[3 - i - 1] = (0, pad)
                image_array = np.pad(
                    image_array,
                    label_padding_at_dim_i,
                    mode="constant",
                    constant_values=0,
                )
                if label_array is not None:
                    label_array = np.pad(
                        label_array,
                        label_padding_at_dim_i,
                        mode="constant",
                        constant_values=0,
                    )
        return image_array, label_array

    @staticmethod
    def convert_toothfairy2_labels(label_array: npt.NDArray):
        label_array[label_array >= 21] -= 2
        label_array[label_array >= 31] -= 2
        label_array[label_array >= 41] -= 2
        return label_array
