import sys
from pathlib import Path
from typing import List, cast

from config import DATASETS_INPUT_PATH, DATASETS_OUTPUT_PATH
from preprocessing.raw_dataset import RawDataset
from preprocessing.dataset_stats import DatasetStats
from preprocessing.preprocessor import Preprocessor

if __name__ == "__main__":
    is_debug = "debugpy" in sys.modules

    # read all the folder in datasets/
    datasets_path = list(DATASETS_INPUT_PATH.glob("*"))
    datasets_path = [dataset for dataset in datasets_path if dataset.is_dir() and (dataset / "dataset.json").exists()]

    datasets = [RawDataset(dataset / "dataset.json") for dataset in datasets_path]

    print(f"Found {len(datasets)} datasets: {[dataset.name for dataset in datasets]}")

    for dataset in datasets:
        if (DATASETS_OUTPUT_PATH / dataset.name).exists():
            print(f"Dataset {dataset.name} already preprocessed. Skipping.")
            continue
        # dataset_stats = DatasetStats(dataset)
        preprocessor = Preprocessor(dataset)
        preprocessor.run(DATASETS_OUTPUT_PATH)
