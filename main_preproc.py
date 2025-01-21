import sys
from pathlib import Path
from typing import List, cast

import configs
from preprocessing.preprocessor import Preprocessor
from preprocessing.raw_dataset import RawDataset

if __name__ == "__main__":
    is_debug = "debugpy" in sys.modules

    # read all the folder in datasets/
    datasets_path = list(configs.DataConfig.DATA_RAW_PATH.glob("*"))
    datasets_path = [
        dataset
        for dataset in datasets_path
        if dataset.is_dir() and (dataset / "dataset.json").exists()
    ]

    datasets = [RawDataset(dataset / "dataset.json") for dataset in datasets_path]

    print(f"Found {len(datasets)} datasets: {[dataset.name for dataset in datasets]}")

    for dataset in datasets:
        if (configs.DataConfig.DATA_PREPROCESSED_PATH / dataset.name).exists():
            print(f"Dataset {dataset.name} already preprocessed. Skipping.")
            continue
        # dataset_stats = DatasetStats(dataset)
        print(f"Dataset {dataset.name}.")
        preprocessor = Preprocessor(dataset)
        preprocessor.run_parallel(configs.DataConfig.DATA_PREPROCESSED_PATH)
