import os

import submitit

# Directory to save generated configuration files
config_dir = "grid_search_configs"
os.makedirs(config_dir, exist_ok=True)

# Base configuration as a template
base_config = """[DataConfig]
NAME = "ComposedDataset"
DATASET_NAMES = ["ToothFairy2"]
INCLUDE_ONLY_CLASSES = ['{class_name}']
BATCH_SIZE = 2
NUM_WORKERS = 2
DATA_PREPROCESSED_PATH = "/work/grana_maxillo/UNetMergingData/preprocessed_data"
DATA_RAW_PATH = "/work/grana_maxillo/UNetMergingData/raw_data"

[BackboneConfig]
NAME = "UNet3D"
COMPILE = true
IN_CHANNELS = 1
PRETRAIN_CHECKPOINTS = "/work/grana_maxillo/UNetMerging/checkpoints/checkpoint_190_2025-01-19 04:55:49.994931.pth"

[OptimizerConfig]
NAME = "AdamW"
BACKBONE_LR = {backbone_learning_rate}
HEAD_LR = 0.001
WEIGHT_DECAY = {weight_decay}

[LossConfig]
NAME = "DiceBCELoss"

[TrainConfig]
EPOCHS = 51
SAVE_EVERY = 10
"""

# All classes
classes = {
    "1": "Lower Jawbone",
    "2": "Upper Jawbone",
    "3": "Left Inferior Alveolar Canal",
    "4": "Right Inferior Alveolar Canal",
    "5": "Left Maxillary Sinus",
    "6": "Right Maxillary Sinus",
    "7": "Pharynx",
    "8": "Bridge",
    "9": "Crown",
    "10": "Implant",
    "11": "Upper Right Central Incisor",
    "12": "Upper Right Lateral Incisor",
    "13": "Upper Right Canine",
    "14": "Upper Right First Premolar",
    "15": "Upper Right Second Premolar",
    "16": "Upper Right First Molar",
    "17": "Upper Right Second Molar",
    "18": "Upper Right Third Molar (Wisdom Tooth)",
    "19": "Upper Left Central Incisor",
    "20": "Upper Left Lateral Incisor",
    "21": "Upper Left Canine",
    "22": "Upper Left First Premolar",
    "23": "Upper Left Second Premolar",
    "24": "Upper Left First Molar",
    "25": "Upper Left Second Molar",
    "26": "Upper Left Third Molar (Wisdom Tooth)",
    "27": "Lower Left Central Incisor",
    "28": "Lower Left Lateral Incisor",
    "29": "Lower Left Canine",
    "30": "Lower Left First Premolar",
    "31": "Lower Left Second Premolar",
    "32": "Lower Left First Molar",
    "33": "Lower Left Second Molar",
    "34": "Lower Left Third Molar (Wisdom Tooth)",
    "35": "Lower Right Central Incisor",
    "36": "Lower Right Lateral Incisor",
    "37": "Lower Right Canine",
    "38": "Lower Right First Premolar",
    "39": "Lower Right Second Premolar",
    "40": "Lower Right First Molar",
    "41": "Lower Right Second Molar",
    "42": "Lower Right Third Molar (Wisdom Tooth)",
}

# Create a Submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")

executor.update_parameters(
    timeout_min=240,
    slurm_partition="all_usr_prod",
    slurm_account="grana_maxillo",
    cpus_per_task=2,
    tasks_per_node=1,
    nodes=1,
    slurm_mem="32G",
    slurm_gres="gpu:1",
)


def run_job(config_path):
    import subprocess

    subprocess.run(
        [
            "python",
            "main.py",
            "--config",
            config_path,
            "--name",
            f"TaskVector_{config_path}",
        ]
    )


GRID_SEARCH_PARAMS = {
    "backbone_learning_rate": [0.001, 0.0001, 0.00001],
    "weight_decay": [10, 5, 2, 1, 0.1],
}
# Generate config files and submit jobs
for wd in GRID_SEARCH_PARAMS["weight_decay"]:
    for blr in GRID_SEARCH_PARAMS["backbone_learning_rate"]:
        class_name = "Pharynx"
        config_filename = (
            f"taskvector_tf2_{class_name.replace(' ', '_')}_BLR_{blr}_WD_{wd}.toml"
        )
        config_path = os.path.join(config_dir, config_filename)

        # Write the config file
        with open(config_path, "w") as f:
            f.write(
                base_config.format(
                    class_name=class_name, backbone_learning_rate=blr, weight_decay=wd
                )
            )

        # Submit the job
        job_name = f"TaskVector_{class_name.replace(' ', '_')}_BLR_{blr}_WD_{wd}"
        executor.update_parameters(name=job_name)
        job = executor.submit(run_job, config_path)
        print(f"Submitted job for class {class_name} with job ID {job.job_id}")
