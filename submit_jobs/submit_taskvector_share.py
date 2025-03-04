import os
from pathlib import Path

import submitit

# Create a Submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs_share")

if "ailb" in os.uname().nodename:
    slurm_partition = "all_usr_prod"
    slurm_account = "grana_maxillo"
    cpus_per_task = 4
    slurm_constraint = "gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G"
else:
    slurm_partition = "boost_usr_prod"
    slurm_account = "IscrB_FeeCO"
    cpus_per_task = 8
    slurm_constraint = None

executor.update_parameters(
    timeout_min=24 * 60,
    slurm_partition=slurm_partition,
    slurm_account=slurm_account,
    cpus_per_task=cpus_per_task,
    tasks_per_node=1,
    nodes=1,
    setup=["source /leonardo/home/userexternal/llumetti/.bashrc"],
    slurm_mem="64G",
    slurm_gres="gpu:1",
    slurm_constraint=slurm_constraint,
)


def run_job(config_path, checkpoint_path, class_to_predict, name):
    postfix = name
    import subprocess

    args_to_run = [
        "python",
        "main.py",
        "--experiment",
        "TaskVectorShareExperiment",
        "--config",
        config_path,
        "--expname",
        postfix,
        "--override",
        f"DataConfig.PRETRAIN_CHECKPOINTS = {checkpoint_path}",
        f"DataConfig.DATASET_CLASSES = {class_to_predict}",
    ]

    subprocess.run(args_to_run)


base_config = "/leonardo/home/userexternal/llumetti/projects/UNetMerging/configs/leonardo/share.toml"

base_path_checkpoints = Path(
    "/leonardo_scratch/large/userexternal/llumetti/output_UNetMergingcheckpoints"
)

task_vectors_checkpoints = [
    Path("TaskVectorTrainExperiment_nicd7j0f__Mandible_Stable1_yg0hh"),
    Path("TaskVectorTrainExperiment_3bkhi755__Skull_Stable1_yg0hh"),
    Path("TaskVectorTrainExperiment_ln5cghm4__Canals_Stable1_yg0hh"),
    Path("TaskVectorTrainExperiment_cijno6gn__Pharynx_Stable1_yg0hh"),
    Path("TaskVectorTrainExperiment_bv8s3hwv__Implants_Stable1_yg0hh"),
    Path("TaskVectorTrainExperiment_z8gwfgf3__Teeth_Stable1_yg0hh"),
]

task_vectors_classes = {
    "Mandible": ["1"],
    "Skull": ["2"],
    "Canals": ["3-4"],
    "Pharynx": ["7"],
    "Implants": ["8,9,10"],
    "Teeth": ["11-18,21-28,31-38,41-48"],
}

task_vectors_checkpoints = [base_path_checkpoints / c for c in task_vectors_checkpoints]


task_vectors_checkpoints = [
    list(checkpoint_path.glob("epoch0010*.pth"))[0]
    for checkpoint_path in task_vectors_checkpoints
]

jobs = []
executor.update_parameters(name="TaskVectorShare")

with executor.batch():
    for tv_checkpoint in task_vectors_checkpoints:
        tv_class_name = str(tv_checkpoint).split("__")[1].split("_", 1)[0]
        for class_name, class_id in task_vectors_classes.items():
            if tv_class_name == class_name:
                continue
            name = f"{tv_class_name[:2]}->{class_name[:2]}"
            # print(f"\nRunning {name}:\n\t{tv_checkpoint=}\n\t{class_id=}")
            job = executor.submit(run_job, base_config, tv_checkpoint, class_id, name)
            jobs.append(job)

print(jobs)
print(f"Submitted {len(jobs)} jobs")
print(f"Logs will be available in {executor.folder}")
