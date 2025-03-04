import os
from pathlib import Path

import submitit

# Create a Submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")

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


def run_job(tv1_checkpoint, tv2_checkpoint):
    import subprocess

    args_to_run = [
        "python",
        "merge_grid_search.py",
        "--tv1_checkpoint",
        tv1_checkpoint,
        "--tv2_checkpoint",
        tv2_checkpoint,
        "--merge_class",
        "TaskVectorTies",
        # "TaskVector",
    ]

    subprocess.run(args_to_run)


base_path = "/leonardo/home/userexternal/llumetti/projects/UNetMerging"
base_config = f"{base_path}/configs/leonardo/merge.toml"

task_vectors_classes = [
    ("Mandible", ["1"]),
    ("Skull", ["2"]),
    ("Canals", ["3-4"]),
    ("Pharynx", ["7"]),
    ("Implants", ["8,9,10"]),
    ("Teeth", ["11-18,21-28,31-38,41-48"]),
]
task_vectors_classes = [x[0] for x in task_vectors_classes]

checkpoints_path = Path(
    "/leonardo_scratch/large/userexternal/llumetti/output_UNetMergingcheckpoints"
)
checkpoints_path = [
    x
    for x in checkpoints_path.glob("*")
    if x.is_dir() and "+" not in x.name and "TaskVectorTrainExperiment" in x.name
]

grouped_checkpoints = {}
for checkpoint_path in checkpoints_path:
    tv_task_name = checkpoint_path.name.split("__", 1)[1].split("_", 1)[0]
    tv_pretrain_kind = checkpoint_path.name.split("__", 1)[1].split("_", 1)[1]
    if "+" in tv_task_name:
        continue
    if "Naive" not in tv_pretrain_kind:
        continue
    if tv_pretrain_kind not in grouped_checkpoints:
        grouped_checkpoints[tv_pretrain_kind] = {}
    grouped_checkpoints[tv_pretrain_kind][tv_task_name] = checkpoint_path

sorted_checkpoints = {}
for pretrain_kind, tv_checkpoints in grouped_checkpoints.items():
    sorted_checkpoints[pretrain_kind] = []
    for task_name, checkpoint_path in sorted(
        tv_checkpoints.items(), key=lambda item: task_vectors_classes.index(item[0])
    ):
        sorted_checkpoints[pretrain_kind].append(checkpoint_path)

print(sorted_checkpoints)

jobs = []

with executor.batch():
    for pretrain_kind, tv_checkpoints in sorted_checkpoints.items():
        for i, tv1_checkpoint in enumerate(tv_checkpoints):
            for j, tv2_checkpoint in enumerate(tv_checkpoints):
                if i >= j:
                    continue
                job = executor.submit(run_job, tv1_checkpoint, tv2_checkpoint)
                jobs.append(job)

print(f"Submitted {len(jobs)} jobs")
print(f"Logs will be available in {executor.folder}")
print(jobs)
