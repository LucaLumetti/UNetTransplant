import os
from pathlib import Path

import submitit

# Create a Submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs_pretrain_AMOS")

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


def run_job(n, config_path, lr, batch_size, dropout):
    postfix = f"_AMOS_{n=}_{lr=}_{batch_size=}_{dropout=}"
    import subprocess

    args_to_run = [
        "python",
        "main.py",
        "--experiment",
        "PretrainExperiment",
        "--config",
        config_path,
        "--expname",
        postfix,
        "--override",
        f"DataConfig.BATCH_SIZE = {batch_size}",
        f"OptimizerConfig.BACKBONE_LR = {lr}",
        f"OptimizerConfig.HEAD_LR = {lr}",
        f"BackboneConfig.DROPOUT_PROB = {dropout}",
    ]

    subprocess.run(args_to_run)


base_path = "/leonardo/home/userexternal/llumetti/projects/UNetMerging"
base_config = f"{base_path}/configs/leonardo/pretrain2.toml"

batch_size_values = [2, 4, 8]
# lr_values = [0.001, 0.0025, 0.005, 0.0075, 0.01]
lr_values = [0.0001, 0.0005, 0.001]
dropout_values = [0.1, 0.3, 0.5]

jobs = []

with executor.batch():
    for i in range(3):
        job = executor.submit(
            run_job,
            n=i,
            config_path=base_config,
            batch_size=4,
            lr=0.001,
            dropout=0.5,
        )
        jobs.append(job)

        job = executor.submit(
            run_job,
            n=i,
            config_path=base_config,
            batch_size=4,
            lr=0.005,
            dropout=0.5,
        )
        jobs.append(job)
    # job = executor.submit(
        # run_job,
        # n=0,
        # config_path=base_config,
        # batch_size=8,
        # lr=0.0001,
        # dropout=0.0,
    # )
    # jobs.append(job)

print(f"Submitted {len(jobs)} jobs")
print(f"Logs will be available in {executor.folder}")
print(jobs)
