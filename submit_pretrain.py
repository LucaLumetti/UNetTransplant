import os

import submitit

# Create a Submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs_pretraining")

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


def run_job(config_path):
    import subprocess

    subprocess.run(
        [
            "python",
            "main.py",
            "--experiment",
            "PretrainExperiment",
            "--config",
            config_path,
        ]
    )


configs_to_run = [
    "/leonardo/home/userexternal/llumetti/projects/UNetMerging/configs/leonardo/pretrain_stable_small.toml",
    "/leonardo/home/userexternal/llumetti/projects/UNetMerging/configs/leonardo/pretrain_stable_large.toml",


    # "/work/grana_maxillo/UNetMerging/configs/ailb/pretrain_stable_small.toml",
    # "/work/grana_maxillo/UNetMerging/configs/pretrain_stable.toml",
]

for config_path in configs_to_run:
    executor.update_parameters(name=config_path)
    job = executor.submit(run_job, config_path)
    print(f"Submitted job {job.job_id}")
