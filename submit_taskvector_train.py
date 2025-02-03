import os

import submitit

# Create a Submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")

executor.update_parameters(
    timeout_min=12 * 60,
    # slurm_partition="boost_usr_prod",
    # timeout_min=4 * 60,
    slurm_partition="all_usr_prod",
    slurm_account="grana_maxillo",
    cpus_per_task=4,
    tasks_per_node=1,
    nodes=1,
    slurm_mem="64G",
    slurm_gres="gpu:1",
    # slurm_constraint="gpu_A40_48G",
    slurm_constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G",
)


def run_job(config_path):
    import subprocess

    subprocess.run(
        [
            "python",
            "main.py",
            "--experiment",
            "TaskVectorExperiment",
            # "PretrainExperiment",
            "--config",
            config_path,
            "--name",
            f"TaskVector_{config_path}",
        ]
    )


configs_to_run = [
    "/work/grana_maxillo/UNetMerging/configs/taskvector_tf_mandible.toml",
    "/work/grana_maxillo/UNetMerging/configs/taskvector_tf_lriac.toml",
    "/work/grana_maxillo/UNetMerging/configs/taskvector_tf_pharynx.toml",
    "/work/grana_maxillo/UNetMerging/configs/finetune_tf_mandible.toml",
    "/work/grana_maxillo/UNetMerging/configs/finetune_tf_lriac.toml",
    "/work/grana_maxillo/UNetMerging/configs/finetune_tf_pharynx.toml",
    # "/work/grana_maxillo/UNetMerging/configs/pretrain.toml",
]

for config_path in configs_to_run:
    executor.update_parameters(name=config_path)
    job = executor.submit(run_job, config_path)
    print(f"Submitted job {job.job_id}")
