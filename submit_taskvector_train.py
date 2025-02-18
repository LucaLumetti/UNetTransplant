import os

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


def run_job(config_path, pretrain_path, classes_to_predict, postfix):
    import subprocess

    args_to_run = [
        "python",
        "main.py",
        "--experiment",
        "TaskVectorTrainExperiment",
        # "PretrainExperiment",
        "--config",
        config_path,
        "--expname",
        postfix,
        "--override",
        "DataConfig.DATASET_CLASSES = ['" + ",".join(classes_to_predict) + "']",
        f"BackboneConfig.PRETRAIN_CHECKPOINTS = {pretrain_path}",
        f"DataConfig.NUM_WORKERS = {cpus_per_task}",
    ]

    subprocess.run(args_to_run)


base_path = "/leonardo/home/userexternal/llumetti/projects/UNetMerging"
base_config = f"{base_path}/configs/leonardo/base.toml"

pretrains = [
    (
        "Stable1_yg0hh",
        f"/leonardo_scratch/large/userexternal/llumetti/output_UNetMergingcheckpoints/PretrainExperiment_yg0hhcbw__3ddo_n=2_lr=0.001_batch_size=4_dropout=0.5/checkpoint_50_2025-02-17 14:17:13.066854.pth",
    ),
]
task_vectors_classes = [
    ("Mandible", ["1"]),
    ("Skull", ["2"]),
    ("Canals", ["3-4"]),
    ("Pharynx", ["7"]),
    ("Implants", ["8,9,10"]),
    ("Teeth", ["11-18,21-28,31-38,41-48"]),
]

jobs = []

with executor.batch():
    for pretrain_name, pretrain_path in pretrains:
        for class_name, tv_class in task_vectors_classes:
            # executor.update_parameters(name=config_path)
            job = executor.submit(
                run_job,
                base_config,
                pretrain_path,
                tv_class,
                f"_{class_name}_{pretrain_name}",
            )
            # job = run_job(
            #     base_config, pretrain_path, tv_class, f"_{class_name}_{pretrain_name}"
            # )
            jobs.append(job)

        for i, (class_name1, tv_class1) in enumerate(task_vectors_classes):
            for j, (class_name2, tv_class2) in enumerate(task_vectors_classes):
                if i >= j:
                    continue
                # executor.update_parameters(name=config_path)
                job = executor.submit(
                    run_job,
                    base_config,
                    pretrain_path,
                    tv_class1 + tv_class2,
                    f"_{class_name1}+{class_name2}_{pretrain_name}",
                )
                # job = run_job(base_config, pretrain_path, tv_class1 + tv_class2)
                jobs.append(job)

print(f"Submitted {len(jobs)} jobs")
print(f"Logs will be available in {executor.folder}")
print(jobs)
