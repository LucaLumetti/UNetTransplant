#!/bin/bash -l

#SBATCH --job-name=Task
#SBATCH --partition=all_usr_prod  
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --time=12:00:00
#SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --account=debiasing

. /usr/local/anaconda3/etc/profile.d/conda.sh

srun python test_merge.py