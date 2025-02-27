#!/bin/bash -l
#SBATCH --job-name=Naaamo
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=1   
#SBATCH --time=00:30:00
#SBATCH --account=IscrB_FeeCO

python3 wandb-sync.py -w 24
#sbatch --begin=now+30minutes wandb-sync-sbatch.sh

