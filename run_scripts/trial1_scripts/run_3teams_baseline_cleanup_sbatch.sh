#!/bin/bash
#SBATCH --job-name=3t-tr1-clean

#SBATCH --partition=p100,t4v1,t4v2,rtx6000

#SBATCH --open-mode=append

#SBATCH --gres=gpu:1

#SBATCH --qos=normal

#SBATCH --cpus-per-task=15

#SBATCH --mem=40G

#SBATCH -x gpu131

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# prepare your environment here
module load tensorflow2.5-gpu-cuda11.0-python3.6 

# put your command here
. ../../venv/bin/activate
bash run_3teams_baseline_cleanup.sh
