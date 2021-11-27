#!/bin/bash
#SBATCH --job-name=collective

#SBATCH --partition=p100,t4v1,t4v2,rtx6000

#SBATCH --open-mode=append

#SBATCH --gres=gpu:1

#SBATCH --qos=normal

#SBATCH --cpus-per-task=15

#SBATCH --mem-per-cpu=1G

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# prepare your environment here
module load tensorflow2.5-gpu-cuda11.0-python3.6 

# put your command here
. ../venv/bin/activate
bash run_baseline_cleanup.sh
