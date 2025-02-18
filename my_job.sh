#!/bin/bash

#SBATCH --job-name=my_job  # Name of your job
#SBATCH --output=outputs/my_job.out     # Name of the output file
#SBATCH --error=outputs/file-my_job.err # Name of the error file
#SBATCH --mem=24G               # Memory
#SBATCH --cpus-per-task=15      # CPUs per task
#SBATCH --gres=gpu:8            # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time

python3 main.py