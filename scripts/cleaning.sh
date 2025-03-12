#!/bin/bash

#SBATCH --job-name=Cleaning  # Name of your job
#SBATCH --output=outputs/my_job.out     # Name of the output file
#SBATCH --error=outputs/file-my_job.err # Name of the error file
#SBATCH --mem=200G               # Memory
#SBATCH --cpus-per-task=128      # CPUs per task
#SBATCH --gres=gpu:4            # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time

python3 cleaning.py aisdk-2025-02-14.csv