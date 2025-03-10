#!/bin/bash

#SBATCH --job-name=TidyTidy # Name of your job
#SBATCH --output=outputs/my_job.out     # Name of the output file
#SBATCH --error=outputs/file-my_job.err # Name of the error file
#SBATCH --mem=400G               # Memory
#SBATCH --cpus-per-task=128   # CPUs per task
#SBATCH --gres=gpu:2           # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

python3 main.py outputs/cleaned_data.csv