#!/bin/bash

#SBATCH --job-name=Plotty # Name of your job
#SBATCH --output=outputs/plot.out     # Name of the output file
#SBATCH --error=outputs/plot.err # Name of the error file
#SBATCH --mem=300G               # Memory
#SBATCH --cpus-per-task=128   # CPUs per task
#SBATCH --gres=gpu:2           # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

python3 Cleaning/Plot.py outputs/cleaned_data.csv