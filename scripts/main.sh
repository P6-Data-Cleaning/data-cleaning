#!/bin/bash

#SBATCH --job-name=TidyTidy # Name of your job
#SBATCH --output=outputs/main.out     # Name of the output file
#SBATCH --error=outputs/main.err # Name of the error file
#SBATCH --mem=400G               # Memory
#SBATCH --cpus-per-task=128   # CPUs per task
#SBATCH --gres=gpu:2           # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

python3 Cleaning/main.py