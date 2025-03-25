#!/bin/bash

#SBATCH --job-name=TidyTidy # Name of your job
#SBATCH --output=outputs/main%j.out     # Name of the output file
#SBATCH --error=outputs/main%j.err # Name of the error file
#SBATCH --mem=400G               # Memory
#SBATCH --cpus-per-task=116   # CPUs per task
#SBATCH --gres=gpu:0           # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

python3 Cleaning/main.py