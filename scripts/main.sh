#!/bin/bash

#SBATCH --job-name=TidyTidy # Name of your job
#SBATCH --output=outputs/main%j.out     # Name of the output file
#SBATCH --error=outputs/main.err # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=400G               # Memory
#SBATCH --cpus-per-task=120   # CPUs per task
#SBATCH --gres=gpu:2           # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

python3 -u Cleaning/main.py