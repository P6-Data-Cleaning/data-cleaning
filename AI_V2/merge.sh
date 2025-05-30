#!/bin/bash

#SBATCH --job-name=Merge  # Name of your job
#SBATCH --output=merge%j.out     # Name of the output file
#SBATCH --error=merge%j.err      # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=500G               # Memory
#SBATCH --cpus-per-task=150      # CPUs per task
#SBATCH --gres=gpu:2            # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now             # Start immediately


# Run inside the container, binding the current path
python3 -u merge.py