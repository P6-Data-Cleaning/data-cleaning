#!/bin/bash

#SBATCH --job-name=ShipSplit # Name of your job
#SBATCH --output=outputs/split.out     # Name of the output file
#SBATCH --error=outputs/file-split.err # Name of the error file
#SBATCH --mem=300G               # Memory
#SBATCH --cpus-per-task=120   # CPUs per task
#SBATCH --gres=gpu:2           # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time

python3 ShipSplit.py