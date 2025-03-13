#!/bin/bash

#SBATCH --job-name=Reducer  # Name of your job
#SBATCH --output=outputs/reducer.out     # Name of the output file
#SBATCH --error=outputs/reducer.err # Name of the error file
#SBATCH --mem=200G               # Memory
#SBATCH --cpus-per-task=128      # CPUs per task
#SBATCH --gres=gpu:4            # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time

python3 Cleaning/trajectory_reducer.py outputs/csv/246539000.csv