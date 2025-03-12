#!/bin/bash

#SBATCH --job-name=Extractor # Name of your job
#SBATCH --output=outputs/extractor.out     # Name of the output file
#SBATCH --error=outputs/extractor.err # Name of the error file
#SBATCH --mem=100G               # Memory
#SBATCH --cpus-per-task=10   # CPUs per task
#SBATCH --gres=gpu:2           # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

python3 extractor.py outputs/cleaned_data27.csv 246539000