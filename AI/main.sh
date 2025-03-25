#!/bin/bash

#SBATCH --job-name=AI-wuhuu      # Name of your job
#SBATCH --output=main5.out%j     # Name of the output file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --error=main5.err        # Name of the error file
#SBATCH --mem=200G               # Memory
#SBATCH --cpus-per-task=24       # CPUs per task
#SBATCH --gres=gpu:8             # Allocated GPUs
#SBATCH --time=12:00:00          # Maximum run time
#SBATCH --begin=now              # Start immediately

# Use a container that already has PyTorch installed
singularity exec --nv ai_container.sif python -u main.py