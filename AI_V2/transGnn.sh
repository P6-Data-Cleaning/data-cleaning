#!/bin/bash

#SBATCH --job-name=AI-Train  # Name of your job
#SBATCH --output=transGnn%j.out     # Name of the output file
#SBATCH --error=transGnn%j.err      # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=50G               # Memory
#SBATCH --cpus-per-task=24      # CPUs per task
#SBATCH --gres=gpu:8            # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now             # Start immediately

# Use a container that already has PyTorch installed
singularity exec --nv ai_container.sif python -u transGnn.py