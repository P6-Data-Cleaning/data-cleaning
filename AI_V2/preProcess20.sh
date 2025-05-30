#!/bin/bash

#SBATCH --job-name=PreProcc  # Name of your job
#SBATCH --output=preProcc%j.out     # Name of the output file
#SBATCH --error=preProcc%j.err      # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=500G               # Memory
#SBATCH --cpus-per-task=60      # CPUs per task
#SBATCH --gres=gpu:2            # Allocated GPUs
#SBATCH --time=72:00:00         # Maximum run time
#SBATCH --begin=now             # Start immediately

# Enable better CUDA error reporting
export CUDA_LAUNCH_BLOCKING=1

# Set the correct working directory
cd /home/project/P6-data-cleaning/sprint2/AI_V2

# Run inside the container, binding the current path
singularity exec --nv --bind $(pwd):/workspace containers/ai_container_l4.sif \
    bash -c "cd /workspace && python -u preproccess20.py"