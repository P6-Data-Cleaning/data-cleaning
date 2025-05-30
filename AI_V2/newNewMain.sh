#!/bin/bash

#SBATCH --job-name=AI-Train  # Name of your job
#SBATCH --output=newMain%j.out     # Name of the output file
#SBATCH --error=newMain%j.err      # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=400G               # Memory
#SBATCH --cpus-per-task=2      # CPUs per task
#SBATCH --gres=gpu:16            # Allocated GPUs
#SBATCH --time=72:00:00         # Maximum run time
#SBATCH --begin=now             # Start immediately

# Set the correct working directory
cd /home/project/P6-data-cleaning/sprint2/AI_V2

# Run inside the container, binding the current path
singularity exec --nv --bind $(pwd):/workspace ai_container.sif \
    bash -c "cd /workspace && python -u newNewMain.py"