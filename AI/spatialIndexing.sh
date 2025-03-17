#!/bin/bash

#SBATCH --job-name=Spatial  # Name of your job
#SBATCH --output=spatialIndexing.out     # Name of the output file
#SBATCH --error=spatialIndexing.err      # Name of the error file (fixed typo)
#SBATCH --mem=24G               # Memory
#SBATCH --cpus-per-task=15      # CPUs per task
#SBATCH --gres=gpu:2            # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time
#SBATCH --begin=now             # Start immediately

# Use a container that already has PyTorch installed
singularity exec --nv /ceph/container/pytorch/pytorch_25.01.sif python spatialIndexing.py