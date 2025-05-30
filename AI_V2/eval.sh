#!/bin/bash

#SBATCH --job-name=AI-Trans  # Name of your job
#SBATCH --output=newNewTransMain%j.out     # Name of the output file
#SBATCH --error=newNewTransMain%j.err      # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=700G               # Memory
#SBATCH --cpus-per-task=48      # CPUs per task
#SBATCH --gres=gpu:4            # Allocated GPUs
#SBATCH --time=72:00:00         # Maximum run time
#SBATCH --begin=now             # Start immediately

# Enable better CUDA error reporting
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

# Set the correct working directory
cd /home/project/P6-data-cleaning/sprint2/AI_V2

# Run inside the container, binding the current path
singularity exec --nv --bind $(pwd):/workspace containers/ai_container_l4.sif \
    bash -c "cd /workspace && torchrun --standalone --nnodes=1 --nproc_per_node=4 eval.py"
