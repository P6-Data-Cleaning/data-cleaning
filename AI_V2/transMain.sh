#!/bin/bash
#SBATCH --job-name=AI-TransJake
#SBATCH --output=transMainJake%j.out
#SBATCH --error=transMainJake%j.err
#SBATCH --open-mode=append
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --begin=now

export CUDA_LAUNCH_BLOCKING=1

cd /home/project/P6-data-cleaning/sprint2/AI_V2

# Expand $PWD now—no parentheses
export BIND_PATH="$PWD:/workspace"

singularity exec --nv \
  --bind "${BIND_PATH}" \
  /home/project/pytorch_24.02.sif \
    bash -c "cd /workspace && pip install --user linformer torch-geometric==2.4.0 && python -u /workspace/transMain.py"
