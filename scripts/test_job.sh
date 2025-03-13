#!/bin/bash

#SBATCH --job-name=TestTest # Name of your job
#SBATCH --output=outputs/test_job.out     # Name of the output file
#SBATCH --error=outputs/file-test_job.err # Name of the error file
#SBATCH --mem=300G               # Memory
#SBATCH --cpus-per-task=128   # CPUs per task
#SBATCH --gres=gpu:0           # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time

python3 Cleaning/testShipMove.py