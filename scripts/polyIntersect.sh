#!/bin/bash

#SBATCH --job-name=PolyIntersect # Name of your job
#SBATCH --output=outputs/polyIntersect%j.out     # Name of the output file
#SBATCH --error=outputs/polyIntersect%j.err # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=300G               # Memory
#SBATCH --cpus-per-task=112   # CPUs per task
#SBATCH --gres=gpu:0           # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

python3 Cleaning/poly_intersect.py