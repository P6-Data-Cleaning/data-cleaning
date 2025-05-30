#!/bin/bash

#SBATCH --job-name=TidyTidy # Name of your job
#SBATCH --output=outputs/mainfigur%j.out     # Name of the output file
#SBATCH --error=outputs/mainfigur%j.err # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=200G               # Memory
#SBATCH --cpus-per-task=72   # CPUs per task
#SBATCH --gres=gpu:0           # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

# Set Dask temporary directory
export DASK_TEMPORARY_DIRECTORY=/ceph/project/P6-data-cleaning/tmp

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

python3 Cleaning/mainFigur.py