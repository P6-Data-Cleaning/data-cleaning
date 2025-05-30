#!/bin/bash

#SBATCH --job-name=ClusterShit # Name of your job
#SBATCH --output=outputs/clusterRunner%j.out     # Name of the output file
#SBATCH --error=outputs/clusterRunner%j.err # Name of the error file
#SBATCH --open-mode=append       # Append output as it's produced
#SBATCH --mem=400G               # Memory
#SBATCH --cpus-per-task=128   # CPUs per task
#SBATCH --gres=gpu:8           # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time
#SBATCH --begin=now         # Start immediately

# Set Dask temporary directory to a local disk
export DASK_TEMPORARY_DIRECTORY=/ceph/project/P6-data-cleaning/tmp/dask_tmp

# Set worker memory limits to better manage shuffling
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.85
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.90
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.95

# Create the local temporary directory
mkdir -p $DASK_TEMPORARY_DIRECTORY

python3 Cleaning/clusterRunner3.py

# Clean up the temporary directory
rm -rf $DASK_TEMPORARY_DIRECTORY