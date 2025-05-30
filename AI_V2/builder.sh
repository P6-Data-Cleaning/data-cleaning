#!/bin/bash

#SBATCH --job-name=SingBuild
#SBATCH --output=sing_build_%j.out
#SBATCH --error=sing_build_%j.err
#SBATCH --cpus-per-task=20           # 8 CPUs for faster build (adjust as needed)
#SBATCH --mem=100G                   # 32GB RAM for build (adjust as needed)
#SBATCH --time=12:00:00              # 12 hours max

# Build the container using srun and fakeroot
singularity build --fakeroot ai_container_l4_v2.sif ai_container_l4.def