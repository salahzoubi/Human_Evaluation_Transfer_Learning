#!/bin/bash
# SBATCH -c 24  # Number of Cores per Task
# SBATCH --mem=40G  # Requested Memory
# SBATCH -p gpu-long  # Partition
# SBATCH -G 8  # Number of GPUs
# SBATCH -t 36:00:00  # Job time limit
# SBATCH -o slurm-%j.out  # %j = job ID