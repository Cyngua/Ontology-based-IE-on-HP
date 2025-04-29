#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=6:00:00
#SBATCH --output=logs/bertcrf_%j.out  # Save standard output to log file
#SBATCH --error=logs/bertcrf_%j.err   # Save error output to log file

# Run the Python script
python models/bertcrf.py

# Print job finish time
echo "Job finished at $(date)"