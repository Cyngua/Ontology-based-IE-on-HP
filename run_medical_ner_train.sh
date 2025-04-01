#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=6:00:00
#SBATCH --output=logs/medical_ner_train_%j.out  # Save standard output to log file
#SBATCH --error=logs/medical_ner_train_%j.err   # Save error output to log file

# Run the Python script
python models/medical_ner_train.py

# Print job finish time
echo "Job finished at $(date)"