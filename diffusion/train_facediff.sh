#!/bin/bash
#SBATCH --job-name=facediff
#SBATCH --time=5-00:00:00
#SBATCH --priority=TOP

srun python3.9 main.py