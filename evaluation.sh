#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --time=1-00:00:00
#SBATCH --priority=TOP

srun python3.9 \
    evaluation.py \
    folder=model_weights/ProbTalk3D/stage_2 \
    number_of_samples=10 \

srun python3.9 \
    evaluation.py \
    folder=model_weights/VAE_variant/stage_2 \
    number_of_samples=10 \
