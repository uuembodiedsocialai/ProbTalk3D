#!/bin/bash
#SBATCH --job-name=vqvae_prior
#SBATCH --time=1-00:00:00
#SBATCH --priority=TOP

srun python3.9 train_all.py \
    experiment=vqvae_prior \
    state=new \
    data=mead_prior \
    model=model_vqvae_prior \