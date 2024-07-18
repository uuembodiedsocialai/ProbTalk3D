#!/bin/bash
#SBATCH --job-name=vae_prior
#SBATCH --time=3-00:00:00
#SBATCH --priority=TOP

srun python3.9 train_all.py \
    experiment=vae_prior \
    state=new \
    data=mead_prior \
    model=model_vae_prior \


