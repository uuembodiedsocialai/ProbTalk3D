#!/bin/bash
#SBATCH --job-name=vqvae_pred
#SBATCH --time=3-00:00:00
#SBATCH --priority=TOP

srun python3.9 train_all.py \
    experiment=vqvae_pred \
    state=new \
    data=mead_pred \
    model=model_vqvae_pred \
    model.folder_prior=outputs/MEAD/vqvae_prior/XXX \
    model.version_prior=0 \

