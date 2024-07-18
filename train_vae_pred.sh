#!/bin/bash
#SBATCH --job-name=vae_pred
#SBATCH --time=3-00:00:00
#SBATCH --priority=TOP

srun python3.9 train_all.py \
    experiment=vae_pred \
    state=new \
    data=mead_pred \
    model=model_vae_pred \
    model.folder_prior=outputs/MEAD/vae_prior/XXX \
    model.version_prior=0 \