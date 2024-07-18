#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --time=1-00:00:00
#SBATCH --priority=TOP


srun python3.9 evaluation_facediff.py --save_path "../model_weights/FaceDiffuser" --max_epoch 50
