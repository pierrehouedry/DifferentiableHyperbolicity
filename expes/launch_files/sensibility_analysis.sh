#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH -C m24

conda activate hyperenv_py311

python ../grid_search.py