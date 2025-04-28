#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH -C m16

conda activate base

python /share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/delta_approxmiation.py