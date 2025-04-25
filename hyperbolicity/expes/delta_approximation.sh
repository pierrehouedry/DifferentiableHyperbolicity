#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 32G
#SBATCH -c 1
#SBATCH -w sn4

conda activate base

python /share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/delta_approxmiation.py