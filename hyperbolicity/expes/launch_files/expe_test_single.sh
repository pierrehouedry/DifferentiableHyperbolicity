#!/bin/bash
#SBATCH --job-name=my_job           # Optional: name of your job
#SBATCH --gres gpu:2
#SBATCH --ntasks=1                  # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=4      # Number of CPU cores per task
#SBATCH --mem=80G                  # Total memory required (e.g., 16 GB)
#SBATCH --time=48:00:00             # Optional: time limit (HH:MM:SS)
#SBATCH --output=job_%j.out         # Standard output file
#SBATCH --error=job_%j.err          # Standard error file


source /share/home/houedry/anaconda3/bin/activate
conda activate base

export OMP_NUM_THREADS=1
mkdir -p /share/home/houedry/projects/hyperbolic/DifferentiableHyperbolicity/hyperbolicity/results_expes/expe_test/
python /share/home/houedry/projects/hyperbolic/DifferentiableHyperbolicity/hyperbolicity/expes/launch_distance_hyperbolicity_learning.py -r /share/home/houedry/projects/hyperbolic/DifferentiableHyperbolicity/hyperbolicity/results_expes/expe_test/ -dp /share/home/houedry/projects/hyperbolic/DifferentiableHyperbolicity/hyperbolicity/datasets/ -ds airport -lr 0.001 -reg 0.5 -ssp 10000 -ssd 10000 -sssm 10000 -ep 500 -bs 32 -rn 0