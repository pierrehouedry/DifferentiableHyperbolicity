#!/bin/bash
#SBATCH -p longrun
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH --mem=12G

conda activate hyperenv_py311
python ./hyperbolicity/expes/benchmark.py --dataset zeisel --num_expe 100 --methods treerep hcc gromov