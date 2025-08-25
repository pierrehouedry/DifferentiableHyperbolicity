#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH -p longrun
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH --mem=12G

eval "$(conda shell.bash hook)" || source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate hyperenv_py311

PROJECT_ROOT="/share/home/houedry/projects/DifferentiableHyperbolicity"

cd "$PROJECT_ROOT"

python "expes/grid_search.py" \
  --config-path "expes/configs/grid_search.yaml"