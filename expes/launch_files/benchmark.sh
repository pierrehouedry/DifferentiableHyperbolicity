#!/bin/bash
#SBATCH -p longrun
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH --mem=12G

# Ensure conda can be activated in non-interactive shells
eval "$(conda shell.bash hook)" || source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate hyperenv_py311

# Use an absolute project root because SLURM may run a copied script from /var/spool
PROJECT_ROOT="" #to be filled

cd "$PROJECT_ROOT"

python "expes/benchmark_map_copy.py" \
  --dataset celegan \
  --optimized_dataset "results_expes/D_celegan2_2025-07-29_14-25-22/lr_0.1_dr_0.1_sd_10.0_epoch_1000_batch_32_n_batches_100.pt" \
  --num_expe 100 \
  --methods gromov \
  --output_dir "expes/results_expes"