# %%
from hyperbolicity.tree_fitting_methods.gromov import gromov_tree
import itertools
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from hyperbolicity.utils import construct_weighted_matrix
import re
# %%
path_results = './results_expes/expe_celegan_sensitivity/'


def merge_csv_files(directory):
    all_files = []
    print('yo')
    # Traverse directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('hyperbolicity_results.csv'):
                all_files.append(os.path.join(root, file))

    df_list = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df


# %%
df = merge_csv_files(path_results)
# %%
# df.loc[df['mean_optim_distortion'].idxmin()]
print(df.shape)
# %%
cols = ['std_optim_distortion', 'mean_optim_distortion',
        'scale_delta', 'distance_reg']
small_df = df[(df['learning_rate'] == 0.1) & (
    df['batch_size'] == 32) & (df['n_batches'] == 100)][cols]
print(small_df.shape)

# %%
cmap = plt.cm.get_cmap('tab10')
fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# Subfigure 1: distance_reg vs mean_optim_distortion for all scale_delta
unique_scales = small_df["scale_delta"].unique()
for k, scale in enumerate(unique_scales):
    subset = small_df[small_df["scale_delta"]
                      == scale].sort_values("distance_reg")
    axs[0].plot(
        subset["distance_reg"],
        subset["mean_optim_distortion"],
        label=f"scale_delta={scale}",
        marker='o',
        lw=2,
        color=cmap(k)
    )
    axs[0].fill_between(
        subset["distance_reg"],
        subset["mean_optim_distortion"] - subset["std_optim_distortion"],
        subset["mean_optim_distortion"] + subset["std_optim_distortion"],
        color=cmap(k),
        alpha=0.2
    )

axs[0].set_xlabel("distance_reg")
axs[0].set_ylabel("mean_optim_distortion")
axs[0].set_xscale('log')
axs[0].set_title("Distance_reg vs Mean_optim_distortion")
axs[0].legend()
axs[0].grid()

# Subfigure 2: scale_delta vs mean_optim_distortion for all distance_reg
# unique_distances = small_df["distance_reg"].unique()
unique_distances = [0.1]
for i, distance in enumerate(unique_distances):
    subset = small_df[small_df["distance_reg"]
                      == distance].sort_values("scale_delta")
    axs[1].plot(
        subset["scale_delta"],
        subset["mean_optim_distortion"],
        label=f"distance_reg={distance}",
        color=cmap(i),
        lw=2,
        marker='o'
    )
    axs[1].fill_between(
        subset["scale_delta"],
        subset["mean_optim_distortion"] - subset["std_optim_distortion"],
        subset["mean_optim_distortion"] + subset["std_optim_distortion"],
        color=cmap(i),
        alpha=0.2
    )

axs[1].set_xlabel("scale_delta")
axs[1].set_ylabel("mean_optim_distortion")
axs[1].set_xscale('log')
axs[1].set_title("Scale_delta vs Mean_optim_distortion")
axs[1].legend()
axs[1].grid()


# %%
fs = 15
cmap = plt.cm.get_cmap('tab10')
fig, axs = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

unique_distances = [0.1]
for i, distance in enumerate(unique_distances):
    subset = small_df[small_df["distance_reg"]  # pour en prendre qu'un unique car il y a des doublons dans les expés
                      == distance].sort_values("scale_delta").groupby('scale_delta', as_index=False).mean()[['scale_delta', 'std_optim_distortion', 'mean_optim_distortion']]

# aggregated_data = small_df
    axs.plot(
        subset["scale_delta"],
        subset["mean_optim_distortion"],
        label=f"Reg. $\\lambda$ = {distance}",
        color=cmap(i),
        lw=2,
        marker='o'
    )
    axs.fill_between(
        subset["scale_delta"],
        subset["mean_optim_distortion"] - subset["std_optim_distortion"],
        subset["mean_optim_distortion"] + subset["std_optim_distortion"],
        color=cmap(i),
        alpha=0.2
    )

axs.set_xlabel("Scale of $\\delta$", fontsize=fs)
axs.set_ylabel("$\ell_\\infty$ error", fontsize=fs)
axs.set_xscale('log')
axs.set_title("Scale vs perf", fontsize=fs)
axs.legend(fontsize=fs - 2)
axs.grid()
plt.savefig('./results_expes/scale_vs_perf.pdf')
# %%
