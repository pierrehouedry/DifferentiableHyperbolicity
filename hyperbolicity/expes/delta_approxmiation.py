import socket
from time import time
import argparse
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm
from hyperbolicity.delta import compute_hyperbolicity_batch, compute_hyperbolicity, compute_delta_from_distances_batched
from hyperbolicity.utils import floyd_warshall, soft_max, construct_weighted_matrix, make_batches
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib


base_path = '/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/datasets'

c_elegan = 'D_csphd.pkl'
c_elegan_path = os.path.join(base_path, c_elegan)
with open(c_elegan_path, 'rb') as f:
    distances = pickle.load(f)
distances = torch.tensor(distances).to('cuda').type(torch.float32)
true_delta = 6.5

delta_scale = [0.01, 0.1, 1, 10, 100, 1000, 10000]
n_batches = 50
batch_size = [4, 8, 16, 32, 48]
n_exp = 5
results = {b: [] for b in batch_size}

for b in batch_size:
    for i in delta_scale:
        exp = []
        for k in range(n_exp):
            M_batch = make_batches(distances, b, n_batches)
            delta_soft = soft_max(compute_delta_from_distances_batched(M_batch, i), i)
            exp.append(delta_soft.item())
        # Save mean for each (batch_size, delta_scale)
        results[b].append(np.mean(exp))

colors = ['#648FFF', '#56B4E9', '#DC267F', '#FE6100', '#FFB000', '#785EF0', '#009E73']

plt.figure(figsize=(10, 6))
for idx, b in enumerate(batch_size):
    plt.plot(delta_scale, results[b], label=f'Batch size {b}', marker='o', color=colors[idx])

# Add horizontal line for true delta
plt.axhline(y=true_delta, color='gray', linewidth=2, label=f'True δ = {true_delta}')



# Add horizontal line for true delta
plt.axhline(y=true_delta, color='gray', linewidth=2, label='True δ = 6.5')

# Set log-log scale for both axes
plt.xscale('log')
plt.yscale('log')


""" delta_scale = 1000
n_batches = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
batch_size = [4, 8, 16, 32, 48]
n_exp = 5
results = {i: [] for i in batch_size}

for i in batch_size:
    for j in n_batches:
        exp = []
        for k in range(n_exp):
            M_batch = make_batches(distances, i, j)
            delta_soft = soft_max(compute_delta_from_distances_batched(M_batch, 100), 100)
            exp.append(delta_soft.item())
        results[i].append(np.mean(exp))
        print(f'n_batches {j} done !')
    print(f'batch_size {i} done !')

colors = ['#648FFF', '#56B4E9', '#DC267F', '#FE6100', '#FFB000']
plt.figure(figsize=(10, 6))
for idx, i in enumerate(batch_size):
    plt.plot(n_batches, results[i], label=f'Batch size {i}', marker='o', color=colors[idx])

# Add horizontal line for true delta
plt.axhline(y=true_delta, color='gray', linewidth=2, label='True δ = 6.5')

# Set log scale for x-axis
plt.xscale('log') """

# Labels and legend
plt.xlabel("Number of Batches (log scale)")
plt.ylabel("Mean Hyperbolicity (δ-soft)")
plt.title("Hyperbolicity vs Number of Batches for Different Batch Sizes (Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()
tikzplotlib.save("/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/hyperbolicity_plot.tex")
plt.savefig("/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/hyperbolicity_plot.png", dpi=300)
plt.show()
print(results[32])