# %%
import pickle
from hyperbolicity.delta import compute_hyperbolicity
from torch_geometric.datasets import Planetoid
import networkx as nx
from hyperbolicity.expes.launch_distance_hyperbolicity_learning import load_data
import torch
from hyperbolicity.utils import create_log_dir, setup_logger, str2bool, soft_max, datasp, construct_weighted_matrix
from hyperbolicity.delta import make_batches, compute_hyperbolicity_batch
import torch.optim as optim
from networkx import gnp_random_graph, shortest_path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
cmap = plt.cm.get_cmap('tab10')


def generate_random_graph(n, p):
    G = gnp_random_graph(n, p=p)
    distances = torch.tensor(nx.floyd_warshall_numpy(G)).float()
    num_nodes = distances.shape[0]

    edges = torch.triu_indices(num_nodes, num_nodes, offset=1)
    upper_adjency = torch.triu(distances, diagonal=1)
    weights_opt = upper_adjency[upper_adjency != 0].float()
    return weights_opt, edges, distances


def matrix_from_weights(w, num_nodes, edges):
    weighted_matrix = torch.full((num_nodes, num_nodes), float(0))
    weighted_matrix[edges[0, :], edges[1, :]] = w
    W = weighted_matrix+weighted_matrix.t()
    return W


def allsp_baseline(w, num_nodes, edges, scale):
    dist = matrix_from_weights(w, num_nodes, edges)
    for k in range(num_nodes):
        sum_costs = dist[k, :, None] + dist[None, :, k]
        dist = soft_max(torch.stack([sum_costs, dist], dim=-1), -scale)
    return dist


def allsp_baseline_no_grad(w, num_nodes, edges, scale):
    dist = matrix_from_weights(w, num_nodes, edges)
    for k in range(num_nodes):
        with torch.no_grad():
            sum_costs = dist[k, :, None] + dist[None, :, k]
        dist = soft_max(torch.stack([sum_costs, dist], dim=-1), -scale)
    return dist


def allsp_rowwise(w, num_nodes, edges, scale):
    # Initialize the distance matrix
    dist = matrix_from_weights(w, num_nodes, edges)

    for k in range(num_nodes):
        # Create an updated distance matrix
        updated_dist = torch.empty_like(dist)

        for i in range(num_nodes):
            # Compute `sum_costs` row-wise for the current row `i`
            sum_costs = dist[i, k] + dist[k, :]  # Row of `dist[i, :]` + column of `dist[:, k]`
            combined = torch.stack([sum_costs, dist[i, :]], dim=-1)  # Stack the costs for softmax
            updated_dist[i, :] = soft_max(combined, -scale)  # Update the row using softmax

        dist = updated_dist  # Update the matrix after processing all rows

    return dist


def allsp_naive(w, num_nodes, edges, scale):  # memorey efficient but long
    dist = matrix_from_weights(w, num_nodes, edges)
    for k in range(num_nodes):
        updated_dist = dist.clone()
        for i in range(num_nodes):
            for j in range(num_nodes):
                sum_costs = dist[i, k] + dist[k, j]
                Mij = dist[i, j]
                updated_dist[i, j] = soft_max(torch.tensor([sum_costs, Mij]), -scale)
        dist = updated_dist
    return dist


def allsp_wrong(w, num_nodes, edges, scale):  # naive implem but wrong
    dist = matrix_from_weights(w, num_nodes, edges)
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                sum_costs = dist[i, k]+dist[k, j]
                dist[i, j] = soft_max(torch.tensor([sum_costs, dist[i, j]]), -scale)
    return dist


@torch.jit.script
def allsp_naive_jit(w: torch.Tensor, num_nodes: int, edges: torch.Tensor, scale: float) -> torch.Tensor:
    # Initialize the distance matrix
    dist = torch.zeros((num_nodes, num_nodes), dtype=w.dtype, device=w.device)
    dist[edges[0, :], edges[1, :]] = w
    dist = dist + dist.t()  # Ensure symmetry

    for k in range(num_nodes):
        updated_dist = dist.clone()
        for i in range(num_nodes):
            for j in range(num_nodes):
                sum_costs = dist[i, k] + dist[k, j]
                Mij = dist[i, j]
                combined = torch.stack([sum_costs, Mij])  # Replace `torch.tensor()` with `torch.stack`
                updated_dist[i, j] = (-1.0 / scale) * torch.logsumexp(-scale * combined, dim=0)
        dist = updated_dist

    return dist


@torch.jit.script
def allsp_rowwise_jit(w: torch.Tensor, num_nodes: int, edges: torch.Tensor, scale: float) -> torch.Tensor:
    # Initialize the distance matrix
    dist = torch.zeros((num_nodes, num_nodes), dtype=w.dtype, device=w.device)
    dist[edges[0, :], edges[1, :]] = w
    dist = dist + dist.t()  # Ensure symmetry

    for k in range(num_nodes):
        # Clone the distance matrix to store updates
        updated_dist = torch.empty_like(dist)

        for i in range(num_nodes):
            # Compute row-wise `sum_costs` and update `dist`
            sum_costs = dist[i, k] + dist[k, :]
            combined = torch.stack([sum_costs, dist[i, :]], dim=-1)
            updated_dist[i, :] = (-1.0/scale) * torch.logsumexp(-scale * combined, dim=-1)

        dist = updated_dist  # Update the distance matrix for the next iteration

    return dist


def sample_batch_indices(N: int, size_batches: int = 32, nb_batches: int = 32, device: str = 'cpu') -> list[torch.Tensor]:
    """
    Randomly samples node indices to create batches.
    """
    all_indices = torch.arange(N, device=device)
    batches = []
    for _ in range(nb_batches):
        permuted = all_indices[torch.randperm(N)]
        selected = permuted[:size_batches]
        batches.append(selected)

    return batches


def batched_datasp_submatrices(w: torch.Tensor,
                               num_nodes: int,
                               edges: torch.Tensor,
                               batch_indices: list[torch.Tensor],
                               beta: float = 1.0) -> torch.Tensor:
    """
    Computes batched softened shortest-path submatrices using a vectorized update
    for the rows required by the batches.
    """

    all_indices = torch.cat(batch_indices)
    unique_indices, _ = torch.unique(all_indices, return_inverse=True)

    weighted_matrix = matrix_from_weights(w, num_nodes, edges)

    for k in range(num_nodes):
        via_k = weighted_matrix[unique_indices, k].unsqueeze(1) + weighted_matrix[k, :].unsqueeze(0)
        current = weighted_matrix[unique_indices, :]
        stacked = torch.stack([via_k, current], dim=1)  # (len(unique_indices), 2, num_nodes)
        weighted_matrix[unique_indices, :] = soft_max(stacked, -beta, dim=1)

    row_cache = weighted_matrix[unique_indices]  # (num_unique, num_nodes)

    batched_distances = []
    index_map = {int(idx): i for i, idx in enumerate(unique_indices)}
    for indices in batch_indices:
        row_idxs = [index_map[int(i)] for i in indices]
        batch_rows = row_cache[row_idxs]         # (size_batch, num_nodes)
        submatrix = batch_rows[:, indices]
        batched_distances.append(submatrix)

    return torch.stack(batched_distances)  # (nb_batches, size_batch, size_batch)

# %%

# data_path = '../datasets/'
# distances = load_data('cora', data_path)
# print(distances.shape)
# num_nodes = distances.shape[0]
# edges = torch.triu_indices(num_nodes, num_nodes, offset=1)
# upper_adjency = torch.triu(distances, diagonal=1)
# weights_opt = upper_adjency[upper_adjency != 0].float().requires_grad_(True)


num_nodes = 2708
weights_opt, edges, distances = generate_random_graph(num_nodes, p=0.6)
weights_opt.requires_grad_(True)

scale_sp = 1000
scale_delta = 1000
scale_soft_max = 1000
batch_size = 32
nb_batches = 10
distance_reg = 1e-1
# %%
scale = 10
# method = allsp_baseline_no_grad
# method = allsp_rowwise
method = allsp_baseline


def loss_fn(w):
    dist = method(w, num_nodes, edges, scale)

    # M_batch = make_batches(dist, batch_size, nb_batches)
    # delta = soft_max(compute_hyperbolicity_batch(M_batch, scale=scale_delta), scale=scale_soft_max)
    # return (distances-sp_matrix).pow(2).sum()
    return dist.sum()


n_batches = 150
batch_size = 32
# def loss_fn(w):
#     batch_indices = sample_batch_indices(num_nodes, batch_size, n_batches)
#     M_batch = batched_datasp_submatrices(w, num_nodes, edges, batch_indices, beta=scale_sp)
#     delta = soft_max(compute_hyperbolicity_batch(M_batch, scale=scale_delta), scale=scale_soft_max)
#     true_batches = torch.stack([distances[idx[:, None], idx] for idx in batch_indices])
#     err = (M_batch- true_batches).pow(2).sum()

#     return delta + distance_reg*err

imax = 2
i = 0
learning_rate = 1e-2
optimizer = optim.Adam([weights_opt], lr=learning_rate)
while i < imax:
    optimizer.zero_grad()
    w = weights_opt
    # with torch.no_grad():
    # w = weights_opt.detach()  # pete pas
    #  # pete
    loss = loss_fn(w)
    loss.backward()
    optimizer.step()
    i += 1
    print('{} done '.format(i))

# %%
weights_opt.grad.shape
# %%
