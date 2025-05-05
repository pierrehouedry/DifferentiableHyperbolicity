import socket
from time import time
import argparse
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm
from hyperbolicity.delta import compute_hyperbolicity
from hyperbolicity.utils import soft_max, floyd_warshall, soft_max, construct_weighted_matrix, make_batches
import numpy as np
import time
import ot

def train_distance_matrix_GW(distances: torch.Tensor,
                             num_leaves: int,
                             scale_delta: float,
                             distance_reg: float,
                             num_epochs: int,
                             learning_rate: float,
                             verbose: bool,
                             gpu: bool):

    if gpu:
        distances = distances.to('cuda')

    edges = torch.triu_indices(num_leaves, num_leaves, offset=1)
    # init distances with mean distances from source distances
    distances_init =torch.ones((num_leaves, num_leaves), dtype=torch.float32, device=distances.device)*distances.mean()
    upper_adjency = torch.triu(distances_init, diagonal=1).type(torch.float32)
    weights_opt = upper_adjency[upper_adjency != 0].requires_grad_(True)
    optimizer = optim.Adam([weights_opt], lr=learning_rate)

    losses = []
    deltas = []
    errors = []

    def projection(weight, num_nodes, edges):
        update_dist = construct_weighted_matrix(weight, num_nodes, edges)
        update_dist = floyd_warshall(update_dist)

        return update_dist[edges[0], edges[1]]

    def loss_fn(w):
        update_dist = construct_weighted_matrix(w, num_leaves, edges)
        # compute full hyperbolicity
        delta = compute_hyperbolicity(update_dist, scale=scale_delta)
        # compute GW loss (thanks POT)
        err = ot.gromov_wasserstein2(distances,update_dist,loss_fun='square_loss', max_iter=1000, log=False, verbose=False)

        return delta + distance_reg*err, delta, err

    patience = 50
    best_loss = float('inf')
    best_weights = weights_opt.detach().clone().cpu()
    patience_counter = 0
    start = time.time()
    with tqdm(range(num_epochs), desc="Training Weights", disable=not verbose, leave="False") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            loss, delta, err = loss_fn(weights_opt)

            pbar.set_description(f"loss = {loss.item():.5f}, delta = {delta:.5f}, error = {err:.5f}")
            losses.append(loss.item())
            deltas.append(delta.item())
            errors.append(err.item())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                weights_opt.data[weights_opt.data<0] = 0
                weights_opt.data = projection(weights_opt, num_leaves, edges)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = weights_opt.detach().clone().cpu()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                pbar.set_description("Early stopping triggered")
                break
    end = time.time()

    return best_weights, losses, deltas, errors, end-start
