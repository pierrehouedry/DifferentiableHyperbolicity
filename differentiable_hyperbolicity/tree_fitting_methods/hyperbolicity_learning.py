import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from differentiable_hyperbolicity.delta import compute_delta_from_distances_batched
from differentiable_hyperbolicity.utils import (
    construct_weighted_matrix,
    floyd_warshall,
    make_batches,
    soft_max,
)


def train_distance_matrix(
    distances: torch.Tensor,
    scale_delta: float,
    distance_reg: float,
    num_epochs: int,
    n_batches: int,
    batch_size: int,
    learning_rate: float,
    verbose: bool,
    gpu: bool,
):
    if gpu:
        distances = distances.to("cuda")

    K = int(np.ceil(n_batches / 32))
    num_nodes = distances.shape[0]
    edges = torch.triu_indices(num_nodes, num_nodes, offset=1)
    upper_adjency = torch.triu(distances, diagonal=1).type(torch.float32)
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
        update_dist = construct_weighted_matrix(w, num_nodes, edges)
        for _ in range(K):
            M_batch = make_batches(update_dist, size_batches=batch_size, nb_batches=32)
            delta = soft_max(
                compute_delta_from_distances_batched(M_batch, scale=scale_delta),
                scale=scale_delta,
            )
            delta.backward(delta, retain_graph=True)
        err = (distances - update_dist).pow(2).mean()
        (distance_reg * err).backward()

        return delta + distance_reg * err, delta, err

    patience = 50
    best_loss = float("inf")
    best_weights = weights_opt.detach().clone().cpu()
    patience_counter = 0
    start = time.time()
    with tqdm(
        range(num_epochs), desc="Training Weights", disable=not verbose, leave="False"
    ) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            loss, delta, err = loss_fn(weights_opt)
            pbar.set_description(
                f"loss = {loss.item():.5f}, delta = {delta:.5f}, error = {err:.5f}"
            )
            losses.append(loss.item())
            deltas.append(delta.item())
            errors.append(err.item())
            # loss.backward()
            optimizer.step()
            with torch.no_grad():
                weights_opt.data[weights_opt.data < 0] = 0
                weights_opt.data = projection(weights_opt, num_nodes, edges)
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
    return best_weights, losses, deltas, errors, end - start
