import torch
import os
import datetime
import logging
import argparse
from dateutil import tz


def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Tricks here: log(sum(exp(x))) = log(sum(exp(x - m)*exp(m))) = log(exp(m)*sum(exp(x - m))) = m + log(sum(exp(x - m)))
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')
    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)

    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


def soft_max(points: torch.Tensor,  scale: float, dim=-1) -> torch.Tensor:
    """
    Computes the log-sum-exp with a scaling factor.
    """

    return (1/scale) * torch.logsumexp(scale * points, dim=dim)


def make_batches(M, size_batches=10, nb_batches=1):
    """
    Samples random submatrices from a given distance matrix for batched hyperbolicity estimation.

    Parameters:
        M (torch.Tensor): A (N x N) distance matrix.
        size_batches (int): Number of points in each batch (submatrix size).
        nb_batches (int): Number of batches to sample.

    Returns:
        torch.Tensor: A tensor of shape (nb_batches, size_batches, size_batches) containing sampled submatrices.
    """
    N = M.size(0)
    all_indices = torch.arange(N).to(M.device)
    batches = []
    for _ in range(nb_batches):
        # Shuffle the indices to ensure random selection without replacement
        shuffled_indices = all_indices[torch.randperm(N)]
        # Select the first `size_batches` indices to form a submatrix
        selected_indices = shuffled_indices[:size_batches]
        # Create the submatrix using the selected indices
        submatrix = M[selected_indices[:, None], selected_indices]
        # Add the submatrix to the list of batches
        batches.append(submatrix)
    # Stack the list of batches into a single tensor

    return torch.stack(batches)

def datasp(weights: torch.Tensor, num_nodes, edges, beta: float = 1.0) -> torch.Tensor:
    """
    Computes a softened version of shortest path distances using the DataSP algorithm with a temperature parameter beta.
    Inspired by: DataSP: A Differential All-to-All Shortest Path Algorithm for  Learning Costs and Predicting Paths with Context by Alan A. Lahoud, Erik Schaffernicht, and Johannes A. Stork
    """
    weighted_matrix = construct_weighted_matrix(weights, num_nodes, edges)

    for k in range(num_nodes):
        for i in range(num_nodes):
            via_k = weighted_matrix[i, k] + weighted_matrix[k, :]
            current = weighted_matrix[i, :]            
            stacked = torch.stack([via_k, current], dim=0)  # shape (2, num_nodes)
            weighted_matrix[i, :] = soft_max(stacked, -beta, dim=0)
    
    return weighted_matrix


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

def construct_weighted_matrix(weights: torch.Tensor, num_nodes: int, edges: torch.Tensor) -> torch.Tensor:
    """
    Constructs a weighted adjacency matrix from given edge weights.
    """
    weighted_matrix = torch.full((num_nodes, num_nodes), float(0)).to(device=weights.device)
    weighted_matrix[edges[0, :], edges[1, :]] = weights
    weighted_matrix = weighted_matrix+weighted_matrix.t()

    return weighted_matrix


def create_log_dir(FLAGS, add_name=None):
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if add_name is not None:
        log_dir = FLAGS.log_dir + 'launch_distance_hyperbolicity_learning' + "_" + timestamp + add_name
    else:
        log_dir = FLAGS.log_dir + 'launch_distance_hyperbolicity_learning' + "_" + timestamp
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save command line arguments
    with open(log_dir + "/hyperparameters_" + timestamp + ".csv", "w") as f:
        for arg in FLAGS.__dict__:
            f.write(arg + "," + str(FLAGS.__dict__[arg]) + "\n")

    return log_dir


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def soft_min(x: torch.Tensor, scale: float = 1000, dim=-1) -> torch.Tensor:
    return soft_max(x, scale=-scale, dim=dim) 

def floyd_warshall(adj_matrix):

    N = adj_matrix.size(0)
    dist = adj_matrix.clone()

    for k in range(N):
        dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

    return dist
