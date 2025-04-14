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
    if scale == 0:
        raise ValueError("scale must be non-zero.")

    return (1/scale) * torch.logsumexp(scale * points, dim=dim)


def datasp(weighted_matrix: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Computes a softened version of shortest path distances using the DataSP algorithm with a temperature parameter beta.
    Inspired by: DataSP: A Differential All-to-All Shortest Path Algorithm for  Learning Costs and Predicting Paths with Context by Alan A. Lahoud, Erik Schaffernicht, and Johannes A. Stork
    """
    num_nodes = weighted_matrix.shape[0]
    for k in range(num_nodes):
        # sum_costs = weighted_matrix[k, :] + weighted_matrix[:, k]
        sum_costs = weighted_matrix[k:k+1, :] + weighted_matrix[:, k:k+1]
        weighted_matrix = soft_max(torch.stack([sum_costs, weighted_matrix], dim=-1), -beta)

    return weighted_matrix


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
