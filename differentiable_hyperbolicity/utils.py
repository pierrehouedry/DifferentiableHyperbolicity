import torch


def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Tricks here: log(sum(exp(x))) = log(sum(exp(x - m)*exp(m))) = log(exp(m)*sum(exp(x - m))) = m + log(sum(exp(x - m)))
    m, _ = x.max(dim=dim)
    mask = m == -float("inf")
    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)

    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


def soft_max(points: torch.Tensor, scale: float, dim=-1) -> torch.Tensor:
    """
    Computes the log-sum-exp with a scaling factor.
    """

    return (1 / scale) * torch.logsumexp(scale * points, dim=dim)


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


def construct_weighted_matrix(
    weights: torch.Tensor, num_nodes: int, edges: torch.Tensor
) -> torch.Tensor:
    """
    Constructs a weighted adjacency matrix from given edge weights.
    """
    weighted_matrix = torch.full((num_nodes, num_nodes), float(0)).to(
        device=weights.device
    )
    weighted_matrix[edges[0, :], edges[1, :]] = weights
    weighted_matrix = weighted_matrix + weighted_matrix.t()

    return weighted_matrix


def floyd_warshall(adj_matrix):
    """
    Implements the Floyd-Warshall algorithm to compute the shortest paths between all pairs of nodes in a graph.

    Parameters:
        adj_matrix (torch.Tensor): A (N x N) adjacency matrix representing the graph.
                                   The value at (i, j) represents the weight of the edge from node i to node j.
                                   Use float('inf') for no direct edge between nodes.

    Returns:
        torch.Tensor: A (N x N) matrix where the value at (i, j) represents the shortest path distance from node i to node j.
    """
    N = adj_matrix.size(0)
    dist = adj_matrix.clone()
    for k in range(N):
        dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

    return dist
