import torch 

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
        sum_costs = weighted_matrix[k, :] + weighted_matrix[:, k]
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