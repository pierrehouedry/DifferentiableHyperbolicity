import torch

# Inspired by: Enumeration of far-apart pairs by decreasing distance for faster hyperbolicity computation by David Coudert, Andre Nusser, and Laurent Viennot

def filter_farapart_memory_inefficient(metric):
    """
    Identifies pairs (u, v) such that for all w ≠ u,v, the triangle inequality conditions
    (u-w, u-v, w-v) are satisfied. This version is fast but memory-inefficient (uses large tensors).

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Tensor of shape (M,) with indices u where some valid v satisfies the condition.
            - Tensor of shape (K x 2) containing valid index pairs (u, v) satisfying the condition.
    """
    N = metric.shape[0]
    # Create index grids
    u_idx, v_idx, w_idx = torch.meshgrid(torch.arange(N), torch.arange(N), torch.arange(N), indexing="ij")
    # Conditions
    cond1 = metric[w_idx, u_idx] + metric[u_idx, v_idx] > metric[w_idx, v_idx]
    cond2 = metric[w_idx, v_idx] + metric[u_idx, v_idx] > metric[w_idx, u_idx]
    # Ensure w ≠ u and w ≠ v
    valid_w = (w_idx != u_idx) & (w_idx != v_idx)
    # Apply conditions only for valid w
    Fpart = torch.all(cond1 & cond2 | ~valid_w, dim=2)

    ind = Fpart.tril(diagonal=0).nonzero()
    
    return torch.where(Fpart.any(dim=1))[0], ind

def filter_farapart(metric):
    """
    Identifies (u, v) pairs such that the triangle inequality conditions hold
    for all w ≠ u,v, using a more memory-efficient approach by iterating over w.

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Tensor of shape (M,) with indices u where some valid v satisfies the condition.
            - Tensor of shape (K x 2) containing valid index pairs (u, v) satisfying the condition.
    """

    N = metric.shape[0]
    Fpart = torch.ones((N, N), dtype=torch.bool, device=metric.device)  # Initialize to True

    # Precompute u_idx and v_idx once (saves memory & computation)
    u_idx, v_idx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    u_idx = u_idx.to(metric.device)
    v_idx = v_idx.to(metric.device)

    # Loop over w instead of creating (N, N, N) tensors
    for w in range(N):
        # Conditions
        cond1 = metric[w, u_idx] + metric[u_idx, v_idx] > metric[w, v_idx]
        cond2 = metric[w, v_idx] + metric[u_idx, v_idx] > metric[w, u_idx]

        # Ignore w == u or w == v
        valid_w = (w != u_idx) & (w != v_idx)

        # Update Fpart using element-wise AND
        Fpart &= cond1 & cond2 | ~valid_w

    # Extract indices efficiently
    ind = Fpart.tril(diagonal=0).nonzero()

    return torch.where(Fpart.any(dim=1))[0], ind

def filter_farapart_memory_efficient(metric):
    """
    Computes far-apart pairs (u, v) in a fully memory-efficient but slower way
    by using three nested loops to check the triangle inequality for each triplet (u, v, w).

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Tensor of shape (M,) with indices u where some valid v satisfies the condition.
            - Tensor of shape (K x 2) containing valid index pairs (u, v) satisfying the condition.
    """
    N = metric.shape[0]
    Fpart = torch.ones((N, N), dtype=torch.bool)  # Initialize to True

    for u in range(N):
        for v in range(N):
            for w in range(N):
                if  (w != u) and (w != v):
                    if not(metric[w,u] + metric[u,v] > metric[w,v]) or not(metric[w,v] + metric[u,v] > metric[w,u]):
                        Fpart[u,v]=0
    ind = Fpart.tril(diagonal=0).nonzero()

    return torch.where(Fpart.any(dim=1))[0], ind