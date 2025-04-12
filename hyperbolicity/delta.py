from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn


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

##---------------\delta-hyperbolicity related methods

def filter_farapart_memory_inefficient(metric):
# memory inefficient but rapid if you can !
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

def compute_hyperbolicity(M,scale=0):
        # Compute S1, S2, S3 for all combinations (i, j, k, l)
    S1 = M.unsqueeze(2).unsqueeze(3) + M.unsqueeze(0).unsqueeze(1)
    S2 = M.unsqueeze(1).unsqueeze(3) + M.unsqueeze(0).unsqueeze(2)
    S3 = M.unsqueeze(1).unsqueeze(2) + M.unsqueeze(0).unsqueeze(3)

    # Stack S1, S2, S3 along a new dimension
    Stot = torch.stack([S1, S2, S3], dim=-1)

    # Sort Stot along the last dimension and compute the difference
    Stot_sorted = Stot.sort(dim=-1, descending=True)[0]
    delta = (Stot_sorted[..., 0] - Stot_sorted[..., 1]) / 2
    # Find the maximum value of delta
    if scale:
        return soft_max(delta,scale,dim=(0,1,2,3))
    else:
        return torch.max(delta)

def make_batches(M, size_batches=10, nb_batches=1):
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

def compute_hyperbolicity_batch(M_batch, scale=0):
    # Compute S1, S2, S3 for all combinations (i, j, k, l) across the batch
    S1 = M_batch.unsqueeze(3).unsqueeze(4) + M_batch.unsqueeze(1).unsqueeze(2)
    S2 = M_batch.unsqueeze(2).unsqueeze(4) + M_batch.unsqueeze(1).unsqueeze(3)
    S3 = M_batch.unsqueeze(2).unsqueeze(3) + M_batch.unsqueeze(1).unsqueeze(4)

    # Stack S1, S2, S3 along a new dimension
    Stot = torch.stack([S1, S2, S3], dim=-1)

    # Sort Stot along the last dimension and compute the difference
    Stot_sorted = Stot.sort(dim=-1, descending=True)[0]
    delta = (Stot_sorted[..., 0] - Stot_sorted[..., 1]) / 2

    # Find the maximum value of delta for each matrix in the batch
    if scale:
        return soft_max(delta, scale, dim=(1, 2, 3, 4))
    else:
        return torch.max(delta, dim=(1, 2, 3, 4))

def compute_exact_hyperbolicity_naive(metric):
    N = metric.shape[0]
    maxi=torch.tensor([0], device=metric.device)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    S1 = metric[i,j] + metric[k,l]
                    S2 = metric[i,k] + metric[j,l]
                    S3 = metric[i,l] + metric[j,k]
                    Stot = torch.stack([S1, S2,S3], dim=-1)
                    Stot = Stot.sort(descending=True)[0]
                    maxi = torch.max(maxi,(Stot[0]-Stot[1])/2)
    return maxi

def compute_hyperbolicity_from_pairs(metric, ind, scale=0):
    # Extract (x, y) index pairs
    x, y = ind[:, 0], ind[:, 1]  # Shape: (P,)

    # Expand (x, y) to compare against all (u, v) pairs
    x_exp, y_exp = x[:, None], y[:, None]  # Shape: (P, 1)
    u, v = ind[:, 0], ind[:, 1]  # Shape: (P,)

    # Compute S1, S2, S3 efficiently
    S1 = metric[x_exp, y_exp] + metric[u[None, :], v[None, :]]  # (P, P)
    S2 = metric[x_exp, u[None, :]] + metric[y_exp, v[None, :]]  # (P, P)
    S3 = metric[x_exp, v[None, :]] + metric[y_exp, u[None, :]]  # (P, P)

    # Stack and sort
    Stot = torch.stack([S1, S2, S3], dim=-1)  # Shape: (P, P, 3)
    Stot_sorted = Stot.sort(dim=-1, descending=True)[0]

    # Compute K
    K = (Stot_sorted[..., 0] - Stot_sorted[..., 1]) / 2  # Shape: (P, P)

    # Get the maximum value for each pair
    #return K.max()
    if scale:
        return soft_max(K,scale,dim=(0,1))
    else:
        return torch.max(K)

def gromov_product_from_distances(metric, i,j,k):
    d_i_k = metric[i, k]
    d_j_k = metric[j, k]
    d_i_j = metric[i, j]
    return (d_i_k + d_j_k - d_i_j) / 2

def delta_hyperbolicity_fixed_basepoint(metric,base_point, alpha, soft=True):
    row = metric[base_point,:]
    XX_p = 0.5 * (row.unsqueeze(0) + row.unsqueeze(1) - metric) # could be optimized if base_point is 0
    # naive  implem
    return torch.logsumexp(alpha*(torch.min(XX_p[:, :, None], XX_p[None, :, :])-XX_p[:, None, :]), dim=(0,1,2))/alpha

def delta_hyperbolicity_fixed_basepoint2(metric, base_point, alpha, soft=True):
    row = metric[base_point, :]
    N = metric.size(0)
    XX_p = 0.5 * (row.unsqueeze(0) + row.unsqueeze(1) - metric)

    max_logsumexp = -float('inf')

    for i in range(N):
        XX_p_i = XX_p[i, :]
        min_values = torch.min(XX_p_i[:, None], XX_p)
        logsumexp_value = torch.logsumexp(alpha * (min_values - XX_p_i), dim=1)

        if torch.max(logsumexp_value) > max_logsumexp:
            max_logsumexp = torch.max(logsumexp_value)

    return max_logsumexp / alpha