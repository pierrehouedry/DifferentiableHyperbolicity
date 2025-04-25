from tqdm import tqdm
import torch
from hyperbolicity.utils import soft_max


def compute_hyperbolicity(M, scale=0):
    """
    Computes the Gromov delta-hyperbolicity of a metric space using the 4-point condition.

    Parameters:
        M (torch.Tensor): A (N x N) distance matrix (assumed to be symmetric).
        scale (float): If non-zero, uses a smooth soft-max approximation instead of hard max.

    Returns:
        torch.Tensor: Scalar value representing the delta-hyperbolicity.
    """
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
        return soft_max(delta, scale, dim=(0, 1, 2, 3))
    else:
        return torch.max(delta)


def compute_hyperbolicity_batch(M_batch, scale=0):
    """
    Computes delta-hyperbolicity over a batch of distance matrices using the 4-point condition.

    Parameters:
        M_batch (torch.Tensor): A batch of distance matrices (B x N x N).
        scale (float): If non-zero, uses a soft-max approximation over the computed deltas.

    Returns:
        torch.Tensor: Tensor of shape (B,) with one hyperbolicity value per batch matrix.
    """
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


def compute_exact_hyperbolicity_naive(metric, scale=0):
    """
    Computes the exact delta-hyperbolicity using the 4-point condition via brute-force enumeration.

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.

    Returns:
        torch.Tensor: Scalar tensor representing the exact hyperbolicity.
    """

    N = metric.shape[0]
    maxi = torch.tensor([0], device=metric.device)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    S1 = metric[i, j] + metric[k, l]
                    S2 = metric[i, k] + metric[j, l]
                    S3 = metric[i, l] + metric[j, k]
                    Stot = torch.stack([S1, S2, S3], dim=-1)
                    Stot = Stot.sort(descending=True)[0]
                    if scale != 0:
                        maxi = soft_max(torch.tensor([maxi, (Stot[0] - Stot[1]) / 2]), scale)
                    else:
                        maxi = torch.max(maxi, (Stot[0]-Stot[1])/2)
    return maxi


def compute_hyperbolicity_from_pairs(metric, ind, scale=0):
    """
    Computes the delta-hyperbolicity over a selected set of index pairs.

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.
        ind (torch.Tensor): A (P x 2) tensor of index pairs (i, j).
        scale (float): If non-zero, uses a soft-max approximation over delta values.

    Returns:
        torch.Tensor: Scalar tensor representing the (approximate) hyperbolicity over selected pairs.
    """

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
    Stot = torch.stack([S2, S3], dim=-1)  # Shape: (P, P, 3)
    Stot_sorted = Stot.sort(dim=-1, descending=True)[0]

    # Compute K
    K = (S1 - Stot_sorted[..., 0]) / 2  # Shape: (P, P)

    # Get the maximum value for each pair
    # return K.max()
    if scale:
        return soft_max(K, scale, dim=(0, 1))
    else:
        return torch.max(K)


def gromov_product_from_distances(metric, i, j, k):
    """
    Computes the Gromov product between points i and j with respect to base point k.

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.
        i (int): Index of point i.
        j (int): Index of point j.
        k (int): Index of base point k.

    Returns:
        torch.Tensor: Scalar tensor with the Gromov product (i·j)_k.
    """
    d_i_k = metric[i, k]
    d_j_k = metric[j, k]
    d_i_j = metric[i, j]

    return (d_i_k + d_j_k - d_i_j) / 2


def delta_hyperbolicity_fixed_basepoint(metric, base_point, alpha, soft=True):
    """
    Computes a basepoint-dependent notion of delta-hyperbolicity using Gromov products.

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.
        base_point (int): Index of the base point for computing Gromov products.
        alpha (float): Smoothing parameter used in log-sum-exp aggregation.
        soft (bool): Whether to use soft-max aggregation (default: True).

    Returns:
        torch.Tensor: Scalar value representing the smoothed hyperbolicity estimate.
    """
    row = metric[base_point, :]
    XX_p = 0.5 * (row.unsqueeze(0) + row.unsqueeze(1) - metric)  # could be optimized if base_point is 0

    return torch.logsumexp(alpha*(torch.min(XX_p[:, :, None], XX_p[None, :, :])-XX_p[:, None, :]), dim=(0, 1, 2))/alpha


def delta_hyperbolicity_fixed_basepoint2(metric, base_point, alpha, soft=True):
    """
    Alternative version of delta_hyperbolicity_fixed_basepoint with explicit loop for aggregation.

    Parameters:
        metric (torch.Tensor): A (N x N) distance matrix.
        base_point (int): Index of the base point for computing Gromov products.
        alpha (float): Smoothing parameter used in log-sum-exp aggregation.
        soft (bool): Whether to use soft-max aggregation (default: True).

    Returns:
        torch.Tensor: Scalar value representing the alternative smoothed hyperbolicity estimate.
    """
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

def compute_delta_from_distances_batched(dist_matrices: torch.Tensor, scale: float) -> torch.Tensor:         
    """
    Batched variant of ‘delta_from_distances’.

    Parameters
    ----------
    dist_matrices : torch.Tensor
        Pairwise distances for B graphs (shape: B × N × N).
    scale : float
        Temperature parameter in the soft-min / soft-max (must be non-zero).

    Returns
    -------
    torch.Tensor
        delta for each batch element (shape: B).
    """
    if dist_matrices.ndim != 3 or dist_matrices.size(-1) != dist_matrices.size(-2):
        raise ValueError("`dist_matrices` must have shape (B, N, N).")

    B, N, _ = dist_matrices.shape
    device = dist_matrices.device

    # Pre-compute the quadruple index grid *once* and broadcast across the batch
    #   idx.shape = (4, N⁴)
    idx = torch.cartesian_prod(*(torch.arange(N, device=device) for _ in range(4))).T
    i_idx, j_idx, k_idx, l_idx = idx             # (N⁴,)

    # Gather all pairwise distances needed for the three Gromov products.
    # Each gather creates a tensor of shape (B, N⁴)
    d_il = dist_matrices[:, i_idx, l_idx]
    d_jl = dist_matrices[:, j_idx, l_idx]
    d_kl = dist_matrices[:, k_idx, l_idx]
    d_ij = dist_matrices[:, i_idx, j_idx]
    d_jk = dist_matrices[:, j_idx, k_idx]
    d_ik = dist_matrices[:, i_idx, k_idx]

    gp_01_3 = (d_il + d_jl - d_ij) / 2           # (B, N⁴)
    gp_12_3 = (d_jl + d_kl - d_jk) / 2
    gp_02_3 = (d_il + d_kl - d_ik) / 2

    # Soft-min over {gp_01_3, gp_12_3}
    minimum   = torch.stack((gp_01_3, gp_12_3), dim=-1)      # (B, N⁴, 2)
    soft_min  = soft_max(minimum, -scale)           # (B, N⁴)

    # Δ₍i,j,k,l₎  =  soft-min(gp_01_3, gp_12_3)  −  gp_02_3
    delta_ijkl = soft_min - gp_02_3                           # (B, N⁴)

    # Finally take the soft-max (log-sum-exp) over all quadruples
    delta = soft_max(delta_ijkl,  scale, dim=-1)     # (B,)

    return delta