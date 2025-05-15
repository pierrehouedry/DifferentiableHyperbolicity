import torch

from differentiable_hyperbolicity.utils import soft_max


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


def compute_delta_from_distances_batched(
    dist_matrices: torch.Tensor, scale: float
) -> torch.Tensor:
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
    i_idx, j_idx, k_idx, l_idx = idx  # (N⁴,)

    # Gather all pairwise distances needed for the three Gromov products.
    # Each gather creates a tensor of shape (B, N⁴)
    d_il = dist_matrices[:, i_idx, l_idx]
    d_jl = dist_matrices[:, j_idx, l_idx]
    d_kl = dist_matrices[:, k_idx, l_idx]
    d_ij = dist_matrices[:, i_idx, j_idx]
    d_jk = dist_matrices[:, j_idx, k_idx]
    d_ik = dist_matrices[:, i_idx, k_idx]

    gp_01_3 = (d_il + d_jl - d_ij) / 2  # (B, N⁴)
    gp_12_3 = (d_jl + d_kl - d_jk) / 2
    gp_02_3 = (d_il + d_kl - d_ik) / 2

    # Soft-min over {gp_01_3, gp_12_3}
    minimum = torch.stack((gp_01_3, gp_12_3), dim=-1)  # (B, N⁴, 2)
    soft_min = soft_max(minimum, -scale)  # (B, N⁴)

    # Δ₍i,j,k,l₎  =  soft-min(gp_01_3, gp_12_3)  −  gp_02_3
    delta_ijkl = soft_min - gp_02_3  # (B, N⁴)

    # Finally take the soft-max (log-sum-exp) over all quadruples
    delta = soft_max(delta_ijkl, scale, dim=-1)  # (B,)

    return delta

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


