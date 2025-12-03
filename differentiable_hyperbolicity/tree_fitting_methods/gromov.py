import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance as ssd


def linkage_to_distance_matrix(Z):
    """
    Converts a linkage matrix Z (from scipy.cluster.hierarchy.linkage)
    into a full pairwise distance matrix D of shape (n, n), where n is the number of leaves.
    """
    N = Z.shape[0] + 1
    clusters = [[i] for i in range(N)]
    D = np.zeros((N, N))
    for i in range(N - 1):
        j, k = int(Z[i, 0]), int(Z[i, 1])
        for x in clusters[j]:
            for y in clusters[k]:
                D[x, y] = D[y, x] = Z[i, 2]
        clusters.append(clusters[j] + clusters[k])
    return D


def gromov_tree(distance_matrix, root):
    """
    Computes a Gromov-style distance matrix from a given root using a hierarchical clustering
    approximation based on the induced ultrametric.

    Parameters:
        distance_matrix (np.ndarray): Original pairwise distance matrix.
        root (int): Index to use as the root node.

    Returns:
        np.ndarray: Gromov-adjusted distance matrix (tree-metric approximation).
    """
    n = distance_matrix.shape[0]
    d_root = distance_matrix[root]
    d_max = d_root.max()

    gp = (
        np.tile(d_root, (n, 1))
        + np.tile(d_root.reshape(n, 1), (1, n))
        - distance_matrix
    )
    gp = gp / 2.0

    d_U = d_max - gp
    np.fill_diagonal(d_U, 0)

    Z = linkage(ssd.squareform(d_U), method="single")
    D_gromov = linkage_to_distance_matrix(Z)

    gp_T = d_max - D_gromov
    d_T = np.tile(d_root, (n, 1)) + np.tile(d_root.reshape(n, 1), (1, n)) - 2.0 * gp_T
    np.fill_diagonal(d_T, 0)

    return d_T


def buneman_extraction_aux(
    tree: nx.Graph,
    distance_matrix: np.ndarray,
    root: int,
    node_map: dict[int, int],
    next_node_id: int,
    fast_mode: bool = False,
    tol: float = 1e-10,
):
    n = distance_matrix.shape[0]

    # base case
    if n == 2:
        # connect root to the other node
        p = 1 - root
        w = max(distance_matrix[root, p], 0.0)
        tree.add_edge(node_map[root], node_map[p], weight=w)
        return

    # choose (p,q) maximizing d(pr)+d(qr)-d(pq)
    if not fast_mode:
        max_value = -np.inf
        p = q = None
        for i in range(n):
            if i == root: 
                continue
            for j in range(n):
                if j == root or j == i:
                    continue
                val = distance_matrix[i, root] + distance_matrix[j, root] - distance_matrix[i, j]
                if val > max_value:
                    max_value = val
                    p, q = i, j
    else:
        D = distance_matrix
        # compute all i!=j, i!=root, j!=root
        M = (D[:, root][:, None] + D[root, :][None, :] - D)
        # mask out diag and rows/cols of root
        mask = np.ones_like(M, dtype=bool)
        np.fill_diagonal(mask, False)
        mask[root, :] = False
        mask[:, root] = False
        M_masked = np.where(mask, M, -np.inf)
        p, q = np.unravel_index(np.argmax(M_masked), M_masked.shape)

    # limb lengths from p and q to new Steiner node t
    d_pq = distance_matrix[p, q]
    d_pr = distance_matrix[p, root]
    d_qr = distance_matrix[q, root]
    d_t_p = 0.5 * (d_pq + d_pr - d_qr)
    d_t_q = 0.5 * (d_pq + d_qr - d_pr)
    # numerical safety
    d_t_p = max(d_t_p, 0.0)
    d_t_q = max(d_t_q, 0.0)

    # distances from new t to all remaining leaves (using p as reference)
    t_distances = distance_matrix[p, :] - d_t_p

    # create Steiner node t in the full graph
    t_name = next_node_id
    next_node_id += 1
    tree.add_node(t_name)
    tree.add_edge(t_name, node_map[p], weight=max(d_t_p, 0.0))
    tree.add_edge(t_name, node_map[q], weight=max(d_t_q, 0.0))

    # build mapping for contracted problem (n-1 nodes): last index is t
    next_node_map = {n - 2: t_name}
    i = 0
    offset = 0
    # Track whether root survives; if not, we'll set it to the new t
    root_removed = (root == p or root == q)
    while i < n - 2:
        idx = i + offset
        if idx == p or idx == q:
            offset += 1
            continue
        next_node_map[i] = node_map[idx]
        i += 1

    # construct contracted distance matrix
    newD = np.zeros((n - 1, n - 1), dtype=distance_matrix.dtype)
    keep = [k for k in range(n) if k not in (p, q)]
    newD[:-1, :-1] = distance_matrix[np.ix_(keep, keep)]
    newD[-1, :-1] = t_distances[keep]
    newD[:-1, -1] = newD[-1, :-1]
    np.fill_diagonal(newD, 0.0)

    # choose new root index
    if root_removed:
        new_root = n - 2  # the Steiner node t is the last index
    else:
        # old root survived; find its new index in 'keep'
        new_root = keep.index(root)

    # recurse
    buneman_extraction_aux(
        tree,
        newD,
        new_root,
        next_node_map,
        next_node_id,
        fast_mode=fast_mode,
        tol=tol,
    )

def buneman_extraction(
    distance_matrix: np.ndarray,
    root: int,
    fast_mode: bool = False,
    tol: float = 1e-10,
) -> nx.Graph:
    """
    Reconstruct a tree from an additive distance matrix (Buneman/additive-phylogeny style).

    Parameters
    ----------
    distance_matrix : (n,n) np.ndarray
        Symmetric, zero-diagonal, additive distances between leaves 0..n-1.
    root : int
        Index to use as the working root during recursion.
        If the chosen pair removes this root in a step,
        the new root becomes the created Steiner node.
    fast_mode : bool
        Vectorized selection of (p,q). Same output for exact additive data.
    tol : float
        Tolerance for numeric checks / clipping.

    Returns
    -------
    nx.Graph
        Weighted undirected tree whose leaf set is {0,...,n-1}.
        Internal Steiner nodes start at id n, n+1, ...
    """
    D = np.array(distance_matrix, dtype=float, copy=True)
    n = D.shape[0]
    if not (0 <= root < n):
        raise ValueError("root must be a valid index in [0, n)")
    tree = nx.Graph()
    # start with identity mapping for leaves
    node_map = {i: i for i in range(n)}
    buneman_extraction_aux(
        tree, D, root, node_map, next_node_id=n, fast_mode=fast_mode, tol=tol
    )
    # clip tiny negative weights from round-off
    for u, v, data in tree.edges(data=True):
        if data.get("weight", 0.0) < 0 and data["weight"] > -tol:
            data["weight"] = 0.0
    return tree
