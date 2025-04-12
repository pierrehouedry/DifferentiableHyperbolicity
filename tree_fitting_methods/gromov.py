import numpy as np
from scipy.spatial import distance as ssd
from scipy.cluster.hierarchy import linkage

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

def gromov(distance_matrix, root):
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
    
    gp = np.tile(d_root, (n, 1)) + np.tile(d_root.reshape(n, 1), (1, n)) - distance_matrix
    gp /= 2.0

    d_U = d_max - gp
    np.fill_diagonal(d_U, 0)

    Z = linkage(ssd.squareform(d_U), method='single')
    D_gromov = linkage_to_distance_matrix(Z)

    gp_T = d_max - D_gromov
    d_T = np.tile(d_root, (n, 1)) + np.tile(d_root.reshape(n, 1), (1, n)) - 2.0 * gp_T
    np.fill_diagonal(d_T, 0)

    return d_T
