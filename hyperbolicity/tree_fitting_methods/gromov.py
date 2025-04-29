import numpy as np
from scipy.spatial import distance as ssd
from scipy.cluster.hierarchy import linkage
import networkx as nx

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

    gp = np.tile(d_root, (n, 1)) + np.tile(d_root.reshape(n, 1), (1, n)) - distance_matrix
    gp = gp/2.0

    d_U = d_max - gp
    np.fill_diagonal(d_U, 0)

    Z = linkage(ssd.squareform(d_U), method='single')
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
):
    if distance_matrix.shape[0] == 2:
        p = [i for i in range(2) if i != root][0]
        tree.add_edge(
            node_map[root],
            node_map[p],
            weight=distance_matrix[root, p],
        )
        return

    num_nodes = distance_matrix.shape[0]
    if not fast_mode:
        max_value = -float("inf")
        for p in range(num_nodes):
            for q in range(num_nodes):
                if p == root or q == root or p == q:
                    continue
                value = (
                    distance_matrix[p, root]
                    + distance_matrix[q, root]
                    - distance_matrix[p, q]
                )
                if value > max_value:
                    max_value = value
                    max_nodes = (p, q)
        p, q = max_nodes
    else:
        values = (
            distance_matrix[None, :, root]
            + distance_matrix[:, None, root]
            - distance_matrix
        ) * (1 - np.diag(np.ones(num_nodes)))
        values = values - 99999 * np.diag(np.ones(num_nodes))
        values[:, root] = -999999
        values[root, :] = -999999
        p, q = np.unravel_index(np.argmax(values, axis=None), values.shape)

    d_t_p = (
        1
        / 2
        * (distance_matrix[p, q] + distance_matrix[p, root] - distance_matrix[q, root])
    )
    d_t_q = (
        1
        / 2
        * (distance_matrix[p, q] + distance_matrix[q, root] - distance_matrix[p, root])
    )
    t_distances = distance_matrix[p, :] - d_t_p

    # add nodes to tree
    t_name = next_node_id
    next_node_id += 1
    tree.add_node(t_name)
    tree.add_edge(t_name, node_map[p], weight=d_t_p)
    tree.add_edge(t_name, node_map[q], weight=d_t_q)
    next_node_map = {num_nodes - 2: t_name}
    i = 0
    offset = 0
    new_root = root
    while i < num_nodes - 2:
        if (i + offset) == root:
            new_root = i
        if (i + offset) == p or (i + offset) == q:
            offset += 1
            continue
        next_node_map[i] = node_map[i + offset]
        i += 1

    # construct new distance matrix
    new_distance_matrix = np.zeros((num_nodes - 1, num_nodes - 1))
    new_distance_matrix[:-1, :-1] = np.delete(
        np.delete(distance_matrix, [p, q], axis=0), [p, q], axis=1
    )
    new_distance_matrix[-1, :-1] = np.delete(t_distances, [p, q], axis=0)
    new_distance_matrix[:-1, -1] = new_distance_matrix[-1, :-1]
    buneman_extraction_aux(
        tree,
        new_distance_matrix,
        new_root,
        next_node_map,
        next_node_id,
        fast_mode=fast_mode,
    )


def buneman_extraction(
    distance_matrix: np.ndarray,
    root: int,
    fast_mode: bool = False,
) -> nx.Graph:
    """
    Extracts a Buneman tree from a distance matrix.

    Parameters:
        distance_matrix (np.ndarray): Pairwise distance matrix.
        root (int): Index to use as the root node.
        fast_mode (bool): If True, uses a faster but less accurate method.
            Default is False.

    Returns:
        nx.Graph: Buneman tree represented as a NetworkX graph.
    """
    tree = nx.Graph()
    buneman_extraction_aux(
        tree,
        distance_matrix,
        root,
        {i: i for i in range(distance_matrix.shape[0])},
        next_node_id=distance_matrix.shape[0],
        fast_mode=fast_mode,
    )
    return tree
