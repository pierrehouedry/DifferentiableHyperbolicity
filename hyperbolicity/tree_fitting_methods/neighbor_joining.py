import numpy as np
import networkx as nx

import numpy as np
import networkx as nx

def NJ(distance_matrix):
    """
    Construct a tree using the Neighbor‑Joining algorithm.

    Parameters
    ----------
    distance_matrix : array_like, shape (n, n)
        Symmetric matrix of pairwise distances between n taxa.

    Returns
    -------
    tree : networkx.Graph
        An unrooted tree (graph) whose nodes are indexed 0..2n−3, where
        the first n nodes correspond to the original taxa and the remaining
        nodes are internal nodes created during joining. Edges have a
        'weight' attribute for branch lengths.
    """
    
    D = np.array(distance_matrix, dtype=float)
    n = D.shape[0]

    tree = nx.Graph()
    tree.add_nodes_from(range(n))

    if n <= 1:
        return tree

    vertices = list(range(n))
    next_node_id = n

    while n > 2:
        total_dist = D.sum(axis=0)
        Q = (n - 2) * D - total_dist[np.newaxis, :] - total_dist[:, np.newaxis]
        np.fill_diagonal(Q, np.inf)

        i, j = divmod(np.argmin(Q), n)
        delta = (total_dist[i] - total_dist[j]) / (n - 2)
        limb_length_i = 0.5 * (D[i, j] + delta)
        limb_length_j = 0.5 * (D[i, j] - delta)

        u = next_node_id
        tree.add_node(u)
        tree.add_edge(u, vertices[i], weight=limb_length_i)
        tree.add_edge(u, vertices[j], weight=limb_length_j)

        new_row = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        D = np.vstack([D, new_row])
        new_col = np.append(new_row, 0.0)[:, np.newaxis]
        D = np.hstack([D, new_col])

        for idx in sorted((i, j), reverse=True):
            D = np.delete(D, idx, axis=0)
            D = np.delete(D, idx, axis=1)

        v_i, v_j = vertices[i], vertices[j]
        vertices.pop(max(i, j))
        vertices.pop(min(i, j))
        vertices.append(u)

        next_node_id += 1
        n -= 1

    v1, v2 = vertices
    tree.add_edge(v1, v2, weight=D[0, 1])

    return tree


