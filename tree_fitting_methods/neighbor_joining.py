import numpy as np
import networkx as nx

def neighbor_join(distance_matrix):
    """
    Constructs a phylogenetic tree using the Neighbor-Joining algorithm from a distance matrix.
    
    Parameters:
        distance_matrix (numpy.ndarray): A symmetric n x n distance matrix.
        
    Returns:
        networkx.Graph: A tree with edge weights representing the reconstructed phylogeny.
    """
    n = distance_matrix.shape[0]
    D = np.array(distance_matrix, dtype=float)
    vertices = list(range(n))
    tree = nx.Graph()
    tree.add_nodes_from(vertices)

    if n <= 1:
        return tree

    next_node = n  # ID for new internal nodes

    while True:
        if n == 2:
            # Only two nodes left, just connect them
            tree.add_edge(vertices[0], vertices[1], weight=D[0, 1])
            break

        # Compute the Q-matrix used to find the pair to join
        total_dist = np.sum(D, axis=0)
        Q = (n - 2) * D - total_dist[:, None] - total_dist
        np.fill_diagonal(Q, np.inf)  # prevent selecting diagonal elements

        # Find the pair (i, j) with minimal Q value
        i, j = divmod(np.argmin(Q), n)

        # Compute branch lengths from new node to i and j
        delta = (total_dist[i] - total_dist[j]) / (n - 2)
        li = (D[i, j] + delta) / 2
        lj = (D[i, j] - delta) / 2

        # Create distances from new node to remaining nodes
        d_new = (D[i, :] + D[j, :] - D[i, j]) / 2
        d_new = np.append(d_new, 0.0)  # Distance to itself is zero

        # Update distance matrix: remove rows/cols i and j, add new row/col
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        D = np.vstack([D, d_new[:-1]])
        d_new = np.append(d_new[:-1], 0.0)
        D = np.column_stack([D, d_new])

        # Update the tree
        vi, vj = vertices[i], vertices[j]
        tree.add_node(next_node)
        tree.add_edge(next_node, vi, weight=li)
        tree.add_edge(next_node, vj, weight=lj)

        # Update the list of vertices
        vertices.pop(max(i, j))  # remove larger index first to avoid reindexing issues
        vertices.pop(min(i, j))
        vertices.append(next_node)

        next_node += 1
        n -= 1

    return tree
