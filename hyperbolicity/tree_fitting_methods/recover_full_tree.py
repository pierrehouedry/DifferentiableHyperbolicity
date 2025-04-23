import networkx as nx
import numpy as np
import itertools


def recover_tree_graph(D, root):
    """
    Recover a tree as a networkx.Graph from a distance matrix and root.

    Parameters:
        D (np.ndarray): n x n distance matrix (tree metric)
        root (int): index of the root node (0-based)

    Returns:
        G (networkx.Graph): Reconstructed tree with weights
    """
    n = D.shape[0]
    depths = D[root]
    nodes = list(range(n))
    nodes.sort(key=lambda v: depths[v])
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for v in nodes:
        if v == root:
            continue
        for u in nodes:
            if depths[u] < depths[v] and np.isclose(D[u, v], depths[v] - depths[u]):
                weight = depths[v] - depths[u]
                G.add_edge(u, v, weight=weight)
                break
        else:
            raise ValueError(f"No parent found for node {v} — check if input is a valid tree metric.")
    
    return G

def enforce_gromov_closure(G, D, root):
    """
    Enforce Gromov closure on a tree G by inserting Steiner nodes when necessary,
    removing direct edges from root to x/y if a common ancestor is introduced.

    Parameters:
        G (nx.Graph): Tree graph with edge weights
        D (np.ndarray): Distance matrix
        root (int): Root node index

    Returns:
        G_augmented (nx.Graph): New tree with added Steiner-like nodes
    """

    G = G.copy()
    n = D.shape[0]
    next_node = max(G.nodes) + 1

    for x, y in itertools.combinations(range(n), 2):

        if x == root or y == root:
            continue

        gromov = 0.5 * (D[root, x] + D[root, y] - D[x, y])
        print(gromov)
        # Check if any node z in G has distance gromov from root
        existing_nodes = [z for z in G.nodes if (D[root, z]==gromov) and (D[z, x] == D[root, x] - gromov) and (D[z, y] == D[root, y] - gromov) and z!=root]

        print(existing_nodes)
        if not existing_nodes:
            # Create new Steiner node
            s = next_node
            G.add_node(s)
            next_node += 1

            # Remove any edge (root, x) or (root, y) if present
            if G.has_edge(root, x):
                G.remove_edge(root, x)
            if G.has_edge(root, y):
                G.remove_edge(root, y)

            # Add edge from root to s
            G.add_edge(root, s, weight=gromov)

            # Add edges from s to x and y
            G.add_edge(s, x, weight=D[root, x] - gromov)
            G.add_edge(s, y, weight=D[root, y] - gromov)

        D = nx.floyd_warshall_numpy(G)

    return G

