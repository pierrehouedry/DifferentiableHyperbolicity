import numpy as np
import networkx as nx

#Inspired by: Additive Spanners and Distance and Routing Labeling Schemes for Hyperbolic Graphs by Victor Chepoi, Feodor F. Dragan2, Bertrand Estellon, Michel Habib, Yann Vaxes, and Yang Xiang

def layering(graph, source):
    dist = nx.floyd_warshall_numpy(graph)
    layers = defaultdict(list)
    for node in graph.nodes():
        d = dist[source][node]
        layers[d].append(node)
    return layers

def layering_partition(graph, source):
    layers = layering(graph, source)
    dist = nx.floyd_warshall_numpy(graph)
    partition = {}
    for d, nodes_in_layer in layers.items():
        allowed = [v for v in graph.nodes() if dist[source][v] >= d]
        G_allowed = graph.subgraph(allowed)
        clusters = []
        for comp in nx.connected_components(G_allowed):
            cluster = [v for v in comp if v in nodes_in_layer]
            if cluster:
                clusters.append(cluster)
        partition[d] = clusters
    return partition

def layering_approx_tree(graph, source):

    partition = layering_partition(graph, source)
    T = nx.Graph()
    T.add_nodes_from(graph.nodes())
    layers_sorted = sorted(partition.keys())
    
    for d in layers_sorted:
        if d == 0:
            continue
        prev_layer = d - 1
        prev_nodes = set()
        for cluster_prev in partition.get(prev_layer, []):
            prev_nodes.update(cluster_prev)
        for cluster in partition[d]:
            rep = None
            for u in cluster:
                for v in graph.neighbors(u):
                    if v in prev_nodes:
                        rep = v
                        break
                if rep is not None:
                    break
            if rep is None:
                rep = source 
            for u in cluster:
                T.add_edge(rep, u)
    return T