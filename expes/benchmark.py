import argparse
import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.datasets import AttributedGraphDataset, Planetoid
from torch_geometric.utils import to_networkx

from differentiable_hyperbolicity.tree_fitting_methods.gromov import gromov_tree
from differentiable_hyperbolicity.tree_fitting_methods.hccfit import HccLinkage
from differentiable_hyperbolicity.tree_fitting_methods.layering_tree import (
    layering_approx_tree,
)
from differentiable_hyperbolicity.tree_fitting_methods.treerep import TreeRep

parser = argparse.ArgumentParser(description="Benchmark tree fitting methods.")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset name (e.g., 'wiki', 'airport', 'celegan', 'csphd', 'cora').",
)
parser.add_argument(
    "--num_expe", type=int, default=100, help="Number of experiments to run."
)
parser.add_argument(
    "--methods",
    type=str,
    nargs="+",
    default=["hcc", "gromov", "treerep", "layering"],
    help="Methods to run (e.g., 'hcc', 'gromov', 'treerep', 'layering').",
)
args = parser.parse_args()

dataset = args.dataset
num_expe = args.num_expe
methods = args.methods
# load the distances
dataset_path = f"./datasets/D_{dataset}.pkl"

with open(dataset_path, "rb") as f:
    distances = pickle.load(f)


# load the graph
if dataset == "wiki":
    wiki_dataset = AttributedGraphDataset("./datasets/wiki", name="Wiki")
    data = wiki_dataset[0]
    G_data = to_networkx(data, to_undirected=True)
    largest_cc_data = max(nx.connected_components(G_data), key=len)
    G_largest_cc = G_data.subgraph(largest_cc_data).copy()
    # Relabel Nodes for Consistency
    mapping = {node: i for i, node in enumerate(G_largest_cc.nodes())}
    G_largest_cc = nx.relabel_nodes(G_largest_cc, mapping)

if dataset == "airport":
    airport = pickle.load(open("./datasets/airport/airport.p", "rb"))
    # For Airport, we need to do this
    li = [airport.subgraph(c) for c in nx.connected_components(airport)]
    G_largest_cc = nx.Graph(li[0])
    G_largest_cc.remove_edges_from(nx.selfloop_edges(G_largest_cc))
    num_nodes = G_largest_cc.number_of_nodes()
    mapping = {node: i for i, node in enumerate(G_largest_cc.nodes())}
    G_largest_cc = nx.relabel_nodes(G_largest_cc, mapping)

if dataset == "celegan":
    # Load the bio-celegans dataset
    celegans_path = "./datasets/bio-celegans.csv"
    df = pd.read_csv(celegans_path)
    G_celegans = nx.from_pandas_edgelist(df, source="id1", target="id2")
    largest_cc_celegans = max(nx.connected_components(G_celegans), key=len)
    G_largest_cc = G_celegans.subgraph(largest_cc_celegans).copy()
    mapping = {node: i for i, node in enumerate(G_largest_cc.nodes())}
    G_largest_cc = nx.relabel_nodes(G_largest_cc, mapping)

if dataset == "csphd":
    csphd_path = "./datasets/ca-CSphd.csv"
    df = pd.read_csv(csphd_path)
    G_csphd = nx.from_pandas_edgelist(df, source="id1", target="id2")
    largest_cc_csphd = max(nx.connected_components(G_csphd), key=len)
    G_largest_cc = G_csphd.subgraph(largest_cc_csphd).copy()
    mapping = {node: i for i, node in enumerate(G_largest_cc.nodes())}
    G_largest_cc = nx.relabel_nodes(G_largest_cc, mapping)

if dataset == "cora":
    cora_dataset = Planetoid(root="./datasets/cora", name="Cora")
    data = cora_dataset[0]
    G_data = to_networkx(data, to_undirected=True)
    largest_cc_data = max(nx.connected_components(G_data), key=len)
    G_largest_cc = G_data.subgraph(largest_cc_data).copy()
    mapping = {node: i for i, node in enumerate(G_largest_cc.nodes())}
    G_largest_cc = nx.relabel_nodes(G_largest_cc, mapping)


num_nodes = distances.shape[0]
np.random.seed(42)
indices = np.random.choice(num_nodes, num_expe, replace=False)
denom = num_nodes * (num_nodes - 1)

if "hcc" in methods:
    l1 = []
    distortion = []
    elapsed_time = []

    for j in indices:
        start_time = time.time()
        tree_hcc = HccLinkage(distances)
        tree_hcc.fit_tree(j)
        elapsed_time.append(time.time() - start_time)
        l1.append(np.abs(distances - tree_hcc.d_T).sum() / denom)
        distortion.append(np.abs(distances - tree_hcc.d_T).max())

    avg_l1 = np.mean(l1)
    std_l1 = np.std(l1)
    avg_distortion = np.mean(distortion)
    std_distortion = np.std(distortion)

    # Save HCC Linkage Results
    with open(f"./expes/results_expes/benchmark_{dataset}.txt", "a") as result_file:
        result_file.write("== HCC Linkage Results ==\n")
        result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
        result_file.write(
            f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n"
        )
        result_file.write(
            f"Average Time: {np.mean(elapsed_time):.4f} ± {np.std(elapsed_time):.4f}\n"
        )
        result_file.write("=========================\n")

if "gromov" in methods:
    l1 = []
    distortion = []
    elapsed_time = []

    for j in indices:
        start_time = time.time()
        distances_gromov = gromov_tree(distances, j)
        elapsed_time.append(time.time() - start_time)
        l1.append(np.abs(distances - distances_gromov).sum() / denom)
        distortion.append(np.abs(distances - distances_gromov).max())

    avg_l1 = np.mean(l1)
    std_l1 = np.std(l1)
    avg_distortion = np.mean(distortion)
    std_distortion = np.std(distortion)

    # Save Gromov Tree Results
    with open(f"./expes/results_expes/benchmark_{dataset}.txt", "a") as result_file:
        result_file.write("== GROMOV TREE RESULTS ==\n")
        result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
        result_file.write(
            f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n"
        )
        result_file.write(
            f"Average Time: {np.mean(elapsed_time):.4f} ± {np.std(elapsed_time):.4f}\n"
        )
        result_file.write("==========================\n")

if "treerep" in methods:
    l1 = []
    distortion = []
    elapsed_time = []

    for _ in range(num_expe):
        start_time = time.time()
        tree_TR = TreeRep(distances)
        tree_TR.learn_tree()
        for e in tree_TR.G.edges():
            if tree_TR.G[e[0]][e[1]]["weight"] < 0:
                tree_TR.G[e[0]][e[1]]["weight"] = 0
        distances_tr = dict(nx.shortest_path_length(tree_TR.G, weight="weight"))
        distances_tr = np.array(
            [
                [
                    distances_tr[i][j] if j in distances_tr[i] else float("inf")
                    for j in range(num_nodes)
                ]
                for i in range(num_nodes)
            ]
        )
        elapsed_time.append(time.time() - start_time)
        l1.append(np.abs(distances - distances_tr).sum() / denom)
        distortion.append(np.abs(distances - distances_tr).max())

    avg_l1 = np.mean(l1)
    std_l1 = np.std(l1)
    avg_distortion = np.mean(distortion)
    std_distortion = np.std(distortion)

    # Save TreeRep Results
    with open(f"./expes/results_expes/benchmark_{dataset}.txt", "a") as result_file:
        result_file.write("== TREEREP RESULTS ==\n")
        result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
        result_file.write(
            f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n"
        )
        result_file.write(
            f"Average Time: {np.mean(elapsed_time):.4f} ± {np.std(elapsed_time):.4f}\n"
        )
        result_file.write("==========================\n")

if "layering" in methods and dataset not in ["zeisel", "cbmc", "microbiote"]:
    l1 = []
    distortion = []
    elapsed_time = []

    for j in indices:
        start_time = time.time()
        layering_tree = layering_approx_tree(G_largest_cc, j)
        distances_lt = dict(nx.shortest_path_length(layering_tree, weight="weight"))
        distances_lt = np.array(
            [
                [
                    distances_lt[i][j] if j in distances_lt[i] else float("inf")
                    for j in range(num_nodes)
                ]
                for i in range(num_nodes)
            ]
        )
        elapsed_time.append(time.time() - start_time)
        l1.append(np.abs(distances - distances_lt).sum() / denom)
        distortion.append(np.abs(distances - distances_lt).max())

    avg_l1 = np.mean(l1)
    std_l1 = np.std(l1)
    avg_distortion = np.mean(distortion)
    std_distortion = np.std(distortion)

    # Save Layering Tree Results
    with open(f"./expes/results_expes/benchmark_{dataset}.txt", "a") as result_file:
        result_file.write("== LAYERING TREE RESULTS ==\n")
        result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
        result_file.write(
            f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n"
        )
        result_file.write(
            f"Average Time: {np.mean(elapsed_time):.4f} ± {np.std(elapsed_time):.4f}\n"
        )
        result_file.write("==========================\n")
