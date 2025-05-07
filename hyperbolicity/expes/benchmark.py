import numpy as np
from hyperbolicity.tree_fitting_methods.treerep import TreeRep
from hyperbolicity.tree_fitting_methods.hccfit import HccLinkage
from hyperbolicity.tree_fitting_methods.gromov import gromov_tree
from hyperbolicity.tree_fitting_methods.layering_tree import layering_approx_tree
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import networkx as nx
from torch_geometric.utils import to_networkx
import torch
import pickle
from torch_geometric.datasets import AttributedGraphDataset


dataset ='zeisel'
num_expe = 100
dataset_path = '/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/datasets/D_zeisel.pkl'

with open(dataset_path, 'rb') as f:
    distances = pickle.load(f)

num_nodes = distances.shape[0]
np.random.seed(42)
indices = np.random.choice(num_nodes, num_expe, replace=False)


l1 = []
distortion = []

for j in indices:
    tree_hcc = HccLinkage(distances)
    tree_hcc.fit_tree(j)
    l1.append(np.abs(distances - tree_hcc.d_T).mean())
    distortion.append(np.abs(distances - tree_hcc.d_T).max())

avg_l1 = np.mean(l1)
std_l1 = np.std(l1)
avg_distortion = np.mean(distortion)
std_distortion = np.std(distortion)

# Save HCC Linkage Results
with open(f'/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/results_expes/benchmark_{dataset}_hcc.txt', 'w') as result_file:
    result_file.write('== HCC Linkage Results ==\n')
    result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
    result_file.write(f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n")
    result_file.write('=========================\n')


# Gromov Tree Evaluation
l1 = []
distortion = []
for j in indices:
    distances_gromov = gromov_tree(distances, j)
    l1.append(np.abs(distances - distances_gromov).mean())
    distortion.append(np.abs(distances - distances_gromov).max())

avg_l1 = np.mean(l1)
std_l1 = np.std(l1)
avg_distortion = np.mean(distortion)
std_distortion = np.std(distortion)

# Save Gromov Tree Results
with open(f'/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/results_expes/benchmark_{dataset}_gromov.txt', 'w') as result_file:
    result_file.write('== GROMOV TREE RESULTS ==\n')
    result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
    result_file.write(f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n")
    result_file.write('==========================\n')

l1 = []
distortion = []

for _ in range(num_expe):
    tree_TR = TreeRep(distances)
    tree_TR.learn_tree()
    distances_tr = nx.floyd_warshall_numpy(tree_TR.G)[:num_nodes, :num_nodes]
    l1.append(np.abs(distances - distances_tr).mean())
    distortion.append(np.abs(distances - distances_tr).max())

avg_l1 = np.mean(l1)
std_l1 = np.std(l1)
avg_distortion = np.mean(distortion)
std_distortion = np.std(distortion)

# Save TreeRep Results
with open(f'/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/results_expes/benchmark_{dataset}_treerep.txt', 'w') as result_file:
    result_file.write('== LAYERING TREEREP ==\n')
    result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
    result_file.write(f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n")
    result_file.write('==========================\n')
    

""" # Load Dataset and Extract Largest Connected Component
dataset = AttributedGraphDataset(
    '/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/datasets', 
    name='Wiki'
)
data = dataset[0]
G_data = to_networkx(data, to_undirected=True)
largest_cc_data = max(nx.connected_components(G_data), key=len)
G_largest_cc = G_data.subgraph(largest_cc_data).copy()

# Relabel Nodes for Consistency
mapping = {node: i for i, node in enumerate(G_largest_cc.nodes())}
G_largest_cc = nx.relabel_nodes(G_largest_cc, mapping)

# Evaluate Layering Approximation Tree
l1 = []
distortion = []

for j in indices:
    layering_tree = layering_approx_tree(G_largest_cc, j)
    distances_lt = nx.floyd_warshall_numpy(layering_tree)[:num_nodes, :num_nodes]
    l1.append(np.abs(distances - distances_lt).mean())
    distortion.append(np.abs(distances - distances_lt).max())

avg_l1 = np.mean(l1)
std_l1 = np.std(l1)
avg_distortion = np.mean(distortion)
std_distortion = np.std(distortion)

# Save Layering Tree Results
with open(f'/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/expes/results_expes/benchmark_{dataset}_layering.txt', 'w') as result_file:
    result_file.write('== LAYERING TREE RESULTS ==\n')
    result_file.write(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}\n")
    result_file.write(f"Average Distortion: {avg_distortion:.4f} ± {std_distortion:.4f}\n")
    result_file.write('==========================\n') """
