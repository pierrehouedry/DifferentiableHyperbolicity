import numpy as np
import pandas as pd
import pickle
from hyperbolicity.tree_fitting_methods.hccfit import HccLinkage
from hyperbolicity.tree_fitting_methods.gromov import gromov_tree
from hyperbolicity.tree_fitting_methods.neighbor_joining import NJ
from hyperbolicity.tree_fitting_methods.treerep import TreeRep
import networkx as nx


path_dataset = '/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/datasets/D_diseases.pkl'

with open(path_dataset, 'rb') as f:
    distances = pickle.load(f)

N = distances.shape[0]

#NJ
tree_nj = NJ(distances)
tree_nj_distances = nx.floyd_warshall_numpy(tree_nj)
distortion_nj = np.abs(tree_nj_distances[:N, :N]-distances).max()
print(f"NJ Tree Distortion: {distortion_nj:.4f}")

#TreeRep
denom = N * (N - 1)
tr_l1_errors = []
tr_distortions = []
for _ in range(100):
    tree_TR = TreeRep(distances)
    tree_TR.learn_tree()
    tree_TR_distances = nx.floyd_warshall_numpy(tree_TR.G)
    tr_distortions.append(np.abs(tree_TR_distances[:N, :N] - distances).max())
    tr_l1_errors.append(np.abs(tree_TR_distances[:N, :N] - distances).sum() / denom)

print("TreeRep Results:")
print(f"Mean L1 Error: {np.mean(tr_l1_errors):.4f}, Std L1 Error: {np.std(tr_l1_errors):.4f}")
print(f"Mean Distortion: {np.mean(tr_distortions):.4f}, Std Distortion: {np.std(tr_distortions):.4f}")

np.random.seed(42)
indices = np.random.choice(N, size=100, replace=False)

#HCC
hcc_l1_errors = []
hcc_distortions = []
denom = N * (N - 1)

for root in indices:
    tree_hcc = HccLinkage(distances)
    tree_hcc.fit_tree(root)
    hcc_distortions.append(np.abs(tree_hcc.d_T - distances).max())
    hcc_l1_errors.append(np.abs(tree_hcc.d_T - distances).sum() / denom)

print("HCC Linkage Results:")
print(f"Mean L1 Error: {np.mean(hcc_l1_errors):.4f}, Std L1 Error: {np.std(hcc_l1_errors):.4f}")
print(f"Mean Distortion: {np.mean(hcc_distortions):.4f}, Std Distortion: {np.std(hcc_distortions):.4f}")


#Gromov
gromov_l1_errors = []
gromov_distortions = []

for root in indices:
    tree_gromov = gromov_tree(distances, root)
    gromov_distortions.append(np.abs(tree_gromov - distances).max())
    gromov_l1_errors.append(np.abs(tree_gromov - distances).sum() / denom)

print("\nGromov Tree Results:")
print(f"Mean L1 Error: {np.mean(gromov_l1_errors):.4f}, Std L1 Error: {np.std(gromov_l1_errors):.4f}")
print(f"Mean Distortion: {np.mean(gromov_distortions):.4f}, Std Distortion: {np.std(gromov_distortions):.4f}")