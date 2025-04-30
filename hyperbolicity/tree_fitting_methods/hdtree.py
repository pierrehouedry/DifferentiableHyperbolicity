import torch
from hyperbolicity.tree_fitting_methods.gromov import gromov_tree
from hyperbolicity.utils import construct_weighted_matrix
from hyperbolicity.tree_fitting_methods.hyperbolicity_learning import train_distance_matrix

def hdtree(distances: torch.Tensor, 
           root: int, 
           lr: float, 
           n_batches: int, 
           batch_size: int, 
           scale_delta: float, 
           distance_reg: float,
           num_epochs: int,
           gpu: bool,
           verbose: bool):
    
       weights, losses, deltas, errors,duratio = train_distance_matrix(distances, scale_delta, distance_reg, num_epochs, n_batches, batch_size, lr, verbose, gpu)

       num_nodes = distances.shape[0]
       edges = torch.triu_indices(num_nodes, num_nodes, offset=1)
       distance_optimized = construct_weighted_matrix(weights, num_nodes, edges)
       distance_optimized_cpu = distance_optimized.cpu().numpy()
       T_opt = gromov_tree(distance_optimized_cpu, root)

       return T_opt