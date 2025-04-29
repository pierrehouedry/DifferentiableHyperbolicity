# %%
import numpy as np
import networkx
import torch
import pickle
import os
import re
from hyperbolicity.utils import construct_weighted_matrix
from hyperbolicity.tree_fitting_methods.gromov import gromov_tree
import pandas as pd
import argparse

# c_elegan = 'D_celegan.pkl'
# c_elegan_path = os.path.join(base_path, c_elegan)
# with open(c_elegan_path, 'rb') as f:
#     distances = pickle.load(f)
# distances = torch.tensor(distances).to('cuda').type(torch.float32)
# num_nodes = distances.shape[0]
# edges = torch.triu_indices(num_nodes, num_nodes, offset=1)


class NanError(Exception):
    pass

score_fields = [
    "n_epochs",
    "intermediate_distortion",
    "intermediate_l1",
    "mean_optim_l1",
    "min_optim_l1",
    "std_optim_l1",
    "mean_optim_distortion",
    "min_optim_distortion",
    "std_optim_distortion",
]


def scores(res, indices, distances):
    # indices = np.random.choice(num_nodes, size=100, replace=False)

    is_nan = res['nan']
    n_epochs_attained = len(res['loss'])

    if not is_nan:
        D_gt = distances
        weights = res['weights']
        num_nodes = D_gt.shape[0]
        edges = torch.triu_indices(num_nodes, num_nodes, offset=1)

        denom = num_nodes * (num_nodes - 1)
        D_hat = construct_weighted_matrix(weights, num_nodes, edges)
        D_hat_cpu = D_hat.numpy()
        D_gt_cpu = D_gt.numpy()

        intermediate_distortion = torch.abs(D_hat - D_gt).max().item()
        intermediate_l1 = torch.abs(D_hat - D_gt).mean().item()

        optim_l1 = []
        optim_distortion = []

        for j in indices:
            T_hat = gromov_tree(D_hat_cpu, j)
            optim_distortion.append(np.abs(T_hat - D_gt_cpu).max())
            optim_l1.append(np.abs(T_hat - D_gt_cpu).sum() / denom)

        def stats(x): return (np.mean(x), np.min(x), np.std(x))

        return (
            n_epochs_attained,
            intermediate_distortion,
            intermediate_l1,
            *stats(optim_l1),
            *stats(optim_distortion),
        )
    else:
        return (
            n_epochs_attained,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )


# %%
if __name__ == '__main__':
    # Read and write the results for one experiment
    # Main parameters
    parser = argparse.ArgumentParser(
        description='Read results for tree fitting experiment')
    parser.add_argument('-r', '--dir', nargs='?',
                        type=str, help='the path to the directory of the experiments', required=True)
    parser.add_argument('-re', '--expe_folder', nargs='?',
                        type=str, help='the name of the expe folder in the directory', required=True)
    parser.add_argument('-s', '--seed', nargs='?', type=int,
                        help='Seed for the perf (graines)', default=42)
    parser.add_argument('-ng', '--n_graines', nargs='?', type=int,
                        help='Number of graines', default=100)
    
    parser.add_argument('-ds', '--dataset', nargs='?', type=str,
                        help='Name of the experiment', default='')

    args = parser.parse_args()
    np.random.seed(args.seed)
    path = args.dir
    expe_folder = args.expe_folder
    dataset = args.dataset

    base_path = '/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/datasets/'

    if dataset == 'celegan':
        dataset_file = 'D_celegan.pkl'
    elif dataset == 'phd':
        dataset_file = 'D_csphd.pkl'
    elif dataset == 'cora':
        dataset_file = 'D_cora.pkl'
    elif dataset == 'airport':
        dataset_file = 'D_airport.pkl'
    else:
        raise ValueError(f"Unknown experiment name: {dataset}")

    dataset_path = os.path.join(base_path, dataset_file)
    with open(dataset_path, 'rb') as f:
        distances = pickle.load(f)
    distances = torch.tensor(distances)

    with open(path+expe_folder+'/res.pickle', 'rb') as f:
        res = pickle.load(f)
    num_nodes = distances.shape[0]
    indices = np.random.choice(num_nodes, size=args.n_graines, replace=False)

    the_scores = scores(res, indices, distances)
    score_dict = dict(zip(score_fields, the_scores)) if the_scores else {}
    full_score = {key: score_dict.get(key, np.nan) for key in score_fields}

    # Find all matches
    pattern = r'([a-zA-Z_]+)-([0-9\.]+|[a-zA-Z_]+)'
    matches = re.findall(pattern, expe_folder)
    # Convert matches to a dictionary
    parsed_dict = {key: (float(value) if '.' in value else int(value) if value.isdigit() else value)
                   for key, value in matches}
    full_score.update(parsed_dict)

    df = pd.DataFrame([full_score])
    df.to_csv(path+expe_folder+"/hyperbolicity_results.csv", index=False, float_format="%.6f")
    print('---- Done ----')

