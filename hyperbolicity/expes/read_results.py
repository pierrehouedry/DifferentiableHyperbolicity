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
    "mean_no_optim_l1",
    "min_no_optim_l1",
    "std_no_optim_l1",
    "mean_optim_distortion",
    "min_optim_distortion",
    "std_optim_distortion",
    "mean_no_optim_distortion",
    "min_no_optim_distortion",
    "std_no_optim_distortion",
]


def scores(res, indices):
    # indices = np.random.choice(num_nodes, size=100, replace=False)

    is_nan = res['nan']
    n_epochs_attained = len(res['loss'])

    if not is_nan:
        D_gt = res['distances']
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
        no_optim_l1 = []
        no_optim_distortion = []

        for j in indices:
            T_hat = gromov_tree(D_hat_cpu, j)
            T_gt = gromov_tree(D_gt_cpu, j)

            optim_distortion.append(np.abs(T_hat - D_gt_cpu).max())
            optim_l1.append(np.abs(T_hat - D_gt_cpu).sum() / denom)

            no_optim_distortion.append(np.abs(T_gt - D_gt_cpu).max())
            no_optim_l1.append(np.abs(T_gt - D_gt_cpu).sum() / denom)

        def stats(x): return (np.mean(x), np.min(x), np.std(x))

        return (
            n_epochs_attained,
            intermediate_distortion,
            intermediate_l1,
            *stats(optim_l1),
            *stats(no_optim_l1),
            *stats(optim_distortion),
            *stats(no_optim_distortion)
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

    args = parser.parse_args()
    np.random.seed(args.seed)
    path = args.dir
    expe_folder = args.expe_folder

    with open(path+expe_folder+'/res.pickle', 'rb') as f:
        res = pickle.load(f)
    num_nodes = res['distances'].shape[0]
    indices = np.random.choice(num_nodes, size=args.n_graines, replace=False)

    the_scores = scores(res, indices)
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


# base_folder = "/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/results_expes/expe_celegan_gpu_batch_32/expe_celegan_gpu"
# results = []

# pattern = re.compile(
#     r"learning_rate-(?P<learning_rate>[\d\.eE+-]+)(?=-distance_reg|-[a-zA-Z_]+=|\Z).*?"
#     r"distance_reg-(?P<distance_reg>[\d\.eE+-]+)(?=-scale_delta|-[a-zA-Z_]+=|\Z).*?"
#     r"scale_delta-(?P<scale_delta>[\d\.eE+-]+)(?=-|$)"
# )

# for root, dirs, files in os.walk(base_folder):
#     for name in dirs:
#         if name.startswith("launch_distance_hyperbolicity_learning"):
#             folder_path = os.path.join(root, name)
#             match = pattern.search(name)
#             if not match:
#                 print(f"Skipping (pattern mismatch): {name}")
#                 continue

#             scale_delta = float(match.group("scale_delta"))
#             distance_reg = float(match.group("distance_reg"))
#             learning_rate = float(match.group("learning_rate"))

#             res_path = os.path.join(folder_path, "res.pickle")
#             if os.path.exists(res_path):
#                 try:
#                     with open(res_path, 'rb') as f:
#                         data = pickle.load(f)
#                         values = scores(data)

#                         score_dict = dict(zip(score_fields, values)) if values else {}
#                         full_score = {key: score_dict.get(key, np.nan) for key in score_fields}

#                         full_score.update({
#                             "scale_delta": scale_delta,
#                             "distance_reg": distance_reg,
#                             "learning_rate": learning_rate,
#                         })

#                         results.append(full_score)
#                         print(f"Processed: {folder_path}")
#                 except Exception as e:
#                     print(f" Error in {folder_path}: {e}")

# df = pd.DataFrame(results)
# df.to_csv("hyperbolicity_results.csv", index=False, float_format="%.6f")

# if "mean_optim_distortion" in df.columns:
#     best_row = df.loc[df["mean_optim_distortion"].idxmin()]
#     print("\nBest Hyperparameters based on mean_optim_distortion:")
#     print(f"  scale_delta       : {best_row['scale_delta']}")
#     print(f"  distance_reg      : {best_row['distance_reg']}")
#     print(f"  learning_rate     : {best_row['learning_rate']}")
#     print(f"  n_epochs          : {int(best_row['n_epochs'])}")
#     print(f"  mean_optim_distortion : {best_row['mean_optim_distortion']:.6f}")
#     print(f"  mean_no_optim_distortion : {best_row['mean_no_optim_distortion']:.6f}")
# else:
#     print("Column 'mean_optim_distortion' not found in results.")
