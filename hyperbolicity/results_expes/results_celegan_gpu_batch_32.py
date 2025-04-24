import numpy as np
import networkx
import torch
import pickle
import os
import re
from hyperbolicity.utils import construct_weighted_matrix
from hyperbolicity.tree_fitting_methods.gromov import gromov_tree
import pandas as pd

base_path = '/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/datasets'

c_elegan = 'D_celegan.pkl'
c_elegan_path = os.path.join(base_path, c_elegan)
with open(c_elegan_path, 'rb') as f:
    distances = pickle.load(f)
distances = torch.tensor(distances).to('cuda').type(torch.float32)
num_nodes = distances.shape[0]
edges = torch.triu_indices(num_nodes, num_nodes, offset=1)

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

def scores(p):
    weights = p['weights']
    n_epochs = len(p['loss'])

    denom = num_nodes * (num_nodes - 1)
    D_gt = distances.cpu().numpy()
    W = construct_weighted_matrix(weights, num_nodes, edges)
    D_hat = W.cpu().numpy()

    intermediate_distortion = np.abs(D_hat - D_gt).max()
    intermediate_l1 = np.abs(D_hat - D_gt).mean()

    indices = np.random.choice(num_nodes, size=100, replace=False)

    optim_l1 = []
    optim_distortion = []
    no_optim_l1 = []
    no_optim_distortion = []

    for j in indices:
        T_hat = gromov_tree(D_hat, j)
        T_gt = gromov_tree(D_gt, j)

        optim_distortion.append(np.abs(T_hat - D_gt).max())
        optim_l1.append(np.abs(T_hat - D_gt).sum() / denom)

        no_optim_distortion.append(np.abs(T_gt - D_gt).max())
        no_optim_l1.append(np.abs(T_gt - D_gt).sum() / denom)

    def stats(x): return (np.mean(x), np.min(x), np.std(x))

    return (
        n_epochs,
        intermediate_distortion,
        intermediate_l1,
        *stats(optim_l1),
        *stats(no_optim_l1),
        *stats(optim_distortion),
        *stats(no_optim_distortion)
    )

base_folder = "/share/home/houedry/projects/DifferentiableHyperbolicity/hyperbolicity/results_expes/expe_celegan_gpu_batch_32/expe_celegan_gpu"  
results = []

pattern = re.compile(
    r"learning_rate-(?P<learning_rate>[\d\.eE+-]+)(?=-distance_reg|-[a-zA-Z_]+=|\Z).*?"
    r"distance_reg-(?P<distance_reg>[\d\.eE+-]+)(?=-scale_delta|-[a-zA-Z_]+=|\Z).*?"
    r"scale_delta-(?P<scale_delta>[\d\.eE+-]+)(?=-|$)"
)

for root, dirs, files in os.walk(base_folder):
    for name in dirs:
        if name.startswith("launch_distance_hyperbolicity_learning"):
            folder_path = os.path.join(root, name)
            match = pattern.search(name)
            if not match:
                print(f"Skipping (pattern mismatch): {name}")
                continue

            scale_delta = float(match.group("scale_delta"))
            distance_reg = float(match.group("distance_reg"))
            learning_rate = float(match.group("learning_rate"))

            res_path = os.path.join(folder_path, "res.pickle")
            if os.path.exists(res_path):
                try:
                    with open(res_path, 'rb') as f:
                        data = pickle.load(f)
                        values = scores(data)

                        score_dict = dict(zip(score_fields, values)) if values else {}
                        full_score = {key: score_dict.get(key, np.nan) for key in score_fields}

                        full_score.update({
                            "scale_delta": scale_delta,
                            "distance_reg": distance_reg,
                            "learning_rate": learning_rate,
                        })

                        results.append(full_score)
                        print(f"Processed: {folder_path}")
                except Exception as e:
                    print(f" Error in {folder_path}: {e}")

df = pd.DataFrame(results)
df.to_csv("hyperbolicity_results.csv", index=False, float_format="%.6f")

if "mean_optim_distortion" in df.columns:
    best_row = df.loc[df["mean_optim_distortion"].idxmin()]
    print("\nBest Hyperparameters based on mean_optim_distortion:")
    print(f"  scale_delta       : {best_row['scale_delta']}")
    print(f"  distance_reg      : {best_row['distance_reg']}")
    print(f"  learning_rate     : {best_row['learning_rate']}")
    print(f"  n_epochs          : {int(best_row['n_epochs'])}")
    print(f"  mean_optim_distortion : {best_row['mean_optim_distortion']:.6f}")
    print(f"  mean_no_optim_distortion : {best_row['mean_no_optim_distortion']:.6f}")
else:
    print("Column 'mean_optim_distortion' not found in results.")

