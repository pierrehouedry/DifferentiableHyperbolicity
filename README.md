# Differentiable Hyperbolicity

Official implementation of the paper **“Bridging Arbitrary and Tree Metrics via Differentiable Gromov Hyperbolicity”** (accepted to NeurIPS 2025). The library provides differentiable estimators of Gromov hyperbolicity and several tree-metric fitting algorithms for graph data.


- Paper: [Bridging Arbitrary and Tree Metrics via Differentiable Gromov Hyperbolicity](https://openreview.net/forum?id=rIudtwY0VM) (NeurIPS 2025). The code in this repository was used to produce the experiments in the paper.
- If you build on this work, please cite the paper above.

![Figure 1 from the paper](expes/figures/fig1.pdf)


## Installation

```bash
pip install -e .
```

## What’s inside

- Differentiable hyperbolicity computation with smooth max/min (`differentiable_hyperbolicity/delta.py`).
- Tree fitting methods: DeltaZero (differentiable), Neighbor Joining, TreeRep, HCCRootedTreeFit, Gromov tree construction, and Layering approximation (`differentiable_hyperbolicity/tree_fitting_methods/`).
- Experiment scripts used in the paper (`expes/`) and preprocessed distance matrices (`datasets/`).

## Quick start

```python
import torch
import networkx as nx
from differentiable_hyperbolicity.delta import compute_hyperbolicity
from differentiable_hyperbolicity.tree_fitting_methods.deltazero import deltazero_tree

# Build a toy graph and its distance matrix
graph = nx.random_geometric_graph(20, 0.35, seed=0)
distances = nx.floyd_warshall_numpy(graph)
distances_t = torch.tensor(distances, dtype=torch.float32)

# Fit a tree metric with the differentiable DeltaZero method
tree = deltazero_tree(
    distances=distances_t,
    root=0,
    lr=0.1,
    n_batches=1,
    batch_size=64,
    scale_delta=1e-2,
    distance_reg=1.0,
    num_epochs=200,
    gpu=False,
    verbose=False,
)
```

## Reproducing paper experiments

- Datasets used in the paper are stored in `datasets/`.
- Benchmark scripts live in `expes/`; for example:

```bash
python expes/benchmark.py --dataset cora --methods hcc gromov treerep layering --num_expe 20
```

## Repository structure

```
.
├── differentiable_hyperbolicity/   # Core library (delta computation, utilities, tree fitting)
├── datasets/                       # Preprocessed distance matrices and raw graphs
└── expes/                          # Experiment scripts and results
```

## License

This project is licensed under the MIT License.
