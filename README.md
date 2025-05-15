# DifferentiableHyperbolicity

A Python library for computing and optimizing tree metrics using differentiable hyperbolicity. This repository implements various tree fitting methods and provides tools for analyzing graph hyperbolicity.

## Installation

Install the package using pip:

```bash
pip install -e .
```

## Features

- Multiple tree fitting algorithms:
  - Neighbor Joining (NJ)
  - TreeRep
  - HCCRootedTreeFit
  - Gromov Tree Construction
  - Layering Tree Approximation
  - DeltaZero (our differentiable method)

- Graph analysis tools:
  - Hyperbolicity computation
  - Distance matrix manipulation
  - Tree metric optimization

## Usage Examples

```python
import numpy as np
import networkx as nx
from differentiable_hyperbolicity.tree_fitting_methods.neighbor_joining import NJ
from differentiable_hyperbolicity.tree_fitting_methods.deltazero import deltazero_tree
from differentiable_hyperbolicity.tree_fitting_methods.treerep import TreeRep

# Create a distance matrix from your graph
graph = nx.random_geometric_graph(10, 0.5)
distances = nx.floyd_warshall_numpy(graph)

# Fit trees using different methods
# Neighbor Joining
tree_nj = NJ(distances)

# TreeRep
tree_TR = TreeRep(distances)
tree_TR.learn_tree()

# DeltaZero(our differentiable method)
opt_tree = deltazero_tree(torch.tensor(distances),
                  root=0,
                  lr=0.1,
                  scale_delta=1e-2,
                  distance_reg=1,
                  batch_size=10,
                  n_batches=1,
                  num_epochs=2000)
```

## Repository Structure

```
.
├── differentiable_hyperbolicity/
│   ├── delta.py            # Hyperbolicity computation
│   ├── utils.py            # Utility functions
│   ├── tree_fitting_methods/ # Different tree fitting implementations
└── expes/                  # Experiments and analysis notebooks
```

## License

This project is licensed under the MIT License.
