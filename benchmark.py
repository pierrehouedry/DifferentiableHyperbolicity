import pickle
import os
import torch
import torch.optim as optim
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from tqdm import tqdm 

n_experiments = 100

base_path = 'hyperbolic/DifferentiableHyperbolicitiy/datasets'

# Load the datasets
c_elegan = 'D_celegan.pkl'
c_elegan_path = os.path.join(base_path, c_elegan)
with open(c_elegan_path, 'rb') as f:
    c_elegan_distances = pickle.load(f)

cs_phd = 'D_csphd.pkl'
cs_phd_path = os.path.join(base_path, cs_phd)
with open(cs_phd_path, 'rb') as f:
    cs_phd_distances = pickle.load(f)

airport = 'airport/airport.p'
airport_path = os.path.join(base_path, airport)
with open(cs_phd_path, 'rb') as f:
    aiport_graph = pickle.load(f)
    airport_distances = nx.floyd_warshall_numpy(aiport_graph)

cora = 'cora'
cora_path = os.path.join(base_path, cora)
cora_dataset = Planetoid(root=cora_path, name='Cora')
cora_graph = to_networkx(cora_dataset[0], to_undirected=True)
cora_distances = nx.floyd_warshall_numpy(cora_graph)

print("Datasets Loaded:")
print(f"- C. Elegans Distances: {len(c_elegan_distances)} nodes")
print(f"- CS PhD Distances: {len(cs_phd_distances)} nodes")
print(f"- Airport Graph: {len(aiport_graph.nodes)} nodes, {len(aiport_graph.edges)} edges")
print(f"- Cora Graph: {len(cora_graph.nodes)} nodes, {len(cora_graph.edges)} edges")

# Create the optimizing funciton
def train_distance_matrix(distances: torch.Tensor,
                        scale_sp: float,
                        scale_delta: float,
                        scale_soft_max: float,
                        distance_reg: float,
                        num_epochs: int = 100,
                        learning_rate: float = 0.01,
                        patience: int = 10) -> torch.Tensor:

    num_nodes = distances.shape[0]
    edges = torch.triu_indices(num_nodes, num_nodes, offset=1)
    upper_adjency = torch.triu(distances, diagonal=1)
    weights_opt = upper_adjency[upper_adjency!=0].float().clone().detach().requires_grad_(True)
    optimizer = optim.Adam([weights_opt], lr=learning_rate)

    def loss_fn(w):

        adj = construct_weighted_matrix(w, num_nodes, edges)
        sp_matrix = datasp(adj, scale_sp)
        M_batch = make_batches(HG.metric, 32, 32)
        delta = soft_max(compute_hyperbolicity_batch(M_batch,scale=scale_delta),scale=scale_soft_max)

        return  delta + (distances-sp_matrix).pow(2).sum()

    best_loss = float('inf')
    best_weights = None
    patience_counter = 0

    with tqdm(range(num_epochs), desc="Training Weights") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            loss = loss_fn(weights_opt)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = weights_opt.detach().clone()
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            pbar.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step()

    return best_weights.detach(), best_loss

#Launch the experiments
print("\n Launching Experiments...\n")

datasets = {
    "C. Elegans": torch.tensor(c_elegan_distances),
    "CS PhD": torch.tensor(cs_phd_distances),
    "Airport": torch.tensor(airport_distances),
    "Cora": torch.tensor(cora_distances),
}

print("All experiments completed.")