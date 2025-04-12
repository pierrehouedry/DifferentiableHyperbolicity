import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

from delta import logsumexp, soft_max

def convert_adjacency_matrix(adj_matrix, large_v=torch.inf):#torch.inf):
    """
    Converts an adjacency matrix to a metric matrix.
    """
    metric = adj_matrix.clone().float()
    metric[~adj_matrix] = large_v
    metric.fill_diagonal_(1)
    return metric

class HyperbolicGraph():
    '''
    Class to handle hyperbolic graphs
    '''

    def __init__(self, G, F=None, mode='weight', device='cpu'):
        super(HyperbolicGraph, self).__init__()

        self.device = device
        # A graph is defined by its metric and its feature information

        # topological part
        self.G = G  # is a networkx graph
        self.num_nodes = self.G.number_of_nodes()
        self.edges = torch.tensor(list(self.G.edges()),device=device)
        self.n_edges = self.edges.shape[0]
        self.adjacency_matrix = self.compute_adjacency_matrix()
        self.metric_adjacency_matrix = convert_adjacency_matrix(self.adjacency_matrix)

        self.metric = torch.zeros(self.num_nodes, self.num_nodes, device=device)


        # those are the quantities we are going to optimize
        self.weight_by_edges = torch.ones(self.n_edges, device=device)

        # feature vector
        if F is not None:
            self.F = F.clone().to(device)
        else:
            self.F =  torch.zeros(self.num_nodes, device=device)

        # Mode of optimization
        # can be 'weight' or 'feature' or 'both'
        self.mode=mode
        self.set_mode_optim()

        # properties of the graph
        self.diameter = self.get_diameter()

    # choose which parameter is going to be updated when optimizing hyperbolicity
    def set_mode_optim(self):
        if self.mode=='weight':
            self.weight_by_edges.requires_grad_(True)
            self.F.requires_grad_(False)
        elif self.mode=='feature':
            self.weight_by_edges.requires_grad_(False)
            self.F.requires_grad_(True)
        elif self.mode=='both':
            self.weight_by_edges.requires_grad_(True)
            self.F.requires_grad_(True)

    def zero_grad(self):
        if self.weight_by_edges.grad is not None:
            self.weight_by_edges.grad.detach_()
            self.weight_by_edges.grad.zero_()

        if self.F.grad is not None:
            self.F.grad.detach_()
            self.F.grad.zero_()

    def get_diameter(self):
        self.update_metric_sp()
        return torch.max(self.metric)

    def compute_adjacency_matrix(self):
        adj_matrix =  torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool, device=self.device)
        adj_matrix[self.edges[:, 0], self.edges[:, 1]] = True
        adj_matrix[self.edges[:, 1], self.edges[:, 0]] = True
        return adj_matrix

    def get_normalized_Laplacian(self):
        degree_matrix = torch.diag(self.adjacency_matrix.int().sum(dim=1))
        degree_inv_sqrt = torch.diag(torch.pow(degree_matrix.diag(), -0.5))
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # Handle isolated nodes
        laplacian = torch.eye(self.num_nodes, device=self.device) - degree_inv_sqrt @ self.adjacency_matrix.float() @ degree_inv_sqrt
        return laplacian

    def update_metric_sp(self, scale=0, bigv=torch.inf):
        if self.mode=='weight':
            self.metric = torch.zeros_like(self.metric,device=self.device)
            self.metric[self.edges[:, 0], self.edges[:, 1]] =  torch.relu(self.weight_by_edges)
            self.metric = self.metric + self.metric.T
            self.metric[self.metric==0] = bigv
            self.metric.fill_diagonal_(0.0)
            #self.metric = torch.mul(self.metric_adjacency_matrix,self.metric)
        elif self.mode=='feature':
            assert self.F is not None, '[update_metric_sp] No Feature vector detected'
            self.metric= torch.cdist(self.F,self.F,p=2).to(self.device)
            #self.metric[~self.adjacency_matrix | torch.eye(*self.metric.shape).bool()] = torch.inf
            self.metric = torch.mul(self.metric_adjacency_matrix,self.metric)


        if scale==0:
            self.update_metric_Floyd_Warshall()
        else:
            self.update_metric_DataSP(scale)

    def init_metric_from_weight_edges(self):
        self.weight_matrix[self.edges[:, 0], self.edges[:, 1]] =  self.weight_by_edges
        self.weight_matrix[self.edges[:, 1], self.edges[:, 0]] =  self.weight_by_edges

    def update_metric_Floyd_Warshall(self):
        for k in range(self.num_nodes):
            self.metric = torch.minimum(self.metric[:, k].unsqueeze(1) + self.metric[k, :].unsqueeze(0), self.metric)

    def update_metric_DataSP(self,scale=100):
        for k in range(self.num_nodes):
            self.metric = soft_max(torch.stack([self.metric[:, k].unsqueeze(1) + self.metric[k, :].unsqueeze(0), self.metric], dim=-1), scale=-scale)
        self.metric = self.metric - self.metric.min()

    ##---------------\delta-hyperbolicity related methods

    def filter_farapart_memory_inefficient(self):
    # memory inefficient but rapid if you can !
        N = self.num_nodes
        # Create index grids
        u_idx, v_idx, w_idx = torch.meshgrid(torch.arange(N), torch.arange(N), torch.arange(N), indexing="ij")
        # Conditions
        cond1 = self.metric[w_idx, u_idx] + self.metric[u_idx, v_idx] > self.metric[w_idx, v_idx]
        cond2 = self.metric[w_idx, v_idx] + self.metric[u_idx, v_idx] > self.metric[w_idx, u_idx]
        # Ensure w ≠ u and w ≠ v
        valid_w = (w_idx != u_idx) & (w_idx != v_idx)
        # Apply conditions only for valid w
        Fpart = torch.all(cond1 & cond2 | ~valid_w, dim=2)

        ind = Fpart.tril(diagonal=0).nonzero()
        return torch.where(Fpart.any(dim=1))[0], ind

    def filter_farapart(self):
        N = self.num_nodes
        Fpart = torch.ones((N, N), dtype=torch.bool)  # Initialize to True

        # Precompute u_idx and v_idx once (saves memory & computation)
        u_idx, v_idx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")

        # Loop over w instead of creating (N, N, N) tensors
        for w in range(N):
            # Conditions
            cond1 = self.metric[w, u_idx] + self.metric[u_idx, v_idx] > self.metric[w, v_idx]
            cond2 = self.metric[w, v_idx] + self.metric[u_idx, v_idx] > self.metric[w, u_idx]

            # Ignore w == u or w == v
            valid_w = (w != u_idx) & (w != v_idx)

            # Update Fpart using element-wise AND
            Fpart &= cond1 & cond2 | ~valid_w

        # Extract indices efficiently
        ind = Fpart.tril(diagonal=0).nonzero()
        return torch.where(Fpart.any(dim=1))[0], ind

    def filter_farapart_memory_efficient(self):
        # memory efficient but slow as hell !
        Fpart = torch.ones(self.num_nodes,self.num_nodes,dtype=torch.bool)
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                for w in range(self.num_nodes):
                    if  (w != u) and (w != v):
                        if not(self.metric[w,u] + self.metric[u,v] > self.metric[w,v]) or not(self.metric[w,v] + self.metric[u,v] > self.metric[w,u]):
                            Fpart[u,v]=False
        ind = Fpart.tril(diagonal=0).nonzero()
        return torch.where(Fpart.any(dim=1))[0], ind

    def compute_hyperbolicity(self,M,scale=0):
            # Compute S1, S2, S3 for all combinations (i, j, k, l)
        S1 = M.unsqueeze(2).unsqueeze(3) + M.unsqueeze(0).unsqueeze(1)
        S2 = M.unsqueeze(1).unsqueeze(3) + M.unsqueeze(0).unsqueeze(2)
        S3 = M.unsqueeze(1).unsqueeze(2) + M.unsqueeze(0).unsqueeze(3)

        # Stack S1, S2, S3 along a new dimension
        Stot = torch.stack([S1, S2, S3], dim=-1)

        # Sort Stot along the last dimension and compute the difference
        Stot_sorted = Stot.sort(dim=-1, descending=True)[0]
        delta = (Stot_sorted[..., 0] - Stot_sorted[..., 1]) / 2
        # Find the maximum value of delta
        if scale:
            return soft_max(delta,scale,dim=(0,1,2,3))
        else:
            return torch.max(delta)

    def compute_exact_hyperbolicity_naive(self):
        maxi=torch.tensor([0])
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                for k in range(self.num_nodes):
                    for l in range(self.num_nodes):
                        S1 = self.metric[i,j] + self.metric[k,l]
                        S2 = self.metric[i,k] + self.metric[j,l]
                        S3 = self.metric[i,l] + self.metric[j,k]
                        #print(S1,S2,S3)
                        Stot = torch.stack([S1, S2,S3], dim=-1)
                        Stot = Stot.sort(descending=True)[0]
                        #print("i={}, j={} delta={}".format(i,j,(Stot[0]-Stot[1])/2))
                        maxi = torch.max(maxi,(Stot[0]-Stot[1])/2)
        return maxi

    def compute_hyperbolicity_from_pairs(self,ind, scale=0):
        M = self.metric  # (N, N)
        P = ind.shape[0]  # Number of index pairs

        # Extract (x, y) index pairs
        x, y = ind[:, 0], ind[:, 1]  # Shape: (P,)

        # Expand (x, y) to compare against all (u, v) pairs
        x_exp, y_exp = x[:, None], y[:, None]  # Shape: (P, 1)
        u, v = ind[:, 0], ind[:, 1]  # Shape: (P,)

        # Compute S1, S2, S3 efficiently
        S1 = self.metric[x_exp, y_exp] + self.metric[u[None, :], v[None, :]]  # (P, P)
        S2 = self.metric[x_exp, u[None, :]] + self.metric[y_exp, v[None, :]]  # (P, P)
        S3 = self.metric[x_exp, v[None, :]] + self.metric[y_exp, u[None, :]]  # (P, P)

        # Stack and sort
        Stot = torch.stack([S1, S2, S3], dim=-1)  # Shape: (P, P, 3)
        Stot_sorted = Stot.sort(dim=-1, descending=True)[0]

        # Compute K
        K = (Stot_sorted[..., 0] - Stot_sorted[..., 1]) / 2  # Shape: (P, P)

        # Get the maximum value for each pair
        #return K.max()
        if scale:
            return soft_max(K,scale,dim=(0,1))
        else:
            return torch.max(K)

    def compute_h_with_filter(self,scale=0):
        idx,_ = self.filter_farapart()
        return self.compute_hyperbolicity(self.metric[idx,:][:,idx],scale)

    def compute_h_with_filter2(self,scale=0):
        _,ind = self.filter_farapart()
        return self.compute_hyperbolicity_from_pairs(ind,scale)

    def compute_h_with_filter3(self,scale=0):
        _,ind = self.filter_farapart_memory_efficient()
        return self.compute_hyperbolicity_from_pairs(ind,scale)

    def check_all_fapart_pairs(self, ind):
    # computation intensive but memory efficient
        for p in range(ind.shape[0]):
            u = ind[p,0]
            v = ind[p,1]
            for i in range(HG.num_nodes):
                if (i != u) and (i != v):
                    assert self.metric[u,i] + self.metric[u,v] > self.metric[i,v]
                    assert self.metric[i,v] + self.metric[u,v] > self.metric[i,u]

    def gromov_product_from_distances(self, i,j,k):
        d_i_k = self.metric[i, k]
        d_j_k = self.metric[j, k]
        d_i_j = self.metric[i, j]
        return (d_i_k + d_j_k - d_i_j) / 2

    def compute_delta_from_distances(self, scale):
        idx = torch.cartesian_prod(
            torch.arange(self.num_nodes),
            torch.arange(self.num_nodes),
            torch.arange(self.num_nodes),
            torch.arange(self.num_nodes)
        )

        gp01_3 = self.gromov_product_from_distances(idx[:, 0], idx[:, 1], idx[:, 3])
        gp12_3 = self.gromov_product_from_distances(idx[:, 1], idx[:, 2], idx[:, 3])
        gp02_3 = self.gromov_product_from_distances(idx[:, 0], idx[:, 2], idx[:, 3])

        minimum = torch.stack([gp01_3, gp12_3], dim=-1)
        results = soft_max(minimum, -scale) - gp02_3

        return soft_max(results, scale)

    def compute_exact_delta(self, scale):
        idx = torch.cartesian_prod(
            torch.arange(self.num_nodes),
            torch.arange(self.num_nodes),
            torch.arange(self.num_nodes),
            torch.arange(self.num_nodes)
        )

        gp01_3 = self.gromov_product_from_distances(idx[:, 0], idx[:, 1], idx[:, 3])
        gp12_3 = self.gromov_product_from_distances(idx[:, 1], idx[:, 2], idx[:, 3])
        gp02_3 = self.gromov_product_from_distances(idx[:, 0], idx[:, 2], idx[:, 3])

        minimum = torch.stack([gp01_3, gp12_3], dim=-1)
        results = soft_max(minimum, -scale) - gp02_3

        return soft_max(results, scale)

    def delta_hyperbolicity_fixed_basepoint(self,base_point, alpha, soft=True):
        row = self.metric[base_point,:]
        XX_p = 0.5 * (row.unsqueeze(0) + row.unsqueeze(1) - self.metric) # could be optimized if base_point is 0
        # naive  implem
        return torch.logsumexp(alpha*(torch.min(XX_p[:, :, None], XX_p[None, :, :])-XX_p[:, None, :]), dim=(0,1,2))/alpha
