# import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import random
import time

# Code of the article: https://proceedings.neurips.cc/paper_files/paper/2023/file/16bce4070c4e23434451b180348e3814-Paper-Conference.pdf

class HccLinkage():

    def __init__(self, d, alt = False, rand = False, tol = 1e-5):
        self.d = d
        self.n = self.d.shape[0]
        self.alt = alt
        self.tol = tol
        self.nextroots = list(range(self.n,2*self.n))
        self.nextroots.reverse()
        self.d_U = np.zeros((self.n, self.n))
        self.A = np.zeros((self.n, self.n))
        self.N = np.zeros((self.n, 2 * self.n))
        self.H = np.zeros((self.n, 2 * self.n))
        self.M = np.zeros((2 * self.n, 2 * self.n))
        self.S = np.ones(2 * self.n)
        self.membership = np.arange(self.n)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        self.heights = np.zeros(2 * self.n)
        self.Z = np.zeros((self.n - 1, 4))
        self.fitted = False
        self.debug = False
        self.elapse_sort = 0.0
        self.elapse_fit = 0.0
        self.d_T = np.zeros((self.n, self.n))

    
    def get_edge_seq(self, d, n, rand = False):
        entries = []
        edges = []
        for i in range(n):
            for j in range(i):
                entries.append((i, j, d[i,j]))
        if rand:
            random.shuffle(entries)
        entries.sort(key = lambda e: e[2])
        for e in entries:
            edges.append((e[0],e[1]))
        return edges

    def update_matrices(self, i, j):
        k = self.membership[i]
        l = self.membership[j]
        self.A[i, j] += 1
        self.A[j, i] += 1
        self.N[i, l] += 1
        self.N[j, k] += 1
        if k != l:
            if self.H[i, l] == 0 and 2 * self.N[i, l] >= self.S[l]:
                self.H[i, l] = 1
                self.M[k, l] += 1
            if self.H[j, k] == 0 and 2 * self.N[j, k] >= self.S[k]:
                self.H[j, k] = 1
                self.M[l, k] += 1

    def merge_clusters(self, k, l, distance):
        r = self.nextroots.pop(-1)
        new_size = self.S[k] + self.S[l]
        self.S[r] = new_size
        self.N[:, r] = self.N[:, k] + self.N[:, l]
        X = []
        Y = []
        for v in range(self.n):
            if 2 * self.N[v, r] >= new_size:
                self.H[v, r] = 1
                self.M[self.membership[v], r] += 1
            if self.membership[v] == k:
                X.append(v)
            if self.membership[v] == l:
                Y.append(v)
        self.M[r, :] = self.M[k, :] + self.M[l, :]
        for x in X:
            for y in Y:
                self.d_U[x,y] = distance
                self.d_U[y,x] = distance
        self.G.add_node(r)
        self.G.add_edge(r, k, length = distance - self.heights[k])
        self.G.add_edge(r, l, length = distance - self.heights[l])
        self.heights[r] = distance
        self.Z[r - self.n] = np.array([k, l, distance, new_size])
        for x in X:
            self.membership[x] = r
        for y in Y:
            self.membership[y] = r
        if self.debug:
            print(X, Y, distance)


    def learn_UM(self):
        start = time.time()
        E = self.get_edge_seq(self.d, self.n)
        end = time.time()
        self.elapse_sort = end - start
        t = 0
        while(len(self.nextroots) > 1):
            i, j = E[t][0], E[t][1]
            self.update_matrices(i, j)
            k, l = self.membership[i], self.membership[j]
            if(k != l and self.M[k,l] + self.M[l,k] == self.S[k] + self.S[l]):
                self.merge_clusters(k, l, self.d[i,j])
            t += 1
        self.fitted = True
        end = time.time()
        self.elapse_fit = end - start


    def draw_dendrogram(self, labels = None, figsize = (20,10)):
        if self.fitted == False:
            print("Warning: d_U is not obtained yet!\n")
        else:
            fig = plt.figure(figsize = figsize)
            dn = dendrogram(self.Z, labels = labels)
            plt.show()

    def fit_tree(self, root):

        d_root = self.d[root]
        d_max = d_root.max()
        gp = np.tile(d_root, (self.n, 1)) + np.tile(d_root.reshape(self.n, 1), (1, self.n)) - self.d
        gp /= 2.0
        d_U = d_max - gp
        np.fill_diagonal(d_U, 0)
        U = self.__class__(d_U)
        U.learn_UM()
        gp_T = d_max - U.d_U
        self.d_T = np.tile(d_root, (self.n, 1)) + np.tile(d_root.reshape(self.n, 1), (1, self.n)) - 2.0 * gp_T
        np.fill_diagonal(self.d_T, 0)
        self.G = U.G

