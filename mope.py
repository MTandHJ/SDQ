from collections import defaultdict

import numpy as np
import torch
from freerec.data.datasets import RecDataSet
from freerec.data.tags import ID, ITEM, SEQUENCE
from freerec.utils import timemeter
from scipy.sparse import coo_array, csr_array


class Mope:
    def __init__(
        self,
        dataset: RecDataSet,
    ):
        self.dataset = dataset
        self.Item = self.dataset.fields[ITEM, ID]
        self.ISeq = self.Item.fork(SEQUENCE)

    def estimate(self):
        seqs = self.dataset.train().to_seqs(maxlen=50)
        A = defaultdict(int)
        for seq in seqs:
            seq = seq[self.ISeq]
            for i in range(len(seq) - 1):
                for j in range(i + 1, len(seq)):
                    A[(seq[i], seq[j])] += 1
                    A[(seq[j], seq[i])] += 1

        edge_index, edge_weight = list(A.keys()), list(A.values())
        edge_index = np.array(edge_index).T
        edge_weight = np.array(edge_weight)
        row, col = edge_index[0], edge_index[1]
        adj = coo_array((edge_weight, (row, col)))
        return adj

    def single(self, adj: coo_array):
        groups = np.zeros(adj.shape[0], dtype=np.int64)
        return groups

    def random(self, adj: coo_array, ngroups: int = 16):
        nums = adj.shape[0]
        group_size = nums // ngroups
        groups = [[ngroups - i] * group_size for i in range(ngroups)]
        groups.append([0] * (nums % ngroups))
        groups = np.concatenate(groups)
        return np.random.permutation(groups)

    def spectral(self, adj, ngroups: int = 16, seed: int = 0):
        from sklearn.cluster import SpectralClustering

        clustering = SpectralClustering(n_clusters=ngroups, affinity="precomputed", random_state=seed)

        adj_coo = adj.tocoo()
        row = adj_coo.row.astype(np.int32)
        col = adj_coo.col.astype(np.int32)
        data = adj_coo.data

        adj = csr_array(
            (data, (row, col)),
            shape=adj_coo.shape,
            dtype=np.float32,  # or adj_coo.dtype
        )

        clustering.fit(adj)
        return clustering.labels_

    def metis(self, adj, ngroups: int = 16, ufactor: int = 7999, seed: int = 0):
        import pymetis

        adj = adj.tocsr()
        xadj = adj.indptr
        adjncy = adj.indices
        eweights = adj.data

        n_cuts, groups = pymetis.part_graph(
            ngroups,
            xadj=xadj,
            adjncy=adjncy,
            eweights=eweights,
            options=pymetis.Options(seed=seed, ufactor=ufactor),
        )
        return np.array(groups)

    def louvain(self, adj, resolution: float = 1.0, seed: int = 0):
        import networkx as nx

        row = adj.row
        col = adj.col
        data = adj.data
        graph = nx.Graph()
        for i, j, weight in zip(row, col, data):
            graph.add_edge(i, j, weight=weight)
        communities = nx.community.louvain_communities(graph, resolution=resolution, seed=seed)

        groups = []
        indices = []
        for i, community in enumerate(communities):
            indices.append(list(community))
            groups.append([i] * len(community))

        indices = np.concatenate(indices)
        groups = np.concatenate(groups)
        return groups[np.argsort(indices)]

    def leiden(self, adj, resolution: float = 1.0, seed: int = 0):
        import networkx as nx

        row = adj.row
        col = adj.col
        data = adj.data
        graph = nx.Graph()
        for i, j, weight in zip(row, col, data):
            graph.add_edge(i, j, weight=weight)
        communities = nx.community.leiden_communities(graph, resolution=resolution, seed=seed)

        groups = []
        indices = []
        for i, community in enumerate(communities):
            indices.append(list(community))
            groups.append([i] * len(community))

        indices = np.concatenate(indices)
        groups = np.concatenate(groups)
        return groups[np.argsort(indices)]

    @timemeter
    def partition(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        method: str = "metis",
        ngroups: int = 16,
        ufactor: int = 7999,
        resolution: float = 1.0,
        seed: int = 0,
    ) -> torch.Tensor:

        edge_index, edge_weight = edge_index.numpy(), edge_weight.numpy()
        row, col = edge_index[0], edge_index[1]
        adj = coo_array((edge_weight, (row, col)))

        if method == "single":
            groups = self.single(adj)
        elif method == "random":
            groups = self.random(adj, ngroups)
        elif method == "spectral":
            groups = self.spectral(adj, ngroups, seed)
        elif method == "metis":
            groups = self.metis(adj, ngroups, ufactor, seed)
        elif method == "louvain":
            groups = self.louvain(adj, resolution, seed)
        elif method == "ledien":
            groups = self.leiden(adj, resolution, seed)
        groups = torch.tensor(groups)
        return groups
