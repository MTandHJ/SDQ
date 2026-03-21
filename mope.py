

import torch
import numpy as np
from scipy.sparse import coo_array, csr_array
from freerec.data.datasets import RecDataSet
from collections import defaultdict
from freerec.data.tags import ITEM, ID, SEQUENCE
from freerec.utils import timemeter


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
        Adj = coo_array(
            (edge_weight, (row, col))
        )
        return Adj

    def rowise(self, Adj: coo_array):
        groups = np.arange(Adj.shape[0])
        return groups

    def single(self, Adj: coo_array):
        groups = np.zeros(Adj.shape[0], dtype=np.int64)
        return groups

    def random(self, Adj: coo_array, ngroups: int = 16):
        nums = Adj.shape[0]
        group_size = nums // ngroups
        groups = [[ngroups - i] * group_size for i in range(ngroups)]
        groups.append([0] * (nums % ngroups))
        groups = np.concatenate(groups)
        return np.random.permutation(groups)

    def spectral(self, Adj, ngroups: int = 16, seed: int = 0):
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters=ngroups, affinity='precomputed', random_state=seed)

        Adj_coo = Adj.tocoo()
        row = Adj_coo.row.astype(np.int32)
        col = Adj_coo.col.astype(np.int32)
        data = Adj_coo.data
        
        Adj = csr_array(
            (data, (row, col)), 
            shape=Adj_coo.shape,
            dtype=np.float32  # 或 Adj_coo.dtype
        )

        clustering.fit(Adj)
        return clustering.labels_

    def metis(self, Adj, ngroups: int = 16, ufactor: int = 7999, seed: int = 0):
        import pymetis
        Adj = Adj.tocsr()
        xadj = Adj.indptr
        adjncy = Adj.indices
        eweights = Adj.data

        n_cuts, groups = pymetis.part_graph(
            ngroups, xadj=xadj, adjncy=adjncy, eweights=eweights,
            options=pymetis.Options(seed=seed, ufactor=ufactor),
        )
        return np.array(groups)

    def louvain(self, Adj, resolution: float = 1., seed: int = 0):
        import networkx as nx
        row = Adj.row
        col = Adj.col
        data = Adj.data
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
        groups  = np.concatenate(groups)
        return groups[np.argsort(indices)]
    
    @timemeter
    def partition(
        self,
        method: str = 'metis',
        ngroups: int = 16,
        ufactor: int = 7999,
        resolution: float = 1.,
        num_pads: int = 0,
        seed: int = 0
    ) -> torch.Tensor:
        Adj = self.estimate()
        if method == 'single':
            groups = self.single(Adj) + int(num_pads > 0)
        elif method == 'random':
            groups = self.random(Adj, ngroups) + int(num_pads > 0)
        elif method == 'spectral':
            groups = self.spectral(Adj, ngroups, seed) + int(num_pads > 0)
        elif method == 'metis':
            groups = self.metis(Adj, ngroups, ufactor, seed) + int(num_pads > 0)
        elif method == 'louvain':
            groups = self.louvain(Adj, resolution, seed) + int(num_pads > 0)
        groups = np.concatenate((np.zeros(num_pads, dtype=np.int64), groups), 0)
        groups = torch.tensor(groups)
        return groups