from typing import Iterable, Optional, Tuple, Literal

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from freerec.data.tags import ID, ITEM, SEQUENCE
from torch_geometric.utils import scatter, subgraph
from torchdiffeq import odeint

from utils import sinkhorn_algorithm


class CodeBook(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )

        self.requires_kmeans_init_ = False

    @torch.no_grad()
    def reinit_kmeans_codebook(self, z: torch.Tensor) -> torch.Tensor:
        if self.requires_kmeans_init_:
            from k_means_constrained import KMeansConstrained

            z = z.detach().cpu().numpy()
            size_min = max(1, min(len(z) // (self.num_embeddings * 2), 50))
            clf = KMeansConstrained(
                n_clusters=self.num_embeddings,
                size_min=size_min,
                max_iter=10,
                n_init=10,
                n_jobs=10,
                verbose=False,
            )
            clf.fit(z)
            codebook = torch.from_numpy(clf.cluster_centers_)

            self.weight.data.copy_(codebook.to(device=self.weight.device, dtype=self.weight.dtype))

            self.requires_kmeans_init_ = False
        return self.weight


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        dataset: freerec.data.datasets.base.RecDataSet,
        hidden_size: int,
        num_codebooks: int = 3,
        num_codewords: int = 256,
        apply_shared_codebook: bool = False,
        commit_weight: float = 0.25,
        sk_iters: int = 50,
        sk_epsilons: Optional[Tuple[float]] = None,
    ):
        super().__init__()

        self.dataset = dataset
        self.Item = self.dataset.fields[ITEM, ID]

        self.codebooks: Iterable[CodeBook]
        if apply_shared_codebook:
            self.codebooks = nn.ModuleList(
                [
                    CodeBook(
                        num_codewords,
                        hidden_size,
                    )
                ]
                * num_codebooks
            )
        else:
            self.codebooks = nn.ModuleList(
                [
                    CodeBook(
                        num_codewords,
                        hidden_size,
                    )
                    for _ in range(num_codebooks)
                ]
            )

        self.sk_iters = sk_iters
        self.sk_epsilons = sk_epsilons
        self.commit_weight = commit_weight

    @staticmethod
    def cdist(x, y, p=2):
        return torch.cdist(x, y, p=p)

    def commit(self, x: torch.Tensor, y: torch.Tensor):
        return F.mse_loss(x, y.detach(), reduction="sum") / x.size(0)

    def step(self, r: torch.Tensor, l: int):
        codebook = self.codebooks[l].reinit_kmeans_codebook(r)
        dist = self.cdist(r, codebook)  # (B, K)
        if self.sk_epsilons[l] > 0.0:
            dist = -sinkhorn_algorithm(dist, self.sk_epsilons[l], self.sk_iters)
        ids = torch.argmin(dist, dim=-1)  # (B,)
        q = codebook[ids]
        return ids, q

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        loss = 0

        ids = []
        z_res = z  # residual
        z_hat = 0.0  # estimation
        L = len(self.codebooks)
        for l in range(L):
            ids_, q = self.step(z_res, l)
            z_hat = z_hat + q
            loss += self.commit(q, z_res) + self.commit(z_res, q) * self.commit_weight
            z_res = z_res - q

            ids.append(ids_)

        return z + (z_hat - z).detach(), loss / L, torch.stack(ids, dim=-1)


class StructureDiffusionQuantizer(ResidualQuantizer):
    def __init__(
        self,
        dataset: freerec.data.datasets.base.RecDataSet,
        hidden_size: int,
        features: Optional[torch.Tensor] = None,
        knn_graph_topk: int = 5,
        num_codebooks: int = 3,
        num_codewords: int = 256,
        apply_shared_codebook: bool = False,
        commit_weight: float = 0.25,
        sk_iters: int = 50,
        sk_epsilons: Optional[Tuple[float]] = None,
        solver: Literal['euler', 'midpoint', 'heun2', 'heun3', 'rk4', 'bosh3', 'adaptive_heun', 'fehlberg2', 'dopri5'] = 'dopri5',
    ):
        super().__init__(
            dataset,
            hidden_size,
            num_codebooks,
            num_codewords,
            apply_shared_codebook,
            commit_weight,
            sk_iters,
            sk_epsilons,
        )

        freerec.infoLogger(f"[SDQ] >>> ODE Solver: {solver}")
        self.solver = solver

        if features is None:
            freerec.infoLogger("[SDQ] >>> Construct co-occurrence graph ...")
            edge_index, edge_weight = self.get_freq_graph()
        else:
            freerec.infoLogger(f"[SDQ] >>> Construct knn graph with topk: {knn_graph_topk} ...")
            edge_index, edge_weight = self.get_knn_graph(features, knn_graph_topk)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

    def get_freq_graph(self) -> Tuple[torch.Tensor]:
        from collections import defaultdict

        ISeq = self.Item.fork(SEQUENCE)
        data = defaultdict(int)
        seqs = self.dataset.train().to_seqs(maxlen=50)
        for row in seqs:
            seq = row[ISeq]
            for i, j in zip(seq[:-1], seq[1:]):
                data[(i, j)] = 1.0
                data[(j, i)] = 1.0
        for i in range(self.Item.count):
            data[(i, i)] = 1.0

        edge_index, edge_weight = list(data.keys()), list(data.values())
        edge_index, edge_weight = torch.tensor(edge_index).T, torch.tensor(edge_weight)
        return edge_index, edge_weight

    def get_knn_graph(self, features: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor]:
        features = F.normalize(features, dim=-1)  # (N, D)
        sim = features @ features.t()  # (N, N)
        sim.fill_diagonal_(-10.0)

        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        edge_weight = torch.ones_like(edge_index[0])
        edge_index, edge_weight = freerec.graph.add_remaining_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
        )

        return edge_index, edge_weight

    def reset_local_graph(self, nodes: torch.Tensor):
        local_edge_index, local_edge_weight = subgraph(
            nodes,
            self.edge_index,
            self.edge_weight,
            relabel_nodes=True,
        )
        self._local_edge_index, self._local_edge_weight = freerec.graph.to_normalized(
            edge_index=local_edge_index,
            edge_weight=local_edge_weight,
            normalization="sym",
        )

    def diffusion_kernel(self, t: float, x: torch.Tensor):
        dst, src = self._local_edge_index[0], self._local_edge_index[1]
        x_src = x[src] * self._local_edge_weight.unsqueeze(1)  # (N, D)
        z = scatter(x_src, index=dst, reduce="sum")
        return z - x

    def solve(self, x: torch.Tensor, t: torch.Tensor):
        if t[0] == t[1]:
            z = x
        else:
            z = odeint(
                self.diffusion_kernel,
                x,
                t,
                method=self.solver,
            )[1]
        return z

    def forward(self, z):
        loss = 0

        ids = []
        z_res = z  # residual
        z_hat = 0.0  # estimation
        L = len(self.codebooks)
        for l in range(L):
            conved = self.solve(z_res, torch.tensor([0.0, L - l - 1], device=z.device))
            ids_, q = self.step(conved, l)
            deconved = self.solve(q, torch.tensor([L - l - 1, 0.0], device=z.device))
            z_hat = z_hat + deconved
            loss += self.commit(z_hat, z) + self.commit(z, z_hat) * self.commit_weight
            z_res = z_res - deconved

            ids.append(ids_)

        return z + (z_hat - z).detach(), loss / L, torch.stack(ids, dim=-1)
