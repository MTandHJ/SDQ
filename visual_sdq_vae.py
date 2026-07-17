import csv
import json
import os
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple

import freerec
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from converter import SemIDConverter
from partition import Mope
from quantizer import StructureDiffusionQuantizer as BaseStructureDiffusionQuantizer


def ensure_single_process() -> None:
    r"""Reject distributed launch modes unsupported by the diagnostics."""
    if os.environ.get("WORLD_SIZE") is not None or (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "visual_sdq_vae.py only supports direct single-process execution; "
            "do not launch it with torchrun."
        )


ensure_single_process()
freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codebooks", type=int, default=3, help="number of codebooks")
cfg.add_argument(
    "--num-codewords",
    type=int,
    default=256,
    help="number of codewords per codebook",
)
cfg.add_argument("--codebook-dim", type=int, default=128, help="dimension of codebook vector")
cfg.add_argument(
    "--apply-shared-codebook",
    type=eval,
    default=False,
    help="whether sharing the codebook",
)
cfg.add_argument(
    "--sk-epsilons",
    type=str,
    default="0.,0.,0.03",
    help="epsilon for sinkhorn iteration",
)
cfg.add_argument(
    "--sk-iters",
    type=float,
    default=50,
    help="number of iterations for sinkhorn iteration",
)

cfg.add_argument("--hidden-dims", type=str, default="512,128", help="hidden sizes")
cfg.add_argument("--commit-weight", type=float, default=1.0, help="weight for commitment loss")
cfg.add_argument("--dropout-rate", type=float, default=0.0, help="dropout rate")

cfg.add_argument(
    "--solver",
    type=str,
    choices=("euler", "rk4", "dopri5"),
    default="dopri5",
)

cfg.add_argument("--partition-method", type=str, default="louvain")
cfg.add_argument("--ngroups", type=int, default=16, help="spectral & metis")
cfg.add_argument("--ufactor", type=int, default=7999, help="metis")
cfg.add_argument("--resolution", type=float, default=1.0, help="for louvain & leiden")

cfg.add_argument(
    "--sem-feat-file",
    type=str,
    default="sentence-t5-xl_title_categories_brand.pkl",
    help="file of semantic features",
)

cfg.set_defaults(
    description="SDQ",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=200,
    batch_size=256,
    optimizer="AdamW",
    lr=5.0e-4,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(",")))
cfg.sk_epsilons = list(map(float, cfg.sk_epsilons.split(",")))


class AcceptedStepDiffusion(nn.Module):
    r"""Record accepted adaptive steps without changing the diffusion field."""

    def __init__(self, kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.kernel = kernel
        self.accepted_steps: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(t, x)

    def callback_accept_step(self, t0: torch.Tensor, _x0: torch.Tensor, dt: torch.Tensor) -> None:
        self.accepted_steps.append((t0.detach().clone(), dt.detach().clone()))


class DiagnosticStructureDiffusionQuantizer(BaseStructureDiffusionQuantizer):
    r"""SDQ quantizer with support and same-stage Gram diagnostics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_durs: Tuple[float, ...] = ()
        self.last_pbbp: Optional[torch.Tensor] = None

    @staticmethod
    def zero_norm(x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(())

    def solve_with_accepted_steps(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        recorder = AcceptedStepDiffusion(self.diffusion_kernel)
        z = odeint(recorder, x, t, method=self.solver)[-1]
        return z, recorder.accepted_steps

    @torch.no_grad()
    def replay_dopri5(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        accepted_steps: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if not accepted_steps:
            raise RuntimeError("DOPRI5 completed a nonzero interval without an accepted step.")

        direction = torch.sign(t[-1] - t[0]).to(accepted_steps[0][0])
        step_t = torch.stack([t0 + direction * dt for t0, dt in accepted_steps])

        # DOPRI5 limits normal growth to 10x. Using 11x forces every recorded
        # endpoint to truncate the replay step, including the final overshoot.
        options = {
            "first_step": accepted_steps[0][1] * 2,
            "step_t": step_t,
            "ifactor": 11.0,
            "norm": self.zero_norm,
        }
        return odeint(
            self.diffusion_kernel,
            x,
            t,
            method="dopri5",
            options=options,
        )[-1]

    @torch.no_grad()
    def diffuse_assignments(
        self,
        assignments: torch.Tensor,
        t: torch.Tensor,
        accepted_steps: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if t[0] == t[-1]:
            return assignments
        if self.solver == "dopri5":
            if accepted_steps is None:
                raise RuntimeError("The DOPRI5 accepted-step schedule is missing.")
            return self.replay_dopri5(assignments, t, accepted_steps)
        return self.solve(assignments, t)

    def forward(
        self,
        z: torch.Tensor,
        *,
        collect_dur: bool = False,
        collect_pbbp: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss = 0
        ids = []
        durs = []
        pbbp = []
        z_res = z
        z_hat = 0.0
        num_stages = len(self.codebooks)
        collect_diagnostics = collect_dur or collect_pbbp

        for stage in range(num_stages):
            diffusion_time = num_stages - stage - 1
            forward_t = torch.tensor([0.0, diffusion_time], device=z.device)
            backward_t = torch.tensor([diffusion_time, 0.0], device=z.device)

            conved = self.solve(z_res, forward_t)
            stage_ids, c = self.match(conved, stage)

            accepted_steps = None
            if collect_diagnostics and self.solver == "dopri5" and diffusion_time > 0:
                deconved, accepted_steps = self.solve_with_accepted_steps(c, backward_t)
            else:
                deconved = self.solve(c, backward_t)

            if collect_diagnostics:
                assignments = F.one_hot(
                    stage_ids,
                    num_classes=self.codebooks[stage].num_embeddings,
                ).to(dtype=c.dtype)
                bp = self.diffuse_assignments(assignments, backward_t, accepted_steps)
                if collect_dur:
                    durs.append(torch.count_nonzero(bp).item() / bp.numel())
                if collect_pbbp:
                    pbbp.append((bp.T @ bp).detach().cpu())

            z_hat = z_hat + deconved
            loss += self.commit(z_hat, z) + self.commit(z, z_hat) * self.commit_weight
            z_res = z_res - deconved
            ids.append(stage_ids)

        self.last_durs = tuple(durs)
        self.last_pbbp = torch.stack(pbbp) if pbbp else None

        q = z + (z_hat - z).detach()
        return q, loss / num_stages, torch.stack(ids, dim=-1)


class SDQ(freerec.models.RecSysArch):
    r"""Train SDQ-VAE while exposing quantization diagnostics."""

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                self.normalize(
                    freerec.utils.import_pickle(os.path.join(self.dataset.path, cfg.sem_feat_file))
                ),
                freeze=True,
            ),
        )

        dims = [self.Item.embeddings.weight.size(1)] + cfg.hidden_dims + [cfg.codebook_dim]
        activation = nn.SiLU

        self.encoder = nn.Sequential()
        for layer, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.encoder.append(nn.Dropout(cfg.dropout_rate))
            self.encoder.append(nn.Linear(input_dim, output_dim, bias=False))
            if layer < len(dims) - 1:
                self.encoder.append(activation())

        self.quantizer = DiagnosticStructureDiffusionQuantizer(
            self.dataset,
            cfg.codebook_dim,
            features=None,
            solver=cfg.solver,
            num_codebooks=cfg.num_codebooks,
            num_codewords=cfg.num_codewords,
            apply_shared_codebook=cfg.apply_shared_codebook,
            commit_weight=cfg.commit_weight,
            sk_iters=cfg.sk_iters,
            sk_epsilons=cfg.sk_epsilons,
        )

        self.decoder = nn.Sequential()
        dims = dims[::-1]
        for layer, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.decoder.append(nn.Dropout(cfg.dropout_rate))
            self.decoder.append(nn.Linear(input_dim, output_dim, bias=False))
            if layer < len(dims) - 1:
                self.decoder.append(activation())

        self.partition = Mope(
            dataset,
            edge_index=self.quantizer.edge_index,
            edge_weight=self.quantizer.edge_weight,
            method=cfg.partition_method,
            resolution=cfg.resolution,
            ngroups=cfg.ngroups,
            ufactor=cfg.ufactor,
        )
        self.register_buffer("groups", torch.from_numpy(self.partition(cfg.seed)).long())

        self.criterion = nn.MSELoss(reduction="sum")
        self.reset_parameters()

    def normalize(self, feats: torch.Tensor) -> torch.Tensor:
        return F.normalize(feats - feats.mean(dim=0, keepdim=True), dim=-1)

    @freerec.utils.timemeter
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        with torch.no_grad():
            for codebook in self.quantizer.codebooks:
                codebook.requires_kmeans_init_ = True

            self.encoder.to(cfg.device)
            self.quantizer.to(cfg.device)

            num_samples = min(self.Item.count, cfg.num_codewords * 5)
            items = torch.randperm(self.Item.count)[:num_samples]
            items, _ = torch.sort(items)

            self.quantizer.reset_local_graph(items.clone().to(cfg.device))
            x = self.Item.embeddings(items).to(cfg.device)
            self.quantizer(self.encode(x))

    def sure_trainpipe(self, batch_size: int = 512):
        return self.dataset.train().shuffled_seqs_source(maxlen=50).batch_(batch_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.decoder(q), dim=-1)

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, ...], int]:
        items = set(chain(*data[self.ISeq]))
        items = torch.tensor(list(items), dtype=torch.long, device=self.device)
        items, _ = torch.sort(items)
        self.quantizer.reset_local_graph(items)

        x = self.Item.embeddings(items)
        z = self.encode(x)
        q, auxiliary_loss, _ = self.quantizer(z, collect_dur=True)
        x_hat = self.decode(q)

        recon_loss = self.criterion(x_hat, x) / len(items)
        return recon_loss, auxiliary_loss, self.quantizer.last_durs, len(items)

    @torch.no_grad()
    def generate_sem_ids(self) -> torch.Tensor:
        is_training = self.training
        self.eval()
        try:
            sem_ids = torch.zeros((self.Item.count, cfg.num_codebooks), dtype=torch.long)
            for group in self.groups.unique():
                items = torch.where(self.groups == group)[0]
                items, _ = torch.sort(items.flatten())
                self.quantizer.reset_local_graph(items)

                x = self.Item.embeddings(items)
                z = self.encode(x)
                _, _, ids = self.quantizer(z)
                sem_ids[items.detach().cpu()] = ids.detach().cpu()
            return sem_ids
        finally:
            self.train(is_training)

    @torch.no_grad()
    def compute_full_graph_pbbp(self) -> torch.Tensor:
        r"""Compute one same-stage PBBP snapshot over all item nodes."""
        is_training = self.training
        self.eval()
        try:
            items = torch.arange(self.Item.count, device=self.device)
            self.quantizer.reset_local_graph(items)

            x = self.Item.embeddings(items)
            z = self.encode(x)
            self.quantizer(z, collect_pbbp=True)
            if self.quantizer.last_pbbp is None:
                raise RuntimeError("The full-graph PBBP snapshot was not produced.")
            return self.quantizer.last_pbbp
        finally:
            self.train(is_training)


class CoachForSDQ(freerec.launcher.Coach):
    r"""Run SDQ-VAE training and serialize DUR and PBBP data."""

    @freerec.ddp.main_process_only
    def save_sid_vocab(self) -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids()
        sid_vocab = {}
        for item_id, sids in enumerate(sem_ids.tolist()):
            sids = [
                SemIDConverter.SID_FORMAT.format(level=level, id=sid)
                for level, sid in enumerate(sids)
            ]
            sid_vocab[SemIDConverter.format(item_id)] = tuple(sids)

        vocab_file = os.path.join(self.cfg.LOG_PATH, "sid_vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as file:
            json.dump(sid_vocab, file)

    def set_other(self) -> None:
        self.register_metric("RECON_LOSS", lambda x: x, best_caster=min)
        self.register_metric("COMMIT_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for stage in range(self.cfg.num_codebooks):
            self.register_metric(f"PPL#{stage}", lambda x: x, best_caster=max)

        self.global_step = 0
        os.makedirs(self.cfg.LOG_PATH, exist_ok=True)
        dur_path = os.path.join(self.cfg.LOG_PATH, "dur.csv")
        self.dur_file = open(dur_path, "w", newline="", encoding="utf-8")
        self.dur_writer = csv.writer(self.dur_file)
        self.dur_writer.writerow(
            [
                "epoch",
                "global_step",
                "batch_items",
                *[f"dur_{stage}" for stage in range(self.cfg.num_codebooks)],
            ]
        )
        self.dur_file.flush()

    def fit(self) -> None:
        try:
            super().fit()
        finally:
            self.dur_file.close()

    def train_per_epoch(self, epoch: int) -> None:
        for data in self.dataloader:
            data = self.dict_to_device(data)
            recon_loss, auxiliary_loss, durs, batch_items = self.model(data)
            loss = recon_loss + auxiliary_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            self.dur_writer.writerow([epoch, self.global_step, batch_items, *durs])
            self.dur_file.flush()

            self.monitor(
                recon_loss.item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["RECON_LOSS"],
            )
            self.monitor(
                auxiliary_loss.item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["COMMIT_LOSS"],
            )

        if epoch % self.cfg.eval_freq == 0:
            self.save_sid_vocab()

    def evaluate(self, epoch: int, step: int = -1, mode: str = "valid") -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids().cpu()
        counts = torch.zeros((self.cfg.num_codewords, self.cfg.num_codebooks))
        counts.scatter_add_(0, sem_ids, torch.ones_like(sem_ids, dtype=torch.float))
        uniques = {tuple(ids) for ids in sem_ids.tolist()}

        freqs = counts.div(counts.sum(dim=0, keepdim=True))
        perplexity = ((freqs + 1.0e-8).log() * freqs).sum(dim=0).neg().exp().tolist()

        ppls = []
        for stage, ppl in enumerate(perplexity):
            ppls.append(ppl)
            self.monitor(ppl, n=1, mode="valid", pool=[f"PPL#{stage}"])

        self.monitor(
            sum(ppls),
            n=len(ppls),
            mode=mode,
            reduction="sum",
            pool=["PPL"],
        )
        self.monitor(
            (self.Item.count - len(uniques)) / self.Item.count,
            n=1,
            mode=mode,
            reduction="mean",
            pool=["COLLISION_RATE"],
        )

    @freerec.ddp.main_process_only
    def save_full_graph_pbbp(self) -> None:
        model = self.get_res_sys_arch()
        pbbp = model.compute_full_graph_pbbp().detach().cpu()
        payload = {
            "pbbp": pbbp,
            "num_items": self.Item.count,
            "num_codebooks": self.cfg.num_codebooks,
            "num_codewords": self.cfg.num_codewords,
            "solver": self.cfg.solver,
            "global_step": self.global_step,
        }
        path = os.path.join(self.cfg.LOG_PATH, "pbbp.pt")
        torch.save(payload, path)
        freerec.infoLogger(f"[SDQ] >>> Save full-graph PBBP to {path}")


def main() -> None:
    ensure_single_process()

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root,
            cfg.dataset,
            tasktag=cfg.tasktag,
        )

    model = SDQ(dataset)
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    coach = CoachForSDQ(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=trainpipe,
        testpipe=None,
        model=model,
        cfg=cfg,
    )
    coach.fit()
    coach.save_full_graph_pbbp()


if __name__ == "__main__":
    main()
