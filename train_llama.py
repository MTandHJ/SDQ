

from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec
from transformers import LlamaModel, LlamaConfig
from transformers.cache_utils import DynamicCache

from tokenizer import SemIDTokenzier


DTYPE = torch.bfloat16


freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50, help="the maximum (item id) sequence length")
cfg.add_argument("--num-heads", type=int, default=4)
cfg.add_argument("--num-blocks", type=int, default=8)
cfg.add_argument("--embedding-dim", type=int, default=128)
cfg.add_argument("--dropout-rate", type=float, default=0.4)
cfg.add_argument("--sem-id-ckpt", type=str, default=None, help="checkpoint file of 'sem_ids'")
cfg.add_argument("--num-beams", type=int, default=32)

cfg.set_defaults(
    description="GenRec-Llama",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=500,
    batch_size=512,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
)
cfg.compile()


class LlamaGenRec(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = cfg.embedding_dim

        self.tokenizer = SemIDTokenzier(
            sem_ids=freerec.utils.import_pickle(cfg.sem_id_ckpt),
        )
        self.num_codebooks: int = self.tokenizer.num_codebooks
        self.num_codewords: List[int] = self.tokenizer.num_codewords
        self.maxlen = (cfg.maxlen + 1) * self.num_codebooks # Every Item occupies #levels tokens

        self.model_config = LlamaConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=cfg.embedding_dim,
            intermediate_size=cfg.embedding_dim,
            num_hidden_layers=cfg.num_blocks,
            num_attention_heads=cfg.num_heads,
            max_position_embeddings=self.maxlen,
            tie_word_embeddings=False,
            attention_dropout=cfg.dropout_rate,
            pad_token_id=0,
            bos_token_id=None,
            eos_token_id=None
        )

        self.output_projector = nn.Linear(
            cfg.embedding_dim, self.tokenizer.vocab_size, bias=False
        )

        self.model = LlamaModel(self.model_config)

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self): ...

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_seqs_source(
           maxlen=maxlen 
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def shrink_paddings(self, seqs: torch.Tensor):
        mask = seqs.ne(self.PADDING_VALUE)
        keeped = mask.any(dim=0, keepdim=False)
        return seqs[:, keeped] # (B, S)
    
    def encode(
        self, seqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out = self.model(
            input_ids=seqs,
            attention_mask=seqs.not_equal(self.PADDING_VALUE),
            output_attentions=False
        )
        
        return out.last_hidden_state

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        seqs = data[self.ISeq]
        seqs = self.tokenizer.encode(seqs)
        seqs = self.shrink_paddings(seqs)
        seqs, labels = seqs[:, :-1], seqs[:, 1:]
        userEmbds = self.encode(seqs)

        indices = seqs != self.PADDING_VALUE
        userEmbds = userEmbds[indices] # (M, D)
        logits = self.output_projector(userEmbds) # (M, N)

        labels = labels[indices] # (M,)
        loss = self.criterion(logits, labels)

        return loss

    def beam_search(
        self, 
        data: Dict[freerec.data.fields.Field, torch.Tensor],
        temperature: float = 1.,
    ) -> torch.Tensor:
        r"""
        Beam search for item recommendation generation.

        Parameters
        ----------
        data : Dict[freerec.data.fields.Field, torch.Tensor]
            Contains the historical sequence.
        temperature : float
            Sampling temperature for controlling exploration. 
            Note: Probabilistic sampling is not supported.

        Returns
        -------
        torch.Tensor
            Recommended candidates, returned as original item IDs (not semantic IDs).
        """

        # historical sequence
        seqs: torch.Tensor = self.tokenizer.encode(data[self.ISeq])
        seqs = self.shrink_paddings(seqs)
        itemEmbds = self.output_projector.weight

        (B, S), K = seqs.shape, cfg.num_beams

        seqs = seqs.repeat_interleave(K, dim=0) # (B * K, S)
        scores = torch.zeros((B, K), device=self.device)
        # mask the last K-1 beams to avoid repeated sampling at the first turn
        scores[:, 1:] = -1e9 

        seqs, scores = seqs.view(B * K, -1), scores.view(B * K, 1)

        out = self.model(
            input_ids=seqs,
            attention_mask=seqs.not_equal(self.PADDING_VALUE),
            use_cache=True, return_dict=True,
            output_attentions=False
        )

        start = self.NUM_PADS
        for l in range(self.num_codebooks):
            N = self.num_codewords[l]
            end = start + N

            userEmbds = out.last_hidden_state[:, -1, :] # (B * K, D)
            past_key_values = out.past_key_values

            logits = torch.einsum("MD,ND->MN", userEmbds, itemEmbds[start:end])
            logp = F.log_softmax(logits / temperature, dim=-1).add(scores).view(B, -1) # (B, K * N)
            scores, indices = logp.topk(k=K, dim=-1, largest=True, sorted=True) # (B, K)

            prev_beam_tokens = indices // N # (B, K)
            next_beam_tokens = indices % N # (B, K)
            prev_beam_tokens += (torch.arange(B, dtype=torch.long, device=self.device) * K).unsqueeze(-1)
            next_beam_tokens += start

            next_beam_tokens = next_beam_tokens.flatten().unsqueeze(-1) # (B * K, 1)
            prev_beam_tokens = prev_beam_tokens.flatten()

            seqs = seqs[prev_beam_tokens] # (B * K, L)

            new_past_key_values = []
            for layer_cache in past_key_values:
                if layer_cache is not None:
                    key_cache, value_cache = layer_cache
                    new_key_cache = key_cache[prev_beam_tokens]
                    new_value_cache = value_cache[prev_beam_tokens]
                    new_past_key_values.append((new_key_cache, new_value_cache))
                else:
                    new_past_key_values.append(None)
            past_key_values = DynamicCache(new_past_key_values)

            seqs = torch.cat(
                (
                    seqs,
                    next_beam_tokens
                ),
                dim=-1
            ) # (B * K, L + 1)

            scores = scores.view(B * K, 1)

            if l < self.num_codebooks - 1:
                out = self.model(
                    input_ids=next_beam_tokens,
                    use_cache=True, return_dict=True,
                    past_key_values=past_key_values,
                    output_attentions=False,
                    attention_mask=seqs.not_equal(self.PADDING_VALUE),
                )

            start += N

        seqs = seqs.view(B, K, -1)
        ranked_sem_ids = seqs[..., -self.num_codebooks:] # (B, K, num_codebooks)
        ranked_item_ids = self.tokenizer.decode(ranked_sem_ids) # (B, K)
        num_invalid_item_ids = (ranked_item_ids == self.PADDING_VALUE).sum()
        return ranked_item_ids, num_invalid_item_ids

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        ranked_item_ids, num_invalid_item_ids = self.beam_search(data)
        B, K, N = ranked_item_ids.size(0), ranked_item_ids.size(1), self.Item.count + self.NUM_PADS

        scores = torch.rand((B, N), device=self.device) * 0.001
        scores = scores.scatter(
            dim=1, index=ranked_item_ids,
            src=torch.arange(K + 1, 0, -1, device=self.device, dtype=torch.float).repeat((B, 1))
        )
        return scores[:, self.NUM_PADS:], num_invalid_item_ids

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        ranked_item_ids, num_invalid_item_ids = self.beam_search(data)
        B, K, N = ranked_item_ids.size(0), ranked_item_ids.size(1), self.Item.count + self.NUM_PADS

        scores = torch.rand((B, N), device=self.device) * 0.001
        scores = scores.scatter(
            dim=1, index=ranked_item_ids,
            src=torch.arange(K + 1, 0, -1, device=self.device, dtype=torch.float).repeat((B, 1))
        )
        scores = scores[:, self.NUM_PADS:]
        candidates = data[self.IUnseen]
        scores = scores.gather(
            dim=1, index=candidates
        )
        return scores, num_invalid_item_ids


class CoachForGenRec(freerec.launcher.Coach):

    def set_other(self):
        self.register_metric(
            "NUM_INVALID", lambda x: x,
            best_caster=min
        )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )

    def evaluate(self, epoch: int, step: int = -1, mode: str = 'valid'):
        self.get_res_sys_arch().reset_ranking_buffers()
        for data in self.dataloader:
            bsz = data[self.Size]

            data = self.dict_to_device(data)
            if self.cfg.ranking == 'full':
                scores, num_invalid_item_ids = self.model(data, ranking='full')
                if self.remove_seen:
                    seen = self.Item.to_csr(data[self.ISeen]).to(self.device).to_dense().bool()
                    scores[seen] = -1e23
                targets = self.Item.to_csr(data[self.IUnseen]).to(self.device).to_dense()
            elif self.cfg.ranking == 'pool':
                scores, num_invalid_item_ids = self.model(data, ranking='pool')
                if self.Label in data:
                    targets = data[self.Label]
                else:
                    targets = torch.zeros_like(scores)
                    targets[:, 0].fill_(1)
            else:
                raise NotImplementedError(
                    f"`ranking` should be 'full' or 'pool' but {self.cfg.ranking} received ..."
                )

            self.monitor(
                scores, targets,
                n=bsz, reduction="mean", mode=mode,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )

            self.monitor(
                num_invalid_item_ids,
                n=bsz, reduction="sum", mode=mode,
                pool=['NUM_INVALID']
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = LlamaGenRec(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=64)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=64)

    coach = CoachForGenRec(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()