"""The model: a Llama decoder whose token embedding is factorized through bond
features, mirroring the simulator's own generative token model.

Runs on the GPU box (needs torch + transformers). Kept out of the package
__init__ so the torch-free core imports without a training stack.

The one non-standard piece is FactorizedTokenEmbedding. In the simulator a bond
vector is v = s * normalize(B @ x + eps): shared attribute geometry plus a
per-token residual. Ninety thousand free anchor rows would be a larger table
than the whole transformer, and would ignore that structure. So the anchor
(TOK) rows are built as

    embed(TOK) = features @ B  +  sector_emb[sector]  +  sense_emb[side]
                                +  residual[token]

while every other family (specials, TDLT, CLI, SZ, NDLR, OUT) keeps an ordinary
learned row -- CLI especially, since the per-client row is exactly where the
model gets to store each client's tilt u_k.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("rfqfm.model needs torch; install it on the GPU box") from e

import numpy as np

from .config import ModelConfig
from .features import BondFeatures
from .vocab import FmVocab


def build_llama_config(mc: ModelConfig, vocab_size: int):
    """ModelConfig -> transformers.LlamaConfig."""
    from transformers import LlamaConfig
    head_dim = mc.hidden_size // mc.num_heads
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=mc.hidden_size,
        num_hidden_layers=mc.num_layers,
        num_attention_heads=mc.num_heads,
        num_key_value_heads=mc.num_kv_heads,
        intermediate_size=mc.intermediate_size,
        max_position_embeddings=mc.max_position_embeddings,
        rope_theta=mc.rope_theta,
        hidden_act="silu",
        rms_norm_eps=mc.rms_norm_eps,
        attention_dropout=0.0,
        head_dim=head_dim,
        tie_word_embeddings=mc.tie_word_embeddings,
        bos_token_id=None, eos_token_id=None, pad_token_id=None,
    )


class FactorizedTokenEmbedding(nn.Module):
    """Drop-in replacement for the decoder's `embed_tokens`.

    Assigned via `model.set_input_embeddings(...)`; HF calls it with input_ids
    of shape (B, T) and expects (B, T, hidden).
    """

    def __init__(self, vocab: FmVocab, feats: BondFeatures, hidden: int,
                 residual_scale: float = 0.02):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab.size
        tok_fam = vocab.families["TOK"]
        self.tok_offset = tok_fam.offset
        self.n_tok = tok_fam.size

        # ordinary rows for every family (used for all non-TOK ids); the TOK
        # slice of this table is unused and left at init
        self.plain = nn.Embedding(vocab.size, hidden)

        # factorized anchor construction
        self.B = nn.Linear(feats.d_numeric, hidden, bias=True)
        self.sector_emb = nn.Embedding(feats.n_sectors, hidden)
        self.sense_emb = nn.Embedding(3, hidden)       # buy / sell / undisc
        self.residual = nn.Embedding(self.n_tok, hidden)
        nn.init.normal_(self.residual.weight, std=residual_scale)

        # per-global-TOK-id -> instrument row / sector / sense, as buffers
        # global TOK id g in [tok_offset, tok_offset+n_tok); raw token id via
        # vocab, then features maps raw token -> instrument row / sense
        raw_of_global = vocab.tok_raw_of_global(
            np.arange(self.tok_offset, self.tok_offset + self.n_tok))
        instr_row = feats.tok_instr_row[raw_of_global]
        sense = feats.tok_sense[raw_of_global]
        # guard against any -1 (shouldn't happen for alive tokens)
        instr_row = np.where(instr_row < 0, 0, instr_row)
        sense = np.where(sense < 0, 2, sense)
        self.register_buffer("X", torch.tensor(feats.X, dtype=torch.float32))
        self.register_buffer("sector_of_tok",
                             torch.tensor(feats.sector_idx[instr_row], dtype=torch.long))
        self.register_buffer("instr_of_tok",
                             torch.tensor(instr_row, dtype=torch.long))
        self.register_buffer("sense_of_tok",
                             torch.tensor(sense, dtype=torch.long))

    def anchor_table(self) -> "torch.Tensor":
        """(n_tok, hidden) embedding for every anchor token, rebuilt from
        features each call (cheap; lets the geometry stay tied to B)."""
        feat = self.X[self.instr_of_tok]                       # (n_tok, d_feat)
        return (self.B(feat)
                + self.sector_emb(self.sector_of_tok)
                + self.sense_emb(self.sense_of_tok)
                + self.residual.weight)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        out = self.plain(input_ids)
        is_tok = (input_ids >= self.tok_offset) & \
                 (input_ids < self.tok_offset + self.n_tok)
        if is_tok.any():
            local = (input_ids[is_tok] - self.tok_offset)
            table = self.anchor_table()
            out = out.clone()
            out[is_tok] = table[local]
        return out


def build_model(mc: ModelConfig, vocab: FmVocab, feats: BondFeatures):
    """Assemble the HF causal-LM with the factorized embedding installed."""
    from transformers import LlamaForCausalLM
    config = build_llama_config(mc, vocab.size)
    model = LlamaForCausalLM(config)
    if mc.factorized_embedding:
        emb = FactorizedTokenEmbedding(vocab, feats, mc.hidden_size)
        model.set_input_embeddings(emb)
    return model
