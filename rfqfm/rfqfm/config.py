"""Configuration: how a line becomes tokens, and the three model sizes.

All pure dataclasses, no torch, so the config ladder and the tokenizer settings
can be imported and reasoned about on any machine (including the one that
generates the data and never sees a GPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Tokenizer / corpus configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenizerConfig:
    """The anchored-composite block scheme.

    Each RFQ leg becomes a short block whose anchor is the (CUSIP, side) token.
    The surrounding fields are few and bucketed so the anchor stays dominant in
    the model's next-token budget. Bucketing edges are fixed here (not fit from
    data) so the vocabulary is stable across runs and seeds -- the same design
    choice the simulator makes for its enum registries.
    """
    # size buckets on qty_par (par amount). Finer than qty_bucket so the model
    # has resolution on the size-dependent behaviour we plan to add generatively.
    size_edges: Tuple[float, ...] = (
        2.5e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2.5e7)          # -> 8 buckets
    # dealer-count buckets on n_dealers (competition intensity)
    ndealer_edges: Tuple[int, ...] = (1, 2, 3, 4, 5, 7, 10)   # -> 8 buckets
    # time-gap buckets on seconds since this client's previous RFQ, log-spaced.
    # 0 (first RFQ of a session) gets its own bucket 0; the rest are log bins
    # from ~1s to ~30 days.
    tdelta_n_bins: int = 24
    tdelta_min_s: float = 1.0
    tdelta_max_s: float = 30.0 * 86400.0

    # sequence assembly
    context_tokens: int = 4096          # window length in tokens
    session_axis: bool = True           # per-client-history sequences
    tape_axis: bool = True              # interleaved market-tape windows
    tape_fraction: float = 0.25         # share of tape windows in the mix
    tape_stride_frac: float = 0.5       # sliding-window stride as frac of ctx

    # per-leg block field order (documentation of the layout; the tokenizer
    # emits exactly these families around the anchor)
    leg_fields: Tuple[str, ...] = ("TDLT", "CLI", "TOK", "SZ", "NDLR", "OUT")


# ---------------------------------------------------------------------------
# Model ladder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """A decoder-only config on the NVIDIA/Llama architecture spine.

    Sized to *which structure we want back*, not only to token count. The
    transformer stays small by LLM standards; the interesting parameter mass is
    the token embedding, which we factorize (see features.py / model.py) so a
    ~100k vocabulary does not blow up the table.
    """
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    context_tokens: int = 4096
    max_position_embeddings: int = 8192
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    factorized_embedding: bool = True   # TOK rows = features @ B + residual
    tie_word_embeddings: bool = False

    def transformer_params(self, vocab_size: int) -> int:
        """Rough transformer parameter count (excludes the input embedding
        table; includes attention, MLP, norms, and the output head)."""
        h, L, i = self.hidden_size, self.num_layers, self.intermediate_size
        kv = self.num_kv_heads * (h // self.num_heads)
        per_layer = (h * h + 2 * h * kv + h * h) + (3 * h * i) + 2 * h
        head = 0 if self.tie_word_embeddings else vocab_size * h
        return L * per_layer + head


# The three rungs. hidden/layers/heads follow the standard shapes; the medium
# rung is deliberately the NVIDIA ~29M-parameter point so it reads as the
# obvious baseline.
SMALL = ModelConfig(
    name="small", hidden_size=256, num_layers=6, num_heads=8, num_kv_heads=2,
    intermediate_size=704)
MEDIUM = ModelConfig(
    name="medium", hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=2,
    intermediate_size=1408)
LARGE = ModelConfig(
    name="large", hidden_size=1024, num_layers=20, num_heads=16, num_kv_heads=4,
    intermediate_size=2816)

LADDER = {"small": SMALL, "medium": MEDIUM, "large": LARGE}
