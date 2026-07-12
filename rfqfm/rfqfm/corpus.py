"""Sequence assembly: turn per-line blocks into training sequences.

Two axes, per the dual-corpus decision:

  * session -- one client's history in time order, greedily cut into windows
    that fit the context. This is the personalization signal and the bulk of
    the corpus.

  * tape -- all clients interleaved in time order, sliding windows with a
    stride. Each line still carries its CLI token, so authorship survives; this
    is the view in which cross-client geometry and the clientele covariance
    become recoverable.

The mix is controlled by cfg.tape_fraction. Packing then wraps each sequence in
<bos>..<eos> and lays everything down as parallel token / family / event_id
arrays with an offset index (see packed.py).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import TokenizerConfig
from .packed import PackedCorpus, ragged_gather
from .tokenize import LineBlocks
from .vocab import FmVocab


def _greedy_windows(lengths: np.ndarray, budget: int) -> List[slice]:
    """Cut a sequence of per-item token counts into contiguous windows each
    summing to <= budget (each item is atomic -- never split a line)."""
    cuts, start, run = [], 0, 0
    for i, L in enumerate(lengths):
        if run + L > budget and run > 0:
            cuts.append(slice(start, i)); start, run = i, 0
        run += int(L)
    if start < len(lengths):
        cuts.append(slice(start, len(lengths)))
    return cuts


def build_sequences(blocks: LineBlocks, cfg: TokenizerConfig,
                    axis: str) -> List[np.ndarray]:
    """Return a list of line-index arrays, one per training sequence."""
    budget = cfg.context_tokens - 2      # room for <bos>/<eos>
    ll = blocks.line_len
    seqs: List[np.ndarray] = []

    if axis == "session":
        cli = blocks.client_id
        order = np.lexsort((blocks.ts_us, cli))
        c_ord = cli[order]
        # boundaries between clients in the ordered array
        bnd = np.nonzero(np.r_[True, c_ord[1:] != c_ord[:-1]])[0]
        bnd = np.r_[bnd, len(order)]
        for a, b in zip(bnd[:-1], bnd[1:]):
            idx = order[a:b]
            for sl in _greedy_windows(ll[idx], budget):
                seqs.append(idx[sl])

    elif axis == "tape":
        order = np.argsort(blocks.ts_us, kind="stable")
        lo = ll[order]
        stride = max(1, int(cfg.tape_stride_frac * budget))
        # cumulative tokens to place window boundaries by token budget
        cum = np.r_[0, np.cumsum(lo)]
        start_line = 0
        n = len(order)
        while start_line < n:
            # take lines until we hit the budget
            tgt = cum[start_line] + budget
            end_line = int(np.searchsorted(cum, tgt, side="right") - 1)
            end_line = max(end_line, start_line + 1)
            seqs.append(order[start_line:end_line])
            # advance by stride tokens
            nxt = cum[start_line] + stride
            start_line = max(int(np.searchsorted(cum, nxt, side="right") - 1),
                             start_line + 1)
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return seqs


def pack(blocks: LineBlocks, sequences: List[np.ndarray],
         vocab: FmVocab, axis: str) -> PackedCorpus:
    """Assemble sequences into a PackedCorpus, wrapping each in <bos>..<eos>."""
    n_seq = len(sequences)
    perm = np.concatenate(sequences) if n_seq else np.zeros(0, np.int64)
    seq_line_counts = np.array([len(s) for s in sequences], np.int64)

    # gather all block tokens in sequence order, once
    tok = ragged_gather(blocks.stream, blocks.line_start, blocks.line_len, perm)
    fam = ragged_gather(blocks.family, blocks.line_start, blocks.line_len, perm)
    evd = ragged_gather(blocks.event_id, blocks.line_start, blocks.line_len, perm)

    # tokens per sequence (sum of its lines' block lengths)
    line_lens_in_order = blocks.line_len[perm]
    seg_start = np.zeros(n_seq, np.int64)
    if n_seq:
        seg_start[1:] = np.cumsum(seq_line_counts)[:-1]
    tok_per_seq = np.add.reduceat(line_lens_in_order, seg_start) if n_seq \
        else np.zeros(0, np.int64)

    # final layout: each sequence is [bos, blocks..., eos]
    final_offsets = np.zeros(n_seq + 1, np.int64)
    final_offsets[1:] = np.cumsum(tok_per_seq + 2)
    T = int(final_offsets[-1])

    out_tok = np.empty(T, np.int64)
    out_fam = np.empty(T, np.uint8)
    out_evd = np.zeros(T, np.uint64)

    bos_pos = final_offsets[:-1]
    eos_pos = final_offsets[1:] - 1
    out_tok[bos_pos] = vocab.bos_id
    out_tok[eos_pos] = vocab.eos_id
    out_fam[bos_pos] = 0
    out_fam[eos_pos] = 0

    # scatter block tokens into the positions that are neither bos nor eos
    block_slots = np.ones(T, bool)
    block_slots[bos_pos] = False
    block_slots[eos_pos] = False
    out_tok[block_slots] = tok
    out_fam[block_slots] = fam
    out_evd[block_slots] = evd

    meta = dict(vocab_size=int(vocab.size), context_tokens=int(vocab.cfg.context_tokens),
                axis=axis, n_sequences=int(n_seq),
                pad_id=int(vocab.pad_id), bos_id=int(vocab.bos_id),
                eos_id=int(vocab.eos_id))
    return PackedCorpus(tokens=out_tok, family=out_fam, event_id=out_evd,
                        seq_offsets=final_offsets, meta=meta)


def build_corpus(blocks: LineBlocks, vocab: FmVocab) -> dict:
    """Build both corpus views per the config. Returns {'session':..,'tape':..}
    for whichever axes are enabled."""
    cfg = vocab.cfg
    out = {}
    if cfg.session_axis:
        out["session"] = pack(blocks, build_sequences(blocks, cfg, "session"),
                              vocab, "session")
    if cfg.tape_axis:
        out["tape"] = pack(blocks, build_sequences(blocks, cfg, "tape"),
                          vocab, "tape")
    return out
