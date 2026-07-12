"""Packed token corpus: the on-disk training format.

A corpus is three parallel arrays -- token ids, family codes, and the owning
event_id -- concatenated across all sequences, plus an offset index marking
sequence boundaries. This is the nanoGPT-style packed layout adapted to carry
two extra channels: the family code (so the loss/probes can restrict to anchor
positions) and the event_id at each anchor (so the model's predicted
probability there can be joined to the true log_p_chosen for the Test-G floor).

Large enough corpora are memory-mapped; the arrays are plain .npy so there is
no bespoke reader to maintain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def ragged_gather(values: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                  order: np.ndarray) -> np.ndarray:
    """Concatenate variable-length slices values[starts[i]:starts[i]+lengths[i]]
    for i in order, fully vectorized (no python loop over slices).

    This is the ragged-range construction: build one index array into values
    and gather once."""
    order = np.asarray(order, dtype=np.int64)
    seg_len = lengths[order]
    total = int(seg_len.sum())
    if total == 0:
        return values[:0].copy()
    out_seg_start = np.zeros(len(order), np.int64)
    out_seg_start[1:] = np.cumsum(seg_len)[:-1]
    src_base = starts[order] - out_seg_start
    idx = np.repeat(src_base, seg_len) + np.arange(total, dtype=np.int64)
    return values[idx]


@dataclass
class PackedCorpus:
    tokens: np.ndarray        # (T,) uint32/int64 concatenated sequences (bos..eos)
    family: np.ndarray        # (T,) uint8
    event_id: np.ndarray      # (T,) uint64 (0 for non-anchor / special positions)
    seq_offsets: np.ndarray   # (n_seq+1,) int64 boundaries into the above
    meta: dict                # vocab_size, context_tokens, axis, ...

    @property
    def n_sequences(self) -> int:
        return len(self.seq_offsets) - 1

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0])

    def sequence(self, i: int):
        a, b = int(self.seq_offsets[i]), int(self.seq_offsets[i + 1])
        return self.tokens[a:b], self.family[a:b], self.event_id[a:b]

    def lengths(self) -> np.ndarray:
        return np.diff(self.seq_offsets)

    # -- persistence ------------------------------------------------------
    def save(self, out_dir) -> None:
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        np.save(p / "tokens.npy", self.tokens.astype(np.uint32))
        np.save(p / "family.npy", self.family.astype(np.uint8))
        np.save(p / "event_id.npy", self.event_id.astype(np.uint64))
        np.save(p / "seq_offsets.npy", self.seq_offsets.astype(np.int64))
        (p / "meta.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, out_dir, mmap: bool = True) -> "PackedCorpus":
        p = Path(out_dir)
        mm = "r" if mmap else None
        return cls(
            tokens=np.load(p / "tokens.npy", mmap_mode=mm),
            family=np.load(p / "family.npy", mmap_mode=mm),
            event_id=np.load(p / "event_id.npy", mmap_mode=mm),
            seq_offsets=np.load(p / "seq_offsets.npy"),
            meta=json.loads((p / "meta.json").read_text()))
