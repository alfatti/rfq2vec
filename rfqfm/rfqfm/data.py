"""Torch data plumbing over a PackedCorpus (runs on the GPU box).

Each item is one packed sequence, padded or truncated to the context length.
Labels follow the HuggingFace convention (pad -> -100). Two knobs:

  * all-token loss (default, as in the NVIDIA blueprint): the model predicts
    every next token, so it also learns size / outcome / timing structure.

  * anchor-restricted loss (mask_to_anchors=True): labels are kept only at TOK
    positions, so the model is trained/graded purely on predicting the next
    (CUSIP, side). Useful for the recovery probes where we care about the
    choice distribution, not the scaffolding.

The event_id channel rides along so evaluation can join anchor predictions to
the true log_p_chosen for the Test-G floor.
"""
from __future__ import annotations

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise ImportError("rfqfm.data needs torch; install it on the GPU box") from e

import numpy as np

from .packed import PackedCorpus
from .tokenize import FAMILY

_TOK = FAMILY["TOK"]


class PackedDataset(Dataset):
    def __init__(self, corpus: PackedCorpus, context_tokens: int,
                 pad_id: int, mask_to_anchors: bool = False,
                 return_event_id: bool = False):
        self.c = corpus
        self.ctx = context_tokens
        self.pad_id = pad_id
        self.mask_to_anchors = mask_to_anchors
        self.return_event_id = return_event_id

    def __len__(self) -> int:
        return self.c.n_sequences

    def __getitem__(self, i: int):
        tok, fam, ev = self.c.sequence(i)
        tok = np.asarray(tok[: self.ctx], dtype=np.int64)
        fam = np.asarray(fam[: self.ctx], dtype=np.int64)
        n = len(tok)

        input_ids = np.full(self.ctx, self.pad_id, dtype=np.int64)
        input_ids[:n] = tok
        labels = np.full(self.ctx, -100, dtype=np.int64)
        labels[:n] = tok
        if self.mask_to_anchors:
            keep = np.zeros(self.ctx, dtype=bool)
            keep[:n] = fam == _TOK
            labels[~keep] = -100

        item = {"input_ids": torch.from_numpy(input_ids),
                "labels": torch.from_numpy(labels)}
        if self.return_event_id:
            evid = np.zeros(self.ctx, dtype=np.uint64)
            evid[:n] = np.asarray(ev[: self.ctx], dtype=np.uint64)
            famf = np.zeros(self.ctx, dtype=np.int64)
            famf[:n] = fam
            item["event_id"] = torch.from_numpy(evid.astype(np.int64))
            item["family"] = torch.from_numpy(famf)
        return item


class MixtureDataset(Dataset):
    """Interleave the session and tape corpora at a fixed ratio.

    Length is driven by the session corpus; tape items are drawn with the
    configured fraction so a fresh tape window appears in roughly that share of
    positions without duplicating the whole corpus in memory.
    """

    def __init__(self, session: PackedDataset, tape: PackedDataset,
                 tape_fraction: float, seed: int = 0):
        self.session, self.tape = session, tape
        self.p = float(tape_fraction)
        self.rng = np.random.default_rng(seed)
        self.n = len(session) + int(len(session) * self.p / max(1e-9, 1 - self.p))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self.rng.random() < self.p and len(self.tape):
            return self.tape[self.rng.integers(len(self.tape))]
        return self.session[i % len(self.session)]
