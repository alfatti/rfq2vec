"""The foundation-model vocabulary.

One global id space carved into disjoint per-family ranges, in the same spirit
as the NVIDIA pipeline's offset scheme. The families:

    <special>   pad / bos / eos / sep / leg / unk
    TDLT_*      log-binned gap since the client's previous RFQ   (intensity)
    CLI_*       client id                                        (the author)
    TOK_*       the (CUSIP, side) atom                           (the anchor)
    SZ_*        par-size bucket
    NDLR_*      dealer-count bucket
    OUT_*       enquiry_outcome x our_result                     (the label)

TOK dominates the count (3 senses per instrument). Its embedding row is the
Arora word vector, which is why I keep it as a single first-class entry rather
than shattering it -- and why model.py factorizes those rows through bond
features instead of learning ~90k of them free.

The mappers from raw ids (token_id, client_id) to global ids are numpy lookup
arrays so tokenizing a day of lines is vectorized, not a python dict loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import TokenizerConfig
from .contract import RfqDataset

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<leg>", "<unk>"]
_N_ENQ = 4     # EnquiryOutcome cardinality
_N_OUR = 6     # OurResult cardinality


@dataclass
class _Family:
    name: str
    offset: int
    size: int


class FmVocab:
    """Global vocabulary + vectorized raw-id -> global-id mappers."""

    def __init__(self, cfg: TokenizerConfig, token_ids: np.ndarray,
                 client_ids: np.ndarray):
        self.cfg = cfg
        self._tok_raw = np.sort(np.asarray(token_ids, dtype=np.int64))
        self._cli_raw = np.sort(np.asarray(client_ids, dtype=np.int64))

        n_tdlt = cfg.tdelta_n_bins + 1          # +1 for the first-of-session bin
        n_sz = len(cfg.size_edges) + 1
        n_ndlr = len(cfg.ndealer_edges) + 1
        n_out = _N_ENQ * _N_OUR

        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.families: Dict[str, _Family] = {}

        cur = 0
        for s in SPECIALS:
            self._add(cur, s); cur += 1
        self._n_special = cur

        cur = self._add_family("TDLT", cur, n_tdlt, lambda i: f"TDLT_{i}")
        # CLI/TOK carry the raw id in the token string for debuggability
        self._cli_off = cur
        cur = self._add_indexed("CLI", cur, self._cli_raw)
        self._tok_off = cur
        cur = self._add_indexed("TOK", cur, self._tok_raw)
        cur = self._add_family("SZ", cur, n_sz, lambda i: f"SZ_{i}")
        cur = self._add_family("NDLR", cur, n_ndlr, lambda i: f"NDLR_{i}")
        cur = self._add_family("OUT", cur, n_out, lambda i: f"OUT_{i}")
        self.size = cur

        # vectorized raw -> global lookup arrays
        self._tok_lut = np.full(int(self._tok_raw.max()) + 1, self.unk_id, np.int64)
        self._tok_lut[self._tok_raw] = self._tok_off + np.arange(len(self._tok_raw))
        self._cli_lut = np.full(int(self._cli_raw.max()) + 1, self.unk_id, np.int64)
        self._cli_lut[self._cli_raw] = self._cli_off + np.arange(len(self._cli_raw))

    # -- construction helpers --------------------------------------------
    def _add(self, gid: int, tok: str) -> None:
        self.id_to_token[gid] = tok
        self.token_to_id[tok] = gid

    def _add_family(self, name, offset, size, fmt) -> int:
        for i in range(size):
            self._add(offset + i, fmt(i))
        self.families[name] = _Family(name, offset, size)
        return offset + size

    def _add_indexed(self, name, offset, raw_ids) -> int:
        for j, rid in enumerate(raw_ids):
            self._add(offset + j, f"{name}_{int(rid)}")
        self.families[name] = _Family(name, offset, len(raw_ids))
        return offset + len(raw_ids)

    # -- special ids ------------------------------------------------------
    @property
    def pad_id(self): return self.token_to_id["<pad>"]
    @property
    def bos_id(self): return self.token_to_id["<bos>"]
    @property
    def eos_id(self): return self.token_to_id["<eos>"]
    @property
    def sep_id(self): return self.token_to_id["<sep>"]
    @property
    def leg_id(self): return self.token_to_id["<leg>"]
    @property
    def unk_id(self): return self.token_to_id["<unk>"]

    # -- vectorized mappers ----------------------------------------------
    def tok_global(self, raw_token_id: np.ndarray) -> np.ndarray:
        r = np.asarray(raw_token_id, dtype=np.int64)
        out = np.full(r.shape, self.unk_id, np.int64)
        ok = (r >= 0) & (r < len(self._tok_lut))
        out[ok] = self._tok_lut[r[ok]]
        return out

    def cli_global(self, client_id: np.ndarray) -> np.ndarray:
        r = np.asarray(client_id, dtype=np.int64)
        out = np.full(r.shape, self.unk_id, np.int64)
        ok = (r >= 0) & (r < len(self._cli_lut))
        out[ok] = self._cli_lut[r[ok]]
        return out

    def family_global(self, name: str, local_idx: np.ndarray) -> np.ndarray:
        fam = self.families[name]
        li = np.clip(np.asarray(local_idx, dtype=np.int64), 0, fam.size - 1)
        return fam.offset + li

    def is_family(self, gid: np.ndarray, name: str) -> np.ndarray:
        fam = self.families[name]
        g = np.asarray(gid)
        return (g >= fam.offset) & (g < fam.offset + fam.size)

    def tok_raw_of_global(self, gid: np.ndarray) -> np.ndarray:
        """Inverse for TOK positions: global id -> raw token_id (-1 elsewhere).
        Needed by the truth sidecar and the geometry probes."""
        g = np.asarray(gid, dtype=np.int64)
        out = np.full(g.shape, -1, np.int64)
        m = self.is_family(g, "TOK")
        out[m] = self._tok_raw[g[m] - self._tok_off]
        return out

    # -- persistence ------------------------------------------------------
    @classmethod
    def from_dataset(cls, ds: RfqDataset, cfg: TokenizerConfig) -> "FmVocab":
        tok = ds.token_map().column("token_id").to_numpy()
        cli = ds.clients().column("client_id").to_numpy()
        return cls(cfg, tok, cli)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "cfg": self.cfg.__dict__,
            "tok_raw": self._tok_raw.tolist(),
            "cli_raw": self._cli_raw.tolist(),
        }))

    @classmethod
    def load(cls, path) -> "FmVocab":
        d = json.loads(Path(path).read_text())
        cfg = TokenizerConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                                 for k, v in d["cfg"].items()})
        return cls(cfg, np.array(d["tok_raw"]), np.array(d["cli_raw"]))

    def summary(self) -> str:
        parts = [f"{f.name}:{f.size}" for f in self.families.values()]
        return f"FmVocab(size={self.size}, specials={self._n_special}, " \
               + " ".join(parts) + ")"
