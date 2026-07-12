"""rfq_lines -> per-line token blocks (the anchored-composite scheme).

I build the blocks vectorized rather than looping over lines, because the
production corpus is a decade of a full market (~10^8 lines) and a python loop
would not finish. The trick: lay every line out in a fixed 7-wide matrix
[delim, TDLT, CLI, TOK, SZ, NDLR, OUT], blank the fields that a continuation
leg does not carry with a DROP sentinel, flatten row-major, and strip the
sentinels. A first leg flattens to <sep> TDLT CLI TOK SZ NDLR OUT; a
continuation leg flattens to <leg> TOK SZ -- exactly the layout I want, and the
package structure is preserved for the switch/list probes.

Every TOK position is an anchor and maps back to its line's event_id, so the
model's predicted probability there can be graded against the true
log_p_chosen in event_truth (Test G). The output keeps lines in input order;
corpus.py does the session / tape ordering and windowing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .config import TokenizerConfig
from .vocab import FmVocab

_US = 1_000_000
DROP = np.int64(-1)


@dataclass
class LineBlocks:
    stream: np.ndarray        # (P,) int64 global token ids, lines in input order
    family: np.ndarray        # (P,) int8 family code (see FAMILY)
    event_id: np.ndarray      # (P,) uint64, the owning line's event_id (all positions)
    line_start: np.ndarray    # (n_lines,) int64 offset of each line's block in stream
    line_len: np.ndarray      # (n_lines,) int64 block length
    anchor_pos: np.ndarray    # (n_lines,) int64 absolute index of the TOK anchor
    client_id: np.ndarray     # (n_lines,) int64 per-line
    ts_us: np.ndarray         # (n_lines,) int64 per-line event time (microseconds)
    ev_id: np.ndarray         # (n_lines,) uint64 per-line event_id

    @property
    def n_lines(self) -> int:
        return len(self.line_start)


FAMILY = {"SPECIAL": 0, "TDLT": 1, "CLI": 2, "TOK": 3, "SZ": 4, "NDLR": 5, "OUT": 6}


def _bucket(edges, values) -> np.ndarray:
    """Right-open bucketing: idx = # edges strictly below value, in [0, len(edges)]."""
    e = np.asarray(edges, dtype=np.float64)
    return np.searchsorted(e, np.asarray(values, dtype=np.float64), side="right")


def _tdelta_bucket(gap_s: np.ndarray, cfg: TokenizerConfig) -> np.ndarray:
    """gap<=0 (first RFQ of a client) -> bucket 0; else log-spaced 1..n_bins."""
    g = np.asarray(gap_s, dtype=np.float64)
    lo, hi, n = cfg.tdelta_min_s, cfg.tdelta_max_s, cfg.tdelta_n_bins
    logg = np.log(np.clip(g, lo, hi))
    frac = (logg - np.log(lo)) / (np.log(hi) - np.log(lo))
    idx = 1 + np.floor(frac * (n - 1e-9)).astype(np.int64)
    idx = np.clip(idx, 1, n)
    return np.where(g > 0, idx, 0)


def tokenize_lines(table: pa.Table, vocab: FmVocab) -> LineBlocks:
    cfg = vocab.cfg
    n = table.num_rows

    client = table.column("client_id").to_numpy().astype(np.int64)
    line_no = table.column("line_no").to_numpy().astype(np.int64)
    tok_raw = table.column("token_id").to_numpy().astype(np.int64)
    qty = table.column("qty_par").to_numpy().astype(np.float64)
    ndlr = table.column("n_dealers").to_numpy().astype(np.float64)
    enq = table.column("enquiry_outcome").to_numpy().astype(np.int64)
    our = table.column("our_result").to_numpy().astype(np.int64)
    evid = table.column("event_id").to_numpy().astype(np.uint64)
    ts_us = (table.column("ts").cast(pa.int64()).to_numpy()).astype(np.int64)

    first = line_no == 0

    # per-client gap since previous RFQ, computed on first legs (one per event)
    gap = np.zeros(n, np.float64)
    fi = np.nonzero(first)[0]
    order = fi[np.lexsort((ts_us[fi], client[fi]))]
    c_sorted = client[order]
    t_sorted = ts_us[order]
    prev_t = np.empty_like(t_sorted)
    prev_t[0] = -1
    prev_t[1:] = t_sorted[:-1]
    same = np.empty_like(c_sorted, dtype=bool)
    same[0] = False
    same[1:] = c_sorted[1:] == c_sorted[:-1]
    g = np.where(same, (t_sorted - prev_t) / _US, -1.0)
    gap[order] = g

    # global ids per family
    delim = np.where(first, np.int64(vocab.sep_id), np.int64(vocab.leg_id))
    tdlt_g = vocab.family_global("TDLT", _tdelta_bucket(gap, cfg))
    cli_g = vocab.cli_global(client)
    tok_g = vocab.tok_global(tok_raw)
    sz_g = vocab.family_global("SZ", _bucket(cfg.size_edges, qty))
    ndlr_g = vocab.family_global("NDLR", _bucket(cfg.ndealer_edges, ndlr))
    out_local = np.clip(enq, 0, 3) * 6 + np.clip(our, 0, 5)
    out_g = vocab.family_global("OUT", out_local)

    # 7-wide matrix; continuation legs drop TDLT/CLI/NDLR/OUT
    d = DROP
    M = np.stack([
        delim,
        np.where(first, tdlt_g, d),
        np.where(first, cli_g, d),
        tok_g,
        sz_g,
        np.where(first, ndlr_g, d),
        np.where(first, out_g, d),
    ], axis=1)                                                   # (n, 7)
    fam = np.array([FAMILY["SPECIAL"], FAMILY["TDLT"], FAMILY["CLI"],
                    FAMILY["TOK"], FAMILY["SZ"], FAMILY["NDLR"], FAMILY["OUT"]],
                   dtype=np.int8)
    Fam = np.broadcast_to(fam, (n, 7))
    Ev = np.broadcast_to(evid[:, None], (n, 7))
    is_anchor_col = np.broadcast_to(np.arange(7) == 3, (n, 7))

    keep = (M != DROP)                                           # (n, 7)
    line_len = keep.sum(axis=1).astype(np.int64)
    line_start = np.zeros(n, np.int64)
    line_start[1:] = np.cumsum(line_len)[:-1]

    flat_keep = keep.reshape(-1)
    stream = M.reshape(-1)[flat_keep].astype(np.int64)
    family = Fam.reshape(-1)[flat_keep].astype(np.int8)
    event_id = Ev.reshape(-1)[flat_keep].astype(np.uint64)
    # anchor (TOK) positions: find them in the pre-strip flat space, then map to
    # post-strip positions via the running count of kept entries
    anchor_flat = is_anchor_col.reshape(-1) & flat_keep
    post = np.cumsum(flat_keep) - 1
    anchor_pos = post[np.nonzero(anchor_flat)[0]].astype(np.int64)

    return LineBlocks(
        stream=stream, family=family, event_id=event_id,
        line_start=line_start, line_len=line_len, anchor_pos=anchor_pos,
        client_id=client, ts_us=ts_us, ev_id=evid)
