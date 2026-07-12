"""Attribute-vs-residual decomposition of what a model recovered.

This is the instrument for the inductive-bias study. In the generator a bond
direction is normalize(B x + eps): B x is the low-rank, attribute-driven part
(rank <= p, and side-blind -- buy and sell of a bond share it), while eps is the
full-rank idiosyncratic residual that also carries the side split. So "how much
geometry did the model recover" splits cleanly into two questions the token
representation trades off:

    attribute_recovery -- does the model's token geometry match the B x Gram?
    residual_recovery  -- the *partial* correlation with the full geometry,
                          controlling for the attribute Gram: does it capture
                          the part attributes cannot explain?

A token scheme that drops bond identity (attributes only) caps residual_recovery
at ~0 by construction -- eps is information-theoretically absent from the input.
One that keeps identity can reach it, given scale. Feeding the oracle's own
attribute directions in makes residual_recovery read ~0; feeding the full
realized vectors makes it read 1. That is the self-test, and it is also a faithful
simulation of the two ceilings, so the metric is guaranteed to detect the
tokenization effect when it is there.

Everything works through token x token Gram matrices, so a model embedding of any
width is comparable to the d-dimensional oracle geometry. Torch-free; the
recovered vectors come from extract.py (a static embedding row for the identity
schemes, a pooled contextual state for the shattered ones).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contract import RfqDataset


@dataclass
class ReferenceGeometry:
    token_id: np.ndarray      # (n_tok,)
    attr_dir: np.ndarray      # (n_tok, d) normalized B x  (attribute part, side-blind)
    full_dir: np.ndarray      # (n_tok, d) normalized realized v  (full geometry)
    attr_share: float         # fraction of full-geometry variance in the attr subspace


def _x_at_run_start(ds: RfqDataset) -> tuple:
    """Per-instrument feature matrix x at the run's first day, from
    instrument_state_daily (the same x the generator fed to B)."""
    files = sorted(glob.glob(str(ds.root / "tables/instrument_state_daily/**/*.parquet"),
                             recursive=True))
    t = pa.concat_tables([pq.read_table(f) for f in files])
    day0 = t.column("trade_date").to_numpy().min()
    m = t.column("trade_date").to_numpy() == day0
    x = np.asarray(t.column("x").to_pylist(), dtype=np.float64)[m]
    iid = t.column("instrument_id").to_numpy()[m]
    return x, iid


def load_reference_geometry(ds: RfqDataset) -> ReferenceGeometry:
    tv = ds.dim("oracle_token_vectors")
    tok = tv.column("token_id").to_numpy()
    v = np.asarray(tv.column("v").to_pylist(), dtype=np.float64)
    eps = np.asarray(ds.dim("oracle_embeddings").column("eps").to_pylist(),
                     dtype=np.float64)
    B = np.asarray(ds.dim("oracle_attribute_gain").column("gain").to_pylist(),
                   dtype=np.float64)
    tm = ds.token_map()
    tm_tok = tm.column("token_id").to_numpy()
    tm_instr = tm.column("instrument_id").to_numpy()
    lut = np.full(int(tm_tok.max()) + 1, -1, np.int64)
    lut[tm_tok] = tm_instr

    x, iid = _x_at_run_start(ds)
    row_of = np.full(int(iid.max()) + 1, -1, np.int64)
    row_of[iid] = np.arange(len(iid))
    x_tok = x[row_of[lut[tok]]]                       # (n_tok, p)

    attr = x_tok @ B                                  # B x, side-blind
    # exact-reconstruction guard: normalize(attr + eps) must equal v-hat
    recon = attr + eps
    recon /= np.linalg.norm(recon, axis=1, keepdims=True)
    vhat = v / np.linalg.norm(v, axis=1, keepdims=True)
    if np.abs(recon - vhat).max() > 1e-4:
        raise ValueError("oracle geometry does not reconstruct; check schema version")

    attr_dir = attr / np.clip(np.linalg.norm(attr, axis=1, keepdims=True), 1e-12, None)

    # share of full-geometry variance living in the attribute subspace
    Vt = np.linalg.svd(attr_dir - attr_dir.mean(0), full_matrices=False)[2]
    p = B.shape[0]
    basis = Vt[:p]                                    # (<=p, d) attribute subspace
    c = vhat - vhat.mean(0)
    proj = c @ basis.T
    share = float((proj ** 2).sum() / (c ** 2).sum())
    return ReferenceGeometry(token_id=tok, attr_dir=attr_dir, full_dir=vhat,
                             attr_share=share)


def _offdiag(G: np.ndarray) -> np.ndarray:
    n = G.shape[0]
    return G[~np.eye(n, dtype=bool)]


def _partial_corr(x, y, z) -> float:
    """corr(x, y | z): correlation of x and y after regressing each on [1, z]."""
    Z = np.column_stack([np.ones_like(z), z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx, ry = x - Z @ bx, y - Z @ by
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def decompose_recovery(recovered: np.ndarray, ds: RfqDataset,
                       max_tokens: int = 4000, seed: int = 0) -> dict:
    """recovered : (n_tok, h) model bond vectors in ascending-token order.

    Returns attribute / residual / full recovery of the token geometry, plus the
    oracle's own attribute share for context."""
    ref = load_reference_geometry(ds)
    n = len(ref.token_id)
    idx = (np.arange(n) if n <= max_tokens
           else np.random.default_rng(seed).choice(n, max_tokens, replace=False))

    def gram(M):
        M = M[idx]
        M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
        return _offdiag(M @ M.T)

    g_model = gram(recovered)
    g_attr = gram(ref.attr_dir)
    g_full = gram(ref.full_dir)

    attribute_recovery = float(np.corrcoef(g_model, g_attr)[0, 1])
    full_recovery = float(np.corrcoef(g_model, g_full)[0, 1])
    residual_recovery = _partial_corr(g_model, g_full, g_attr)
    return dict(
        n_tokens=len(idx),
        attribute_recovery=attribute_recovery,
        residual_recovery=residual_recovery,
        full_recovery=full_recovery,
        oracle_attr_share=ref.attr_share)
