"""Vectorized counter-based randomness: Philox4x32-10 over the xp backend.

Why not np.random / cupy.random generators: the reproducibility contract
wants every line's randomness addressable by (key, counter) so a single event
replays in isolation, and the GPU wants all events' randomness computed in one
batched call. Counter-based RNG gives both at once -- you evaluate the random
numbers AT arbitrary counters, in parallel. Philox4x32-10 is the Random123 /
cuRAND workhorse for exactly this reason; ten rounds of 32x32->64 multiplies
and XORs, all of which numpy/cupy uint64 arithmetic does natively.

Stream discipline (production path):
  * key   = first 8 bytes of sha256(f"{seed_root}//{stream}") as two uint32
  * ctr   = a uint64 the caller lays out; emission_batch assigns each
            (day, event, leg) a block of LINE_STRIDE counters with fixed
            purpose offsets inside (see emission_batch._OFF)
  * one counter yields 4 uint32; uniforms map them to (0,1) open interval

This is a DIFFERENT stream family from the reference path's numpy Philox4x64
(numpy does not expose batched evaluation at arbitrary counters). Both paths
are individually deterministic; they are not bit-compatible with each other,
and the reference path remains the semantic spec.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .backend import xp

_M0 = xp.uint64(0xD2511F53)
_M1 = xp.uint64(0xCD9E8D57)
_W0 = xp.uint32(0x9E3779B9)
_W1 = xp.uint32(0xBB67AE85)
_LO = xp.uint64(0xFFFFFFFF)


def derive_key32(seed_root_hex: str, stream: str) -> tuple[int, int]:
    d = hashlib.sha256(f"{seed_root_hex}//{stream}".encode()).digest()
    return int.from_bytes(d[:4], "little"), int.from_bytes(d[4:8], "little")


def philox4x32(key: tuple[int, int], ctr):
    """ctr: uint64 array of any shape -> uint32 array shape + (4,).
    The 64-bit counter fills lanes (c0, c1); lanes (c2, c3) are zero."""
    ctr = xp.asarray(ctr, dtype=xp.uint64)
    c0 = (ctr & _LO).astype(xp.uint32)
    c1 = (ctr >> xp.uint64(32)).astype(xp.uint32)
    c2 = xp.zeros_like(c0)
    c3 = xp.zeros_like(c0)
    k0 = xp.full_like(c0, key[0])
    k1 = xp.full_like(c0, key[1])
    for _ in range(10):
        p0 = _M0 * c0.astype(xp.uint64)
        p1 = _M1 * c2.astype(xp.uint64)
        hi0 = (p0 >> xp.uint64(32)).astype(xp.uint32)
        lo0 = (p0 & _LO).astype(xp.uint32)
        hi1 = (p1 >> xp.uint64(32)).astype(xp.uint32)
        lo1 = (p1 & _LO).astype(xp.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + _W0
        k1 = k1 + _W1
    return xp.stack([c0, c1, c2, c3], axis=-1)


def uniforms(key: tuple[int, int], ctr0, n: int):
    """(E,) counter starts -> (E, n) float64 uniforms in the OPEN interval
    (0, 1). Consumes ceil(n / 4) consecutive counters per row."""
    ctr0 = xp.asarray(ctr0, dtype=xp.uint64)
    blocks = (n + 3) // 4
    ctrs = ctr0[:, None] + xp.arange(blocks, dtype=xp.uint64)[None, :]
    bits = philox4x32(key, ctrs).reshape(len(ctr0), blocks * 4)[:, :n]
    return (bits.astype(xp.float64) + 0.5) / 4294967296.0


def normals(key: tuple[int, int], ctr0, n: int):
    """(E, n) standard normals via Box-Muller; consumes ceil(n/2)*2 uniforms
    (i.e. ceil(n/2) counter... blocks per the uniforms layout)."""
    m = ((n + 1) // 2) * 2
    u = uniforms(key, ctr0, m)
    r = xp.sqrt(-2.0 * xp.log(u[:, 0::2]))
    th = 2.0 * xp.pi * u[:, 1::2]
    out = xp.empty((u.shape[0], m), dtype=xp.float64)
    out[:, 0::2] = r * xp.cos(th)
    out[:, 1::2] = r * xp.sin(th)
    return out[:, :n]


def icdf_choice(u, cum_probs):
    """Inverse-CDF draw from a small fixed table: u (E,), cum_probs (m,)
    -> int64 (E,). Used for list lengths, dealer counts, platforms."""
    cp = xp.asarray(cum_probs, dtype=xp.float64)
    return (u[:, None] > cp[None, :]).sum(axis=1).astype(xp.int64)


def row_categorical(u, weights):
    """One draw per row from unnormalized nonnegative weights (E, m) using a
    single uniform per row -- inverse CDF via cumulative sums. Rows with zero
    total weight return -1."""
    w = xp.asarray(weights, dtype=xp.float64)
    tot = w.sum(axis=1, keepdims=True)
    ok = tot[:, 0] > 0
    cdf = xp.cumsum(w, axis=1)
    x = u * xp.where(ok, tot[:, 0], 1.0)
    idx = (cdf < x[:, None]).sum(axis=1)
    idx = xp.minimum(idx, w.shape[1] - 1)
    return xp.where(ok, idx, -1).astype(xp.int64)


def masked_softmax_draw(u, logits, mask):
    """One softmax draw per row with a candidate mask, plus the exact log
    partition function over the masked set. logits (E, m) float, mask (E, m)
    bool, u (E,) uniforms. Returns (choice int64 with -1 for empty rows,
    log_z float64, logit_chosen float64, n_candidates int64).

    The cumulative-sum inverse-CDF needs one uniform per row instead of m
    Gumbels -- that is what keeps the per-event counter blocks small enough
    for the replay layout."""
    neg = xp.float64(-xp.inf)
    lg = xp.where(mask, logits.astype(xp.float64), neg)
    mx = lg.max(axis=1, keepdims=True)
    any_row = xp.isfinite(mx[:, 0])
    mx = xp.where(xp.isfinite(mx), mx, 0.0)
    ex = xp.where(mask, xp.exp(lg - mx), 0.0)
    z = ex.sum(axis=1)
    log_z = xp.where(any_row, xp.log(xp.maximum(z, 1e-300)) + mx[:, 0], neg)
    cdf = xp.cumsum(ex, axis=1)
    x = u * xp.where(any_row, z, 1.0)
    idx = (cdf < x[:, None]).sum(axis=1)
    idx = xp.minimum(idx, logits.shape[1] - 1)
    choice = xp.where(any_row, idx, -1).astype(xp.int64)
    safe = xp.maximum(choice, 0)
    logit_chosen = xp.where(any_row,
                            xp.take_along_axis(lg, safe[:, None], axis=1)[:, 0],
                            neg)
    n_cand = mask.sum(axis=1).astype(xp.int64)
    return choice, log_z, logit_chosen, n_cand
