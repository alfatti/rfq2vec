"""Arrival layer: the Cox sampling of the grid's intensity surface.

The state layer already produced lambda_sector(t) = lambda_bar_s * m[regime_s]
* (r_t/r0)^gamma on the 1-min grid; this module owns only the "how many, when"
draw: per (grid step, sector), N ~ Poisson(lambda_s), times uniform within the
step. Content-addressed per day ("arrivals" stream, counter = day ordinal), so
any day's arrival panel regenerates in isolation.

Deliberate consequence of the agreed factorization: arrivals are PER-SECTOR,
and the emission layer's softmax runs WITHIN the arrival's sector. The MMPP
envelope is thereby exact by construction; cross-sector co-occurrence carries
the walk's geometry through the shared c_t in each sector's within-sector
choice, plus the intensity co-movement term (shared radial load) -- a
decomposition the oracle can separate exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .state import GridCalendar, derive_stream_key
from .writer import PhiloxLedger

_ARRIVAL_CTR_STRIDE = 1 << 20


@dataclass(frozen=True)
class ArrivalPanel:
    """One day's arrivals, time-ordered."""
    ts_us: np.ndarray       # int64
    grid_idx: np.ndarray    # int64
    step_k: np.ndarray      # int64, step within the day
    sector: np.ndarray      # int64

    def __len__(self) -> int:
        return len(self.ts_us)


def sample_arrivals(context_grid_day: pa.Table, cal: GridCalendar,
                    day_ordinal: int, seed_root_hex: str) -> ArrivalPanel:
    lam = np.asarray(context_grid_day["lambda_sector"].combine_chunks()
                     .flatten()).reshape(context_grid_day.num_rows, -1)
    gidx = context_grid_day["grid_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
    ts0 = context_grid_day["ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)

    key = derive_stream_key(seed_root_hex, "arrivals")
    g = PhiloxLedger.generator(*key, day_ordinal * _ARRIVAL_CTR_STRIDE)

    counts = g.poisson(lam)                      # (steps, sectors)
    total = int(counts.sum())
    step_k = np.repeat(np.arange(lam.shape[0]), counts.sum(axis=1))
    sector = np.concatenate([np.repeat(np.arange(lam.shape[1]), c)
                             for c in counts]) if total else np.empty(0, np.int64)
    offs = (g.random(total) * cal.step_us).astype(np.int64)
    ts = ts0[step_k] + offs

    order = np.argsort(ts, kind="stable")
    return ArrivalPanel(ts_us=ts[order], grid_idx=gidx[step_k][order],
                        step_k=step_k[order].astype(np.int64),
                        sector=sector[order].astype(np.int64))
