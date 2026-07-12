"""The entropy floor -- what "how big does the model need to be" is measured
against.

event_truth stores, per line, the exact probability the generator assigned to
the token it actually emitted (log_p_chosen), computed *with* the true latent
walk position, the client tilt, and the candidate mask in hand. Averaging
-log_p_chosen gives the choice entropy conditional on the latents. Call this
floor one.

A model reading only the observable tape does not know those latents; it is
uncertain about them too, so its cross-entropy cannot fall to floor one. The
quantity it can actually approach is the choice entropy conditional on the
*observable history* -- floor two -- which sits above floor one by exactly the
information the latents carry that the tape does not reveal.

So the honest picture, and the reason this is worth measuring:

    model cross-entropy  =  floor_one
                          + (floor_two - floor_one)   latent information gap
                          + (model - floor_two)       approximation error

The last term shrinks with scale; the middle term does not, and it is itself
the thing a desk cares about -- the ceiling on how predictable RFQ flow is from
the tape alone. We know floor one exactly; we estimate floor two from the
largest model's asymptote. Plotting (model cross-entropy - floor one) against
model size is the Test-G scaling curve, and no real-data model can draw it
because it never knows its floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import RfqDataset


@dataclass
class FloorReport:
    floor_one_nats: float             # E[-log_p_chosen], the given-latent floor
    uniform_baseline_nats: float      # E[log n_candidates], pick-uniformly reference
    n_lines: int
    by_stratum: dict                  # optional stratified floors

    def as_dict(self) -> dict:
        return dict(floor_one_nats=self.floor_one_nats,
                    floor_one_bits=self.floor_one_nats / np.log(2),
                    uniform_baseline_nats=self.uniform_baseline_nats,
                    n_lines=self.n_lines, by_stratum=self.by_stratum)


def entropy_floor(ds: RfqDataset, stratify_by_candidates: bool = True
                  ) -> FloorReport:
    """Compute floor one and the uniform-choice reference from event_truth."""
    t = ds.scan("event_truth",
                columns=["log_p_chosen", "n_candidates"])
    lp = t.column("log_p_chosen").to_numpy()
    ncand = t.column("n_candidates").to_numpy().astype(np.float64)
    finite = np.isfinite(lp)
    lp = lp[finite]
    ncand = ncand[finite]

    floor_one = float(-lp.mean())
    uniform = float(np.log(np.maximum(ncand, 1.0)).mean())

    by = {}
    if stratify_by_candidates and len(ncand):
        # bucket by log2 of candidate-set size; the floor rises with more choices
        b = np.clip(np.floor(np.log2(np.maximum(ncand, 1.0))).astype(int), 0, 12)
        for k in np.unique(b):
            m = b == k
            by[f"log2_ncand={int(k)}"] = dict(
                n=int(m.sum()),
                floor_one_nats=float(-lp[m].mean()),
                uniform_nats=float(np.log(np.maximum(ncand[m], 1.0)).mean()))
    return FloorReport(floor_one_nats=floor_one,
                       uniform_baseline_nats=uniform,
                       n_lines=int(finite.sum()), by_stratum=by)


def excess_over_floor(model_logp: np.ndarray, event_ids: np.ndarray,
                      ds: RfqDataset) -> dict:
    """Given a model's predicted log-prob at anchor positions and the event_ids
    they correspond to, join to the true log_p_chosen and report the
    decomposition terms we can measure directly.

    model_logp, event_ids : 1-D arrays aligned position-for-position.
    """
    t = ds.scan("event_truth", columns=["event_id", "log_p_chosen"])
    eid = t.column("event_id").to_numpy().astype(np.uint64)
    tlp = t.column("log_p_chosen").to_numpy()
    # event_id is a structured 64-bit id (shard in the high bits), far too
    # sparse for an index array -- join by sorted search instead
    srt = np.argsort(eid)
    eid_s, tlp_s = eid[srt], tlp[srt]

    ev = np.asarray(event_ids, dtype=np.uint64)
    pos = np.searchsorted(eid_s, ev)
    pos = np.clip(pos, 0, len(eid_s) - 1)
    hit = eid_s[pos] == ev
    true_lp = np.where(hit, tlp_s[pos], np.nan)

    m = np.isfinite(true_lp) & np.isfinite(model_logp)
    model_ce = float(-np.mean(model_logp[m]))
    floor_one = float(-np.mean(true_lp[m]))
    return dict(
        n=int(m.sum()),
        model_cross_entropy_nats=model_ce,
        floor_one_nats=floor_one,
        excess_over_floor_one_nats=model_ce - floor_one,
        excess_over_floor_one_bits=(model_ce - floor_one) / np.log(2))
