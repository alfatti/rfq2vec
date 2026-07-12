"""Bond features for the factorized token embedding.

The simulator builds a bond vector as v = s(t) * normalize(B @ x(t) + eps):
shared attribute geometry through B @ x, plus a per-token residual eps. I want
the model's TOK embedding to have the same shape, so the embedding is not ~90k
free rows but

    embed(TOK) = g(instrument_features) + sense_vector[sense] + residual[token]

This module produces the pieces the model needs: a standardized per-instrument
feature matrix, small categorical indices (sector, seniority), and the maps
from raw token id to its instrument row and its side sense. Features are taken
at a single reference date; the slow roll-down / aging in x(t) is deliberately
left for the sequence model to recover from context (that is Test E), not fed
through the static embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List

import numpy as np
import pyarrow.compute as pc

from .contract import RfqDataset

# numeric feature columns, in a fixed order (this order is the model's B input)
NUMERIC = ("log_tenor", "log_age", "log_amt", "coupon", "rating_z", "is_144a")


@dataclass
class BondFeatures:
    instrument_ids: np.ndarray          # (n_instr,) sorted raw instrument ids
    X: np.ndarray                       # (n_instr, len(NUMERIC)) float32, standardized
    sector_idx: np.ndarray              # (n_instr,) int
    seniority_idx: np.ndarray           # (n_instr,) int
    n_sectors: int
    n_seniorities: int
    # token-indexed maps (indexed by RAW token_id, 0..max_token_id)
    tok_instr_row: np.ndarray           # raw token_id -> row in X (-1 if none)
    tok_sense: np.ndarray               # raw token_id -> sense code (-1 if none)
    ref_date: date
    feature_names: List[str]

    @property
    def n_instruments(self) -> int:
        return len(self.instrument_ids)

    @property
    def d_numeric(self) -> int:
        return self.X.shape[1]


def _ref_date(ds: RfqDataset) -> date:
    cal = ds.dim("calendar")
    # calendar has a date column; take the first business day of the run
    for col in cal.column_names:
        if "date" in col:
            return cal.column(col).to_pylist()[0]
    raise KeyError("no date column in calendar table")


def build_bond_features(ds: RfqDataset, ref: date = None) -> BondFeatures:
    instr = ds.instruments()
    iss = ds.issuers()
    ref = ref or _ref_date(ds)

    iid = instr.column("instrument_id").to_numpy()
    order = np.argsort(iid)
    iid = iid[order]

    def col(name):
        return np.asarray(instr.column(name).to_pylist(), dtype=object)[order]

    # join sector / is_financial from issuers via issuer_id
    issuer_id = instr.column("issuer_id").to_numpy()[order]
    iss_id = iss.column("issuer_id").to_numpy()
    iss_sector = iss.column("sector_id").to_numpy()
    iss_fin = np.asarray(iss.column("is_financial").to_pylist(), dtype=bool)
    lut = np.full(int(iss_id.max()) + 1, 0, np.int64)
    lut[iss_id] = np.arange(len(iss_id))
    row = lut[issuer_id]
    sector_idx = iss_sector[row].astype(np.int64)
    is_fin = iss_fin[row]

    # time-varying features at the reference date
    def years_between(d_arr, d0, sign=1):
        out = np.empty(len(d_arr), np.float64)
        for i, d in enumerate(d_arr):
            out[i] = sign * (d - d0).days / 365.25
        return out

    maturity = col("maturity_date")
    issue = col("issue_date")
    tenor = np.clip(years_between(maturity, ref), 0.01, 60.0)
    age = np.clip(years_between(issue, ref, sign=-1), 0.0, 60.0)
    amt = instr.column("amt_issued").to_numpy()[order].astype(np.float64)
    coupon = instr.column("coupon").to_numpy()[order].astype(np.float64)
    rating = instr.column("rating_at_issue").to_numpy()[order].astype(np.float64)
    is144 = np.asarray(instr.column("is_144a").to_pylist(), dtype=np.float64)
    seniority_idx = instr.column("seniority").to_numpy()[order].astype(np.int64)

    raw = np.column_stack([
        np.log(tenor),
        np.log1p(age),
        np.log(amt),
        coupon,
        rating,               # standardized below
        is144,
    ]).astype(np.float64)

    # standardize the continuous columns (all but the binary is_144a at the end)
    mu = raw.mean(axis=0)
    sd = raw.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xz = (raw - mu) / sd
    Xz[:, NUMERIC.index("is_144a")] = is144    # keep the binary as 0/1
    X = Xz.astype(np.float32)

    # token -> instrument row and sense
    tm = ds.token_map()
    t_id = tm.column("token_id").to_numpy()
    t_instr = tm.column("instrument_id").to_numpy()
    t_sense = tm.column("sense").to_numpy()
    max_tok = int(t_id.max())
    instr_row_of_id = np.full(int(iid.max()) + 1, -1, np.int64)
    instr_row_of_id[iid] = np.arange(len(iid))
    tok_instr_row = np.full(max_tok + 1, -1, np.int64)
    tok_sense = np.full(max_tok + 1, -1, np.int64)
    tok_instr_row[t_id] = instr_row_of_id[t_instr]
    tok_sense[t_id] = t_sense

    return BondFeatures(
        instrument_ids=iid, X=X, sector_idx=sector_idx,
        seniority_idx=seniority_idx,
        n_sectors=int(sector_idx.max()) + 1,
        n_seniorities=int(seniority_idx.max()) + 1,
        tok_instr_row=tok_instr_row, tok_sense=tok_sense,
        ref_date=ref, feature_names=list(NUMERIC))
