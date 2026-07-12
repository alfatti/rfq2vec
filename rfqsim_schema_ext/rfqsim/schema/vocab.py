"""Tokenizer and vocabulary for the (CUSIP, sense) token space.

Four responsibilities, one contract:

  1. FeatureSpec -- the deterministic morphology x_w(t). Every transform is a
     FIXED affine/RBF map with constants pinned in the spec (never
     data-dependent standardization): oracle and learner must compute
     bit-identical x from the same inputs, and refitting a scaler between
     runs would silently move B. Tenor roll-down is carried entirely by the
     time-varying inputs (age, ttm, OTR, rating), so the residual eps_w can
     stay static and falsifiable.
  2. TokenVocab -- the open vocabulary. Three senses per instrument
     (BUY / SELL / UNDISC), explicit token_id map (stored, not computed, so a
     later sense split -- e.g. by size block -- can't break joins), birth at
     issuance, retirement at maturity.
  3. Tokenization -- sense codes are aligned with SideDisclosed by
     construction (BUY=0, SELL=1, TWO_WAY=UNDISC=2), so observable
     tokenization is `token(instrument, disclosed)` verbatim. Disclosure is
     part of the client's choice: on received rows the observable tokenizer
     reproduces the canonical latent token_id exactly. That is a property
     test, not a hope (see tests/test_vocab.py).
  4. Corpus view -- sentences (packages) and windowed co-occurrence pair
     counts with the same-client / cross-client split that the Sigma_u
     identification needs. Reference-grade numpy implementations for
     validation scale; the production counting path is CuPy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa

from .enums import Sense, SideDisclosed
from .tables import SchemaConfig

# The whole tokenization scheme leans on this alignment; fail loudly at import
# if anyone ever renumbers the enums.
assert int(Sense.BUY) == int(SideDisclosed.BUY) == 0
assert int(Sense.SELL) == int(SideDisclosed.SELL) == 1
assert int(Sense.UNDISC) == int(SideDisclosed.TWO_WAY) == 2

_DAYS_PER_YEAR = 365.25
_RETIRED_OPEN = np.iinfo(np.int64).max  # sentinel: token still live


class VocabError(RuntimeError):
    pass


# ===========================================================================
# 1. FeatureSpec: x_w(t).
# ===========================================================================

@dataclass(frozen=True)
class FeatureSpec:
    """Layout and constants of the attribute vector x_w(t).

    Order: [log1p_age, log1p_ttm, tenor RBF bumps (one per knot), coupon_s,
    log_size_s, rating_unit, is_hy, is_otr, in_index, is_144a, is_financial,
    seniority_s, sector one-hot].

    The RBF basis over log(1+ttm) is deliberate: a scalar ttm would force B
    to express curve geometry linearly, and the cross-issuer analogy probes
    (GM 5s : GM 10s :: F 5s : F 10s) need curve-LOCAL directions. Bumps at
    the benchmark knots give B a curve coordinate system while roll-down
    remains a smooth deterministic drift through it.

    All constants are part of the contract; bump the version if you touch
    them and record it in the run manifest dials.
    """
    sectors: tuple[str, ...]
    tenor_knots: tuple[float, ...] = (2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
    tenor_bandwidth: float = 0.35          # in log1p-ttm space
    coupon_scale: float = 10.0             # coupon_s = coupon / 10
    size_center_log10: float = 8.0         # log_size_s = (log10(amt) - 8) / 2
    size_scale_log10: float = 2.0
    rating_notches: int = 22
    hy_threshold: int = 11                 # rating code >= BB+ => high yield
    seniority_scale: float = 3.0
    version: str = "x-0.1.0"

    # -- layout ---------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return (
            "log1p_age", "log1p_ttm",
            *(f"tenor_rbf_{k:g}" for k in self.tenor_knots),
            "coupon_s", "log_size_s", "rating_unit", "is_hy", "is_otr",
            "in_index", "is_144a", "is_financial", "seniority_s",
            *(f"sector_{s}" for s in self.sectors),
        )

    @property
    def dim(self) -> int:
        # Derived from the layout itself so the two can never disagree; the
        # shape check in compute() then guards the implementation against
        # the declaration.
        return len(self.names)

    def assert_matches(self, cfg: SchemaConfig) -> None:
        if cfg.p != self.dim:
            raise VocabError(f"SchemaConfig.p={cfg.p} != FeatureSpec.dim={self.dim}")
        if cfg.n_sectors != len(self.sectors):
            raise VocabError(
                f"SchemaConfig.n_sectors={cfg.n_sectors} != len(spec.sectors)={len(self.sectors)}")

    # -- computation -----------------------------------------------------------

    def compute(self, asof_days: int, *,
                issue_days: np.ndarray, maturity_days: np.ndarray,
                coupon: np.ndarray, amt_issued: np.ndarray,
                rating: np.ndarray, seniority: np.ndarray,
                is_144a: np.ndarray, is_financial: np.ndarray,
                sector_id: np.ndarray, is_otr: np.ndarray,
                index_mask: np.ndarray) -> np.ndarray:
        """x_w(asof) for a panel of instruments; (n, dim) float32.

        Static inputs come from `instruments`; dynamic ones (rating, is_otr,
        index_mask) from the compiled daily state. This function IS what
        populates instrument_state_daily.x -- one source of truth for both
        planes.
        """
        n = len(issue_days)
        age = np.maximum((asof_days - np.asarray(issue_days)) / _DAYS_PER_YEAR, 0.0)
        ttm = np.maximum((np.asarray(maturity_days) - asof_days) / _DAYS_PER_YEAR, 0.0)

        z = np.log1p(ttm)
        centers = np.log1p(np.asarray(self.tenor_knots))
        bumps = np.exp(-0.5 * ((z[:, None] - centers[None, :]) / self.tenor_bandwidth) ** 2)

        sec = np.zeros((n, len(self.sectors)), dtype=np.float32)
        sid = np.asarray(sector_id)
        if sid.size and (sid.min() < 0 or sid.max() >= len(self.sectors)):
            raise VocabError("sector_id outside the spec's sector registry")
        sec[np.arange(n), sid] = 1.0

        rating = np.asarray(rating, dtype=np.float64)
        cols = [
            np.log1p(age),
            z,
            *bumps.T,
            np.asarray(coupon) / self.coupon_scale,
            (np.log10(np.maximum(np.asarray(amt_issued, dtype=np.float64), 1.0))
             - self.size_center_log10) / self.size_scale_log10,
            (rating - 1.0) / (self.rating_notches - 1.0),
            (rating >= self.hy_threshold).astype(np.float64),
            np.asarray(is_otr, dtype=np.float64),
            (np.asarray(index_mask) != 0).astype(np.float64),
            np.asarray(is_144a, dtype=np.float64),
            np.asarray(is_financial, dtype=np.float64),
            np.asarray(seniority, dtype=np.float64) / self.seniority_scale,
        ]
        x = np.column_stack([*cols, sec]).astype(np.float32)
        if x.shape != (n, self.dim):
            raise VocabError(f"feature layout drift: got {x.shape}, want {(n, self.dim)}")
        return x


# ===========================================================================
# 2. TokenVocab: the open vocabulary.
# ===========================================================================

class TokenVocab:
    """(instrument, sense) -> token_id, with lifecycle.

    Token ids are allocated sequentially at registration and STORED; nothing
    downstream may assume token_id == 3 * instrument + sense. Instruments are
    registered at issuance (primary_calendar.pricing_ts) and retired at
    maturity/call; the alive set at time t is the candidate universe before
    mandate masking.
    """

    SENSES = (Sense.BUY, Sense.SELL, Sense.UNDISC)

    def __init__(self) -> None:
        self._next = 0
        self._instr: list[int] = []             # registration order
        self._tok: list[list[int]] = []         # row -> [tid_buy, tid_sell, tid_undisc]
        self._born: list[int] = []              # per row, us
        self._retired: list[int] = []           # per row, us; _RETIRED_OPEN = live
        self._row_of: dict[int, int] = {}
        self._instr_of_tok: list[int] = []      # token_id -> instrument
        self._sense_of_tok: list[int] = []      # token_id -> sense
        self._index_dirty = True
        self._instr_sorted: np.ndarray | None = None
        self._perm: np.ndarray | None = None

    # -- lifecycle -------------------------------------------------------------

    def register_instrument(self, instrument_id: int, born_ts_us: int) -> tuple[int, int, int]:
        if instrument_id in self._row_of:
            raise VocabError(f"instrument {instrument_id} already registered")
        tids = (self._next, self._next + 1, self._next + 2)
        self._next += 3
        self._row_of[instrument_id] = len(self._instr)
        self._instr.append(int(instrument_id))
        self._tok.append(list(tids))
        self._born.append(int(born_ts_us))
        self._retired.append(_RETIRED_OPEN)
        for s in self.SENSES:
            self._instr_of_tok.append(int(instrument_id))
            self._sense_of_tok.append(int(s))
        self._index_dirty = True
        return tids

    def retire_instrument(self, instrument_id: int, retired_ts_us: int) -> None:
        row = self._row_of.get(instrument_id)
        if row is None:
            raise VocabError(f"instrument {instrument_id} not registered")
        self._retired[row] = int(retired_ts_us)

    # -- lookup ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._next

    @property
    def n_instruments(self) -> int:
        return len(self._instr)

    def token(self, instrument_id: int, sense: int | Sense) -> int:
        row = self._row_of.get(int(instrument_id))
        if row is None:
            raise VocabError(f"instrument {instrument_id} not registered")
        return self._tok[row][int(sense)]

    def instrument_sense(self, token_id: int) -> tuple[int, int]:
        return self._instr_of_tok[token_id], self._sense_of_tok[token_id]

    def _ensure_index(self) -> None:
        if self._index_dirty:
            instr = np.asarray(self._instr, dtype=np.int64)
            self._perm = np.argsort(instr, kind="stable")
            self._instr_sorted = instr[self._perm]
            self._index_dirty = False

    def tokens_for(self, instrument_ids: np.ndarray, senses: np.ndarray) -> np.ndarray:
        """Vectorized (instrument, sense) -> token_id; raises on unknowns."""
        self._ensure_index()
        ids = np.asarray(instrument_ids, dtype=np.int64)
        pos = np.searchsorted(self._instr_sorted, ids)
        bad = (pos >= len(self._instr_sorted)) | (self._instr_sorted[np.minimum(pos, len(self._instr_sorted) - 1)] != ids)
        if bad.any():
            raise VocabError(f"unregistered instruments: {np.unique(ids[bad])[:8].tolist()} ...")
        rows = self._perm[pos]
        tok = np.asarray(self._tok, dtype=np.int64)          # (n_instr, 3)
        return tok[rows, np.asarray(senses, dtype=np.int64)].astype(np.uint32)

    def alive_token_ids(self, ts_us: int) -> np.ndarray:
        """Candidate universe at t: born <= t < retired. Mandate masking is
        the emission layer's job, on top of this."""
        born = np.asarray(self._born, dtype=np.int64)
        retired = np.asarray(self._retired, dtype=np.int64)
        rows = np.nonzero((born <= ts_us) & (ts_us < retired))[0]
        tok = np.asarray(self._tok, dtype=np.int64)
        return np.sort(tok[rows].ravel()).astype(np.uint32)

    # -- arrow round-trip -----------------------------------------------------------

    def to_arrow(self, token_map_schema: pa.Schema) -> pa.Table:
        n = len(self)
        cols = {
            "token_id": np.arange(n, dtype=np.uint32),
            "instrument_id": np.asarray(self._instr_of_tok, dtype=np.uint32),
            "sense": np.asarray(self._sense_of_tok, dtype=np.int8),
            "born_ts": [self._born[self._row_of[i]] for i in self._instr_of_tok],
            "retired_ts": [None if self._retired[self._row_of[i]] == _RETIRED_OPEN
                           else self._retired[self._row_of[i]]
                           for i in self._instr_of_tok],
        }
        return pa.Table.from_pydict(cols, schema=token_map_schema)

    @classmethod
    def from_arrow(cls, table: pa.Table) -> "TokenVocab":
        v = cls()
        tid = np.asarray(table["token_id"])
        instr = np.asarray(table["instrument_id"])
        sense = np.asarray(table["sense"])
        born = table["born_ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
        retired = table["retired_ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
        order = np.argsort(tid, kind="stable")
        seen: dict[int, int] = {}
        for k in order:
            i = int(instr[k])
            if i not in seen:
                row_tids = v.register_instrument(i, int(born[k]))
                seen[i] = row_tids[0]
                r = retired[k]
                if not np.isnan(r):
                    v.retire_instrument(i, int(r))
            expect = seen[i] + int(sense[k])
            if int(tid[k]) != expect:
                raise VocabError(
                    "token_map is not in registration layout; extend from_arrow "
                    "before introducing custom id maps")
        return v


# ===========================================================================
# 3. Tokenization.
# ===========================================================================

def sense_from_disclosed(disclosed: np.ndarray) -> np.ndarray:
    """SideDisclosed codes ARE Sense codes (asserted at import): buy->BUY,
    sell->SELL, two_way->UNDISC. The identity is the design."""
    d = np.asarray(disclosed, dtype=np.int64)
    if d.size and (d.min() < 0 or d.max() > 2):
        raise VocabError("disclosed side outside {BUY, SELL, TWO_WAY}")
    return d


def tokenize_table(vocab: TokenVocab, table: pa.Table,
                   side_col: str = "client_side_disclosed") -> np.ndarray:
    """Observable tokenization: token(instrument, disclosed). On received rows
    this reproduces the canonical latent token_id exactly, because disclosure
    is part of the client's choice -- the parity test in test_vocab.py pins
    that property."""
    instr = table["instrument_id"].to_numpy(zero_copy_only=False)
    sense = sense_from_disclosed(table[side_col].to_numpy(zero_copy_only=False))
    return vocab.tokens_for(instr, sense)


# ===========================================================================
# 4. Corpus view: sentences and windowed pair counts.
# ===========================================================================

@dataclass(frozen=True)
class Sentence:
    package_id: int
    client_id: int
    trade_date: int          # days since epoch
    ts_first_us: int
    token_ids: np.ndarray    # in line_no order


def sentences(table: pa.Table, vocab: TokenVocab | None = None,
              token_col: str = "token_id", strict: bool = True) -> list[Sentence]:
    """Group a (canonical or observable) lines table into package-sentences,
    in tape order. If token_col is absent (observable plane), a vocab must be
    supplied and tokenization is applied. A package with more than one client
    violates the single-author invariant and raises when strict."""
    if token_col in table.column_names:
        tok = table[token_col].to_numpy(zero_copy_only=False).astype(np.int64)
    elif vocab is not None:
        tok = tokenize_table(vocab, table).astype(np.int64)
    else:
        raise VocabError(f"no {token_col} column and no vocab supplied")

    pkg = table["package_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    cli = table["client_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    day = table["trade_date"].cast(pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64)
    ts = table["ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    line = table["line_no"].to_numpy(zero_copy_only=False).astype(np.int64)

    order = np.lexsort((line, ts, pkg))  # group-friendly; tape order restored below
    out: dict[int, Sentence] = {}
    first_ts: dict[int, int] = {}
    for k in order:
        p = int(pkg[k])
        if p not in out:
            out[p] = Sentence(p, int(cli[k]), int(day[k]), int(ts[k]),
                              np.array([tok[k]], dtype=np.int64))
            first_ts[p] = int(ts[k])
        else:
            s = out[p]
            if strict and s.client_id != int(cli[k]):
                raise VocabError(f"package {p} has multiple authors")
            out[p] = Sentence(p, s.client_id, s.trade_date, s.ts_first_us,
                              np.append(s.token_ids, tok[k]))
    return sorted(out.values(), key=lambda s: (s.ts_first_us, s.package_id))


def window_pair_counts(table: pa.Table, vocab: TokenVocab | None = None,
                       token_col: str = "token_id",
                       window: int = 64) -> pa.Table:
    """Reference-grade co-occurrence counting: per trade_date, order events by
    (ts, event_id), cut the stream into disjoint blocks of `window` lines, and
    count unordered token pairs within each block, split by same-client vs
    cross-client. Same-client excess PMI is what identifies Sigma_u, so the
    split is carried in the key, not folded away. Output: (token_a <= token_b,
    same_client, n). Validation scale only -- production counting is the CuPy
    path."""
    if token_col in table.column_names:
        tok = table[token_col].to_numpy(zero_copy_only=False).astype(np.int64)
    elif vocab is not None:
        tok = tokenize_table(vocab, table).astype(np.int64)
    else:
        raise VocabError(f"no {token_col} column and no vocab supplied")

    day = table["trade_date"].cast(pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64)
    cli = table["client_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    ts = table["ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    eid = table["event_id"].to_numpy(zero_copy_only=False).astype(np.uint64)

    counts: Counter[tuple[int, int, bool]] = Counter()
    order = np.lexsort((eid, ts, day))
    tok, cli, day = tok[order], cli[order], day[order]
    start = 0
    n = len(tok)
    while start < n:
        d = day[start]
        stop = start
        while stop < n and day[stop] == d:
            stop += 1
        for b0 in range(start, stop, window):
            b1 = min(b0 + window, stop)
            for i in range(b0, b1):
                for j in range(i + 1, b1):
                    a, b = int(tok[i]), int(tok[j])
                    if a > b:
                        a, b = b, a
                    counts[(a, b, bool(cli[i] == cli[j]))] += 1
        start = stop

    keys = sorted(counts)
    return pa.table({
        "token_a": pa.array([k[0] for k in keys], pa.uint32()),
        "token_b": pa.array([k[1] for k in keys], pa.uint32()),
        "same_client": pa.array([k[2] for k in keys], pa.bool_()),
        "n": pa.array([counts[k] for k in keys], pa.uint64()),
    })
