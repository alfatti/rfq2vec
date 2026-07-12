"""Latent state layer: the regime-modulated walk that writes context_grid.

Architecture (as agreed): the walk is the primary state object; the sector
CTMCs are demoted to modulators. Radial decomposition c_t = r_t * theta_t:

  theta_t   spherical diffusion (thematic rotation); mixing parameterized
            DIRECTLY as a half-life in grid steps, eta = sqrt(2 ln2 / HL),
            because that is the dial the PMI-decay-vs-window-width estimator
            recovers.
  r_t       AR(1) in log space, mean-reverting to a regime-shifted target;
            drives the intensity load lambda ~ (r/r0)^gamma_lambda AND the
            inverse softmax temperature T ~ (r/r0)^(-gamma_T) (floored) --
            stress = busy + narrow is emergent, not stipulated.
  regimes   one 4-state bidimensional CTMC per sector, state = (activity
            bit, stress bit). The activity bit drives the per-sector
            intensity multiplier m[state]; the stress bits aggregate into a
            global s_t = sum_s w_s * stress_s in [0,1] that shifts the radial
            target and speeds the angular rotation. This is the concrete
            per-sector -> global coupling; the maps are config surfaces the
            calibration layer owns.

Reproducibility: the latent layer uses CONTENT-ADDRESSED randomness -- global
stream keys (no shard in the key) with arithmetic counters: the walk's counter
is a pure function of the day ordinal, a regime sojourn's counter a pure
function of (sector, sojourn_no). No stream cursors exist, so a checkpoint is
just the walk/regime state, and any single day or sojourn is regenerable in
isolation. (Per-event draws in the emission layer keep the per-shard
PhiloxLedger.) Regime chains tick in SESSION time; holding times are drawn in
session-hours, and sojourn boundaries are mapped back to wall-clock for the
regime_path table.

The drift budget is physical here: step_l1 is |c_t - c_{t-1}|_1 within a day,
null at the open (overnight mixing is applied as unrecorded steps), so the
intraday budget audit -- realized_drift() -- is a column scan.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator, Mapping

import numpy as np
import pyarrow as pa

from .tables import SchemaBundle
from .writer import PhiloxLedger, ShardWriter

_US_PER_HOUR = 3_600_000_000
_DAY_CTR_STRIDE = 1 << 20          # counter block per trade day (walk stream)
_SOJOURN_CTR_STRIDE = 16           # counter block per sojourn (regime stream)
_SECTOR_CTR_SPAN = 1 << 32         # sojourn address space per sector
_INIT_CTR = 1 << 40                # dedicated block for the theta init draw


class StateError(RuntimeError):
    pass


def derive_stream_key(seed_root_hex: str, stream: str) -> tuple[int, int]:
    """Global (shard-free) Philox key for a latent stream. Deliberately a
    different string format from PhiloxLedger's per-shard recipe, so the two
    key families cannot collide."""
    digest = hashlib.sha256(f"{seed_root_hex}//{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:16], "little")


# ===========================================================================
# Session grid calendar.
# ===========================================================================

@dataclass(frozen=True)
class GridCalendar:
    """In-session 1-min grid over the run's trading calendar. grid_idx is a
    global running index; both it and the session clock are pure functions of
    the calendar, so nothing here is state."""
    days: np.ndarray        # int32 days-since-epoch, ascending
    opens_us: np.ndarray    # int64 wall-clock session opens
    closes_us: np.ndarray   # int64 wall-clock session closes
    step_us: int = 60_000_000

    def __post_init__(self):
        if not (len(self.days) == len(self.opens_us) == len(self.closes_us)):
            raise StateError("calendar arrays must align")
        if np.any(np.asarray(self.closes_us) <= np.asarray(self.opens_us)):
            raise StateError("session close must exceed open")

    @classmethod
    def from_arrow(cls, calendar: pa.Table, step_us: int = 60_000_000) -> "GridCalendar":
        return cls(
            days=calendar["trade_date"].cast(pa.int32()).to_numpy(zero_copy_only=False),
            opens_us=calendar["session_open"].cast(pa.int64()).to_numpy(zero_copy_only=False),
            closes_us=calendar["session_close"].cast(pa.int64()).to_numpy(zero_copy_only=False),
            step_us=step_us,
        )

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def steps_per_day(self) -> np.ndarray:
        return ((np.asarray(self.closes_us) - np.asarray(self.opens_us))
                // self.step_us).astype(np.int64)

    @property
    def cum_steps(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.steps_per_day)])

    @property
    def cum_session_us(self) -> np.ndarray:
        dur = np.asarray(self.closes_us) - np.asarray(self.opens_us)
        return np.concatenate([[0], np.cumsum(dur)])

    def grid_idx(self, day_ordinal: int, k: int) -> int:
        return int(self.cum_steps[day_ordinal]) + k

    def ts_of(self, day_ordinal: int, k: int) -> int:
        return int(self.opens_us[day_ordinal]) + k * self.step_us

    def wall_of_session(self, session_us: int) -> int:
        d = int(np.searchsorted(self.cum_session_us, session_us, side="right")) - 1
        d = min(max(d, 0), self.n_days - 1)
        return int(self.opens_us[d]) + int(session_us - self.cum_session_us[d])

    def session_of_wall(self, ts_us: int) -> int:
        d = int(np.searchsorted(self.opens_us, ts_us, side="right")) - 1
        return int(self.cum_session_us[d]) + int(ts_us - self.opens_us[d])


# ===========================================================================
# Regime layer: per-sector 4-state bidimensional CTMC.
# ===========================================================================

@dataclass(frozen=True)
class RegimeConfig:
    """States ordered (activity, stress) as bits: 0=(lo,calm) 1=(hi,calm)
    2=(lo,stress) 3=(hi,stress). Q in per-session-hour units, rows sum to 0.
    m: per-state intensity multiplier (the MMPP channel). pi0: initial law."""
    Q: np.ndarray
    m: tuple[float, float, float, float] = (1.0, 1.6, 1.2, 2.2)
    pi0: tuple[float, float, float, float] = (0.45, 0.30, 0.15, 0.10)

    def __post_init__(self):
        Q = np.asarray(self.Q, dtype=np.float64)
        if Q.shape != (4, 4):
            raise StateError("Q must be 4x4")
        off = Q.copy(); np.fill_diagonal(off, 0.0)
        if (off < 0).any() or not np.allclose(Q.sum(axis=1), 0.0, atol=1e-12):
            raise StateError("Q must have nonnegative off-diagonals and zero row sums")
        if not np.isclose(sum(self.pi0), 1.0):
            raise StateError("pi0 must sum to 1")

    @staticmethod
    def state_bits(state: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(state)
        return (s & 1).astype(np.float64), (s >> 1).astype(np.float64)

    @classmethod
    def bidimensional(cls, act_up: float = 0.35, act_down: float = 0.5,
                      stress_on: float = 0.04, stress_off: float = 0.12,
                      stress_act_coupling: float = 1.5,
                      m: tuple[float, float, float, float] = (1.0, 1.6, 1.2, 2.2),
                      pi0: tuple[float, float, float, float] = (0.45, 0.30, 0.15, 0.10),
                      ) -> "RegimeConfig":
        """Single-coordinate flips only; under stress the activity up-rate is
        multiplied by stress_act_coupling (busy markets under stress)."""
        Q = np.zeros((4, 4))
        for s in range(4):
            act, stress = s & 1, s >> 1
            up = act_up * (stress_act_coupling if stress else 1.0)
            Q[s, s ^ 1] = act_down if act else up          # flip activity
            Q[s, s ^ 2] = stress_off if stress else stress_on  # flip stress
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return cls(Q=Q, m=m, pi0=pi0)


# ===========================================================================
# Latent state config.
# ===========================================================================

@dataclass(frozen=True)
class LatentStateConfig:
    d: int
    n_sectors: int
    lambda_bar_sector: tuple[float, ...]        # per-minute base arrival rate
    regime: RegimeConfig = field(default_factory=RegimeConfig.bidimensional)
    sector_weights: tuple[float, ...] | None = None   # stress aggregate; default uniform
    grid_step_us: int = 60_000_000
    overnight_steps: int = 120                  # unrecorded mixing steps at each open
    theta_half_life_steps: float = 390.0        # angular mixing HL, calm baseline
    theta_stress_speedup: float = 1.0           # HL_eff = HL / (1 + a * s_t)
    r0: float = 1.0
    r_half_life_steps: float = 390.0
    r_sigma: float = 0.02                       # per-step innovation of log r
    r_stress_uplift: float = 0.5                # log-target = log(r0 * (1 + a * s_t))
    temp0: float = 1.0
    temp_gamma: float = 1.0                     # T = temp0 * (r/r0)^-gamma_T
    temp_min: float = 0.10                      # the temperature floor bound
    lambda_gamma: float = 1.0                   # load = (r/r0)^gamma_lambda

    def __post_init__(self):
        if len(self.lambda_bar_sector) != self.n_sectors:
            raise StateError("lambda_bar_sector must have n_sectors entries")
        if self.sector_weights is not None and len(self.sector_weights) != self.n_sectors:
            raise StateError("sector_weights must have n_sectors entries")
        if min(self.theta_half_life_steps, self.r_half_life_steps) <= 0:
            raise StateError("half-lives must be positive")
        if self.d <= 0 or self.n_sectors <= 0:
            raise StateError("d and n_sectors must be positive")

    @property
    def weights(self) -> np.ndarray:
        if self.sector_weights is None:
            return np.full(self.n_sectors, 1.0 / self.n_sectors)
        w = np.asarray(self.sector_weights, dtype=np.float64)
        return w / w.sum()


# ===========================================================================
# The engine.
# ===========================================================================

class LatentStateEngine:
    """Generates context_grid and regime_path day by day.

    State between recorded steps is exactly: (theta, log_r) plus, per sector,
    (state, sojourn_no, entry/next-switch on the session clock). state_dict()
    serializes precisely that -- feed it to ShardWriter.checkpoint_walk_state
    at block boundaries; restore() resumes bitwise-identically (pinned by
    test_checkpoint_replay_is_bitwise)."""

    def __init__(self, cfg: LatentStateConfig, bundle: SchemaBundle,
                 seed_root_hex: str, calendar: GridCalendar,
                 state: Mapping[str, np.ndarray] | None = None):
        if cfg.d != bundle.config.d or cfg.n_sectors != bundle.config.n_sectors:
            raise StateError("LatentStateConfig disagrees with SchemaBundle dims")
        self.cfg = cfg
        self.bundle = bundle
        self.cal = calendar
        self._walk_key = derive_stream_key(seed_root_hex, "context_grid")
        self._regime_key = derive_stream_key(seed_root_hex, "regime_path")

        if state is None:
            g = PhiloxLedger.generator(*self._walk_key, _INIT_CTR)
            theta = g.standard_normal(cfg.d)
            self._theta = theta / np.linalg.norm(theta)
            self._log_r = np.log(cfg.r0)
            self._day_next = 0
            n = cfg.n_sectors
            self._rstate = np.zeros(n, np.int8)
            self._sojourn_no = np.zeros(n, np.int64)
            self._entry_s = np.zeros(n, np.int64)      # session clock, us
            self._next_s = np.zeros(n, np.int64)
            for s in range(n):
                st, hold, _ = self._sojourn_draws(s, 0, initial=True)
                self._rstate[s] = st
                self._next_s[s] = hold
        else:
            self._theta = np.asarray(state["theta"], np.float64).copy()
            self._log_r = float(np.asarray(state["log_r"]).ravel()[0])
            self._day_next = int(np.asarray(state["day_next"]).ravel()[0])
            self._rstate = np.asarray(state["rstate"], np.int8).copy()
            self._sojourn_no = np.asarray(state["sojourn_no"], np.int64).copy()
            self._entry_s = np.asarray(state["entry_s"], np.int64).copy()
            self._next_s = np.asarray(state["next_s"], np.int64).copy()

    # -- content-addressed randomness ----------------------------------------

    def _day_gen(self, day_ordinal: int) -> np.random.Generator:
        return PhiloxLedger.generator(*self._walk_key, day_ordinal * _DAY_CTR_STRIDE)

    def _sojourn_draws(self, sector: int, sojourn_no: int,
                       initial: bool = False) -> tuple[int, int, int]:
        """(state_entered, holding_us, next_state). The state entered is pi0-
        drawn for sojourn 0 and otherwise supplied by the previous sojourn's
        next-state draw -- we still return it here so a single sojourn is a
        pure function of its address."""
        ctr = (sector * _SECTOR_CTR_SPAN + sojourn_no) * _SOJOURN_CTR_STRIDE
        g = PhiloxLedger.generator(*self._regime_key, ctr)
        if initial:
            st = int(np.searchsorted(np.cumsum(self.cfg.regime.pi0), g.random()))
        else:
            st = -1  # caller already knows it
        state_for_rates = st if initial else int(self._rstate[sector])
        rate = -self.cfg.regime.Q[state_for_rates, state_for_rates]
        holding_us = int(-np.log(g.random()) / max(rate, 1e-12) * _US_PER_HOUR)
        p = self.cfg.regime.Q[state_for_rates].copy()
        p[state_for_rates] = 0.0
        p = p / p.sum()
        nxt = int(np.searchsorted(np.cumsum(p), g.random()))
        return st, holding_us, nxt

    # -- regime advance --------------------------------------------------------

    def _advance_regimes(self, session_us: int) -> list[dict]:
        """Advance every sector chain to the session clock; return closed
        sojourns as regime_path rows (wall-clock timestamps)."""
        closed: list[dict] = []
        for s in range(self.cfg.n_sectors):
            while self._next_s[s] <= session_us:
                # close current sojourn
                _, _, nxt = self._sojourn_draws(s, int(self._sojourn_no[s]),
                                                initial=self._sojourn_no[s] == 0)
                closed.append(dict(
                    sector_id=s, sojourn_no=int(self._sojourn_no[s]),
                    state=int(self._rstate[s]),
                    t_start=self.cal.wall_of_session(int(self._entry_s[s])),
                    t_end=self.cal.wall_of_session(int(self._next_s[s]))))
                # open the next one
                self._rstate[s] = nxt
                self._sojourn_no[s] += 1
                self._entry_s[s] = self._next_s[s]
                _, hold, _ = self._sojourn_draws(s, int(self._sojourn_no[s]))
                self._next_s[s] = self._entry_s[s] + hold
        return closed

    # -- walk step ----------------------------------------------------------------

    def _stress(self) -> float:
        _, stress = RegimeConfig.state_bits(self._rstate)
        return float(np.dot(self.cfg.weights, stress))

    def _step_walk(self, g: np.random.Generator, s_t: float) -> None:
        cfg = self.cfg
        hl = cfg.theta_half_life_steps / (1.0 + cfg.theta_stress_speedup * s_t)
        eta = np.sqrt(2.0 * np.log(2.0) / hl)
        xi = g.standard_normal(cfg.d)
        xi -= np.dot(xi, self._theta) * self._theta
        theta = self._theta + eta * xi / np.sqrt(cfg.d)
        self._theta = theta / np.linalg.norm(theta)

        mu = np.log(cfg.r0 * (1.0 + cfg.r_stress_uplift * s_t))
        phi = 2.0 ** (-1.0 / cfg.r_half_life_steps)
        self._log_r = mu + phi * (self._log_r - mu) + cfg.r_sigma * g.standard_normal()

    # -- one trading day ------------------------------------------------------------

    def simulate_day(self, day_ordinal: int) -> tuple[pa.Table, list[dict]]:
        """Returns (context_grid table for the day, closed regime_path rows).
        Days must be simulated in order; the walk state advances."""
        if day_ordinal != self._day_next:
            raise StateError(f"days must be simulated in order: expected {self._day_next}")
        cfg, cal = self.cfg, self.cal
        n_steps = int(cal.steps_per_day[day_ordinal])
        g = self._day_gen(day_ordinal)
        open_session = int(cal.cum_session_us[day_ordinal])

        closed = self._advance_regimes(open_session)
        if day_ordinal > 0:
            s_open = self._stress()
            for _ in range(cfg.overnight_steps):   # unrecorded mixing
                self._step_walk(g, s_open)

        lam_bar = np.asarray(cfg.lambda_bar_sector, np.float64)
        m = np.asarray(cfg.regime.m, np.float64)

        grid_idx = np.empty(n_steps, np.int64)
        ts = np.empty(n_steps, np.int64)
        r_out = np.empty(n_steps, np.float32)
        temp_out = np.empty(n_steps, np.float32)
        c_out = np.empty((n_steps, cfg.d), np.float32)
        lam_out = np.empty((n_steps, cfg.n_sectors), np.float32)
        reg_out = np.empty((n_steps, cfg.n_sectors), np.int8)
        step_l1: list[float | None] = []

        c_prev: np.ndarray | None = None
        for k in range(n_steps):
            session_us = open_session + k * cfg.grid_step_us
            closed += self._advance_regimes(session_us)
            s_t = self._stress()
            self._step_walk(g, s_t)

            r = float(np.exp(self._log_r))
            load = (r / cfg.r0) ** cfg.lambda_gamma
            c = (r * self._theta).astype(np.float32)

            grid_idx[k] = cal.grid_idx(day_ordinal, k)
            ts[k] = cal.ts_of(day_ordinal, k)
            r_out[k] = r
            temp_out[k] = max(cfg.temp_min, cfg.temp0 * (r / cfg.r0) ** (-cfg.temp_gamma))
            c_out[k] = c
            reg_out[k] = self._rstate
            lam_out[k] = lam_bar * m[self._rstate] * load
            step_l1.append(None if c_prev is None
                           else float(np.abs(c - c_prev).sum()))
            c_prev = c

        self._day_next += 1
        sch = self.bundle.tables["context_grid"]
        tbl = pa.Table.from_pydict({
            "grid_idx": grid_idx, "ts": ts,
            "trade_date": np.full(n_steps, cal.days[day_ordinal], np.int32),
            "regime": _fsl(reg_out, pa.int8()),
            "r": r_out, "temperature": temp_out,
            "c": _fsl(c_out, pa.float32()),
            "lambda_sector": _fsl(lam_out, pa.float32()),
            "step_l1": pa.array(step_l1, pa.float32()),
        }, schema=sch)
        return tbl, closed

    def open_sojourn_rows(self) -> list[dict]:
        """Rows for sojourns still open (t_end null); emit at end of run."""
        return [dict(sector_id=s, sojourn_no=int(self._sojourn_no[s]),
                     state=int(self._rstate[s]),
                     t_start=self.cal.wall_of_session(int(self._entry_s[s])),
                     t_end=None)
                for s in range(self.cfg.n_sectors)]

    # -- checkpoint -------------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        return dict(theta=self._theta.copy(),
                    log_r=np.array([self._log_r]),
                    day_next=np.array([self._day_next], np.int64),
                    rstate=self._rstate.copy(),
                    sojourn_no=self._sojourn_no.copy(),
                    entry_s=self._entry_s.copy(),
                    next_s=self._next_s.copy())

    @classmethod
    def restore(cls, cfg: LatentStateConfig, bundle: SchemaBundle,
                seed_root_hex: str, calendar: GridCalendar,
                state: Mapping[str, np.ndarray]) -> "LatentStateEngine":
        return cls(cfg, bundle, seed_root_hex, calendar, state=state)


def _fsl(mat: np.ndarray, typ: pa.DataType) -> pa.FixedSizeListArray:
    m = np.ascontiguousarray(mat)
    return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel(), typ), m.shape[1])


# ===========================================================================
# Driver + drift audit.
# ===========================================================================

def run_and_write(engine: LatentStateEngine, writer: ShardWriter,
                  day_lo: int, day_hi: int,
                  emit_open_sojourns: bool = False,
                  checkpoint: bool = True) -> dict[str, np.ndarray]:
    """Simulate [day_lo, day_hi), appending context_grid and regime_path to
    the shard writer; optionally checkpoint the walk state at the boundary.
    Returns the final state_dict (what the next shard restores from)."""
    rp_schema = engine.bundle.tables["regime_path"]
    for day in range(day_lo, day_hi):
        tbl, closed = engine.simulate_day(day)
        writer.append("context_grid", tbl)
        if closed:
            writer.append("regime_path", _regime_rows_table(closed, rp_schema))
    if emit_open_sojourns:
        rows = engine.open_sojourn_rows()
        if rows:
            writer.append("regime_path", _regime_rows_table(rows, rp_schema))
    state = engine.state_dict()
    if checkpoint:
        writer.checkpoint_walk_state(engine.cal.grid_idx(day_hi - 1,
                                     int(engine.cal.steps_per_day[day_hi - 1]) - 1), state)
    return state


def _regime_rows_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pydict(
        {name: [r[name] for r in rows] for name in schema.names}, schema=schema)


def realized_drift(context_grid: pa.Table, window_steps: int) -> dict[str, float]:
    """The drift-budget audit as a column scan: rolling window sums of step_l1
    within each day; returns their mean / p99 / max. Note this is TOTAL
    VARIATION -- an upper bound on the displacement that actually biases PMI;
    compare against predicted_displacement for the closed-form counterpart."""
    l1 = context_grid["step_l1"].to_numpy(zero_copy_only=False)
    day = context_grid["trade_date"].cast(pa.int32()).to_numpy(zero_copy_only=False)
    sums: list[np.ndarray] = []
    for d in np.unique(day):
        x = l1[day == d]
        x = x[~np.isnan(x.astype(np.float64))]
        if len(x) >= window_steps:
            c = np.concatenate([[0.0], np.cumsum(x)])
            sums.append(c[window_steps:] - c[:-window_steps])
    if not sums:
        raise StateError("no full window fits inside any day")
    w = np.concatenate(sums)
    return dict(mean=float(w.mean()), p99=float(np.quantile(w, 0.99)),
                max=float(w.max()), n_windows=int(len(w)))


def predicted_displacement(cfg: LatentStateConfig, window_steps: int,
                           stress: float = 0.0) -> dict[str, float]:
    """Closed-form E-displacement of the context over a window of grid steps,
    angular component only (r frozen at r0; radial innovation adds variance
    but no direction bias). With per-step retention rho = 2^(-1/HL_eff):

        E||c_{t+W} - c_t||_2 ~= r0 * sqrt(2 (1 - rho^W))
        E||.||_1            ~= ||.||_2 * sqrt(2 d / pi)

    The l1 number feeds the worst-case (union-over-vocab) bias exponent from
    the paper's A(c,c') machinery; the l2 number is what governs the TYPICAL
    pairwise PMI bias via <v_w + v_w', dc> ~ ||v||_2 * ||dc||_2 / sqrt(d).
    Both are deliberately returned: the window-width rule should be argued
    against the l2 line, audited against the l1 line."""
    hl = cfg.theta_half_life_steps / (1.0 + cfg.theta_stress_speedup * stress)
    rho = 2.0 ** (-1.0 / hl)
    l2 = cfg.r0 * float(np.sqrt(2.0 * (1.0 - rho ** window_steps)))
    return dict(l2=l2, l1=l2 * float(np.sqrt(2.0 * cfg.d / np.pi)))


def max_window_steps(cfg: LatentStateConfig, budget: float,
                     norm: str = "l2", stress: float = 0.0) -> int | None:
    """Largest window (grid steps) whose predicted displacement stays within
    budget; None if even the stationary limit does (unbounded windows)."""
    scale = 1.0 if norm == "l2" else float(np.sqrt(2.0 * cfg.d / np.pi))
    x = (budget / (cfg.r0 * scale)) ** 2 / 2.0
    if x >= 1.0:
        return None
    hl = cfg.theta_half_life_steps / (1.0 + cfg.theta_stress_speedup * stress)
    rho = 2.0 ** (-1.0 / hl)
    return int(np.floor(np.log1p(-x) / np.log(rho)))
