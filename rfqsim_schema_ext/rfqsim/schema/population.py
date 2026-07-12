"""Universe builder: the static (and event-sourced) population a run emits from.

Everything here is drawn from ONE content-addressed Philox stream
("population"), so the universe is a pure function of the seed root and the
dials -- regenerate it anywhere, byte-identical, without touching the tape.

The embedding construction implements the agreed decomposition:

    v_{(w, side)}(t) = s_{(w, side)}(t) * normalize( B x_w(t) + eps_{(w, side)} )

  * norms carry frequency: X = ||v||^2 / 2d ~ Exp(rate beta) gives the
    Pareto(beta) frequency tail (Gini = 1/(2 beta - 1); 0.84 -> beta ~ 1.1),
    with a deterministic schedule s(t) = s_birth * exp(-age/tau) * (1 +
    otr_boost * is_otr) carrying age decay and benchmark boost;
  * per-side log-odds shifts on X give the buy/sell asymmetry and the
    two-way prevalence (the UNDISC sense competes as its own token);
  * directions carry meaning: morphology B x_w(t) plus a residual split
    eps_(w,side) = sqrt(1-phi^2) eps_w^shared + phi eps_w,side -- the phi dial
    trades substitution geometry shared across sides against genuinely signed
    idiosyncrasy. The signed-geometry ensemble is a calibration surface; its
    gates are the analogy and switch probes.

Client tilts: u_k = sigma_u ( sqrt(rho/r_u) V z_k + sqrt((1-rho)/d) g_k ),
V a random d x r_u frame => Sigma_u = sigma_u^2 ( rho/r_u V V' + (1-rho)/d I ).
rho and r_u are the collaborative-transfer dials; Sigma_u ships as an artifact
so the same-client-excess-PMI rung has its ground truth on disk.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from . import enums as E
from .state import GridCalendar, derive_stream_key
from .tables import SchemaBundle
from .vocab import FeatureSpec, TokenVocab
from .writer import PhiloxLedger

_US = 1_000_000
_DAY_US = 86_400 * _US


class PopulationError(RuntimeError):
    pass


# Archetype parameter table: (rating_lo, rating_hi, tenor_lo, tenor_hi,
# max_line, base_activity, size_mu_log, panel_prob)
_ARCH = {
    E.Archetype.REAL_MONEY:  (11, 16, 1.0, 30.0, 25_000_000, 1.0, 14.2, 0.75),
    E.Archetype.HEDGE_FUND:  (11, 22, 0.5, 30.0, 50_000_000, 1.6, 14.6, 0.60),
    E.Archetype.INSURER:     (11, 14, 3.0, 30.0, 20_000_000, 0.5, 14.4, 0.80),
    E.Archetype.PENSION:     (11, 14, 5.0, 30.0, 15_000_000, 0.4, 14.3, 0.70),
    E.Archetype.BANK:        (11, 15, 0.5, 10.0, 30_000_000, 0.7, 14.0, 0.65),
    E.Archetype.PT_DESK:     (11, 22, 0.5, 30.0, 80_000_000, 0.9, 13.8, 0.85),
}
_ARCH_WEIGHTS = (0.34, 0.22, 0.14, 0.10, 0.12, 0.08)


@dataclass(frozen=True)
class UniverseConfig:
    n_issuers: int = 150
    n_clients: int = 80
    mean_instr_per_issuer: float = 3.0
    beta_norm: float = 1.1              # Exp rate of ||v||^2/2d: the Gini dial
    side_logodds_disp: float = 0.35     # buy/sell asymmetry of the norm law
    undisc_logodds: float = -1.0        # two-way prevalence dial
    norm_tau_years: float = 4.0         # age decay of s(t)
    otr_boost: float = 0.6
    sigma_eps: float = 1.0              # residual scale vs ||B x|| (learnability dial)
    phi_side: float = 0.45              # side-specific share of the residual
    gain_sector: float = 1.2            # B row gains by feature family
    gain_curve: float = 1.0
    gain_quality: float = 0.8
    gain_misc: float = 0.4
    sigma_u: float = 0.7                # client tilt strength
    rho_u: float = 0.6                  # low-rank share of Sigma_u
    rank_u: int = 4
    activity_hl_days: float = 15.0      # OU half-life of client activity
    activity_sigma: float = 0.5
    otr_age_years: float = 1.0
    index_min_amt: int = 300_000_000


@dataclass
class Universe:
    cfg: UniverseConfig
    spec: FeatureSpec
    vocab: TokenVocab
    cal: GridCalendar
    # issuers
    issuer_sector: np.ndarray
    issuer_financial: np.ndarray
    # instruments (index = instrument_id)
    instr_issuer: np.ndarray
    instr_sector: np.ndarray
    coupon: np.ndarray
    issue_days: np.ndarray
    maturity_days: np.ndarray
    amt_issued: np.ndarray
    rating: np.ndarray
    seniority: np.ndarray
    is_144a: np.ndarray
    benchmark_tenor: np.ndarray
    # clients (index = client_id)
    archetype: np.ndarray
    region: np.ndarray
    mandate_rating: np.ndarray        # (K, 2)
    mandate_tenor: np.ndarray         # (K, 2)
    mandate_sector_mask: np.ndarray   # (K,) uint32
    max_line: np.ndarray
    base_activity: np.ndarray
    size_mu_log: np.ndarray
    our_panel: np.ndarray             # bool: dealer 0 on the client's panel
    panel_tier: np.ndarray
    # oracle geometry
    B: np.ndarray                     # (p, d)
    u: np.ndarray                     # (K, d)
    Sigma_u: np.ndarray               # (d, d)
    eps: np.ndarray                   # (n_tokens, d)
    norm_params: np.ndarray           # (n_tokens, n_norm_params)
    # latent activity (per calendar day)
    activity: np.ndarray              # (n_days, K) OU state
    intensity_mult: np.ndarray        # (n_days, K)

    # -- derived -----------------------------------------------------------

    @property
    def n_instruments(self) -> int:
        return len(self.instr_issuer)

    @property
    def n_clients(self) -> int:
        return len(self.archetype)

    def token_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(tok_instrument, tok_sense) aligned with token_id."""
        n = len(self.vocab)
        instr = np.empty(n, np.int64)
        sense = np.empty(n, np.int64)
        for t in range(n):
            instr[t], sense[t] = self.vocab.instrument_sense(t)
        return instr, sense

    def s_of(self, token_ids: np.ndarray, asof_days: int) -> np.ndarray:
        """The deterministic norm schedule s_w(t)."""
        p = self.norm_params[token_ids]
        instr, _ = self._tok_cache()
        age = np.maximum((asof_days - self.issue_days[instr[token_ids]]) / 365.25, 0.0)
        is_otr = age < self.cfg.otr_age_years
        return p[:, 0] * np.exp(-age / p[:, 1]) * (1.0 + p[:, 2] * is_otr)

    def _tok_cache(self):
        if not hasattr(self, "_tok"):
            self._tok = self.token_arrays()
        return self._tok


# ===========================================================================
# Calendar.
# ===========================================================================

def business_calendar(start: dt.date, n_days: int,
                      open_utc: tuple[int, int] = (13, 30),
                      session_minutes: int = 390) -> GridCalendar:
    days, opens = [], []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append((d - dt.date(1970, 1, 1)).days)
            o = dt.datetime(d.year, d.month, d.day, *open_utc, tzinfo=dt.timezone.utc)
            opens.append(int(o.timestamp() * _US))
        d += dt.timedelta(days=1)
    days = np.asarray(days, np.int32)
    opens = np.asarray(opens, np.int64)
    return GridCalendar(days=days, opens_us=opens,
                        closes_us=opens + session_minutes * 60 * _US)


# ===========================================================================
# Builder.
# ===========================================================================

def build_universe(cfg: UniverseConfig, spec: FeatureSpec, cal: GridCalendar,
                   seed_root_hex: str, d: int) -> Universe:
    g = PhiloxLedger.generator(*derive_stream_key(seed_root_hex, "population"), 0)
    ns = len(spec.sectors)
    start_day = int(cal.days[0])
    end_day = int(cal.days[-1])

    # -- issuers ------------------------------------------------------------
    issuer_sector = g.integers(0, ns, cfg.n_issuers)
    fin_idx = spec.sectors.index("Financials") if "Financials" in spec.sectors else -1
    issuer_financial = issuer_sector == fin_idx

    # issuer size ~ Pareto: the issuer-Zipf channel of the old composed model,
    # now feeding only the count of instruments (frequency itself is the norm law)
    issuer_size = (1.0 + g.pareto(1.3, cfg.n_issuers))
    n_per = np.clip(np.round(issuer_size * cfg.mean_instr_per_issuer
                             / issuer_size.mean()).astype(int), 1, 40)

    # -- instruments ----------------------------------------------------------
    instr_issuer = np.repeat(np.arange(cfg.n_issuers), n_per)
    n_instr = len(instr_issuer)
    instr_sector = issuer_sector[instr_issuer]
    coupon = g.uniform(4.0, 9.5, n_instr).astype(np.float32)
    tenor = g.choice([3.0, 5.0, 7.0, 10.0], n_instr, p=[0.2, 0.35, 0.25, 0.2])
    tenor = tenor + g.uniform(-0.25, 0.25, n_instr)
    # issue dates: mostly seasoned, ~15% priced during the run (token births)
    seasoned = g.random(n_instr) > 0.15
    issue_days = np.where(
        seasoned,
        start_day - g.integers(30, 2200, n_instr),
        start_day + g.integers(1, max(2, cal.n_days - 2), n_instr) *
        ((end_day - start_day) // max(1, cal.n_days - 1)),
    ).astype(np.int64)
    maturity_days = issue_days + (tenor * 365.25).astype(np.int64)
    alive = maturity_days > start_day + 20
    resample = ~alive
    maturity_days[resample] = start_day + g.integers(400, 3000, resample.sum())
    amt = np.round(np.exp(g.normal(np.log(4e8), 0.7, n_instr)) / 2.5e7) * 2.5e7
    amt_issued = np.clip(amt, 1e8, 4e9).astype(np.int64)
    rating = g.choice(np.arange(11, 18), n_instr,
                      p=[0.16, 0.20, 0.22, 0.18, 0.12, 0.08, 0.04]).astype(np.int8)
    seniority = g.choice([0, 1, 2], n_instr, p=[0.15, 0.75, 0.10]).astype(np.int8)
    is_144a = g.random(n_instr) < 0.5

    # -- token vocabulary -------------------------------------------------------
    vocab = TokenVocab()
    for i in range(n_instr):
        born = int(issue_days[i]) * _DAY_US + 14 * 3600 * _US
        vocab.register_instrument(i, born)
        vocab.retire_instrument(i, int(maturity_days[i]) * _DAY_US)

    # -- clients -------------------------------------------------------------------
    K = cfg.n_clients
    archetype = g.choice(np.arange(6), K, p=_ARCH_WEIGHTS).astype(np.int8)
    region = g.choice([0, 1, 2], K, p=[0.55, 0.30, 0.15]).astype(np.int8)
    arch_rows = np.array([_ARCH[E.Archetype(a)] for a in archetype])
    mandate_rating = arch_rows[:, 0:2].astype(np.int8)
    mandate_tenor = arch_rows[:, 2:4].astype(np.float32)
    max_line = arch_rows[:, 4].astype(np.int64)
    base_activity = arch_rows[:, 5] * np.exp(g.normal(0.0, 0.6, K))
    size_mu_log = arch_rows[:, 6].astype(np.float64)
    panel_prob = arch_rows[:, 7]
    # sector mask: each client covers a random >=60% subset of sectors
    mask = np.zeros(K, np.uint32)
    for k in range(K):
        n_cover = g.integers(max(1, int(0.6 * ns)), ns + 1)
        cover = g.choice(ns, n_cover, replace=False)
        mask[k] = np.bitwise_or.reduce((1 << cover).astype(np.uint32)) if len(cover) else 0
    our_panel = g.random(K) < panel_prob
    panel_tier = np.clip((base_activity / base_activity.max() * 3).astype(np.int8), 0, 2)

    # -- geometry --------------------------------------------------------------------
    names = spec.names
    gains = np.empty(spec.dim)
    for j, nm in enumerate(names):
        if nm.startswith("sector_"):
            gains[j] = cfg.gain_sector
        elif nm.startswith("tenor_rbf_") or nm in ("log1p_ttm", "log1p_age"):
            gains[j] = cfg.gain_curve
        elif nm in ("rating_unit", "is_hy", "coupon_s"):
            gains[j] = cfg.gain_quality
        else:
            gains[j] = cfg.gain_misc
    Bdir = g.standard_normal((spec.dim, d))
    Bdir /= np.linalg.norm(Bdir, axis=1, keepdims=True)
    B = (gains[:, None] * Bdir).astype(np.float64)

    n_tok = len(vocab)
    eps_shared = g.standard_normal((n_instr, d)) * (cfg.sigma_eps / np.sqrt(d))
    eps_side = g.standard_normal((n_tok, d)) * (cfg.sigma_eps / np.sqrt(d))
    tok_instr = np.empty(n_tok, np.int64)
    tok_sense = np.empty(n_tok, np.int64)
    for t in range(n_tok):
        tok_instr[t], tok_sense[t] = vocab.instrument_sense(t)
    phi = cfg.phi_side
    eps = (np.sqrt(1 - phi ** 2) * eps_shared[tok_instr] + phi * eps_side)

    # norm law: X = ||v||^2/2d ~ Exp(rate beta), side log-odds shifts on X
    X_base = g.exponential(1.0 / cfg.beta_norm, n_instr)
    side_shift = np.where(tok_sense == int(E.Sense.BUY), 0.0,
                          np.where(tok_sense == int(E.Sense.SELL),
                                   g.normal(0.0, cfg.side_logodds_disp, n_tok),
                                   cfg.undisc_logodds))
    X = np.maximum(X_base[tok_instr] + side_shift, 0.02)
    s_birth = np.sqrt(2.0 * d * X)
    norm_params = np.zeros((n_tok, 4), np.float32)
    norm_params[:, 0] = s_birth
    norm_params[:, 1] = cfg.norm_tau_years
    norm_params[:, 2] = cfg.otr_boost

    # client tilts: Sigma_u = sigma_u^2 (rho/r V V' + (1-rho)/d I)
    V = np.linalg.qr(g.standard_normal((d, cfg.rank_u)))[0]
    z = g.standard_normal((K, cfg.rank_u))
    w = g.standard_normal((K, d))
    u = cfg.sigma_u * (np.sqrt(cfg.rho_u / cfg.rank_u) * z @ V.T
                       + np.sqrt((1 - cfg.rho_u) / d) * w)
    Sigma_u = cfg.sigma_u ** 2 * (cfg.rho_u / cfg.rank_u * V @ V.T
                                  + (1 - cfg.rho_u) / d * np.eye(d))

    # -- client activity: AR(1) on calendar days, content-addressed per day ---------
    phi_a = 2.0 ** (-1.0 / cfg.activity_hl_days)
    act = np.zeros((cal.n_days, K))
    ga = PhiloxLedger.generator(*derive_stream_key(seed_root_hex, "client_activity"), 0)
    act[0] = ga.standard_normal(K) * cfg.activity_sigma
    for t in range(1, cal.n_days):
        act[t] = phi_a * act[t - 1] + np.sqrt(1 - phi_a ** 2) * cfg.activity_sigma * ga.standard_normal(K)
    mult = np.exp(act - cfg.activity_sigma ** 2 / 2)

    return Universe(
        cfg=cfg, spec=spec, vocab=vocab, cal=cal,
        issuer_sector=issuer_sector, issuer_financial=issuer_financial,
        instr_issuer=instr_issuer, instr_sector=instr_sector, coupon=coupon,
        issue_days=issue_days, maturity_days=maturity_days,
        amt_issued=amt_issued, rating=rating, seniority=seniority,
        is_144a=is_144a, benchmark_tenor=tenor.astype(np.float32),
        archetype=archetype, region=region,
        mandate_rating=mandate_rating, mandate_tenor=mandate_tenor,
        mandate_sector_mask=mask, max_line=max_line,
        base_activity=base_activity, size_mu_log=size_mu_log,
        our_panel=our_panel, panel_tier=panel_tier,
        B=B, u=u, Sigma_u=Sigma_u, eps=eps, norm_params=norm_params,
        activity=act, intensity_mult=mult,
    )


# ===========================================================================
# Daily instrument state + embeddings.
# ===========================================================================

def instrument_state(uni: Universe, asof_days: int) -> dict[str, np.ndarray]:
    age = np.maximum((asof_days - uni.issue_days) / 365.25, 0.0)
    is_otr = age < uni.cfg.otr_age_years
    index_mask = (uni.amt_issued >= uni.cfg.index_min_amt).astype(np.uint32)
    x = uni.spec.compute(
        asof_days,
        issue_days=uni.issue_days, maturity_days=uni.maturity_days,
        coupon=uni.coupon, amt_issued=uni.amt_issued, rating=uni.rating,
        seniority=uni.seniority, is_144a=uni.is_144a,
        is_financial=uni.issuer_financial[uni.instr_issuer],
        sector_id=uni.instr_sector, is_otr=is_otr, index_mask=index_mask)
    return dict(x=x, is_otr=is_otr, index_mask=index_mask,
                rating=uni.rating, age=age)


def token_vectors(uni: Universe, asof_days: int,
                  x: np.ndarray | None = None) -> np.ndarray:
    """v_{(w,side)}(asof) for the whole vocabulary: (n_tokens, d) float64.
    s(t) * normalize(B x + eps) -- the one place the decomposition is wired."""
    if x is None:
        x = instrument_state(uni, asof_days)["x"]
    tok_instr, _ = uni.token_arrays()
    raw = x[tok_instr].astype(np.float64) @ uni.B + uni.eps
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    s = uni.s_of(np.arange(len(uni.vocab)), asof_days)
    return raw * s[:, None]


# ===========================================================================
# Writing the dimension + oracle tables.
# ===========================================================================

def write_universe(uni: Universe, writer, bundle: SchemaBundle) -> None:
    t = bundle.tables
    ni, k = uni.n_instruments, uni.n_clients
    day0 = int(uni.cal.days[0])

    writer.append("issuers", pa.Table.from_pydict({
        "issuer_id": np.arange(uni.cfg.n_issuers, dtype=np.uint32),
        "name": [f"ISSUER{i:04d}" for i in range(uni.cfg.n_issuers)],
        "sector_id": uni.issuer_sector.astype(np.uint16),
        "sub_sector_id": (uni.issuer_sector * 10).astype(np.uint16),
        "is_financial": uni.issuer_financial,
    }, schema=t["issuers"]))

    writer.append("instruments", pa.Table.from_pydict({
        "instrument_id": np.arange(ni, dtype=np.uint32),
        "cusip": [f"SIM{i:06d}" for i in range(ni)],
        "issuer_id": uni.instr_issuer.astype(np.uint32),
        "coupon": uni.coupon,
        "issue_date": uni.issue_days.astype(np.int32),
        "maturity_date": uni.maturity_days.astype(np.int32),
        "amt_issued": uni.amt_issued,
        "rating_at_issue": uni.rating,
        "seniority": uni.seniority,
        "is_144a": uni.is_144a,
        "benchmark_tenor_yrs": uni.benchmark_tenor,
    }, schema=t["instruments"]))

    writer.append("clients", pa.Table.from_pydict({
        "client_id": np.arange(k, dtype=np.uint32),
        "archetype": uni.archetype, "region": uni.region,
        "onboard_date": np.full(k, day0 - 400, np.int32),
        "offboard_date": [None] * k,
    }, schema=t["clients"]))

    writer.append("client_mandates", pa.Table.from_pydict({
        "mandate_id": np.arange(k, dtype=np.uint32),
        "client_id": np.arange(k, dtype=np.uint32),
        "rating_min": uni.mandate_rating[:, 0], "rating_max": uni.mandate_rating[:, 1],
        "tenor_min_yrs": uni.mandate_tenor[:, 0], "tenor_max_yrs": uni.mandate_tenor[:, 1],
        "sector_mask": uni.mandate_sector_mask,
        "max_line_par": uni.max_line,
        "effective_from": np.full(k, day0 - 400, np.int32),
        "effective_to": [None] * k,
    }, schema=t["client_mandates"]))

    writer.append("client_dealer_panel", pa.Table.from_pydict({
        "client_id": np.arange(k, dtype=np.uint32),
        "dealer_id": np.where(uni.our_panel, 0, -1).astype(np.int16),
        "tier": uni.panel_tier,
        "effective_from": np.full(k, day0 - 400, np.int32),
        "effective_to": [None] * k,
    }, schema=t["client_dealer_panel"]))

    writer.append("token_map", uni.vocab.to_arrow(t["token_map"]))

    writer.append("calendar", pa.Table.from_pydict({
        "trade_date": uni.cal.days, "session_open": uni.cal.opens_us,
        "session_close": uni.cal.closes_us,
        "is_half_day": np.zeros(uni.cal.n_days, bool),
    }, schema=t["calendar"]))

    in_run = uni.issue_days >= day0
    ids = np.nonzero(in_run)[0]
    writer.append("primary_calendar", pa.Table.from_pydict({
        "deal_id": np.arange(len(ids), dtype=np.uint32),
        "instrument_id": ids.astype(np.uint32),
        "pricing_ts": (uni.issue_days[ids] * _DAY_US + 14 * 3600 * _US),
        "deal_size": uni.amt_issued[ids],
    }, schema=t["primary_calendar"]))

    writer.append("oracle_clients", pa.Table.from_pydict({
        "client_id": np.arange(k, dtype=np.uint32),
        "u": _fsl32(uni.u), "u_norm": np.linalg.norm(uni.u, axis=1).astype(np.float32),
    }, schema=t["oracle_clients"]))

    writer.append("oracle_embeddings", pa.Table.from_pydict({
        "token_id": np.arange(len(uni.vocab), dtype=np.uint32),
        "eps": _fsl32(uni.eps), "norm_params": _fsl32(uni.norm_params),
    }, schema=t["oracle_embeddings"]))

    # Realized word vectors at run start -- the ground truth the geometry probes
    # align against (residual + norm alone can't reconstruct these without B and
    # the day-0 features, so I dump them here). asof = calendar day 0.
    v0 = token_vectors(uni, day0)
    writer.append("oracle_token_vectors", pa.Table.from_pydict({
        "token_id": np.arange(len(uni.vocab), dtype=np.uint32),
        "v": _fsl32(v0.astype(np.float32)),
    }, schema=t["oracle_token_vectors"]))

    # The attribute gain matrix B (p x d): lets a probe separate the shared
    # feature-driven geometry (B x) from the full-rank residual (eps).
    writer.append("oracle_attribute_gain", pa.Table.from_pydict({
        "feat_idx": np.arange(uni.B.shape[0], dtype=np.uint16),
        "gain": _fsl32(uni.B.astype(np.float32)),
    }, schema=t["oracle_attribute_gain"]))

    writer.append("client_activity_daily", pa.Table.from_pydict({
        "trade_date": np.repeat(uni.cal.days, k).astype(np.int32),
        "client_id": np.tile(np.arange(k, dtype=np.uint32), uni.cal.n_days),
        "activity": uni.activity.ravel().astype(np.float32),
        "intensity_mult": uni.intensity_mult.ravel().astype(np.float32),
    }, schema=t["client_activity_daily"]))


def write_instrument_state_day(uni: Universe, writer, bundle: SchemaBundle,
                               asof_days: int) -> dict[str, np.ndarray]:
    st = instrument_state(uni, asof_days)
    ni = uni.n_instruments
    writer.append("instrument_state_daily", pa.Table.from_pydict({
        "trade_date": np.full(ni, asof_days, np.int32),
        "instrument_id": np.arange(ni, dtype=np.uint32),
        "rating": st["rating"], "age_yrs": st["age"].astype(np.float32),
        "ttm_yrs": np.maximum((uni.maturity_days - asof_days) / 365.25, 0.0).astype(np.float32),
        "is_otr": st["is_otr"],
        "index_mask": st["index_mask"],
        "x": _fsl32(st["x"]),
    }, schema=bundle.tables["instrument_state_daily"]))
    return st


def _fsl32(mat: np.ndarray) -> pa.FixedSizeListArray:
    m = np.ascontiguousarray(mat, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel(), pa.float32()), m.shape[1])
