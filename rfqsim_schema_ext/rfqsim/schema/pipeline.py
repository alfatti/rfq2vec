"""End-to-end run orchestration: dials in, verified dataset out.

generate_run(root, dials):
  1. calendar + universe (population stream)
  2. latent state day loop: context_grid + regime_path (+ checkpoint)
  3. arrivals (Cox sampling of the grid surface)
  4. emission: rfq_lines / event_truth / auction_book
  5. instrument_state_daily (the materialized x_w(t))
  6. observable projection per day, audited, written as rfq_lines_obs
  7. validation battery -> validation_report
  8. manifest with checksums; verify before returning

Single-shard reference driver; the multi-GPU version parallelizes step 2-6 by
contiguous day blocks, restoring the walk checkpoint at each block boundary --
every ingredient here is already content-addressed or checkpointed for that.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pyarrow as pa

from . import enums as E
from .emission import EmissionConfig, Emitter
from .intensity import sample_arrivals
from .manifest import RunManifest
from .population import (UniverseConfig, build_universe, business_calendar,
                         write_instrument_state_day, write_universe)
from .projection import DEFAULT_POLICY, RevelationPolicy, audit_observable, project_observable
from .state import (LatentStateConfig, LatentStateEngine, RegimeConfig,
                    predicted_displacement, realized_drift)
from .tables import SchemaConfig, build_schemas
from .vocab import FeatureSpec
from .writer import EventIdAllocator, PhiloxLedger, ShardWriter


@dataclass(frozen=True)
class RunDials:
    run_id: str
    seed_root_hex: str
    start: tuple[int, int, int] = (2026, 1, 5)
    n_days: int = 20
    d: int = 16
    sectors: tuple[str, ...] = ("Energy", "TMT", "Healthcare", "Utilities")
    lambda_bar_per_min: float = 0.5          # per sector, calm baseline
    theta_half_life_days: float = 10.0
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    policy: RevelationPolicy = DEFAULT_POLICY


def generate_run(root, dials: RunDials) -> dict:
    import datetime as dt
    spec = FeatureSpec(sectors=dials.sectors)
    ns = len(dials.sectors)
    cfg = SchemaConfig(d=dials.d, p=spec.dim, n_sectors=ns, run_id=dials.run_id)
    spec.assert_matches(cfg)
    bundle = build_schemas(cfg)

    cal = business_calendar(dt.date(*dials.start), dials.n_days)
    uni = build_universe(dials.universe, spec, cal, dials.seed_root_hex, dials.d)

    scfg = LatentStateConfig(
        d=dials.d, n_sectors=ns,
        lambda_bar_sector=(dials.lambda_bar_per_min,) * ns,
        theta_half_life_steps=dials.theta_half_life_days * 390.0,
        regime=RegimeConfig.bidimensional())
    engine = LatentStateEngine(scfg, bundle, dials.seed_root_hex, cal)

    w = ShardWriter(root, bundle, shard_id=0)
    write_universe(uni, w, bundle)

    emitter = Emitter(uni, bundle, dials.emission, dials.seed_root_hex)
    alloc = EventIdAllocator(shard_id=0)
    ledger = PhiloxLedger(dials.seed_root_hex, "event_truth", shard_id=0)
    rp_schema = bundle.tables["regime_path"]

    grid_tables, canon_tables = [], []
    for day in range(cal.n_days):
        grid_day, closed = engine.simulate_day(day)
        w.append("context_grid", grid_day)
        if closed:
            w.append("regime_path", pa.Table.from_pydict(
                {n: [r[n] for r in closed] for n in rp_schema.names}, schema=rp_schema))
        write_instrument_state_day(uni, w, bundle, int(cal.days[day]))

        arrivals = sample_arrivals(grid_day, cal, day, dials.seed_root_hex)
        rfq, truth, book = emitter.emit_day(day, grid_day, arrivals, alloc, ledger)
        w.append("rfq_lines", rfq)
        w.append("event_truth", truth)
        w.append("auction_book", book)

        obs = project_observable(rfq, cfg, dials.policy)
        audit_observable(obs, cfg, dials.policy)
        w.append("rfq_lines_obs", obs)

        grid_tables.append(grid_day)
        canon_tables.append(rfq)

    open_rows = engine.open_sojourn_rows()
    if open_rows:
        w.append("regime_path", pa.Table.from_pydict(
            {n: [r[n] for r in open_rows] for n in rp_schema.names}, schema=rp_schema))
    w.checkpoint_walk_state(cal.grid_idx(cal.n_days - 1,
                            int(cal.steps_per_day[-1]) - 1), engine.state_dict())

    grid = pa.concat_tables(grid_tables)
    canon = pa.concat_tables(canon_tables)
    checks = validation_battery(canon, grid, uni, scfg, dials)
    w.append("validation_report", pa.Table.from_pydict(
        {n: [c[n] for c in checks] for n in bundle.tables["validation_report"].names},
        schema=bundle.tables["validation_report"]))

    records = w.close()
    man = RunManifest.new(bundle, dials.seed_root_hex, rfqsim_git_sha="ext-dev",
                          config=_dials_json(dials), dials={
                              "d": dials.d, "beta_norm": dials.universe.beta_norm,
                              "sigma_u": dials.universe.sigma_u,
                              "rho_u": dials.universe.rho_u,
                              "sigma_eps": dials.universe.sigma_eps,
                              "theta_half_life_days": dials.theta_half_life_days,
                              "feature_spec_version": spec.version})
    man.add_files(records)
    man.write(root)
    mismatches = man.verify(root)
    if mismatches:
        raise RuntimeError(f"manifest verify failed: {mismatches}")

    return dict(rows=canon.num_rows, checks=checks,
                tables={t: e["rows"] for t, e in man.tables.items()})


# ===========================================================================
# Validation battery -> validation_report rows.
# ===========================================================================

def _check(run_id, name, statistic, value, lo, hi) -> dict:
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    return dict(run_id=run_id, check=name, stratum="all", statistic=statistic,
                value=float(value), target_lo=lo, target_hi=hi,
                status=int(E.CheckStatus.PASS if ok else E.CheckStatus.FAIL),
                details=None)


def _gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def validation_battery(canon: pa.Table, grid: pa.Table, uni,
                       scfg: LatentStateConfig, dials: RunDials) -> list[dict]:
    rid = dials.run_id
    checks = []
    recv = canon.filter(canon["received"])
    our = recv["our_result"].to_numpy(zero_copy_only=False)
    checks.append(_check(rid, "hit_rate", "won/received",
                         float(np.mean(our == int(E.OurResult.WON))), 0.03, 0.10))
    eo = canon["enquiry_outcome"].to_numpy(zero_copy_only=False)
    checks.append(_check(rid, "traded_share", "traded/all",
                         float(np.mean(eo == int(E.EnquiryOutcome.TRADED))), 0.45, 0.90))
    disc = canon["client_side_disclosed"].to_numpy(zero_copy_only=False)
    checks.append(_check(rid, "two_way_share", "two_way/all",
                         float(np.mean(disc == int(E.SideDisclosed.TWO_WAY))), 0.08, 0.45))
    checks.append(_check(rid, "panel_share", "received/all",
                         float(np.mean(canon["received"].to_numpy(zero_copy_only=False))),
                         0.35, 0.90))

    tok = canon["token_id"].to_numpy(zero_copy_only=False)
    counts = np.bincount(tok, minlength=len(uni.vocab)).astype(np.float64)
    active = counts[counts > 0]
    checks.append(_check(rid, "token_gini", "gini(active token counts)",
                         _gini(active), 0.35, 0.95))
    # Fig 6 analogue: log frequency should track the norm law s^2/2d
    asof = int(uni.cal.days[uni.cal.n_days // 2])
    s = uni.s_of(np.arange(len(uni.vocab)), asof)
    mask = counts >= 3
    corr = float(np.corrcoef(np.log(counts[mask]),
                             (s[mask] ** 2) / (2 * dials.d))[0, 1])
    checks.append(_check(rid, "norm_freq_corr", "corr(log n_w, ||v||^2/2d)",
                         corr, 0.30, None))

    # drift audit: realized TV per 30-step window vs the closed-form prediction
    rep = realized_drift(grid, 30)
    per_step = predicted_displacement(scfg, 1)["l1"]
    checks.append(_check(rid, "drift_tv_ratio", "realized TV / predicted TV (30 steps)",
                         rep["mean"] / (30 * per_step), 0.5, 2.0))
    return checks


def _dials_json(dials: RunDials) -> dict:
    d = asdict(dials)
    d["policy"] = {k: (v.name if hasattr(v, "name") else v)
                   for k, v in asdict(dials.policy).items()}
    return json.loads(json.dumps(d, default=str))
