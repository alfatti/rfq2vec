"""Production run driver: the multi-shard, GPU-oriented path.

Two phases, matching how the pieces were built to parallelize:

  Phase 1 (cheap, sequential): the latent layer. The walk and regime chains
  run once over the whole calendar (~10s per simulated year on CPU), each
  day's context_grid going to the shard that owns that day, with the walk
  state checkpointed at every shard boundary.

  Phase 2 (the hot path, parallel by shard): batched emission. Each shard
  owns a contiguous day block and needs nothing but the universe, its day
  range, and the content-addressed streams -- no cross-shard state at all,
  because arrivals are keyed by day ordinal and emission randomness by
  (day, event, leg) counters. On the H200 box, run one worker process per
  GPU with CUDA_VISIBLE_DEVICES set per worker and shard_worker() as the
  entry point; here the shards run in-process, sequentially, on the NumPy
  backend -- same code, same output.

Everything a worker writes is checksummed into FileRecords; the driver merges
them into one manifest and verifies the tree before returning.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from . import enums as E
from .emission_batch import BatchEmitter
from .intensity import sample_arrivals
from .manifest import RunManifest
from .pipeline import RunDials, _dials_json, validation_battery
from .population import (build_universe, business_calendar,
                         write_instrument_state_day, write_universe)
from .projection import audit_observable, project_observable
from .state import LatentStateConfig, LatentStateEngine, RegimeConfig
from .tables import SchemaConfig, build_schemas
from .vocab import FeatureSpec
from .writer import EventIdAllocator, FileRecord, ShardWriter


def _blocks(n_days: int, n_shards: int) -> list[range]:
    cuts = np.linspace(0, n_days, n_shards + 1).astype(int)
    return [range(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def shard_worker(root, dials: RunDials, shard_id: int, days: range,
                 grid_tables: list[pa.Table]) -> list[FileRecord]:
    """Phase-2 worker: emit one contiguous day block. Safe to run in its own
    process (one per GPU); everything it needs is an argument or content-
    addressed. grid_tables are that block's context_grid days, in order."""
    import datetime as dt
    spec = FeatureSpec(sectors=dials.sectors)
    cfg = SchemaConfig(d=dials.d, p=spec.dim, n_sectors=len(dials.sectors),
                       run_id=dials.run_id)
    bundle = build_schemas(cfg)
    cal = business_calendar(dt.date(*dials.start), dials.n_days)
    uni = build_universe(dials.universe, spec, cal, dials.seed_root_hex, dials.d)

    w = ShardWriter(root, bundle, shard_id=shard_id)
    em = BatchEmitter(uni, bundle, dials.emission, dials.seed_root_hex)
    alloc = EventIdAllocator(shard_id)

    for j, day in enumerate(days):
        grid_day = grid_tables[j]
        arrivals = sample_arrivals(grid_day, cal, day, dials.seed_root_hex)
        rfq, truth, book = em.emit_day_batch(day, grid_day, arrivals, alloc)
        w.append("rfq_lines", rfq)
        w.append("event_truth", truth)
        w.append("auction_book", book)
        obs = project_observable(rfq, cfg, dials.policy)
        audit_observable(obs, cfg, dials.policy)
        w.append("rfq_lines_obs", obs)
        write_instrument_state_day(uni, w, bundle, int(cal.days[day]))
    return w.close()


def generate_run_production(root, dials: RunDials, n_shards: int = 1) -> dict:
    """The production entry point. Sequential in-process shard loop here;
    on the GPU box, dispatch shard_worker(root, dials, k, block, grids[k])
    to one process per GPU and merge the returned FileRecords."""
    import datetime as dt
    spec = FeatureSpec(sectors=dials.sectors)
    cfg = SchemaConfig(d=dials.d, p=spec.dim, n_sectors=len(dials.sectors),
                       run_id=dials.run_id)
    spec.assert_matches(cfg)
    bundle = build_schemas(cfg)
    cal = business_calendar(dt.date(*dials.start), dials.n_days)
    uni = build_universe(dials.universe, spec, cal, dials.seed_root_hex, dials.d)
    blocks = _blocks(cal.n_days, n_shards)

    # -- phase 1: latent layer, sequential, checkpoint at block boundaries --
    scfg = LatentStateConfig(
        d=dials.d, n_sectors=len(dials.sectors),
        lambda_bar_sector=(dials.lambda_bar_per_min,) * len(dials.sectors),
        theta_half_life_steps=dials.theta_half_life_days * 390.0,
        regime=RegimeConfig.bidimensional())
    engine = LatentStateEngine(scfg, bundle, dials.seed_root_hex, cal)

    writers = [ShardWriter(root, bundle, shard_id=1000 + k)
               for k in range(len(blocks))]
    rp = bundle.tables["regime_path"]
    grids: list[list[pa.Table]] = [[] for _ in blocks]
    records: list[FileRecord] = []
    for k, block in enumerate(blocks):
        for day in block:
            g, closed = engine.simulate_day(day)
            grids[k].append(g)
            writers[k].append("context_grid", g)
            if closed:
                writers[k].append("regime_path", pa.Table.from_pydict(
                    {n: [r[n] for r in closed] for n in rp.names}, schema=rp))
        writers[k].checkpoint_walk_state(
            cal.grid_idx(block[-1], int(cal.steps_per_day[block[-1]]) - 1),
            engine.state_dict())
    open_rows = engine.open_sojourn_rows()
    if open_rows:
        writers[-1].append("regime_path", pa.Table.from_pydict(
            {n: [r[n] for r in open_rows] for n in rp.names}, schema=rp))

    # dims + oracle tables ride on the last latent writer
    write_universe(uni, writers[0], bundle)
    for w in writers:
        records += w.close()

    # -- phase 2: emission, per shard (parallelize this loop on the GPU box) --
    for k, block in enumerate(blocks):
        records += shard_worker(root, dials, k, block, grids[k])

    # -- validation + manifest ------------------------------------------------
    import pyarrow.dataset as ds
    canon = ds.dataset(f"{root}/tables/rfq_lines", partitioning="hive").to_table()
    grid = ds.dataset(f"{root}/tables/context_grid", partitioning="hive").to_table()
    checks = validation_battery(canon, grid, uni, scfg, dials)
    vw = ShardWriter(root, bundle, shard_id=999)
    vw.append("validation_report", pa.Table.from_pydict(
        {n: [c[n] for c in checks]
         for n in bundle.tables["validation_report"].names},
        schema=bundle.tables["validation_report"]))
    records += vw.close()

    man = RunManifest.new(bundle, dials.seed_root_hex, rfqsim_git_sha="ext-dev",
                          config=_dials_json(dials),
                          dials={"d": dials.d, "engine": "batch",
                                 "n_shards": n_shards,
                                 "theta_half_life_days": dials.theta_half_life_days,
                                 "feature_spec_version": spec.version})
    man.add_files(records)
    man.write(root)
    bad = man.verify(root)
    if bad:
        raise RuntimeError(f"manifest verify failed: {bad}")
    return dict(rows=canon.num_rows, checks=checks,
                tables={t: e["rows"] for t, e in man.tables.items()})
