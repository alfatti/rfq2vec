# rfqsim.schema — data-plane contract (extension)

Drop-in subpackage for `rfqsim`: pyarrow schemas for every table in the
oracle/observable design, the observable projection with its leakage audit,
the sharded Parquet write path, and the run manifest.

```
rfqsim/schema/
  enums.py        int-coded categorical registries; embedded as JSON in every file's metadata
  tables.py       SchemaConfig (d, p, n_sectors, max_dealers, n_norm_params),
                  all table schemas, RFQ_COLUMN_CLASS, build_schemas() -> SchemaBundle
  projection.py   RevelationPolicy, project_observable(), audit_observable()
  vocab.py        FeatureSpec (x_w(t): fixed affine + tenor-RBF map, deterministic
                  roll-down), TokenVocab (3 senses/instrument, explicit id map,
                  birth/retirement), tokenization, sentences, windowed
                  same/cross-client pair counts
  state.py        GridCalendar, RegimeConfig (per-sector 4-state bidimensional
                  CTMCs), LatentStateEngine (regime-modulated radial walk ->
                  context_grid + regime_path), content-addressed randomness,
                  bitwise checkpoint/restore, realized_drift audit,
                  predicted_displacement / max_window_steps window rule
  population.py   Universe builder: issuers/instruments/clients/mandates/panels,
                  token vocabulary, signed embeddings under the norm law,
                  client tilts with low-rank Sigma_u, activity paths
  intensity.py    Cox arrival sampling of the grid's lambda_sector surface
  emission.py     per-arrival marks: author (activity coupling), mandate-masked
                  within-sector softmax, two-way side truth from the signed
                  pair, switch/list packages, sizes, auction, outcomes ->
                  rfq_lines / event_truth / auction_book
  pipeline.py     generate_run(): end-to-end reference driver
  backend.py      CuPy/NumPy switch (import xp from here in the fast path)
  gpurand.py      counter-based Philox4x32-10 (Random123 KAT-verified),
                  batched uniforms/normals, masked softmax draw
  emission_batch.py  BatchEmitter: the production engine -- per-sector
                  batched softmax, per-line counter-addressed randomness
  production.py   generate_run_production(): two-phase sharded driver
                  (sequential latent layer with block checkpoints, then
                  parallel-by-shard batched emission)
  writer.py       ShardWriter (hive month partitions, sorted files, zstd, checksums),
                  EventIdAllocator, PhiloxLedger (per-event exact replay)
  manifest.py     RunManifest: seed root, git SHA, config, dials, schema
                  fingerprints, per-file sha256, verify()
tests/            17 tests: schema integrity, leakage audit, write path
```

Install: copy `rfqsim/schema/` into the repo (do **not** copy the top-level
`rfqsim/__init__.py` shim — it exists only so this zip imports standalone).
Deps: pyarrow >= 14, numpy. Run `pytest tests` from the zip root.

## Generation-side integration

```python
cfg    = SchemaConfig(d=64, p=24, n_sectors=12, run_id=run_id)
bundle = build_schemas(cfg)
w      = ShardWriter(root, bundle, shard_id)          # one per H200 time block
alloc  = EventIdAllocator(shard_id)
led    = PhiloxLedger(seed_root_hex, "event_truth", shard_id)

gen, k0, k1, ctr = led.spawn_generator()              # per event; store (k0,k1,ctr)
w.append("rfq_lines", cols); w.append("event_truth", truth_cols)
w.checkpoint_walk_state(grid_idx, {"c": c, "r": r})   # at block boundaries
records = w.close()

man = RunManifest.new(bundle, seed_root_hex, git_sha, config, dials)
man.add_files(records); man.write(root)
```

Learner-side: `obs = project_observable(canonical, cfg, policy)` and put
`audit_observable(obs, cfg, policy)` in CI. Polars reads the output directly
(`pl.scan_parquet(root/"tables/rfq_lines/**/*.parquet")`); row groups carry
stats on ts / event_id / trade_date / client_id / instrument_id / token_id for
pruning windowed co-occurrence scans.

## Deliberate deltas vs. the agreed schema (flagged, with reasons)

1. **`outcome` split into `enquiry_outcome` + `our_result`.** The single enum
   {done, cover, traded_away, dnt, cancelled} was ill-posed on canonical rows
   we never received. `enquiry_outcome` is the market-level fact (TRADED / DNT
   / CANCELLED / EXPIRED); `our_result` is our POV (WON / COVER / LOST /
   NO_QUOTE / NO_TRADE / NOT_RECEIVED). Our own behaviour on non-trades lives
   in `action`.
2. **`side_revealed` / `side_reveal_ts` moved out of the canonical table into
   the projection output.** The canonical tape stays policy-free; one oracle
   run supports many leakage regimes because RevelationPolicy is a projection
   parameter. `outcome_ts` was added to canonical to anchor revelation lags.
3. **`grid_idx` classified LATENT.** It is information-free given `ts`, but it
   is a foreign key into the oracle plane, and the observable plane carries no
   oracle-table references at all.
4. **`token_map` is shared-plane.** The vocabulary (instrument x sense) is
   public structure; what's latent is which token an RFQ instantiated
   (`rfq_lines.token_id`) — including the fact that two-ways instantiate the
   UNDISC sense.
5. **`context_grid.regime` is a per-sector vector**, matching the sector-level
   bidimensional CTMC layer; `regime_path` is keyed (sector_id, sojourn_no)
   and shaped for the existing MMPP EM tooling.
6. **`s_w(t)` is not materialized daily.** `oracle_embeddings.norm_params`
   holds the schedule; daily norms are derived (store state, not
   probabilities). `instrument_state_daily.x` IS materialized — it is the
   single source of truth for x_w(t) shared by oracle and learner, which kills
   morphology skew.
7. **Exec/cover carried in both spread and price space** (spread primary;
   px nullable when the reference curve is disabled).
8. **`client_mandates` / `client_dealer_panel` default to the oracle plane.**
   Whether the learner receives mandate/panel side-information is a
   consumption-time config choice, not a schema fact; evaluators join them
   oracle-side for positivity scoring either way.

Reserved, deliberately absent: `inventory_state`, size–token interaction
(S-curve hook), `marks_daily`.

## The leakage tripwire

Every canonical `rfq_lines` column must appear in `RFQ_COLUMN_CLASS` as
OBSERVABLE, LATENT, SELECTOR or POLICY_GATED; `build_schemas` refuses to
construct otherwise. Adding a column without classifying it is a build error,
not a silent leak. `audit_observable` additionally verifies, per policy, the
null patterns of side revelation and exec/cover gating — recomputed from the
same mask functions the projection uses, so the two cannot drift.

## Measured

Projection + audit on 525k canonical lines (≈ one month of full-market tape at
default dials), single-threaded, pyarrow 24: **0.07s project + 0.01s audit**
on a single-chunk table. Caveat that matters operationally: per-chunk overhead
dominates on heavily chunked inputs (the same table in 47k chunks projects in
~15s), so project per month partition or `combine_chunks()` after concatenating
many small files. Five years of tape projects in a few seconds when driven
partition-by-partition.

## vocab.py design points

FeatureSpec constants are **fixed**, never data-fit: a refit scaler between
runs would silently move B and break oracle/learner parity. The tenor basis is
RBF bumps at benchmark knots over log(1+ttm) — a scalar ttm would force curve
geometry through B linearly, and the cross-issuer analogy probes need
curve-local directions; roll-down is then a smooth deterministic drift through
that coordinate system. `spec.dim` derives from `spec.names`, and
`spec.assert_matches(cfg)` pins it to SchemaConfig.p. Sense codes equal
SideDisclosed codes by construction (asserted at import), so observable
tokenization is `token(instrument, disclosed)` and reproduces the canonical
latent token_id exactly on received rows — pinned by a property test, and the
reason PMI estimated on the observable plane needs no censoring correction on
the received subset. `window_pair_counts` carries the same-client /
cross-client split in the key: same-client excess PMI is the Sigma_u
identification channel.

Measured: tokenizing 5M lines over a 90k-token vocabulary takes 1.5s;
a 30k-instrument x-panel (p=30) for a month of daily snapshots, 0.15s.

## state.py design points

The latent layer uses **content-addressed randomness**: global stream keys
(no shard component) with arithmetic counters — the walk's counter is a pure
function of the day ordinal, a regime sojourn's a pure function of
(sector, sojourn_no). No stream cursors exist, so a checkpoint is just the
walk/regime state and any single day or sojourn regenerates in isolation;
checkpoint-restore is bitwise (pinned by test). Regime chains tick in
session time. The per-sector -> global coupling: each sector's 4-state
(activity, stress) chain drives its own intensity multiplier m[state]; the
stress bits aggregate into s_t in [0,1], which lifts the radial target and
speeds the angular rotation, while T ~ (r/r0)^-gamma (floored) and
lambda_s ~ lambda_bar_s * m * (r/r0)^gamma — stress = busy + narrow is
emergent. The angular dial is parameterized directly as a half-life in grid
steps (the quantity the PMI-decay estimator recovers). Overnight gaps are
unrecorded mixing steps, so step_l1 stays a clean intraday audit (total
variation, upper-bounding displacement); predicted_displacement gives the
closed-form l2/l1 displacement and max_window_steps inverts it into the
PMI window-width rule.

## Emission-layer decisions (flagged)

1. **Per-sector arrivals, within-sector softmax.** The MMPP envelope is exact
   by construction; sector one-hots drop out of the within-sector softmax
   exactly, and cross-sector PMI decomposes into walk geometry (shared c_t)
   plus intensity co-movement (shared radial load) -- separable in-oracle.
   A global softmax with log-intensity offsets is the recorded alternative.
2. **v = s(t) * normalize(B x + eps)**: norms carry frequency (the Zipf/Gini
   dial), directions carry meaning; the frequency law then EMERGES -- see
   below.
3. **Two-way side truth** is drawn from the softmax over the instrument's two
   signed senses at the same (c_t + u_k, T_t): directional base rates come
   from the signed norm asymmetry, no separate side model.
4. The **auction layer is a placeholder** (symmetric dealers, logistic DNT,
   our_width_bp as the hit-rate dial); the production auction ports from the
   existing rfqsim.

## Emergence check worth knowing

The Fig-6 analogue (corr of log token frequency with ||v||^2/2d) FAILS on a
4-day tape with a 10-day walk half-life and PASSES (0.56) on a 20-day tape
with a 2-day half-life: frequency-from-norms is a mixing theorem, and the
battery only sees it once the tape spans many independent contexts. Battery
bands are production-scale bands; short tapes will legitimately WARN/FAIL on
mixing-dependent checks.

## Running the production path on the GPU box

Same code, CuPy picked up automatically by backend.py. One worker process per
H200, CUDA_VISIBLE_DEVICES set per worker, each calling
production.shard_worker(root, dials, k, block, grids[k]) for its day block;
merge the returned FileRecords into one manifest. Set BatchEmitter.chunk to
65536 on device. Measured here on the NumPy backend, single CPU core:
13,760 lines/s emission (7.6x the reference loop); a 10-day, 500k-line
full-market run end to end (latent layer, emission, projection, audit,
writing, validation, manifest hashing) in 45s. Five years is ~75 min on one
CPU core; sharded across GPUs the math stops being the bottleneck and disk
writes are.

The production and reference engines are separately deterministic but not
bit-compatible (different Philox families: batched 4x32 vs numpy's 4x64);
the reference engine remains the semantic spec, and the distributional-
agreement test pins the two together. auction_book layouts differ (natural
slots vs compacted) -- both valid under the schema.

## Next against this contract

The calibration ladder (SN objective with Theorem-6 weights, norms <-
frequencies, directions <- cross-client PMI, Sigma_u <- same-client excess,
walk speed <- PMI decay) and the gate notebook; CuPy port of the emission
loop for production scale.
