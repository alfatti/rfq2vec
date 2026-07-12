# RFQ Foundation-Model Workspace

This repository holds a scientific study of a simple question with a rarely
available answer: **can transformer architectures learn a known data-generating
process?** We simulate electronic credit-desk RFQ (request-for-quote) trading in
investment-grade corporate bonds with a fully specified generative model, train
decoder-only transformers on the simulated event stream, and grade what they
recovered against the ground truth — the latent market state, the per-client
tilts, the bond-embedding geometry, and, uniquely, the **exact probability the
generator assigned to every realized choice**. That last piece gives the study a
computable entropy floor: "how well did the model learn" is a measurement in
nats above a known optimum, not a comparison against a proxy baseline. No
real-data foundation model can grade itself this way, which is the reason to do
the study on synthetic data at all.

## Components

The workspace is two sibling Python packages plus data directories:

```
rfq-workspace/
├── rfqsim_schema_ext/        # the simulator
│   ├── rfqsim/schema/        # generative model, schema, GPU production path
│   └── tests/                # 49 tests
├── rfqfm/                    # the foundation-model study package
│   ├── rfqfm/                # contract, tokenizer, model, probes, instruments
│   ├── rfqfm/scripts/        # generate / build_corpus / train / probe / sweep
│   ├── configs/              # small / medium / large training configs
│   └── tests/                # 18 tests (torch-free; run anywhere)
├── data/                     # simulator run roots (large; gitignored)
└── artifacts/                # corpora, checkpoints, probe reports (gitignored)
```

**`rfqsim`** generates the world. It is grounded in Bergault & Guéant's model of
RFQ liquidity: sector-level activity regimes drive arrival intensity (a
Markov-modulated Poisson process), a latent random walk drives *which* bonds are
in demand, each bond token is a vector `v = s · normalize(B·x + ε)` (attribute
geometry plus an idiosyncratic residual), and each client carries a persistent
tilt `u_k` that rotates their substitution pattern. Every choice the generator
makes is logged with its exact log-probability (`event_truth`), and the latent
ground truth is dumped to oracle tables (`oracle_token_vectors`,
`oracle_attribute_gain`, `oracle_clients`, `context_grid`, `regime_path`). The
production path is fully vectorized (counter-based Philox RNG, batched
emission, multi-GPU sharding) and targets H200s, falling back to NumPy on CPU.

**`rfqfm`** consumes the simulator's output from disk and runs the study. It
tokenizes RFQ lines into anchored-composite blocks — the `(CUSIP, side)` pair
stays a single vocabulary entry, wrapped in a short block of time-gap, client,
size, dealer-count, and outcome tokens — assembles two corpus views (per-client
sessions and the interleaved market tape), trains a ladder of Llama-style
decoders (small / medium / large), and grades each checkpoint with a probe
harness that reads the oracle tables. The two packages touch at exactly one
import (`rfqsim.schema.enums`, so token semantics stay single-sourced);
everything else flows through the on-disk schema, which is the contract between
them. Torch never enters the simulator.

## Installation

From the workspace root:

```bash
pip install -e rfqsim_schema_ext
pip install -e "rfqfm[train]"     # [train] adds torch + transformers (GPU box)
pip install -e rfqfm              # torch-free core only (data box / laptop)
```

Editable installs make the cross-package import resolve without path games and
mean a one-line simulator change is picked up immediately.

Verify:

```bash
(cd rfqsim_schema_ext && python -m pytest tests -q)   # 49 passed
(cd rfqfm && python -m pytest tests -q)               # 18 passed, no GPU needed
```

## Running the study end to end

On the GPU box (4× H200 assumed; adjust `--gpus`):

```bash
# 1. Generate a study dataset. sigma_eps sets how much of the substitution
#    geometry lives in the identity-bound residual (see below); the script
#    prints the realized share so the parameter is recorded with the run.
python -m rfqfm.scripts.generate --out data/run-001 --sigma-eps 2.0 \
    --days 2520 --issuers 3000 --clients 400 --shards 4

# 2. The whole sweep in one command: build corpora, train the
#    {small,medium,large} x {factorized,free} grid in waves across the GPUs,
#    probe every checkpoint, and collate the scaling table.
python -m rfqfm.scripts.sweep --root data/run-001 --work artifacts/run-001 \
    --variants factorized free --rungs small medium large --gpus 0 1 2 3
```

The sweep is re-enterable (`--skip build train` to re-probe without
retraining), and each stage can be run by hand (`build_corpus`, `train` under
torchrun, `probe`) — see `rfqfm/README.md` for the individual commands.

The endpoint is `artifacts/run-001/scaling_table.json`, printed as:

```
rung    variant          params   gram   attr  resid    u_k  excessKL
small   factorized      5000000    ...
small   free            5000000    ...
medium  factorized     23000000    ...
...
```

## What the study measures

**The entropy floor (probe G).** `event_truth` stores the generator's exact
log-probability for every realized choice, so the model's cross-entropy
decomposes into: the floor given the latents (known exactly), plus the latent
information gap (what the observable tape cannot reveal — itself the ceiling on
how predictable flow is from the tape alone), plus approximation error (what
shrinks with scale). The `excessKL` column is the distance above the floor, in
nats.

**Structure-recovery probes (A, B, C).** Substitution geometry: does the
model's token-similarity structure match the oracle Gram, out to high rank?
Side-sense polysemy: are buy and sell kept as distinct senses rather than
collapsing to the instrument centroid? Client tilt: is `u_k` recovered as a
conditional rotation of the substitution pattern, not just marginal popularity?
Each probe is self-tested two ways: feeding the oracle quantities in as a
"perfect model" must score ~1, and a deliberately degraded model (rank-reduced
geometry, collapsed sides) must score low — so no probe passes vacuously.

**The inductive-bias experiment (L0 vs L1).** The deeper question the token
representation raises: does the transformer *discover* the generator's
attribute factorization, or only succeed when we bake it in? Two embeddings,
same tokens: **L0 (factorized)** builds each anchor row as
`features @ B + residual`, handing the model the generator's functional form;
**L1 (free)** is a plain per-token table that must earn that structure. The
decomposition instrument (`rfqfm/representation.py`) splits each model's
recovered geometry into an **attribute part** (explained by the `B·x` Gram) and
a **residual part** (the partial correlation with the full geometry controlling
for attributes — the identity-bound piece), and is validated against both
ceilings. The generator's `sigma_eps` dial controls the effect size: at 2.0
(the study default) attributes explain ~0.72 of the substitution Gram and the
residual carries ~48% — attributes still dominant, residual clearly material.

## Reading the results

The organizing thesis: the analytic identification ladder — norms from
frequencies (easy), directions from cross-client co-occurrence (medium), client
covariance from same-client excess co-occurrence (hard), regimes from arrival
timing (hard, separate channel) — should reappear as an empirical capability
ladder, recovered in the same order as model scale grows. The scaling table is
built to show exactly that: recovery metrics climbing down the size column,
`excessKL` falling toward the floor, and the `attr`/`resid` pair revealing how
much of the answer the token representation handed the model versus how much it
discovered.

## Status and roadmap

Built and green: the simulator's full production path with the geometry oracle
(schema 0.2.0); corpus construction on both axes; the factorized-embedding
model; concurrent DDP training; probes A/B/C/G with perfect-model and
degraded-model self-tests; the attribute/residual instrument; the one-command
sweep.

Not yet built: the three probes needing a model's decoded next-token
distribution with context — regime/intensity tracking (D), roll-down/aging (E),
package completion (F) — which wire up once real checkpoints exist; and the
deferred simulator extensions (size-dependent quoting, inventory skew,
event-driven affinity shocks, EM closed-loop recovery of the regime
parameters).

Deeper design rationale lives in `rfq_foundation_model_strategy.md` (the study
design note) and the two package READMEs.
