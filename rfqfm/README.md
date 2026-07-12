# rfqfm — a transaction foundation model over RFQ simulation output

`rfqfm` trains a decoder-only foundation model on the RFQ event stream produced
by `rfqsim`. It takes the NVIDIA "transaction foundation model" blueprint as a
starting point and departs from it wherever owning the data-generating process
lets us do better — most of all in evaluation, where we grade the model against
the simulator's ground-truth latents and the exact generative probability of
every choice.

See `rfq_foundation_model_strategy.md` for the full rationale. This README is
the operational summary.

## The relationship to rfqsim

`rfqfm` is a **consumer** of the simulator's output. The contract between them
is the on-disk schema: the hive-partitioned tables plus `manifest.json`, with
the enum registries versioned inside every file. `rfqfm` imports exactly one
thing from `rfqsim` — `rfqsim.schema.enums`, so token semantics stay
single-sourced — and reads everything else from disk. Torch never enters the
simulator; the simulator's build never enters the training box beyond that one
light import.

## Layout

```
rfqfm/
  contract.py     RfqDataset: the schema contract (the only rfqsim import lives here)
  config.py       TokenizerConfig + the small/medium/large ModelConfig ladder
  vocab.py        FmVocab: global vocabulary, disjoint per-family id ranges
  features.py     BondFeatures: per-bond feature matrix for the factorized embedding
  tokenize.py     rfq_lines -> anchored-composite token blocks (vectorized)
  corpus.py       session + tape sequence assembly, dual-corpus mix
  packed.py       PackedCorpus: packed-token format with a truth (event_id) sidecar
  floor.py        the DGP entropy floor + excess-over-floor (Test G groundwork)
  model.py        FactorizedTokenEmbedding + Llama builder            [needs torch]
  data.py         torch Dataset over a PackedCorpus                   [needs torch]
  scripts/
    build_corpus.py   run root -> vocab + features + packed corpora   (torch-free)
    train.py          torchrun DDP training of one rung               [needs torch]
configs/            small.yaml / medium.yaml / large.yaml
tests/              torch-free core tests, run against a simulator run
```

The core (everything above `model.py`) imports without torch. The model and
training modules pull torch lazily and are meant for the GPU box.

## Tokenization (the short version)

Each RFQ leg becomes a short block whose **anchor** is the `(CUSIP, side)` token
— a single vocabulary entry, so its embedding row is the Arora word vector:

```
<sep> TDLT CLI TOK SZ NDLR OUT        (first leg of an RFQ)
<leg> TOK SZ                          (continuation leg of a package)
```

`TDLT` is a log-binned gap since the client's previous RFQ — the channel through
which the model can see the Markov-modulated intensity; strip it and no model
beats a constant-rate baseline. A sequence is one client's history (session
axis) or an interleaved market-tape window (tape axis); we train on both, mostly
sessions. The `TOK` embedding is **factorized** through bond features
(`features @ B + sector + sense + residual`) rather than ~90k free rows, both to
afford a large vocabulary and to mirror the simulator's own `v = s·normalize(B·x+ε)`.

## Model ladder

| Rung | hidden | layers | heads (Q/KV) | transformer params | recovers |
|------|--------|--------|--------------|--------------------|----------|
| small | 256 | 6 | 8 / 2 | ~5M | popularity, first-order substitution, own recency |
| medium | 512 | 8 | 8 / 2 | ~23M (NVIDIA-scale) | within-sector geometry, side sense, size/regime basics |
| large | 1024 | 20 | 16 / 4 | ~230M | full-rank cross-client geometry, client-conditional substitution, regime switching, roll-down |

Architecture is the NVIDIA/Llama spine throughout (RoPE θ=500k, GQA, SwiGLU,
RMSNorm, no dropout). Even the large rung fits on one H200, so the four cards run
the sweep in parallel, one rung per card — plain data parallel, no model
sharding.

## Running it (on the box)

```bash
# 1. generate a production dataset with the simulator (sharded across GPUs)
#    -> produces a run root with tables/ + manifest.json (now incl. the geometry
#       oracle: oracle_token_vectors + oracle_attribute_gain)

# 2. the whole scaling sweep in one command: build corpus, train the three
#    rungs concurrently (one per GPU), probe each checkpoint, collate the table
python -m rfqfm.scripts.sweep --root /data/run --work /data/fm/run \
    --rungs small medium large --gpus 0 1 2

# or run the stages by hand:
python -m rfqfm.scripts.build_corpus --root /data/run --out /data/fm/run --context 4096
torchrun --nproc-per-node=4 -m rfqfm.scripts.train \
    --config configs/medium.yaml --data /data/fm/run --out /data/fm/ckpt/medium
python -m rfqfm.scripts.probe --root /data/run --data /data/fm/run \
    --ckpt /data/fm/ckpt/medium/final --size medium --out /data/fm/probe/medium.json
```

The sweep writes `scaling_table.json` and prints it — the headline artifact,
each recovery metric as a function of model size:

```
rung          params    gram  topk_hi    side     u_k  excessKL
---------------------------------------------------------------
small        5000000    ...      ...     ...     ...      ...
medium      23000000    ...      ...     ...     ...      ...
large      230000000    ...      ...     ...     ...      ...
```

`build_corpus` also writes `floor.json` — the DGP entropy floor the `excessKL`
column is measured against.

## The recovery probes

`probes.py` grades a trained model against the simulator's ground truth. Four
are implemented and self-tested (`perfect_model_selftest` feeds the oracle
quantities in and every probe reports near-perfect recovery; a degraded-model
control confirms they drop when structure is destroyed):

| Probe | Question | Reads |
|-------|----------|-------|
| A substitution geometry | does the embedding recover the full-rank word geometry, top-k? | anchor embedding table vs `oracle_token_vectors` |
| B side-sense | are buy/sell kept as distinct senses, not collapsed to the instrument centroid? | anchor embedding table |
| C client tilt | is `u_k` recovered as a conditional rotation, not just marginal popularity? | CLI embedding vs `oracle_clients` |
| G calibration | how far is the model above the exact DGP floor? | anchor log-probs vs `event_truth` |

`extract.py` (torch) pulls the arrays each probe needs out of a checkpoint; the
probe logic itself is torch-free. The predictive-distribution probes — regime /
intensity (D), roll-down (E), package completion (F) — are documented metric
contracts that wire up once checkpoints exist, since (unlike geometry) there is
no oracle-only stand-in for a next-token distribution.


## Tests

```bash
pip install -e .            # plus rfqsim on the path
python -m pytest tests -q   # torch-free; runs against a small simulator run
```

The suite checks the schema contract, the vocabulary's disjoint ranges and
mappers, the bond-feature maps, the vectorized block construction (structure,
anchor/truth alignment, time-gap bucketing), the session/tape packing round-trip,
and the entropy floor (including that feeding the generator's own log-probs back
in yields zero excess over the floor).

## The inductive-bias study (L0 vs L1)

The first experiment asks whether the transformer *discovers* the generator's
attribute factorization or only succeeds when we bake it in. Same tokens, two
embeddings:

  * **L0 — factorized**: the anchor row is `features @ B + residual`.
  * **L1 — free table**: a plain per-token embedding, no structure handed over.

It's a one-flag change (`ModelConfig.factorized_embedding`, exposed as
`--factorized/--no-factorized`), so the sweep runs the `{size} x {factorized,
free}` grid and `probes.decompose_recovery` splits each model's recovered
geometry into an **attribute** part (the `B x` Gram) and a **residual** part (the
partial correlation with the full geometry controlling for attributes — the
identity-bound piece). The prediction: L0 leads on attribute-recovery at small
scale (the bias is free) with the gap closing as L1 earns it, and a possible
reversal on residual-recovery where L1's free rows memorize the residual that
L0's regularized rows won't reach.

`sigma_eps` sets how much of the substitution geometry lives in that residual, so
it sets the size of the effect this study measures. Calibrated on the smoke
universe: `sigma_eps=1.0` leaves attributes explaining ~0.92 of the Gram
(residual a thin ~16%); `sigma_eps=2.0` brings that to ~0.72 (residual ~48%) —
attributes still dominant, residual clearly material. Use `scripts/generate.py`,
which takes `--sigma-eps` (default 2.0) and prints the realized residual share so
the study parameter is recorded with the run.

```bash
python -m rfqfm.scripts.generate --out /data/run --sigma-eps 2.0 \
    --days 2520 --issuers 3000 --clients 400 --shards 4
python -m rfqfm.scripts.sweep --root /data/run --work /data/fm/run \
    --variants factorized free --rungs small medium large --gpus 0 1 2 3
```

The sweep prints and writes `scaling_table.json` with the `attr` and `resid`
columns side by side for each `size x variant` cell — the study's headline.

## Not yet built

The three predictive-distribution probes — regime/intensity (D), roll-down (E),
and package completion (F) — need a model's decoded next-token distribution with
context, so their metric functions are documented contracts in `probes.py` that
wire up once real checkpoints exist. Everything else — the geometry oracle in the
simulator, corpus building, the factorized-embedding model, training, the
A/B/C/G probes, and the one-command sweep — is built and green.
