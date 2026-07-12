"""Probe-harness tests.

The perfect-model path feeds the oracle quantities back in and checks every
probe reports near-perfect recovery. A degraded-model control (rank-reduced
embeddings, collapsed sides) checks the probes actually drop when structure is
destroyed -- otherwise a probe that always returns 1.0 would pass the first test
vacuously.
"""
import numpy as np


def test_perfect_model_recovers_everything(ds):
    from rfqfm import perfect_model_selftest
    r = perfect_model_selftest(ds)

    # A: feeding the oracle vectors reproduces the oracle geometry exactly
    A = r["A_substitution"]
    assert A["gram_offdiag_corr"] > 0.999
    assert min(A["topk_alignment"].values()) > 0.98

    # B: model side structure equals oracle side structure
    B = r["B_side_sense"]
    assert abs(B["model"]["side_axis_coherence"]
               - B["oracle"]["side_axis_coherence"]) < 1e-6

    # C: feeding u_k in recovers u_k (canonical corr ~ 1)
    C = r["C_client_tilt"]
    assert C["mean_canonical_corr"] > 0.999

    # G: the generator's own log-probs give zero excess over the floor
    G = r["G_calibration"]
    assert abs(G["excess_over_floor_one_nats"]) < 1e-9


def test_degraded_model_scores_lower(ds):
    """Negative control: a rank-reduced embedding must lose top-k alignment at
    high k, and collapsing buy==sell must kill the side-axis coherence."""
    from rfqfm import (load_oracle_geometry, probe_side_sense,
                       probe_substitution_geometry)
    geo = load_oracle_geometry(ds)

    # rank-1 truncation of the oracle vectors: keeps the top direction only
    U, S, Vt = np.linalg.svd(geo.v - geo.v.mean(0), full_matrices=False)
    low_rank = (U[:, :1] * S[:1]) @ Vt[:1]
    A_full = probe_substitution_geometry(geo.v.copy(), ds)
    A_low = probe_substitution_geometry(low_rank, ds)
    hi_k = max(A_full["topk_alignment"])
    assert A_low["topk_alignment"][hi_k] < A_full["topk_alignment"][hi_k] - 0.1

    # collapse sides: give buy and sell the same (instrument-centroid) vector
    collapsed = geo.v.copy()
    instr = geo.instr
    for i in np.unique(instr):
        m = instr == i
        collapsed[m] = geo.v[m].mean(0)
    B = probe_side_sense(collapsed, ds)
    assert B["model"]["buy_sell_separation"] < 1e-6
    assert B["oracle"]["buy_sell_separation"] > B["model"]["buy_sell_separation"]


def test_scaling_table_sorts_and_renders(ds):
    from rfqfm import (decompose_recovery, load_reference_geometry,
                       perfect_model_selftest, render_scaling_table, scaling_table)
    base = perfect_model_selftest(ds)
    ref = load_reference_geometry(ds)
    # attach a decomposition + variant so the L0/L1 columns are exercised
    D_full = decompose_recovery(ref.full_dir.copy(), ds)
    reports = {}
    for variant in ("factorized", "free"):
        for rung, p in [("small", 5_000_000), ("large", 230_000_000),
                        ("medium", 23_000_000)]:
            reports[f"{variant}/{rung}"] = dict(
                base, size=rung, params=p, variant=variant, D_decomposition=D_full)
    table = scaling_table(reports)
    # within a variant, rows ascend by params
    fac = [r for r in table["rows"] if r["variant"] == "factorized"]
    assert [r["params"] for r in fac] == sorted(r["params"] for r in fac)
    assert all("residual_recovery" in r and "attribute_recovery" in r
               for r in table["rows"])
    txt = render_scaling_table(table)
    assert "variant" in txt and "resid" in txt and "attr" in txt


def test_sweep_orchestrator_imports_without_torch():
    # the sweep driver is a torch-free subprocess orchestrator; it must import
    # even where torch is absent (probe/train are launched as subprocesses)
    import importlib
    m = importlib.import_module("rfqfm.scripts.sweep")
    assert hasattr(m, "main") and hasattr(m, "stage_table")
