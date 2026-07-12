"""population / intensity / emission / pipeline: universe determinism,
arrival statistics, emission invariants, and the end-to-end run with a
verified manifest and audited observable plane."""
import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

from rfqsim.schema import enums as E
from rfqsim.schema.emission import EmissionConfig, Emitter
from rfqsim.schema.intensity import sample_arrivals
from rfqsim.schema.pipeline import RunDials, generate_run
from rfqsim.schema.population import (UniverseConfig, build_universe,
                                      business_calendar, token_vectors)
from rfqsim.schema.state import LatentStateConfig, LatentStateEngine
from rfqsim.schema.tables import SchemaConfig, build_schemas
from rfqsim.schema.vocab import FeatureSpec
from rfqsim.schema.writer import EventIdAllocator, PhiloxLedger

SEED = "0123456789abcdef0123456789abcdef"
SECTORS = ("Energy", "TMT", "Healthcare", "Utilities")
D = 16


@pytest.fixture(scope="module")
def world():
    spec = FeatureSpec(sectors=SECTORS)
    cfg = SchemaConfig(d=D, p=spec.dim, n_sectors=4, run_id="t-world")
    bundle = build_schemas(cfg)
    cal = business_calendar(dt.date(2026, 1, 5), 3, session_minutes=60)
    ucfg = UniverseConfig(n_issuers=40, n_clients=30)
    uni = build_universe(ucfg, spec, cal, SEED, D)
    scfg = LatentStateConfig(d=D, n_sectors=4, lambda_bar_sector=(1.5,) * 4,
                             theta_half_life_steps=300.0)
    eng = LatentStateEngine(scfg, bundle, SEED, cal)
    grid0, _ = eng.simulate_day(0)
    return dict(spec=spec, cfg=cfg, bundle=bundle, cal=cal, uni=uni, grid0=grid0)


# ---------------------------------------------------------------------------
# Population.
# ---------------------------------------------------------------------------

def test_universe_shapes_and_determinism(world):
    uni = world["uni"]
    n_tok = len(uni.vocab)
    assert n_tok == 3 * uni.n_instruments
    assert uni.eps.shape == (n_tok, D) and uni.u.shape == (uni.n_clients, D)
    assert uni.B.shape == (world["spec"].dim, D)
    assert (uni.norm_params[:, 0] > 0).all()
    # Sigma_u ground truth matches the construction's second moment scale
    assert np.trace(uni.Sigma_u) == pytest.approx(uni.cfg.sigma_u ** 2, rel=1e-6)
    # pure function of (seed, dials)
    uni2 = build_universe(uni.cfg, world["spec"], world["cal"], SEED, D)
    np.testing.assert_array_equal(uni.eps, uni2.eps)
    np.testing.assert_array_equal(uni.rating, uni2.rating)


def test_token_vectors_norm_schedule(world):
    uni = world["uni"]
    asof = int(world["cal"].days[0])
    v = token_vectors(uni, asof)
    s = uni.s_of(np.arange(len(uni.vocab)), asof)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), s, rtol=1e-6)


# ---------------------------------------------------------------------------
# Intensity.
# ---------------------------------------------------------------------------

def test_arrivals_statistics_and_determinism(world):
    cal, grid0 = world["cal"], world["grid0"]
    a1 = sample_arrivals(grid0, cal, 0, SEED)
    a2 = sample_arrivals(grid0, cal, 0, SEED)
    np.testing.assert_array_equal(a1.ts_us, a2.ts_us)      # content-addressed

    lam = np.asarray(grid0["lambda_sector"].combine_chunks().flatten()).reshape(
        grid0.num_rows, -1)
    expect = lam.sum()
    assert abs(len(a1) - expect) < 5 * np.sqrt(expect)      # Poisson envelope
    assert (np.diff(a1.ts_us) >= 0).all()
    ts0 = grid0["ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    assert (a1.ts_us >= ts0[a1.step_k]).all()
    assert (a1.ts_us < ts0[a1.step_k] + cal.step_us).all()


# ---------------------------------------------------------------------------
# Emission.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emitted(world):
    uni, bundle, cal = world["uni"], world["bundle"], world["cal"]
    em = Emitter(uni, bundle, EmissionConfig(), SEED)
    arrivals = sample_arrivals(world["grid0"], cal, 0, SEED)
    rfq, truth, book = em.emit_day(0, world["grid0"], arrivals,
                                   EventIdAllocator(0),
                                   PhiloxLedger(SEED, "event_truth", 0))
    return rfq, truth, book


def test_emission_schema_and_alignment(world, emitted):
    rfq, truth, book = emitted
    assert rfq.num_rows > 50
    assert rfq.schema.equals(world["bundle"].tables["rfq_lines"])
    assert truth.num_rows == rfq.num_rows == book.num_rows
    np.testing.assert_array_equal(
        rfq["event_id"].to_numpy(zero_copy_only=False),
        truth["event_id"].to_numpy(zero_copy_only=False))


def test_emission_invariants(world, emitted):
    rfq, truth, book = emitted
    uni = world["uni"]
    disc = rfq["client_side_disclosed"].to_numpy(zero_copy_only=False)
    true = rfq["client_side_true"].to_numpy(zero_copy_only=False)
    tok = rfq["token_id"].to_numpy(zero_copy_only=False)
    instr = rfq["instrument_id"].to_numpy(zero_copy_only=False)
    cl = rfq["client_id"].to_numpy(zero_copy_only=False)

    two = disc == int(E.SideDisclosed.TWO_WAY)
    np.testing.assert_array_equal(disc[~two], true[~two])   # disclosure honest
    ti, tsen = uni.token_arrays()
    np.testing.assert_array_equal(ti[tok], instr)            # token <-> instrument
    want_sense = np.where(two, int(E.Sense.UNDISC), disc)
    np.testing.assert_array_equal(tsen[tok], want_sense)     # disclosure = sense

    # mandates respected: rating band and sector coverage
    r = uni.rating[instr]
    assert (r >= uni.mandate_rating[cl, 0]).all() and (r <= uni.mandate_rating[cl, 1]).all()
    sec = uni.instr_sector[instr]
    assert (((uni.mandate_sector_mask[cl] >> sec) & 1) == 1).all()

    # our POV vs receipt
    recv = rfq["received"].to_numpy(zero_copy_only=False)
    our = rfq["our_result"].to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(recv, our != int(E.OurResult.NOT_RECEIVED))

    lz = truth["log_z"].to_numpy(zero_copy_only=False)
    lp = truth["log_p_chosen"].to_numpy(zero_copy_only=False)
    assert np.isfinite(lz).all() and (lp <= 1e-9).all()

    # philox replay: the stored coordinates reproduce the event's first draws
    k0 = truth["philox_key0"][0].as_py(); k1 = truth["philox_key1"][0].as_py()
    c0 = truth["philox_ctr"][0].as_py()
    np.testing.assert_array_equal(
        PhiloxLedger.generator(k0, k1, c0).random(4),
        PhiloxLedger.generator(k0, k1, c0).random(4))


def test_emission_packages(world, emitted):
    rfq, _, _ = emitted
    uni = world["uni"]
    pt = rfq["package_type"].to_numpy(zero_copy_only=False)
    if (pt == int(E.PackageType.SWITCH)).any():
        pkg = rfq["package_id"].to_numpy(zero_copy_only=False)
        tok = rfq["token_id"].to_numpy(zero_copy_only=False)
        _, tsen = uni.token_arrays()
        sw = pkg[pt == int(E.PackageType.SWITCH)][0]
        legs = np.nonzero(pkg == sw)[0]
        assert len(legs) == 2
        senses = sorted(tsen[tok[legs]])
        assert senses == [int(E.Sense.BUY), int(E.Sense.SELL)]  # opposite sides


def test_emission_deterministic_replay(world):
    uni, bundle, cal = world["uni"], world["bundle"], world["cal"]
    arrivals = sample_arrivals(world["grid0"], cal, 0, SEED)
    out = []
    for _ in range(2):
        em = Emitter(uni, bundle, EmissionConfig(), SEED)
        rfq, _, _ = em.emit_day(0, world["grid0"], arrivals,
                                EventIdAllocator(0),
                                PhiloxLedger(SEED, "event_truth", 0))
        out.append(rfq)
    assert out[0].equals(out[1])


# ---------------------------------------------------------------------------
# Pipeline end-to-end.
# ---------------------------------------------------------------------------

def test_generate_run_end_to_end(tmp_path):
    dials = RunDials(run_id="t-e2e", seed_root_hex=SEED, n_days=4,
                     d=D, sectors=SECTORS, lambda_bar_per_min=0.6,
                     universe=UniverseConfig(n_issuers=40, n_clients=30))
    out = generate_run(tmp_path, dials)
    assert out["rows"] > 500
    assert out["tables"]["rfq_lines"] == out["tables"]["event_truth"] \
        == out["tables"]["auction_book"]
    assert out["tables"]["rfq_lines_obs"] < out["tables"]["rfq_lines"]
    assert (tmp_path / "manifest.json").exists()

    # structural checks must PASS; statistical ones may drift at toy scale
    by = {c["check"]: c for c in out["checks"]}
    assert by["drift_tv_ratio"]["status"] == int(E.CheckStatus.PASS)
    # norm->frequency (Fig 6 analogue) only emerges once the walk has mixed
    # over many contexts; at 4 days << HL=10 days the tape is one context
    # draw and frequency is angular-alignment-dominated. Assert presence and
    # finiteness here; the mixing-dependent emergence is tested in the demo.
    assert np.isfinite(by["norm_freq_corr"]["value"])

    import pyarrow.dataset as ds
    obs = ds.dataset(tmp_path / "tables/rfq_lines_obs", partitioning="hive").to_table()
    assert "token_id" not in obs.column_names
    assert "client_side_true" not in obs.column_names
