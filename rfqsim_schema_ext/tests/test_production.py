"""gpurand / emission_batch / production: RNG known-answer + statistics,
batch-engine invariants and determinism, distributional agreement with the
reference engine, and the two-shard production run end to end."""
import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

from rfqsim.schema import enums as E
from rfqsim.schema.emission import EmissionConfig, Emitter
from rfqsim.schema.emission_batch import BatchEmitter
from rfqsim.schema.gpurand import (derive_key32, masked_softmax_draw, normals,
                                   philox4x32, row_categorical, uniforms)
from rfqsim.schema.intensity import sample_arrivals
from rfqsim.schema.pipeline import RunDials
from rfqsim.schema.population import UniverseConfig, build_universe, business_calendar
from rfqsim.schema.production import generate_run_production
from rfqsim.schema.state import LatentStateConfig, LatentStateEngine
from rfqsim.schema.tables import SchemaConfig, build_schemas
from rfqsim.schema.vocab import FeatureSpec
from rfqsim.schema.writer import EventIdAllocator, PhiloxLedger

SEED = "0123456789abcdef0123456789abcdef"
SECTORS = ("Energy", "TMT", "Healthcare", "Utilities")
D = 16


# ---------------------------------------------------------------------------
# gpurand.
# ---------------------------------------------------------------------------

def test_philox_known_answer():
    out = philox4x32((0, 0), np.array([0], np.uint64))[0]
    assert [int(v) for v in out] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]


def test_uniforms_and_normals():
    key = derive_key32(SEED, "t")
    u = uniforms(key, np.arange(0, 4000, 64, dtype=np.uint64), 32)
    assert u.shape == (63, 32) and (u > 0).all() and (u < 1).all()
    # replay + non-overlap
    np.testing.assert_array_equal(
        u[0], uniforms(key, np.array([0], np.uint64), 32)[0])
    assert not np.array_equal(u[0], u[1])
    z = normals(key, np.arange(0, 64000, 64, dtype=np.uint64), 8)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_row_categorical_and_masked_softmax():
    key = derive_key32(SEED, "t2")
    E_, m = 20000, 5
    w = np.tile(np.array([1.0, 2.0, 3.0, 0.0, 4.0]), (E_, 1))
    u = uniforms(key, np.arange(E_, dtype=np.uint64) * 4, 1)[:, 0]
    pick = row_categorical(u, w)
    freq = np.bincount(pick, minlength=m) / E_
    np.testing.assert_allclose(freq, [0.1, 0.2, 0.3, 0.0, 0.4], atol=0.01)
    # empty row -> -1
    assert row_categorical(np.array([0.5]), np.zeros((1, 3)))[0] == -1

    logits = np.tile(np.log(np.array([1.0, 2.0, 3.0, 99.0, 4.0])), (E_, 1))
    mask = np.tile(np.array([True, True, True, False, True]), (E_, 1))
    ch, lz, lc, nc = masked_softmax_draw(u, logits, mask)
    assert (ch != 3).all() and (nc == 4).all()
    np.testing.assert_allclose(np.exp(lz), 10.0, rtol=1e-9)   # masked partition
    freq = np.bincount(ch, minlength=m) / E_
    np.testing.assert_allclose(freq[[0, 1, 2, 4]], [0.1, 0.2, 0.3, 0.4], atol=0.01)


# ---------------------------------------------------------------------------
# Batch emitter vs reference.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    spec = FeatureSpec(sectors=SECTORS)
    cfg = SchemaConfig(d=D, p=spec.dim, n_sectors=4, run_id="t-batch")
    bundle = build_schemas(cfg)
    cal = business_calendar(dt.date(2026, 1, 5), 2, session_minutes=240)
    uni = build_universe(UniverseConfig(n_issuers=40, n_clients=30),
                        spec, cal, SEED, D)
    scfg = LatentStateConfig(d=D, n_sectors=4, lambda_bar_sector=(2.0,) * 4,
                             theta_half_life_steps=300.0)
    eng = LatentStateEngine(scfg, bundle, SEED, cal)
    grid0, _ = eng.simulate_day(0)
    arrivals = sample_arrivals(grid0, cal, 0, SEED)
    return dict(cfg=cfg, bundle=bundle, cal=cal, uni=uni,
                grid0=grid0, arrivals=arrivals)


@pytest.fixture(scope="module")
def batch_out(world):
    em = BatchEmitter(world["uni"], world["bundle"], EmissionConfig(), SEED)
    return em.emit_day_batch(0, world["grid0"], world["arrivals"],
                             EventIdAllocator(0))


def test_batch_schema_and_invariants(world, batch_out):
    rfq, truth, book = batch_out
    uni = world["uni"]
    assert rfq.num_rows > 100
    assert rfq.schema.equals(world["bundle"].tables["rfq_lines"])
    assert truth.num_rows == rfq.num_rows == book.num_rows

    disc = rfq["client_side_disclosed"].to_numpy(zero_copy_only=False)
    true = rfq["client_side_true"].to_numpy(zero_copy_only=False)
    tok = rfq["token_id"].to_numpy(zero_copy_only=False)
    instr = rfq["instrument_id"].to_numpy(zero_copy_only=False)
    cl = rfq["client_id"].to_numpy(zero_copy_only=False)
    two = disc == int(E.SideDisclosed.TWO_WAY)
    np.testing.assert_array_equal(disc[~two], true[~two])
    ti, tsen = uni.token_arrays()
    np.testing.assert_array_equal(ti[tok], instr)
    np.testing.assert_array_equal(
        tsen[tok], np.where(two, int(E.Sense.UNDISC), disc))

    r = uni.rating[instr]
    assert (r >= uni.mandate_rating[cl, 0]).all()
    assert (r <= uni.mandate_rating[cl, 1]).all()
    sec = uni.instr_sector[instr]
    assert (((uni.mandate_sector_mask[cl] >> sec) & 1) == 1).all()

    recv = rfq["received"].to_numpy(zero_copy_only=False)
    our = rfq["our_result"].to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(recv, our != int(E.OurResult.NOT_RECEIVED))
    won = our == int(E.OurResult.WON)
    assert rfq["exec_sprd_bp"].null_count < rfq.num_rows
    exec_valid = np.asarray(rfq["exec_sprd_bp"].is_valid())
    eo = rfq["enquiry_outcome"].to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(exec_valid,
                                  eo == int(E.EnquiryOutcome.TRADED))
    assert won.sum() > 0

    lz = truth["log_z"].to_numpy(zero_copy_only=False)
    lp = truth["log_p_chosen"].to_numpy(zero_copy_only=False)
    assert np.isfinite(lz).all() and (lp <= 1e-9).all()

    # packages: legs contiguous, switch legs are opposite signed senses
    pkg = rfq["package_id"].to_numpy(zero_copy_only=False)
    ln = rfq["line_no"].to_numpy(zero_copy_only=False)
    pt = rfq["package_type"].to_numpy(zero_copy_only=False)
    for p in np.unique(pkg[pt == int(E.PackageType.SWITCH)])[:5]:
        legs = np.nonzero(pkg == p)[0]
        assert list(ln[legs]) == list(range(len(legs)))
        assert sorted(tsen[tok[legs]]) == [int(E.Sense.BUY), int(E.Sense.SELL)]
        assert len(set(instr[legs])) == len(legs)


def test_batch_replay_from_stored_coordinates(batch_out):
    _, truth, _ = batch_out
    k0 = truth["philox_key0"][0].as_py()
    k1 = truth["philox_key1"][0].as_py()
    c0 = truth["philox_ctr"][0].as_py()
    a = uniforms((k0, k1), np.array([c0], np.uint64), 8)
    b = uniforms((k0, k1), np.array([c0], np.uint64), 8)
    np.testing.assert_array_equal(a, b)


def test_batch_deterministic(world, batch_out):
    em = BatchEmitter(world["uni"], world["bundle"], EmissionConfig(), SEED)
    rfq2, truth2, book2 = em.emit_day_batch(0, world["grid0"],
                                            world["arrivals"],
                                            EventIdAllocator(0))
    assert batch_out[0].equals(rfq2)
    assert batch_out[1].equals(truth2)
    # auction book pads px with NaN, which IEEE-compares unequal under
    # Table.equals; compare NaN-aware column by column
    for col in batch_out[2].column_names:
        a = batch_out[2][col].combine_chunks()
        b = book2[col].combine_chunks()
        if col == "px_sprd_bp":
            np.testing.assert_array_equal(np.asarray(a.flatten()),
                                          np.asarray(b.flatten()))
        else:
            assert a.equals(b), col


def test_batch_agrees_with_reference_distributionally(world, batch_out):
    ref_em = Emitter(world["uni"], world["bundle"], EmissionConfig(), SEED)
    ref, _, _ = ref_em.emit_day(0, world["grid0"], world["arrivals"],
                                EventIdAllocator(0),
                                PhiloxLedger(SEED, "event_truth", 0))
    bat = batch_out[0]

    def stats(t):
        eo = t["enquiry_outcome"].to_numpy(zero_copy_only=False)
        disc = t["client_side_disclosed"].to_numpy(zero_copy_only=False)
        recv = t["received"].to_numpy(zero_copy_only=False)
        return dict(
            n=t.num_rows,
            traded=float(np.mean(eo == int(E.EnquiryOutcome.TRADED))),
            two_way=float(np.mean(disc == int(E.SideDisclosed.TWO_WAY))),
            received=float(np.mean(recv)),
            nd=float(t["n_dealers"].to_numpy(zero_copy_only=False).mean()),
        )

    a, b = stats(ref), stats(bat)
    assert abs(a["n"] - b["n"]) / a["n"] < 0.05        # same arrivals, few skips
    for k, tol in (("traded", 0.06), ("two_way", 0.06),
                   ("received", 0.06), ("nd", 0.3)):
        assert abs(a[k] - b[k]) < tol, (k, a[k], b[k])


# ---------------------------------------------------------------------------
# Production driver.
# ---------------------------------------------------------------------------

def test_production_two_shards(tmp_path):
    dials = RunDials(run_id="t-prod", seed_root_hex=SEED, n_days=4, d=D,
                     sectors=SECTORS, lambda_bar_per_min=0.6,
                     universe=UniverseConfig(n_issuers=40, n_clients=30))
    out = generate_run_production(tmp_path, dials, n_shards=2)
    assert out["rows"] > 400
    assert out["tables"]["rfq_lines"] == out["tables"]["event_truth"]

    import pyarrow.dataset as ds
    canon = ds.dataset(tmp_path / "tables/rfq_lines", partitioning="hive").to_table()
    # both shards contributed, disjoint event-id prefixes, all four days present
    shard = canon["event_id"].to_numpy(zero_copy_only=False) >> 48
    assert set(shard.tolist()) == {0, 1}
    assert len(np.unique(canon["trade_date"].to_numpy(zero_copy_only=False))) == 4

    obs = ds.dataset(tmp_path / "tables/rfq_lines_obs", partitioning="hive").to_table()
    assert "token_id" not in obs.column_names
    assert (tmp_path / "manifest.json").exists()
