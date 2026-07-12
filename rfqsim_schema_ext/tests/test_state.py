"""state.py: calendar arithmetic, engine output conformance, the radial ->
(temperature, intensity) wiring, bitwise checkpoint replay, regime-path
tiling, and the drift-budget scan."""
import numpy as np
import pyarrow as pa
import pytest

from _toy import date_days, ts_us
from rfqsim.schema.state import (GridCalendar, LatentStateConfig,
                                 LatentStateEngine, RegimeConfig, StateError,
                                 realized_drift, run_and_write)
from rfqsim.schema.writer import ShardWriter

SEED = "feedfacefeedfacefeedfacefeedface"
STEP = 60_000_000


def make_cal(n_days=3, steps=30):
    import datetime as dt
    base = dt.datetime(2026, 1, 5, 14, tzinfo=dt.timezone.utc)
    stamps = [base + dt.timedelta(days=i) for i in range(n_days)]
    days = np.array([(s.date() - dt.date(1970, 1, 1)).days for s in stamps], np.int32)
    opens = np.array([int(s.timestamp() * 1_000_000) for s in stamps], np.int64)
    return GridCalendar(days=days, opens_us=opens, closes_us=opens + steps * STEP)


def make_cfg(**kw):
    base = dict(d=8, n_sectors=4, lambda_bar_sector=(2.0, 1.5, 1.0, 0.5),
                overnight_steps=10, theta_half_life_steps=30.0,
                r_half_life_steps=30.0, r_sigma=0.05)
    base.update(kw)
    return LatentStateConfig(**base)


# ---------------------------------------------------------------------------
# Calendar.
# ---------------------------------------------------------------------------

def test_grid_calendar_arithmetic():
    cal = make_cal(3, 30)
    assert list(cal.steps_per_day) == [30, 30, 30]
    assert cal.grid_idx(0, 0) == 0 and cal.grid_idx(2, 5) == 65
    assert cal.ts_of(1, 3) == int(cal.opens_us[1]) + 3 * STEP
    # session <-> wall round trip lands inside the right day
    s = cal.session_of_wall(cal.ts_of(2, 7))
    assert cal.wall_of_session(s) == cal.ts_of(2, 7)
    assert cal.session_of_wall(cal.ts_of(1, 0)) == 30 * STEP


def test_config_guards():
    with pytest.raises(StateError):
        make_cfg(lambda_bar_sector=(1.0,))          # wrong sector count
    with pytest.raises(StateError):
        make_cfg(theta_half_life_steps=0.0)
    with pytest.raises(StateError):
        RegimeConfig(Q=np.ones((4, 4)))              # rows don't sum to 0


# ---------------------------------------------------------------------------
# Engine output.
# ---------------------------------------------------------------------------

def test_day_output_conforms_and_wires_r(bundle):
    cfg = make_cfg()
    eng = LatentStateEngine(cfg, bundle, SEED, make_cal(2, 30))
    t0, _ = eng.simulate_day(0)
    t1, _ = eng.simulate_day(1)

    sch = bundle.tables["context_grid"]
    assert t0.schema.equals(sch)
    assert t0.num_rows == 30

    # grid_idx contiguous across days
    gi = t0["grid_idx"].to_pylist() + t1["grid_idx"].to_pylist()
    assert gi == list(range(60))

    # step_l1 null exactly at each open
    l1 = t0["step_l1"].to_pylist() + t1["step_l1"].to_pylist()
    assert l1[0] is None and l1[30] is None
    assert all(v is not None and v > 0 for i, v in enumerate(l1) if i not in (0, 30))

    # |c| == r, and temperature / lambda are the declared functions of r
    for tbl in (t0, t1):
        c = np.asarray(tbl["c"].combine_chunks().flatten()).reshape(-1, cfg.d)
        r = tbl["r"].to_numpy(zero_copy_only=False)
        np.testing.assert_allclose(np.linalg.norm(c, axis=1), r, rtol=2e-5)

        temp = tbl["temperature"].to_numpy(zero_copy_only=False)
        np.testing.assert_allclose(
            temp, np.maximum(cfg.temp_min, cfg.temp0 * (r / cfg.r0) ** -cfg.temp_gamma),
            rtol=2e-5)

        lam = np.asarray(tbl["lambda_sector"].combine_chunks().flatten()).reshape(-1, cfg.n_sectors)
        reg = np.asarray(tbl["regime"].combine_chunks().flatten()).reshape(-1, cfg.n_sectors)
        m = np.asarray(cfg.regime.m)
        want = np.asarray(cfg.lambda_bar_sector) * m[reg] * ((r / cfg.r0) ** cfg.lambda_gamma)[:, None]
        np.testing.assert_allclose(lam, want, rtol=2e-5)

    with pytest.raises(StateError):
        eng.simulate_day(5)  # out of order


# ---------------------------------------------------------------------------
# Checkpoint replay is bitwise.
# ---------------------------------------------------------------------------

def test_checkpoint_replay_is_bitwise(bundle):
    cfg = make_cfg()
    cal = make_cal(3, 30)

    a = LatentStateEngine(cfg, bundle, SEED, cal)
    a.simulate_day(0); a.simulate_day(1)
    day2_a, reg_a = a.simulate_day(2)

    b = LatentStateEngine(cfg, bundle, SEED, cal)
    b.simulate_day(0); b.simulate_day(1)
    restored = LatentStateEngine.restore(cfg, bundle, SEED, cal, b.state_dict())
    day2_b, reg_b = restored.simulate_day(2)

    assert day2_a.equals(day2_b)
    assert reg_a == reg_b


def test_checkpoint_survives_npz(tmp_path, bundle):
    cfg = make_cfg()
    cal = make_cal(3, 30)
    a = LatentStateEngine(cfg, bundle, SEED, cal)
    a.simulate_day(0)
    np.savez(tmp_path / "ck.npz", **a.state_dict())
    day1_a, _ = a.simulate_day(1)

    with np.load(tmp_path / "ck.npz") as f:
        state = {k: f[k] for k in f.files}
    r = LatentStateEngine.restore(cfg, bundle, SEED, cal, state)
    day1_b, _ = r.simulate_day(1)
    assert day1_a.equals(day1_b)


# ---------------------------------------------------------------------------
# Regime path tiling.
# ---------------------------------------------------------------------------

def test_regime_path_tiles_the_session_clock(bundle):
    # fast chains so every sector switches many times inside 3 short sessions
    fast = RegimeConfig.bidimensional(act_up=40, act_down=40,
                                      stress_on=20, stress_off=20)
    cfg = make_cfg(regime=fast)
    eng = LatentStateEngine(cfg, bundle, SEED, make_cal(3, 30))
    rows = []
    for d in range(3):
        _, closed = eng.simulate_day(d)
        rows += closed
    rows += eng.open_sojourn_rows()

    by_sector = {}
    for r in rows:
        by_sector.setdefault(r["sector_id"], []).append(r)
    assert set(by_sector) == {0, 1, 2, 3}
    for s, rs in by_sector.items():
        rs.sort(key=lambda r: r["sojourn_no"])
        assert [r["sojourn_no"] for r in rs] == list(range(len(rs)))
        assert len(rs) > 3                                # chains actually switched
        for a, b in zip(rs, rs[1:]):
            assert a["t_end"] == b["t_start"]             # tiling, no gaps
            assert a["state"] != b["state"]               # CTMC never self-jumps
            assert 0 <= a["state"] <= 3
        assert rs[-1]["t_end"] is None                    # open at run end


# ---------------------------------------------------------------------------
# Driver + drift audit.
# ---------------------------------------------------------------------------

def test_run_and_write_and_drift(tmp_path, bundle):
    cfg = make_cfg()
    cal = make_cal(3, 30)
    eng = LatentStateEngine(cfg, bundle, SEED, cal)
    w = ShardWriter(tmp_path, bundle, shard_id=0)
    state = run_and_write(eng, w, 0, 3, emit_open_sojourns=True)
    recs = w.close()

    assert int(np.asarray(state["day_next"]).ravel()[0]) == 3
    assert (tmp_path / "tables/context_grid/trade_month=2026-01").is_dir()
    assert any(r.table == "regime_path" for r in recs)
    assert any(r.table == "_walk_checkpoint" for r in recs)

    import pyarrow.dataset as ds
    grid = ds.dataset(tmp_path / "tables/context_grid", partitioning="hive").to_table()
    assert grid.num_rows == 90

    rep = realized_drift(grid, window_steps=10)
    assert rep["n_windows"] == 3 * (29 - 10 + 1)    # 29 non-null steps per day
    assert 0 < rep["mean"] <= rep["p99"] <= rep["max"] < np.inf

    # the dial does what it claims: slower rotation => smaller drift
    calm = make_cfg(theta_half_life_steps=3000.0, r_sigma=0.0)
    eng2 = LatentStateEngine(calm, bundle, SEED, cal)
    g2 = pa.concat_tables([eng2.simulate_day(d)[0] for d in range(3)])
    assert realized_drift(g2, 10)["mean"] < rep["mean"]


def test_displacement_predictor_and_window_rule(bundle):
    from rfqsim.schema.state import max_window_steps, predicted_displacement
    cfg = make_cfg(theta_half_life_steps=200.0, r_sigma=0.0)

    # closed form vs simulated l2 displacement at W = 20
    eng = LatentStateEngine(cfg, bundle, SEED, make_cal(30, 30))
    cs = []
    for d in range(30):
        t, _ = eng.simulate_day(d)
        cs.append(np.asarray(t["c"].combine_chunks().flatten()).reshape(-1, cfg.d))
    c = np.concatenate(cs)
    W = 20
    emp = float(np.mean(np.linalg.norm(c[W:] - c[:-W], axis=1)))
    pred = predicted_displacement(cfg, W)
    assert abs(emp - pred["l2"]) / pred["l2"] < 0.25       # E-level agreement
    assert pred["l1"] == pytest.approx(pred["l2"] * np.sqrt(2 * cfg.d / np.pi))

    # inverter consistency: within budget at W*, over budget at W*+1
    for budget in (0.05, 0.2, 0.5):
        w_star = max_window_steps(cfg, budget, norm="l2")
        assert predicted_displacement(cfg, w_star)["l2"] <= budget
        assert predicted_displacement(cfg, w_star + 1)["l2"] > budget
    assert max_window_steps(cfg, 10.0, norm="l2") is None  # beyond stationary limit
