"""The leakage audit: projection semantics under multiple revelation policies,
tripwires for canonical-tape invariant violations, and tamper detection on the
observable plane."""
import datetime as dt

import pyarrow as pa
import pytest

from _toy import US, make_canonical
from rfqsim.schema import enums as E
from rfqsim.schema.enums import CoverReveal
from rfqsim.schema.projection import (DEFAULT_POLICY, LeakageError,
                                      ProjectionError, RevelationPolicy,
                                      audit_observable, project_observable)
from rfqsim.schema.tables import observable_rfq_schema

BUY, SELL = int(E.SideTrue.BUY), int(E.SideTrue.SELL)


@pytest.fixture
def canon(cfg):
    return make_canonical(cfg)


def _col(t, name):
    return t[name].to_pylist()


# ---------------------------------------------------------------------------
# Default policy semantics. Filtered row order: r0..r7, r10 (9 rows).
# ---------------------------------------------------------------------------

def test_projection_default_policy(canon, cfg):
    obs = project_observable(canon, cfg)

    assert obs.num_rows == 9  # r8 (not received) and r9 (off panel) dropped
    assert obs.schema.names == observable_rfq_schema(cfg).names
    for latent in ("token_id", "client_side_true", "our_in_panel", "received", "grid_idx"):
        assert latent not in obs.column_names

    # side_revealed: disclosed rows always; two-ways only WON/COVER by default
    assert _col(obs, "side_revealed") == [BUY, SELL, BUY, SELL, None, None, None, None, BUY]

    ts = _col(obs, "ts")
    ots = _col(obs, "outcome_ts")
    rev = _col(obs, "side_reveal_ts")
    for i in (0, 1, 8):          # disclosed: revealed at enquiry time
        assert rev[i] == ts[i]
    for i in (2, 3):             # revealed two-ways: at outcome time (+0 lag)
        assert rev[i] == ots[i]
    for i in (4, 5, 6, 7):
        assert rev[i] is None

    # WINNER_ONLY cover; exec only where we won
    exec_px = _col(obs, "exec_px")
    cover_px = _col(obs, "cover_px")
    assert [i for i, v in enumerate(exec_px) if v is not None] == [0, 2]
    assert [i for i, v in enumerate(cover_px) if v is not None] == [0, 2]

    audit_observable(obs, cfg, DEFAULT_POLICY)  # must not raise


def test_projection_policy_variants(canon, cfg):
    # revelation lag (timestamps come back as datetimes)
    import datetime as dt
    lagged = RevelationPolicy(side_reveal_lag_us=5 * US)
    obs = project_observable(canon, cfg, lagged)
    rev = _col(obs, "side_reveal_ts")
    ots = _col(obs, "outcome_ts")
    assert rev[2] - ots[2] == dt.timedelta(seconds=5)
    assert rev[3] - ots[3] == dt.timedelta(seconds=5)
    assert rev[0] == _col(obs, "ts")[0]  # disclosed rows unaffected by lag
    audit_observable(obs, cfg, lagged)

    # cover to all quoters: WON/COVER/LOST see it (rows 0..4), NO_QUOTE doesn't
    allq = RevelationPolicy(cover_view=CoverReveal.ALL_QUOTERS)
    obs = project_observable(canon, cfg, allq)
    assert [i for i, v in enumerate(_col(obs, "cover_px")) if v is not None] == [0, 1, 2, 3, 4]
    audit_observable(obs, cfg, allq)

    # public prints: every traded row's exec is visible, quoting or not
    tape = RevelationPolicy(exec_px_on_loss=True)
    obs = project_observable(canon, cfg, tape)
    assert [i for i, v in enumerate(_col(obs, "exec_px")) if v is not None] == [0, 1, 2, 3, 4, 5]
    audit_observable(obs, cfg, tape)

    # losing a two-way reveals the side (MKAX-ish venue)
    loose = RevelationPolicy(reveal_on_lost=True)
    obs = project_observable(canon, cfg, loose)
    assert _col(obs, "side_revealed")[4] == BUY
    audit_observable(obs, cfg, loose)


# ---------------------------------------------------------------------------
# Canonical-tape tripwires.
# ---------------------------------------------------------------------------

def _replace(t: pa.Table, name: str, values, typ) -> pa.Table:
    i = t.schema.get_field_index(name)
    return t.set_column(i, t.schema.field(name), pa.array(values, typ))


def test_strict_rejects_disclosed_side_mismatch(canon, cfg):
    vals = _col(canon, "client_side_true")
    vals[0] = SELL  # r0 disclosed BUY but "true" SELL: generator bug
    bad = _replace(canon, "client_side_true", vals, pa.int8())
    with pytest.raises(ProjectionError):
        project_observable(bad, cfg)


def test_strict_rejects_received_off_panel(canon, cfg):
    vals = _col(canon, "our_in_panel")
    vals[0] = False  # received=True but our_in_panel=False
    bad = _replace(canon, "our_in_panel", vals, pa.bool_())
    with pytest.raises(ProjectionError):
        project_observable(bad, cfg)


def test_unknown_canonical_column_rejected(canon, cfg):
    sneaky = canon.append_column("sneaky", pa.array([1] * canon.num_rows, pa.int8()))
    with pytest.raises(ProjectionError):
        project_observable(sneaky, cfg)


# ---------------------------------------------------------------------------
# Tamper detection on the observable plane.
# ---------------------------------------------------------------------------

def test_audit_catches_latent_column(canon, cfg):
    obs = project_observable(canon, cfg)
    tampered = obs.append_column("token_id", pa.array([0] * obs.num_rows, pa.uint32()))
    with pytest.raises(LeakageError):
        audit_observable(tampered, cfg)


def test_audit_catches_missing_column(canon, cfg):
    obs = project_observable(canon, cfg)
    with pytest.raises(LeakageError):
        audit_observable(obs.drop_columns(["cover_px"]), cfg)


def test_audit_catches_policy_violation(canon, cfg):
    # project under a leaky policy, audit against the strict default
    obs = project_observable(canon, cfg, RevelationPolicy(reveal_on_lost=True,
                                                          cover_view=CoverReveal.ALL_QUOTERS))
    with pytest.raises(LeakageError):
        audit_observable(obs, cfg, DEFAULT_POLICY)
