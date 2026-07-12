"""vocab.py: feature map determinism and roll-down, vocabulary lifecycle,
tokenization parity across the two planes, sentence grouping, and the
same/cross-client pair counting."""
import numpy as np
import pyarrow as pa
import pytest

from _toy import date_days, make_canonical, ts_us
from rfqsim.schema import enums as E
from rfqsim.schema.projection import project_observable
from rfqsim.schema.tables import SchemaConfig, build_schemas
from rfqsim.schema.vocab import (FeatureSpec, Sentence, TokenVocab, VocabError,
                                 sense_from_disclosed, sentences,
                                 tokenize_table, window_pair_counts)


# ---------------------------------------------------------------------------
# FeatureSpec.
# ---------------------------------------------------------------------------

@pytest.fixture
def spec():
    return FeatureSpec(sectors=("Energy", "TMT", "Healthcare", "Utilities"))


def _panel(n=1):
    return dict(
        issue_days=np.full(n, date_days(2024, 1, 5)),
        maturity_days=np.full(n, date_days(2031, 1, 5)),
        coupon=np.full(n, 7.25), amt_issued=np.full(n, 500_000_000),
        rating=np.full(n, 15),                      # B
        seniority=np.full(n, int(E.Seniority.SR_UNSECURED)),
        is_144a=np.ones(n, bool), is_financial=np.zeros(n, bool),
        sector_id=np.zeros(n, np.int64), is_otr=np.ones(n, bool),
        index_mask=np.full(n, 5, np.uint32),
    )


def test_feature_layout_and_values(spec):
    assert spec.dim == 11 + 7 + 4 == len(spec.names)
    cfg = SchemaConfig(d=8, p=spec.dim, n_sectors=4, run_id="t")
    spec.assert_matches(cfg)  # must not raise
    with pytest.raises(VocabError):
        spec.assert_matches(SchemaConfig(d=8, p=spec.dim + 1, n_sectors=4))

    x = spec.compute(date_days(2026, 1, 5), **_panel())
    assert x.dtype == np.float32 and x.shape == (1, spec.dim)
    nm = list(spec.names)
    assert np.isclose(x[0, nm.index("log1p_ttm")], np.log1p(5.0), atol=2e-2)
    assert np.isclose(x[0, nm.index("log1p_age")], np.log1p(2.0), atol=2e-2)
    assert x[0, nm.index("is_hy")] == 1.0          # B is HY
    assert x[0, nm.index("in_index")] == 1.0
    assert x[0, nm.index("sector_Energy")] == 1.0 and x[0, nm.index("sector_TMT")] == 0.0


def test_rolldown_moves_only_time_features(spec):
    nm = list(spec.names)
    t0, t1 = date_days(2026, 1, 5), date_days(2027, 1, 5)
    x0 = spec.compute(t0, **_panel())[0]
    x1 = spec.compute(t1, **_panel())[0]
    assert x1[nm.index("log1p_ttm")] < x0[nm.index("log1p_ttm")]
    assert x1[nm.index("log1p_age")] > x0[nm.index("log1p_age")]
    time_driven = {i for i, n in enumerate(nm)
                   if n.startswith(("log1p_", "tenor_rbf_"))}
    static = [i for i in range(spec.dim) if i not in time_driven]
    np.testing.assert_array_equal(x0[static], x1[static])  # roll-down is x-only, and only there


# ---------------------------------------------------------------------------
# TokenVocab.
# ---------------------------------------------------------------------------

def test_vocab_lifecycle_and_roundtrip(bundle):
    v = TokenVocab()
    for i in (100, 101, 102):
        v.register_instrument(i, born_ts_us=100)
    v.retire_instrument(101, retired_ts_us=500)

    assert len(v) == 9 and v.n_instruments == 3
    assert v.token(100, E.Sense.SELL) == 1
    assert v.instrument_sense(v.token(102, E.Sense.UNDISC)) == (102, int(E.Sense.UNDISC))
    with pytest.raises(VocabError):
        v.register_instrument(100, 0)
    with pytest.raises(VocabError):
        v.tokens_for(np.array([999]), np.array([0]))

    # vectorized == scalar
    ids = np.array([102, 100, 101])
    ss = np.array([0, 2, 1])
    np.testing.assert_array_equal(
        v.tokens_for(ids, ss),
        np.array([v.token(i, s) for i, s in zip(ids, ss)], dtype=np.uint32))

    assert len(v.alive_token_ids(200)) == 9
    alive_after = v.alive_token_ids(600)
    assert len(alive_after) == 6
    assert v.token(101, E.Sense.BUY) not in alive_after

    # arrow round-trip preserves ids, birth and retirement
    tm = v.to_arrow(bundle.tables["token_map"])
    w = TokenVocab.from_arrow(tm)
    assert len(w) == len(v)
    assert w.token(101, E.Sense.SELL) == v.token(101, E.Sense.SELL)
    assert len(w.alive_token_ids(600)) == 6


# ---------------------------------------------------------------------------
# Tokenization parity across the planes.
# ---------------------------------------------------------------------------

def test_tokenization_parity(cfg):
    canon = make_canonical(cfg)
    v = TokenVocab()
    for i in np.unique(canon["instrument_id"].to_numpy(zero_copy_only=False)):
        v.register_instrument(int(i), born_ts_us=0)

    # canonical tokens: sense IS the disclosed code (UNDISC for two-ways) --
    # rewrite the toy's ad-hoc token_id column with vocab-issued ids
    canon_tok = tokenize_table(v, canon)
    idx = canon.schema.get_field_index("token_id")
    canon = canon.set_column(idx, canon.schema.field("token_id"),
                             pa.array(canon_tok, pa.uint32()))

    # two-ways with opposite TRUE sides share the UNDISC token: the disguise
    # keeps its own company
    r2, r3 = 2, 3
    assert canon["client_side_true"][r2].as_py() != canon["client_side_true"][r3].as_py()
    s2 = v.instrument_sense(int(canon_tok[r2]))[1]
    s3 = v.instrument_sense(int(canon_tok[r3]))[1]
    assert s2 == s3 == int(E.Sense.UNDISC)

    # observable tokenization reproduces the latent token_id on received rows
    obs = project_observable(canon, cfg)
    obs_tok = tokenize_table(v, obs)
    received = canon.filter(canon["received"])
    np.testing.assert_array_equal(obs_tok, received["token_id"].to_numpy(zero_copy_only=False))


def test_sense_from_disclosed_guards():
    with pytest.raises(VocabError):
        sense_from_disclosed(np.array([3]))


# ---------------------------------------------------------------------------
# Corpus view.
# ---------------------------------------------------------------------------

def test_sentences_group_packages(cfg):
    canon = make_canonical(cfg)
    ss = sentences(canon)
    assert len(ss) == 10                       # 11 lines, one SWITCH pair
    assert all(isinstance(s, Sentence) for s in ss)
    assert ss[0].package_id == 0               # tape order
    switch = next(s for s in ss if s.package_id == 2)
    assert len(switch.token_ids) == 2 and switch.client_id == 7
    # line_no order within the sentence
    tok = canon["token_id"].to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(switch.token_ids, tok[2:4])


def test_sentences_reject_multi_author(cfg):
    canon = make_canonical(cfg)
    cli = canon["client_id"].to_pylist()
    cli[3] = 8                                  # second line of the switch
    i = canon.schema.get_field_index("client_id")
    bad = canon.set_column(i, canon.schema.field("client_id"), pa.array(cli, pa.uint32()))
    with pytest.raises(VocabError):
        sentences(bad)


def test_window_pair_counts_hand_check():
    day = date_days(2026, 1, 5)
    t0 = ts_us(2026, 1, 5, 14)
    tbl = pa.table({
        "token_id": pa.array([1, 2, 3, 1, 2, 2], pa.uint32()),
        "client_id": pa.array([7, 7, 8, 7, 7, 7], pa.uint32()),
        "trade_date": pa.array([day] * 6, pa.date32()),
        "ts": pa.array([t0 + k for k in range(6)], pa.timestamp("us", tz="UTC")),
        "event_id": pa.array(list(range(6)), pa.uint64()),
    })
    out = window_pair_counts(tbl, window=3)
    got = {(r["token_a"], r["token_b"], r["same_client"]): r["n"]
           for r in out.to_pylist()}
    assert got == {(1, 2, True): 3, (1, 3, False): 1, (2, 3, False): 1, (2, 2, True): 1}
