"""Torch-free core tests, run against the smoke dataset built by the simulator."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------

def test_contract_reads_dataset(ds):
    assert ds.run_id
    assert ds.schema_version is not None
    tm = ds.token_map()
    assert {"token_id", "instrument_id", "sense"} <= set(tm.column_names)
    # the one allowed simulator import: shared enum registries
    assert ds.enums.Sense.BUY == 0 and ds.enums.Sense.UNDISC == 2
    assert ds.n_rows("rfq_lines") == ds.scan("rfq_lines").num_rows


# ---------------------------------------------------------------------------
# vocab
# ---------------------------------------------------------------------------

def test_vocab_disjoint_ranges_and_mappers(ds, vocab):
    # every id maps to exactly one token and back
    assert len(vocab.id_to_token) == vocab.size
    assert len(vocab.token_to_id) == vocab.size
    # families are contiguous and non-overlapping, covering the whole space
    spans = sorted((f.offset, f.offset + f.size) for f in vocab.families.values())
    cur = vocab._n_special
    for lo, hi in spans:
        assert lo == cur, (lo, cur)
        cur = hi
    assert cur == vocab.size

    tm = ds.token_map()
    raw = tm.column("token_id").to_numpy()
    g = vocab.tok_global(raw)
    assert vocab.is_family(g, "TOK").all()
    # round-trip global -> raw
    np.testing.assert_array_equal(vocab.tok_raw_of_global(g), raw.astype(np.int64))
    # unknown raw ids fall to <unk>
    assert vocab.tok_global(np.array([10**9]))[0] == vocab.unk_id


def test_vocab_save_load(tmp_path, vocab):
    from rfqfm import FmVocab
    p = tmp_path / "vocab.json"
    vocab.save(p)
    v2 = FmVocab.load(p)
    assert v2.size == vocab.size
    assert v2.families.keys() == vocab.families.keys()
    r = np.array([vocab._tok_raw[0], vocab._tok_raw[-1]])
    np.testing.assert_array_equal(v2.tok_global(r), vocab.tok_global(r))


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_bond_features(ds):
    from rfqfm import build_bond_features
    bf = build_bond_features(ds)
    assert bf.X.shape == (bf.n_instruments, bf.d_numeric)
    assert np.isfinite(bf.X).all()
    # continuous columns standardized (is_144a left as 0/1)
    cont = [i for i, n in enumerate(bf.feature_names) if n != "is_144a"]
    assert abs(bf.X[:, cont].mean()) < 1e-4
    # token -> instrument row / sense agree with token_map
    tm = ds.token_map()
    tok = tm.column("token_id").to_numpy()
    sense = tm.column("sense").to_numpy()
    np.testing.assert_array_equal(bf.tok_sense[tok], sense.astype(np.int64))
    assert (bf.tok_instr_row[tok] >= 0).all()
    assert bf.n_sectors >= 1 and bf.n_seniorities >= 1


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_block_structure(lines, blocks, vocab):
    from rfqfm.tokenize import FAMILY
    n = lines.num_rows
    assert blocks.n_lines == n

    line_no = lines.column("line_no").to_numpy()
    # first legs get the full 7-token block; continuation legs get 3 (<leg> TOK SZ)
    assert (blocks.line_len[line_no == 0] == 7).all()
    if (line_no > 0).any():
        assert (blocks.line_len[line_no > 0] == 3).all()

    # anchors are TOK-family and there is exactly one per line
    assert len(blocks.anchor_pos) == n
    assert (blocks.family[blocks.anchor_pos] == FAMILY["TOK"]).all()

    # the anchor's global id maps back to the line's raw token_id
    raw = lines.column("token_id").to_numpy().astype(np.int64)
    got = vocab.tok_raw_of_global(blocks.stream[blocks.anchor_pos])
    np.testing.assert_array_equal(got, raw)

    # event_id at the anchor equals the line's event_id
    ev = lines.column("event_id").to_numpy().astype(np.uint64)
    np.testing.assert_array_equal(blocks.event_id[blocks.anchor_pos], ev)

    # first token of every first-leg block is <sep>; continuation is <leg>
    first_starts = blocks.line_start[line_no == 0]
    assert (blocks.stream[first_starts] == vocab.sep_id).all()
    if (line_no > 0).any():
        cont_starts = blocks.line_start[line_no > 0]
        assert (blocks.stream[cont_starts] == vocab.leg_id).all()


def test_tokenize_tdelta_first_is_zero_bucket(lines, blocks, vocab):
    # a client's earliest RFQ must land in the first-of-session TDLT bucket (0)
    from rfqfm.tokenize import FAMILY
    tdlt_mask = blocks.family == FAMILY["TDLT"]
    fam = vocab.families["TDLT"]
    local = blocks.stream[tdlt_mask] - fam.offset
    assert local.min() == 0 and local.max() <= fam.size - 1
    # at least one client's first RFQ exists -> bucket 0 present
    assert (local == 0).any()


# ---------------------------------------------------------------------------
# corpus + packed
# ---------------------------------------------------------------------------

def test_session_corpus_roundtrip(tmp_path, blocks, vocab):
    from rfqfm import build_sequences, pack
    from rfqfm.packed import PackedCorpus
    from rfqfm.tokenize import FAMILY

    seqs = build_sequences(blocks, vocab.cfg, "session")
    pc = pack(blocks, seqs, vocab, "session")

    # every sequence fits the context and is bos..eos wrapped
    lens = pc.lengths()
    assert (lens <= vocab.cfg.context_tokens).all()
    for i in range(min(pc.n_sequences, 20)):
        tok, fam, ev = pc.sequence(i)
        assert tok[0] == vocab.bos_id and tok[-1] == vocab.eos_id

    # session axis: every line's anchor appears exactly once, so the number of
    # anchor positions equals the line count
    n_anchor = int((pc.family == FAMILY["TOK"]).sum())
    assert n_anchor == blocks.n_lines

    # anchors carry their event_id through packing (event_id 0 is a valid id,
    # so anchors are identified by family, not by a nonzero id)
    anchor_ev = pc.event_id[pc.family == FAMILY["TOK"]]
    assert set(anchor_ev.tolist()) == set(blocks.ev_id.tolist())

    # persistence
    pc.save(tmp_path / "sess")
    pc2 = PackedCorpus.load(tmp_path / "sess", mmap=False)
    np.testing.assert_array_equal(pc.tokens, pc2.tokens)
    np.testing.assert_array_equal(pc.seq_offsets, pc2.seq_offsets)


def test_tape_corpus_covers_lines(blocks, vocab):
    from rfqfm import build_sequences, pack
    from rfqfm.tokenize import FAMILY
    seqs = build_sequences(blocks, vocab.cfg, "tape")
    pc = pack(blocks, seqs, vocab, "tape")
    assert (pc.lengths() <= vocab.cfg.context_tokens).all()
    # sliding windows may overlap, so anchors >= line count, and every line's
    # event_id is represented at least once
    anchor_ev = set(pc.event_id[pc.family == FAMILY["TOK"]].tolist())
    assert set(blocks.ev_id.tolist()) <= anchor_ev


def test_build_corpus_both_axes(blocks, vocab):
    from rfqfm import build_corpus
    corp = build_corpus(blocks, vocab)
    assert set(corp.keys()) == {"session", "tape"}
    assert corp["session"].n_sequences > 0
    assert corp["tape"].n_sequences > 0


# ---------------------------------------------------------------------------
# entropy floor
# ---------------------------------------------------------------------------

def test_entropy_floor(ds):
    from rfqfm import entropy_floor
    fr = entropy_floor(ds)
    assert fr.n_lines > 0
    # choice entropy given latents cannot exceed picking uniformly among the
    # candidate set
    assert fr.floor_one_nats <= fr.uniform_baseline_nats + 1e-9
    assert fr.floor_one_nats >= 0
    assert fr.by_stratum  # stratified by candidate-set size


def test_excess_over_floor_is_zero_for_true_logp(ds):
    # feeding the generator's own log_p_chosen back in must give ~0 excess
    from rfqfm import excess_over_floor
    t = ds.scan("event_truth", columns=["event_id", "log_p_chosen"])
    eid = t.column("event_id").to_numpy()
    tlp = t.column("log_p_chosen").to_numpy()
    ok = np.isfinite(tlp)
    res = excess_over_floor(tlp[ok], eid[ok], ds)
    assert abs(res["excess_over_floor_one_nats"]) < 1e-9
