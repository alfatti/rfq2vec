"""Write path: id allocation, Philox replay, hive layout, sorted files,
manifest round-trip and tamper detection."""
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from _toy import base_row, date_days, ts_us
from rfqsim.schema.manifest import RunManifest
from rfqsim.schema.writer import EventIdAllocator, PhiloxLedger, ShardWriter

SEED = "deadbeefdeadbeefdeadbeefdeadbeef"


def test_event_id_allocator():
    a = EventIdAllocator(shard_id=3)
    ids = a.take(5)
    assert ids.dtype == np.uint64
    assert list(np.diff(ids.astype(np.int64))) == [1, 1, 1, 1]
    shard, ctr = EventIdAllocator.decode(int(ids[0]))
    assert (shard, ctr) == (3, 0)
    assert EventIdAllocator.decode(int(a.take(1)[0])) == (3, 5)


def test_philox_replay_and_isolation():
    led = PhiloxLedger(SEED, "rfq_lines", shard_id=0)
    k0, k1, c0 = led.next_event()
    _, _, c1 = led.next_event()
    assert c1 - c0 == PhiloxLedger.COUNTER_STRIDE

    draws_a = PhiloxLedger.generator(k0, k1, c0).random(8)
    draws_b = PhiloxLedger.generator(k0, k1, c0).random(8)
    np.testing.assert_array_equal(draws_a, draws_b)          # exact replay

    other = PhiloxLedger(SEED, "rfq_lines", shard_id=1)
    assert (other.key0, other.key1) != (k0, k1)              # shard isolation
    tab = PhiloxLedger(SEED, "auction_book", shard_id=0)
    assert (tab.key0, tab.key1) != (k0, k1)                  # table isolation


def _rows(shard: int, specs):
    """specs: list of (month, day). Returns rfq_lines column mapping."""
    alloc = EventIdAllocator(shard)
    ids = alloc.take(len(specs))
    rows = []
    for idx, (eid, (m, d)) in enumerate(zip(ids, specs)):
        r = base_row(idx, ts_us(2026, m, d, 14), date_days(2026, m, d))
        r["event_id"] = int(eid)      # 64-bit shard-prefixed id
        r["package_id"] = int(eid)
        rows.append(r)
    # deliberately out of time order to prove the writer sorts at flush
    rows = rows[::-1]
    return {k: [r[k] for r in rows] for k in rows[0]}


def test_writer_end_to_end(tmp_path, bundle):
    led0 = PhiloxLedger(SEED, "event_truth", 0)

    w0 = ShardWriter(tmp_path, bundle, shard_id=0)
    w0.append("rfq_lines", _rows(0, [(1, 5), (1, 6), (1, 7), (2, 3), (2, 4)]))
    truth = []
    for i in range(5):
        k0, k1, c = led0.next_event()
        truth.append(dict(event_id=i, trade_date=date_days(2026, 1, 5),
                          log_z=10.0, logit_chosen=1.5, log_p_chosen=-8.5,
                          n_candidates=1200, masked_count=34,
                          philox_key0=k0, philox_key1=k1, philox_ctr=c))
    w0.append("event_truth", {k: [t[k] for t in truth] for k in truth[0]})
    w0.checkpoint_walk_state(390, {"c": np.zeros(8, np.float32), "r": np.array([1.0])})
    rec0 = w0.close()

    w1 = ShardWriter(tmp_path, bundle, shard_id=1)
    w1.append("rfq_lines", _rows(1, [(1, 8), (1, 9)]))
    rec1 = w1.close()

    # layout
    assert (tmp_path / "tables/rfq_lines/trade_month=2026-01").is_dir()
    assert (tmp_path / "tables/rfq_lines/trade_month=2026-02").is_dir()
    assert (tmp_path / "_checkpoints/shard=00000").is_dir()

    # dataset reads whole, hive-partitioned
    d = ds.dataset(tmp_path / "tables/rfq_lines", partitioning="hive")
    assert d.to_table().num_rows == 7

    # files are sorted by (ts, event_id) despite reversed input
    one = pq.read_table(tmp_path / "tables/rfq_lines/trade_month=2026-01/part-s00000-00000.parquet")
    ts = one["ts"].to_pylist()
    assert ts == sorted(ts)

    # schema metadata survives the round trip
    md = {k.decode(): v.decode() for k, v in one.schema.metadata.items()}
    assert md["rfqsim.table"] == "rfq_lines"

    # manifest: build, write, load, verify, tamper
    man = RunManifest.new(bundle, seed_root_hex=SEED, rfqsim_git_sha="abc123",
                          config={"demo": True}, dials={"d": bundle.config.d})
    man.add_files(rec0 + rec1)
    man.write(tmp_path)

    loaded = RunManifest.load(tmp_path)
    assert loaded.tables["rfq_lines"]["rows"] == 7
    assert loaded.schema_fingerprints["rfq_lines"] == man.schema_fingerprints["rfq_lines"]
    assert loaded.verify(tmp_path) == []

    victim = tmp_path / loaded.tables["rfq_lines"]["files"][0]["path"]
    with open(victim, "ab") as f:
        f.write(b"x")
    problems = loaded.verify(tmp_path)
    assert len(problems) == 1 and "mismatch" in problems[0]
