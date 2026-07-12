"""Sharded Parquet write path + reproducibility ledger.

Layout (one run root):

    <root>/tables/<name>/trade_month=YYYY-MM/part-sSSSSS-NNNNN.parquet
    <root>/tables/<name>/part-sSSSSS-NNNNN.parquet          (unpartitioned dims)
    <root>/_checkpoints/shard=SSSSS/grid_GGGGGGGGG.npz      (walk state)
    <root>/manifest.json                                     (see manifest.py)

Each H200 owns a ShardWriter over a contiguous block of trade dates. Files
are sorted by the table's declared sort key at flush (so (ts, event_id) scans
never resort), compressed with zstd, and hashed for the manifest.

Reproducibility contract: event_id = (shard << 48) | counter, and every
stochastic event draws from a Philox-4x64 stream at coordinates derived as
sha256(seed_root / table / shard) for the key and a per-event counter block of
COUNTER_STRIDE for the offset. event_truth persists (key0, key1, ctr), so any
single event's randomness is replayable in isolation -- no replaying the run,
no cursor state, just PhiloxLedger.generator(key0, key1, ctr).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .tables import SchemaBundle

# Columns worth parquet footer statistics: these are what windowed
# co-occurrence scans and per-client/per-name pulls prune on.
STAT_COLUMNS = ("ts", "event_id", "trade_date", "grid_idx",
                "client_id", "instrument_id", "token_id")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class FileRecord:
    table: str
    path: str          # relative to run root
    rows: int
    n_bytes: int
    sha256: str


class EventIdAllocator:
    """event_id = (shard_id << 48) | counter. 65k shards, 2^48 events each.
    Not time-ordered across shards by itself -- the global sort key is
    (ts, event_id), with event_id as the deterministic tiebreak."""

    SHARD_BITS = 16
    CTR_BITS = 48

    def __init__(self, shard_id: int):
        if not (0 <= shard_id < (1 << self.SHARD_BITS)):
            raise ValueError(f"shard_id out of range: {shard_id}")
        self.shard_id = shard_id
        self._next = 0

    def take(self, n: int = 1) -> np.ndarray:
        if self._next + n > (1 << self.CTR_BITS):
            raise OverflowError("event counter exhausted for shard")
        base = self.shard_id << self.CTR_BITS
        ids = np.arange(self._next, self._next + n, dtype=np.uint64) + np.uint64(base)
        self._next += n
        return ids

    @staticmethod
    def decode(event_id: int) -> tuple[int, int]:
        return event_id >> EventIdAllocator.CTR_BITS, event_id & ((1 << EventIdAllocator.CTR_BITS) - 1)


class PhiloxLedger:
    """Per-(table, shard) Philox key; per-event counter blocks.

    I derive the 128-bit key as sha256(f"{seed_root}/{table}/{shard:05d}") so
    keys are collision-safe across tables and shards without coordination,
    and I stride the counter by COUNTER_STRIDE per event so an event can draw
    up to STRIDE * 256 bits without touching its neighbour's stream.
    """

    COUNTER_STRIDE = 1 << 20

    def __init__(self, seed_root_hex: str, table: str, shard_id: int):
        digest = hashlib.sha256(f"{seed_root_hex}/{table}/{shard_id:05d}".encode()).digest()
        self.key0 = int.from_bytes(digest[:8], "little")
        self.key1 = int.from_bytes(digest[8:16], "little")
        self.shard_id = shard_id
        self.table = table
        self._ctr = 0

    def next_event(self) -> tuple[int, int, int]:
        """Reserve a counter block; returns (key0, key1, ctr) -- exactly what
        event_truth stores."""
        c = self._ctr
        self._ctr += self.COUNTER_STRIDE
        return self.key0, self.key1, c

    def spawn_generator(self) -> tuple[np.random.Generator, int, int, int]:
        k0, k1, c = self.next_event()
        return self.generator(k0, k1, c), k0, k1, c

    @staticmethod
    def generator(key0: int, key1: int, ctr: int) -> np.random.Generator:
        bg = np.random.Philox(counter=[ctr, 0, 0, 0], key=[key0, key1])
        return np.random.Generator(bg)


class ShardWriter:
    """Buffered, month-partitioned, sorted, checksummed Parquet writer for one
    shard. append() any bundle table as a pa.Table or a column mapping; data
    is cast to the declared schema (a wrong dtype fails HERE, not three joins
    downstream); close() returns FileRecords for the manifest."""

    def __init__(self, root: str | Path, bundle: SchemaBundle, shard_id: int,
                 compression: str = "zstd",
                 target_file_bytes: int = 256 << 20,
                 target_row_group_bytes: int = 192 << 20):
        self.root = Path(root)
        self.bundle = bundle
        self.shard_id = int(shard_id)
        self.compression = compression
        self.target_file_bytes = target_file_bytes
        self.target_row_group_bytes = target_row_group_bytes
        self._buf: dict[tuple[str, str], list[pa.Table]] = {}
        self._bufbytes: dict[tuple[str, str], int] = {}
        self._seq: dict[tuple[str, str], int] = {}
        self._records: list[FileRecord] = []
        self._closed = False

    # -- ingest -------------------------------------------------------------

    def append(self, table: str, data: pa.Table | Mapping) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        sch = self.bundle.tables[table]
        tbl = data if isinstance(data, pa.Table) else pa.Table.from_pydict(dict(data), schema=sch)
        tbl = tbl.select(sch.names).cast(sch)
        for month, sub in self._split_months(table, tbl):
            key = (table, month)
            self._buf.setdefault(key, []).append(sub)
            self._bufbytes[key] = self._bufbytes.get(key, 0) + sub.nbytes
            if self._bufbytes[key] >= self.target_file_bytes:
                self._flush(key)

    def _split_months(self, table: str, tbl: pa.Table) -> Iterable[tuple[str, pa.Table]]:
        if table not in self.bundle.month_partitioned or "trade_date" not in tbl.column_names:
            yield "", tbl
            return
        months = pc.strftime(pc.cast(tbl["trade_date"], pa.timestamp("s")), format="%Y-%m")
        for mv in pc.unique(months).to_pylist():
            yield mv, tbl.filter(pc.equal(months, mv))

    # -- flush --------------------------------------------------------------

    def _flush(self, key: tuple[str, str]) -> None:
        chunks = self._buf.pop(key, [])
        self._bufbytes.pop(key, None)
        if not chunks:
            return
        table, month = key
        tbl = pa.concat_tables(chunks).combine_chunks()
        sort = self.bundle.sort_keys.get(table)
        if sort:
            tbl = tbl.sort_by(list(sort))

        seq = self._seq.get(key, 0)
        self._seq[key] = seq + 1
        rel = Path("tables") / table
        if month:
            rel = rel / f"trade_month={month}"
        rel = rel / f"part-s{self.shard_id:05d}-{seq:05d}.parquet"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)

        n = tbl.num_rows
        rg = max(1, int(n * self.target_row_group_bytes / max(1, tbl.nbytes)))
        stats = [c for c in STAT_COLUMNS if c in tbl.column_names]
        pq.write_table(tbl, path, compression=self.compression,
                       row_group_size=rg,
                       write_statistics=stats if stats else None)
        self._records.append(FileRecord(
            table=table, path=str(rel), rows=n,
            n_bytes=path.stat().st_size, sha256=sha256_file(path)))

    # -- walk-state checkpoints ----------------------------------------------

    def checkpoint_walk_state(self, grid_idx: int, arrays: Mapping[str, np.ndarray]) -> FileRecord:
        """Persist the latent-walk state at a shard boundary so any block is
        regenerable in isolation from (manifest, checkpoint, Philox)."""
        rel = Path("_checkpoints") / f"shard={self.shard_id:05d}" / f"grid_{grid_idx:09d}.npz"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
        rec = FileRecord(table="_walk_checkpoint", path=str(rel), rows=0,
                         n_bytes=path.stat().st_size, sha256=sha256_file(path))
        self._records.append(rec)
        return rec

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> list[FileRecord]:
        if not self._closed:
            for key in sorted(self._buf.keys()):
                self._flush(key)
            self._closed = True
        return list(self._records)
