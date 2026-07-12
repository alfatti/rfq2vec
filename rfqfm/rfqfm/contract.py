"""The dataset contract.

This module is the whole seam between the simulator and the foundation-model
code. Everything rfqfm consumes at runtime comes through here, and it reaches
into rfqsim for exactly one thing -- the enum registries -- so that token
semantics stay single-sourced instead of being re-declared and drifting. Every
other fact about the data is read from the on-disk tables and the manifest that
the production driver already writes.

I keep this deliberately thin. If I ever find myself importing rfqsim.emission
or rfqsim.state in here, that is the signal that I have broken the contract and
should be adding a column to the simulator's output instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

# The one and only import from the simulator: the shared vocab of int codes.
from rfqsim.schema import enums as E

# Tables the foundation-model path is allowed to read. rfq_lines_obs (the
# leakage-safe projection) and the policy-gated recommender columns are held
# out of pretraining on purpose; the model learns the generative process, not
# a policy's redactions.
PRETRAIN_SOURCE = "rfq_lines"
TRUTH_TABLE = "event_truth"
_DIMS = ("token_map", "instruments", "issuers", "clients")
_ORACLE = ("oracle_clients", "oracle_embeddings", "context_grid", "regime_path")


class RfqDataset:
    """A read-only view over one simulator run.

    Parameters
    ----------
    root : path
        The run directory -- the one holding manifest.json and tables/.
    """

    def __init__(self, root):
        self.root = Path(root)
        man_path = self.root / "manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(
                f"no manifest.json under {self.root}; is this a simulator run root?")
        self.manifest = json.loads(man_path.read_text())
        self.schema_version = self.manifest.get("schema_version")
        self.run_id = self.manifest.get("run_id")
        self.enums = E
        self._cache: dict = {}

    # -- table access -----------------------------------------------------
    def _table_dir(self, name: str) -> Path:
        return self.root / "tables" / name

    def dataset(self, name: str) -> pads.Dataset:
        """A pyarrow Dataset for a (possibly hive-partitioned) table."""
        d = self._table_dir(name)
        if not d.exists():
            raise KeyError(f"table {name!r} not present under {d}")
        # hive partitioning is a no-op for the flat dims tables
        return pads.dataset(str(d), partitioning="hive")

    def scan(self, name: str, columns: Optional[Iterable[str]] = None,
             filter=None) -> pa.Table:
        """Read a table (optionally a column subset / row filter) into memory.

        The simulator tables are small enough per run that a full read is fine
        here; the packing step streams by day when that stops being true."""
        cols = list(columns) if columns is not None else None
        return self.dataset(name).to_table(columns=cols, filter=filter)

    def dim(self, name: str) -> pa.Table:
        """Cached read of a small dimension/oracle table."""
        if name not in self._cache:
            self._cache[name] = self.dataset(name).to_table()
        return self._cache[name]

    # -- convenience for the pieces downstream ----------------------------
    def token_map(self) -> pa.Table:
        """token_id -> (instrument_id, sense, born_ts, retired_ts)."""
        return self.dim("token_map")

    def instruments(self) -> pa.Table:
        return self.dim("instruments")

    def issuers(self) -> pa.Table:
        return self.dim("issuers")

    def clients(self) -> pa.Table:
        return self.dim("clients")

    def has(self, name: str) -> bool:
        return self._table_dir(name).exists()

    def n_rows(self, name: str) -> int:
        return sum(pq.read_metadata(f).num_rows
                   for f in self._table_dir(name).rglob("*.parquet"))

    def __repr__(self) -> str:
        return (f"RfqDataset(run_id={self.run_id!r}, "
                f"schema_version={self.schema_version!r}, root={str(self.root)!r})")
