"""Run manifest: the reproducibility contract for a dataset run.

One manifest.json per run root. It pins: the seed root and git SHA, the full
config blob and the dials that shape difficulty (d, norm law, mixing
half-life, temperature schedule, ...), a sha256 fingerprint of every table
schema (so schema drift between writer and reader is detectable, not
discoverable), and a checksum for every file written -- including walk-state
checkpoints. verify() re-hashes the tree; a run that doesn't verify is not a
run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .tables import SCHEMA_VERSION, SchemaBundle
from .writer import FileRecord, sha256_file

MANIFEST_NAME = "manifest.json"


def _schema_fingerprints(bundle: SchemaBundle) -> dict[str, str]:
    return {name: hashlib.sha256(bytes(sch.serialize())).hexdigest()
            for name, sch in bundle.tables.items()}


@dataclass
class RunManifest:
    run_id: str
    created_utc: str
    rfqsim_git_sha: str
    schema_version: str
    seed_root_hex: str
    config: dict
    dials: dict
    schema_fingerprints: dict
    tables: dict = field(default_factory=dict)   # name -> {"rows": int, "files": [FileRecord dict]}

    # -- construction ---------------------------------------------------------

    @classmethod
    def new(cls, bundle: SchemaBundle, seed_root_hex: str, rfqsim_git_sha: str,
            config: dict, dials: dict) -> "RunManifest":
        return cls(
            run_id=bundle.config.run_id,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            rfqsim_git_sha=rfqsim_git_sha,
            schema_version=SCHEMA_VERSION,
            seed_root_hex=seed_root_hex,
            config=dict(config),
            dials=dict(dials),
            schema_fingerprints=_schema_fingerprints(bundle),
        )

    def add_files(self, records: Iterable[FileRecord]) -> None:
        for r in records:
            entry = self.tables.setdefault(r.table, {"rows": 0, "files": []})
            entry["rows"] += r.rows
            entry["files"].append(asdict(r))

    # -- io ---------------------------------------------------------------------

    def write(self, root: str | Path) -> Path:
        path = Path(root) / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, root: str | Path) -> "RunManifest":
        raw = json.loads((Path(root) / MANIFEST_NAME).read_text())
        return cls(**raw)

    # -- verification -------------------------------------------------------------

    def verify(self, root: str | Path) -> list[str]:
        """Re-hash every recorded file. Returns a list of human-readable
        mismatches; empty list == the tree is exactly the run."""
        root = Path(root)
        problems: list[str] = []
        for table, entry in sorted(self.tables.items()):
            for f in entry["files"]:
                p = root / f["path"]
                if not p.exists():
                    problems.append(f"missing: {f['path']}")
                    continue
                if p.stat().st_size != f["n_bytes"]:
                    problems.append(f"size mismatch: {f['path']}")
                    continue
                if sha256_file(p) != f["sha256"]:
                    problems.append(f"sha256 mismatch: {f['path']}")
        return problems
