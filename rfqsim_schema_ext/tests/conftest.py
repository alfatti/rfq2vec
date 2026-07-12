import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pytest  # noqa: E402

from rfqsim.schema.tables import SchemaConfig, build_schemas  # noqa: E402


@pytest.fixture
def cfg():
    return SchemaConfig(d=8, p=6, n_sectors=4, run_id="test-run")


@pytest.fixture
def bundle(cfg):
    return build_schemas(cfg)
