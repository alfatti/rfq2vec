import sys
from pathlib import Path

import pytest

# make both packages importable without installation
sys.path.insert(0, str(Path("/home/claude/rfqsim_schema_ext").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DS_ROOT = "/home/claude/fm_ds"


@pytest.fixture(scope="session")
def ds():
    from rfqfm import RfqDataset
    return RfqDataset(DS_ROOT)


@pytest.fixture(scope="session")
def cfg():
    # small context so the smoke run produces several sequences to exercise
    # windowing, not one big sequence per client
    from rfqfm import TokenizerConfig
    return TokenizerConfig(context_tokens=256, tape_stride_frac=0.5)


@pytest.fixture(scope="session")
def vocab(ds, cfg):
    from rfqfm import FmVocab
    return FmVocab.from_dataset(ds, cfg)


@pytest.fixture(scope="session")
def lines(ds):
    return ds.scan("rfq_lines")


@pytest.fixture(scope="session")
def blocks(lines, vocab):
    from rfqfm import tokenize_lines
    return tokenize_lines(lines, vocab)
