"""rfqfm -- a transaction foundation model over rfqsim RFQ output.

The torch-free core (dataset contract, tokenizer, corpus packer, entropy floor,
config ladder) imports cleanly anywhere. The model / data / train modules pull
torch lazily and are meant for the GPU box; importing this package does not
require torch.
"""
from .config import LADDER, LARGE, MEDIUM, SMALL, ModelConfig, TokenizerConfig
from .contract import RfqDataset
from .corpus import build_corpus, build_sequences, pack
from .features import BondFeatures, build_bond_features
from .floor import FloorReport, entropy_floor, excess_over_floor
from .packed import PackedCorpus, ragged_gather
from .probes import (OracleGeometry, load_oracle_geometry, perfect_model_selftest,
                     probe_calibration, probe_client_tilt_recovery,
                     probe_side_sense, probe_substitution_geometry,
                     render_scaling_table, scaling_table)
from .representation import (ReferenceGeometry, decompose_recovery,
                             load_reference_geometry)
from .tokenize import LineBlocks, tokenize_lines
from .vocab import FmVocab

__all__ = [
    "RfqDataset",
    "TokenizerConfig", "ModelConfig", "LADDER", "SMALL", "MEDIUM", "LARGE",
    "FmVocab",
    "BondFeatures", "build_bond_features",
    "LineBlocks", "tokenize_lines",
    "PackedCorpus", "ragged_gather", "build_corpus", "build_sequences", "pack",
    "FloorReport", "entropy_floor", "excess_over_floor",
    "OracleGeometry", "load_oracle_geometry", "perfect_model_selftest",
    "probe_substitution_geometry", "probe_side_sense",
    "probe_client_tilt_recovery", "probe_calibration",
    "scaling_table", "render_scaling_table",
    "ReferenceGeometry", "load_reference_geometry", "decompose_recovery",
]
