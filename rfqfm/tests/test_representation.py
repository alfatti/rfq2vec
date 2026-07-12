"""The decomposition instrument must separate the attribute part from the
residual part -- otherwise it can't detect the tokenization inductive bias.

Feeding the oracle's own quantities in simulates the ladder's ceilings:
  * full realized vectors  -> what an ideal identity model (L0/L1) could recover
  * attribute directions   -> what an ideal attributes-only model (L3) recovers
The residual_recovery metric must light up for the first and vanish for the
second.
"""
import numpy as np


def test_reconstruction_and_share(ds):
    from rfqfm import load_reference_geometry
    ref = load_reference_geometry(ds)
    # attribute directions are unit vectors; share in [0,1]
    assert np.allclose(np.linalg.norm(ref.attr_dir, axis=1), 1.0, atol=1e-6)
    assert 0.0 <= ref.attr_share <= 1.0
    # attributes are side-blind: a token's attribute direction depends only on
    # its instrument, so buy and sell of the same bond share it exactly
    # (spot check via the oracle geometry loader)
    from rfqfm import load_oracle_geometry
    geo = load_oracle_geometry(ds)
    import collections
    by = collections.defaultdict(dict)
    for r, (i, s) in enumerate(zip(geo.instr, geo.sense)):
        by[int(i)][int(s)] = r
    pair = next(v for v in by.values() if 0 in v and 1 in v)
    assert np.allclose(ref.attr_dir[pair[0]], ref.attr_dir[pair[1]], atol=1e-6)


def test_decomposition_separates_ceilings(ds):
    from rfqfm import decompose_recovery, load_reference_geometry
    ref = load_reference_geometry(ds)

    # ideal identity model: recovers the full geometry
    full = decompose_recovery(ref.full_dir.copy(), ds)
    assert full["full_recovery"] > 0.999
    assert full["residual_recovery"] > 0.99          # captures the eps part
    assert full["attribute_recovery"] > 0.5          # and the attribute part

    # ideal attributes-only model (L3 ceiling): recovers attributes, not residual
    attr = decompose_recovery(ref.attr_dir.copy(), ds)
    assert attr["attribute_recovery"] > 0.999
    assert abs(attr["residual_recovery"]) < 0.05     # residual is out of reach

    # the residual-recovery gap between the two is the tokenization effect
    assert full["residual_recovery"] - attr["residual_recovery"] > 0.8


def test_random_recovers_nothing(ds):
    from rfqfm import decompose_recovery, load_reference_geometry
    ref = load_reference_geometry(ds)
    rng = np.random.default_rng(0)
    r = rng.standard_normal(ref.full_dir.shape)
    d = decompose_recovery(r, ds)
    assert abs(d["attribute_recovery"]) < 0.1
    assert abs(d["residual_recovery"]) < 0.1
