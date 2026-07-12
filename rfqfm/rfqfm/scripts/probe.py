#!/usr/bin/env python
"""Run the recovery probes on one trained checkpoint.

    python -m rfqfm.scripts.probe --root /data/run --data /data/fm/run \
        --ckpt /data/fm/ckpt/medium/final --size medium \
        --out /data/fm/probe/medium.json

Writes a JSON report with the A/B/C/G probe outputs plus the rung's transformer
parameter count, which the sweep driver collates into the scaling table.
"""
from __future__ import annotations

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as e:  # pragma: no cover
    raise ImportError("rfqfm.scripts.probe needs torch; run on the GPU box") from e

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import LADDER
from ..contract import RfqDataset
from ..data import PackedDataset
from ..extract import (anchor_embedding_table, anchor_logprobs, client_tilt_table)
from ..features import BondFeatures
from ..packed import PackedCorpus
from ..probes import (probe_calibration, probe_client_tilt_recovery,
                      probe_side_sense, probe_substitution_geometry)
from ..vocab import FmVocab


def _load_features(data_dir: Path) -> BondFeatures:
    z = np.load(data_dir / "features.npz", allow_pickle=True)
    return BondFeatures(
        instrument_ids=z["instrument_ids"], X=z["X"],
        sector_idx=z["sector_idx"], seniority_idx=z["seniority_idx"],
        n_sectors=int(z["n_sectors"]), n_seniorities=int(z["n_seniorities"]),
        tok_instr_row=z["tok_instr_row"], tok_sense=z["tok_sense"],
        ref_date=None, feature_names=list(z["feature_names"]))


def _load_checkpoint(ckpt: Path, mc, vocab, feats, device):
    """Rebuild the architecture (so the factorized embedding is installed) and
    load the saved weights into it."""
    from ..model import build_model
    model = build_model(mc, vocab, feats)
    state = None
    st = ckpt / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        state = load_file(str(st))
    else:
        bin_ = ckpt / "pytorch_model.bin"
        if bin_.exists():
            state = torch.load(bin_, map_location="cpu")
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [load_state_dict] missing={len(missing)} "
                  f"unexpected={len(unexpected)} (buffers/tied heads are fine)")
    return model.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="simulator run root (oracle tables)")
    ap.add_argument("--data", required=True, help="build_corpus output dir")
    ap.add_argument("--ckpt", required=True, help="checkpoint directory")
    ap.add_argument("--size", required=True, choices=list(LADDER))
    ap.add_argument("--factorized", dest="factorized", action="store_true", default=True)
    ap.add_argument("--no-factorized", dest="factorized", action="store_false")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-corpus", default="session",
                    choices=["session", "tape"])
    ap.add_argument("--eval-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="token subsample for the geometry Gram")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RfqDataset(args.root)
    data_dir = Path(args.data)
    vocab = FmVocab.load(data_dir / "vocab.json")
    feats = _load_features(data_dir)
    import dataclasses
    mc = dataclasses.replace(LADDER[args.size], factorized_embedding=args.factorized)

    model = _load_checkpoint(Path(args.ckpt), mc, vocab, feats, device)

    # A / B: geometry from the anchor embedding table
    emb = anchor_embedding_table(model, vocab)
    A = probe_substitution_geometry(emb, ds, max_tokens=args.max_tokens)
    B = probe_side_sense(emb, ds)
    # attribute-vs-residual decomposition of that same table (the L0/L1 headline)
    from ..representation import decompose_recovery
    D = decompose_recovery(emb, ds, max_tokens=args.max_tokens)

    # C: client tilt from the CLI embedding
    C = probe_client_tilt_recovery(client_tilt_table(model, vocab), ds)

    # G: calibration over a held-out slice of the corpus
    corp = PackedCorpus.load(data_dir / "corpus" / args.eval_corpus)
    dset = PackedDataset(corp, corp.meta["context_tokens"], vocab.pad_id,
                         return_event_id=True)
    n = min(len(dset), args.eval_batches * args.batch_size)
    subset = torch.utils.data.Subset(dset, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    logp, evid = anchor_logprobs(model, loader, device)
    G = probe_calibration(logp, evid, ds)

    report = dict(size=args.size,
                  variant=("factorized" if args.factorized else "free"),
                  factorized=bool(args.factorized),
                  params=mc.transformer_params(vocab.size),
                  A_substitution=A, B_side_sense=B, C_client_tilt=C,
                  G_calibration=G, D_decomposition=D)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}: variant={report['variant']} "
          f"attr={D['attribute_recovery']:.3f} resid={D['residual_recovery']:.3f} "
          f"excess-floor={G['excess_over_floor_one_nats']:.4f} nats")


if __name__ == "__main__":
    main()
