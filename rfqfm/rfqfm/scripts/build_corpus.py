#!/usr/bin/env python
"""Materialize training artifacts from a simulator run.

Reads one rfqsim run root and writes, under --out:
    vocab.json                    the global vocabulary
    features.npz                  the bond feature matrix + token maps
    corpus/session/               packed session corpus
    corpus/tape/                  packed tape corpus (if enabled)
    floor.json                    the DGP entropy floor for this run

Torch-free; run on the data box. For a decade-scale run this streams by day
inside tokenize/pack rather than loading everything at once -- here we keep the
whole-run read since the simulator tables are modest per run.

    python -m rfqfm.scripts.build_corpus --root /data/run --out /data/fm/run \
        --context 4096
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import TokenizerConfig
from ..contract import RfqDataset
from ..corpus import build_corpus
from ..features import build_bond_features
from ..floor import entropy_floor
from ..tokenize import tokenize_lines
from ..vocab import FmVocab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="simulator run root")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--tape-fraction", type=float, default=0.25)
    ap.add_argument("--no-tape", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = RfqDataset(args.root)
    cfg = TokenizerConfig(context_tokens=args.context,
                          tape_fraction=args.tape_fraction,
                          tape_axis=not args.no_tape)

    vocab = FmVocab.from_dataset(ds, cfg)
    vocab.save(out / "vocab.json")
    print(vocab.summary())

    bf = build_bond_features(ds)
    np.savez(out / "features.npz", X=bf.X, sector_idx=bf.sector_idx,
             seniority_idx=bf.seniority_idx, tok_instr_row=bf.tok_instr_row,
             tok_sense=bf.tok_sense, instrument_ids=bf.instrument_ids,
             n_sectors=bf.n_sectors, n_seniorities=bf.n_seniorities,
             feature_names=np.array(bf.feature_names))

    lines = ds.scan("rfq_lines")
    blocks = tokenize_lines(lines, vocab)
    corp = build_corpus(blocks, vocab)
    for axis, pc in corp.items():
        pc.save(out / "corpus" / axis)
        print(f"  {axis}: {pc.n_sequences} seqs, {pc.n_tokens:,} tokens")

    fr = entropy_floor(ds)
    (out / "floor.json").write_text(json.dumps(fr.as_dict(), indent=2))
    print(f"  floor_one: {fr.floor_one_nats:.4f} nats "
          f"({fr.as_dict()['floor_one_bits']:.3f} bits)")
    print(f"artifacts written to {out}")


if __name__ == "__main__":
    main()
