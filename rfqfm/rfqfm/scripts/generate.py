#!/usr/bin/env python
"""Generate a study dataset with the simulator.

This is the one script that legitimately needs both packages -- it drives the
simulator to produce a run the rest of rfqfm then consumes from disk. sigma_eps
is a first-class study dial here: it sets how much of the substitution geometry
lives in the identity-bound residual (versus the attribute-driven B x part), and
so it sets the size of the effect the factorized-vs-free (L0/L1) study measures.

Calibrated on the smoke universe: sigma_eps 1.0 leaves attributes explaining
~0.92 of the substitution Gram (residual a thin ~16%); sigma_eps 2.0 brings that
to ~0.72 (residual ~48%) -- attributes still dominant, residual clearly material.
Read the realized share back with rfqfm.load_reference_geometry /
decompose_recovery after generating.

    python -m rfqfm.scripts.generate --out /data/run --sigma-eps 2.0 \
        --days 2520 --issuers 3000 --clients 400 --shards 4
"""
from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run root to create")
    ap.add_argument("--sigma-eps", type=float, default=2.0,
                    help="residual scale vs ||B x||; sets the L0/L1 effect size")
    ap.add_argument("--phi-side", type=float, default=0.45,
                    help="side-specific share of the residual")
    ap.add_argument("--days", type=int, default=2520)
    ap.add_argument("--issuers", type=int, default=3000)
    ap.add_argument("--clients", type=int, default=400)
    ap.add_argument("--sectors", type=int, default=12)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed-hex", default="ab" * 16)
    ap.add_argument("--run-id", default="rfqfm-study")
    args = ap.parse_args(argv)

    # simulator imports live here, not in the rfqfm core (which reads from disk)
    from rfqsim.schema.pipeline import RunDials
    from rfqsim.schema.population import UniverseConfig
    from rfqsim.schema.production import generate_run_production

    dials = RunDials(
        run_id=args.run_id, seed_root_hex=args.seed_hex, n_days=args.days,
        d=args.d, sectors=tuple(f"S{i}" for i in range(args.sectors)),
        universe=UniverseConfig(n_issuers=args.issuers, n_clients=args.clients,
                                sigma_eps=args.sigma_eps, phi_side=args.phi_side))
    out = generate_run_production(args.out, dials, n_shards=args.shards)
    print(f"generated {out['rows']:,} lines at {args.out} "
          f"(sigma_eps={args.sigma_eps}, phi_side={args.phi_side})")
    # report the realized residual share so the study parameter is recorded
    try:
        from ..contract import RfqDataset
        from ..representation import decompose_recovery, load_reference_geometry
        ds = RfqDataset(args.out)
        ref = load_reference_geometry(ds)
        d = decompose_recovery(ref.full_dir.copy(), ds)
        print(f"  attributes explain {d['attribute_recovery']:.3f} of the "
              f"substitution Gram -> residual carries "
              f"~{1 - d['attribute_recovery']**2:.2f} (the L0/L1 target)")
    except Exception as e:  # pragma: no cover
        print(f"  (could not compute residual share: {e})")


if __name__ == "__main__":
    main()
