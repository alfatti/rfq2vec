#!/usr/bin/env python
"""The scaling sweep, end to end, in one command.

Stages (each can be skipped so you can re-enter partway):
  1. generate     -- run the simulator to produce a dataset (optional; skip if a
                     run root already exists)
  2. build        -- materialize vocab / features / packed corpora / floor
  3. train        -- train the three rungs concurrently, one per GPU
  4. probe        -- run A/B/C/G on each checkpoint
  5. table        -- collate into the scaling table (the headline artifact)

This is deliberately a torch-free subprocess orchestrator: each training rung is
its own process pinned to one card via CUDA_VISIBLE_DEVICES, which is the right
isolation for running the three sizes side by side. The heavy lifting lives in
build_corpus / train / probe; this just sequences them and reads back the
per-rung JSON reports.

    python -m rfqfm.scripts.sweep --root /data/run --work /data/fm/run \
        --rungs small medium large --gpus 0 1 2

To (re)generate the dataset first, pass --generate with the simulator dials you
want; otherwise --root is assumed to already hold a run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..config import LADDER
from ..probes import render_scaling_table, scaling_table

_PY = sys.executable


def _run(cmd, env_extra=None, background=False):
    import os
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print("  $", " ".join(cmd), flush=True)
    if background:
        return subprocess.Popen(cmd, env=env)
    subprocess.run(cmd, env=env, check=True)
    return None


def stage_build(root: str, work: str, context: int, tape_fraction: float):
    _run([_PY, "-m", "rfqfm.scripts.build_corpus", "--root", root,
          "--out", work, "--context", str(context),
          "--tape-fraction", str(tape_fraction)])


_VARIANT_FLAG = {"factorized": "--factorized", "free": "--no-factorized"}


def _ckpt_path(ckpt_dir, variant, rung):
    return str(Path(ckpt_dir) / variant / rung)


def stage_train(work, rungs, variants, gpus, cfg_dir, ckpt_dir):
    """Train the size x variant grid, at most len(gpus) jobs at once, each job
    pinned to one card. The grid (e.g. 3 sizes x {factorized, free} = 6) runs in
    waves across the available GPUs."""
    jobs = [(v, r) for v in variants for r in rungs]
    width = max(1, len(gpus))
    for i in range(0, len(jobs), width):
        wave = jobs[i:i + width]
        procs = []
        for (variant, rung), gpu in zip(wave, gpus):
            cfg = str(Path(cfg_dir) / f"{rung}.yaml")
            out = _ckpt_path(ckpt_dir, variant, rung)
            p = _run([_PY, "-m", "rfqfm.scripts.train", "--config", cfg,
                      "--data", work, "--out", out, _VARIANT_FLAG[variant]],
                     env_extra={"CUDA_VISIBLE_DEVICES": str(gpu)}, background=True)
            procs.append(((variant, rung), p))
        fail = [vr for vr, p in procs if p.wait() != 0]
        if fail:
            raise RuntimeError(f"training failed for: {fail}")


def stage_probe(root, work, rungs, variants, ckpt_dir, probe_dir):
    for variant in variants:
        for rung in rungs:
            ckpt = str(Path(_ckpt_path(ckpt_dir, variant, rung)) / "final")
            out = str(Path(probe_dir) / f"{variant}__{rung}.json")
            _run([_PY, "-m", "rfqfm.scripts.probe", "--root", root, "--data", work,
                  "--ckpt", ckpt, "--size", rung, _VARIANT_FLAG[variant],
                  "--out", out])


def stage_table(rungs, variants, probe_dir, out_path):
    reports = {}
    for variant in variants:
        for rung in rungs:
            p = Path(probe_dir) / f"{variant}__{rung}.json"
            if p.exists():
                reports[f"{variant}/{rung}"] = json.loads(p.read_text())
    table = scaling_table(reports)
    Path(out_path).write_text(json.dumps(table, indent=2))
    print("\n=== scaling table (factorized vs free x size) ===")
    print(render_scaling_table(table))
    print(f"\nwrote {out_path}")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="simulator run root")
    ap.add_argument("--work", required=True, help="corpus/artifact dir")
    ap.add_argument("--rungs", nargs="+", default=["small", "medium", "large"],
                    choices=list(LADDER))
    ap.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--variants", nargs="+", default=["factorized", "free"],
                    choices=["factorized", "free"],
                    help="L0 (factorized embedding) vs L1 (free table)")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--tape-fraction", type=float, default=0.25)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--probe-dir", default=None)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["build", "train", "probe", "table"],
                    help="stages to skip when re-entering")
    args = ap.parse_args(argv)

    work = args.work
    ckpt_dir = args.ckpt_dir or str(Path(work) / "ckpt")
    probe_dir = args.probe_dir or str(Path(work) / "probe")

    if "build" not in args.skip:
        print("[1/4] build corpus"); stage_build(args.root, work, args.context,
                                                 args.tape_fraction)
    if "train" not in args.skip:
        print(f"[2/4] train {args.variants} x {args.rungs} (waves across GPUs)")
        stage_train(work, args.rungs, args.variants, args.gpus, args.configs, ckpt_dir)
    if "probe" not in args.skip:
        print("[3/4] probe checkpoints")
        stage_probe(args.root, work, args.rungs, args.variants, ckpt_dir, probe_dir)
    if "table" not in args.skip:
        print("[4/4] scaling table")
        stage_table(args.rungs, args.variants, probe_dir,
                    str(Path(work) / "scaling_table.json"))


if __name__ == "__main__":
    main()
