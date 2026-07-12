#!/usr/bin/env python
"""Pretrain a rung of the ladder on packed RFQ corpora.

Plain torch + HuggingFace with DistributedDataParallel -- deliberately not NeMo,
so it runs anywhere with torch and is easy to read. Launch one process per H200:

    torchrun --nproc-per-node=4 -m rfqfm.scripts.train \
        --config rfqfm/configs/medium.yaml \
        --data /data/fm/run --out /data/fm/ckpt/medium

Because even the large rung fits on one H200, this is plain data parallel: each
GPU holds a full replica and processes a shard of the batch. The four cards are
a small cluster for the scaling sweep, not a model-sharding rig.
"""
from __future__ import annotations

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
except ImportError as e:  # pragma: no cover
    raise ImportError("rfqfm.scripts.train needs torch; run on the GPU box") from e

import argparse
import math
import os
from pathlib import Path

import numpy as np
import yaml

from ..config import LADDER, TokenizerConfig
from ..data import MixtureDataset, PackedDataset
from ..features import BondFeatures
from ..packed import PackedCorpus
from ..vocab import FmVocab


def _is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _load_features(data_dir: Path) -> BondFeatures:
    z = np.load(data_dir / "features.npz", allow_pickle=True)
    return BondFeatures(
        instrument_ids=z["instrument_ids"], X=z["X"],
        sector_idx=z["sector_idx"], seniority_idx=z["seniority_idx"],
        n_sectors=int(z["n_sectors"]), n_seniorities=int(z["n_seniorities"]),
        tok_instr_row=z["tok_instr_row"], tok_sense=z["tok_sense"],
        ref_date=None, feature_names=list(z["feature_names"]))


def cosine_lr(step, warmup, total, base, min_lr):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="build_corpus output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factorized", dest="factorized", action="store_true", default=None)
    ap.add_argument("--no-factorized", dest="factorized", action="store_false")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())

    rank, world = 0, 1
    if _is_dist():
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0

    data_dir = Path(args.data)
    vocab = FmVocab.load(data_dir / "vocab.json")
    feats = _load_features(data_dir)
    mc = LADDER[cfg["model"]["size"]]
    # L0 (factorized embedding) vs L1 (free table): CLI wins, else config, else default
    import dataclasses
    fac = args.factorized
    if fac is None:
        fac = cfg["model"].get("factorized", mc.factorized_embedding)
    mc = dataclasses.replace(mc, factorized_embedding=bool(fac))
    if main_proc:
        print(f"model={mc.name} factorized_embedding={mc.factorized_embedding}")

    # data: mix session + tape per the tokenizer fraction
    ctx = cfg["data"]["context_tokens"]
    sess = PackedDataset(PackedCorpus.load(data_dir / "corpus/session"),
                         ctx, vocab.pad_id,
                         mask_to_anchors=cfg["data"].get("mask_to_anchors", False))
    tape_dir = data_dir / "corpus/tape"
    if tape_dir.exists():
        tape = PackedDataset(PackedCorpus.load(tape_dir), ctx, vocab.pad_id,
                             mask_to_anchors=cfg["data"].get("mask_to_anchors", False))
        ds = MixtureDataset(sess, tape, cfg["data"].get("tape_fraction", 0.25))
    else:
        ds = sess

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                 shuffle=True) if _is_dist() else None
    loader = DataLoader(ds, batch_size=cfg["train"]["local_batch_size"],
                        sampler=sampler, shuffle=sampler is None,
                        num_workers=cfg["train"].get("num_workers", 4),
                        drop_last=True, pin_memory=True)

    # model
    from ..model import build_model
    model = build_model(mc, vocab, feats).to(device)
    if cfg["train"].get("compile", False):
        model = torch.compile(model)
    if _is_dist():
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])

    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["optim"]["lr"],
                            betas=tuple(cfg["optim"].get("betas", (0.9, 0.95))),
                            weight_decay=cfg["optim"].get("weight_decay", 0.077))
    total = cfg["train"]["max_steps"]
    warmup = cfg["optim"].get("warmup_steps", 10)
    base, min_lr = cfg["optim"]["lr"], cfg["optim"].get("min_lr", 6.5e-6)
    accum = cfg["train"].get("grad_accum", 1)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    step = 0
    model.train()
    while step < total:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lr = cosine_lr(step, warmup, total, base, min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", enabled=cfg["train"].get("amp", True),
                                dtype=torch.bfloat16):
                out_ = model(input_ids=ids, labels=labels)
                loss = out_.loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if main_proc and step % cfg["train"].get("log_every", 50) == 0:
                print(f"step {step:6d}/{total}  loss {out_.loss.item():.4f}  "
                      f"lr {lr:.2e}", flush=True)
            if main_proc and step > 0 and \
                    step % cfg["train"].get("ckpt_every", 1000) == 0:
                (model.module if _is_dist() else model).save_pretrained(
                    out / f"step{step}")
            step += 1
            if step >= total:
                break

    if main_proc:
        (model.module if _is_dist() else model).save_pretrained(out / "final")
        print(f"saved final checkpoint to {out/'final'}")
    if _is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
