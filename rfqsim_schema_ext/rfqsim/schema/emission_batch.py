"""Production emission engine: the reference emitter's semantics as batched
array operations.

No per-event Python loop. One day is processed in chunks of events; per chunk
the hot work is a single (E, d) @ (d, n_tok) matmul for the logits, masked
softmax draws via cumulative sums (one uniform per row instead of n_tok
Gumbels), and fixed-width auction algebra. On the H200 the same code runs on
CuPy; here it runs on NumPy.

Randomness is counter-addressed (gpurand.philox4x32): every line owns a block
of counters at

    ctr = day * 2**28 + (event_in_day * MAX_LEGS + leg) * LINE_STRIDE + offset

with fixed purpose offsets (_OFF below). event_truth stores the line's block
base and the stream key, so any line's draws replay in isolation -- same
contract as the reference path, different (documented) stream family. Up to
2**19 events per day, 8 legs per package.

Semantics preserved from the reference engine: author weights with activity
coupling, mandate-masked within-sector softmax, UNDISC true side from the
signed pair, switch = opposite-side second leg, list = one-way legs, sharpened
package temperature, and the same fair-value model (shared code). Known
deltas, both documented: auction_book uses NATURAL slots (0 = us, 1..7 =
competitors, holes allowed) instead of the reference's compacted layout, and
competitor count is capped at MAX_DEALERS - 1.
"""
from __future__ import annotations

import math

import numpy as np
import pyarrow as pa

from . import enums as E
from .backend import to_numpy, xp
from .emission import Emitter
from .gpurand import (derive_key32, icdf_choice, masked_softmax_draw, normals,
                      row_categorical, uniforms)
from .intensity import ArrivalPanel
from .writer import EventIdAllocator

_US = 1_000_000
MAX_LEGS = 8
LINE_STRIDE = 64
DAY_SHIFT = 28

# Purpose offsets inside a line's counter block. Event-level draws live in
# leg 0's block; the round-r token draw lives in leg r's block.
_OFF = dict(PKG=0, CLIENT=1, TOKEN=2, NLEGS=3, SIDE=4, SIZE=5, NDEALERS=6,
            ACTION=7, OURNOISE=8, COMPVALID=9, COMPNOISE=11, OUTCOME=15,
            DNT=16, DELAYS=17, MISC=18, RECV=20)


def _cum(p):
    return np.cumsum(np.asarray(p, dtype=np.float64))[:-1]


def _poisson_cum(mean: float, kmax: int = 16) -> np.ndarray:
    k = np.arange(kmax + 1)
    pmf = np.exp(-mean) * mean ** k / np.array([math.factorial(i) for i in k])
    pmf /= pmf.sum()
    return _cum(pmf)


class BatchEmitter(Emitter):
    """Same construction as Emitter (shared fair-value stream, token maps,
    day_context); adds the vectorized emit_day_batch."""

    def __init__(self, uni, bundle, cfg, seed_root_hex: str):
        super().__init__(uni, bundle, cfg, seed_root_hex)
        self.key = derive_key32(seed_root_hex, "emission_batch")
        # events per chunk: the memory knob. The softmax runs per SECTOR, so
        # the peak working set is ~5 * (chunk / n_sectors) * (n_tokens /
        # n_sectors) * 8 bytes -- small; 8192 is fine on CPU hosts and the
        # H200 can take 65536.
        self.chunk = 8192
        self._binom_cum = _cum([math.comb(5, k) * 0.65 ** k * 0.35 ** (5 - k)
                                for k in range(6)])
        self._list_cum = _poisson_cum(max(cfg.list_len_mean - 2.0, 0.1))

    # ------------------------------------------------------------------
    def emit_day_batch(self, day_ordinal: int, grid_day: pa.Table,
                       arrivals: ArrivalPanel, alloc: EventIdAllocator
                       ) -> tuple[pa.Table, pa.Table, pa.Table]:
        uni, cfg = self.uni, self.cfg
        asof = int(uni.cal.days[day_ordinal])
        dc = self.day_context(asof)
        n_tok = len(uni.vocab)
        Emax = 1 << (DAY_SHIFT - 3 - 6)          # events/day capacity
        if len(arrivals) > Emax:
            raise RuntimeError("arrival panel exceeds counter layout capacity")

        V = xp.asarray(dc["v"])                                   # (n_tok, d)
        alive_t = xp.asarray(dc["alive_t"])
        tok_instr = xp.asarray(self.tok_instr)
        tok_sense = xp.asarray(self.tok_sense)
        sec_of_tok = xp.asarray(self.uni.instr_sector)[tok_instr]
        # (n_sectors, n_tok) and (K, n_tok) masks, built once per day
        ns = len(uni.spec.sectors)
        sec_alive = (xp.arange(ns)[:, None] == sec_of_tok[None, :]) & alive_t[None, :]
        mand_tok = xp.asarray(dc["mandate"])[:, self.tok_instr]
        signed = tok_sense != int(E.Sense.UNDISC)
        n_sec_alive = sec_alive.sum(axis=1)

        c_grid = xp.asarray(np.asarray(grid_day["c"].combine_chunks().flatten())
                            .reshape(grid_day.num_rows, self.d).astype(np.float64))
        T_grid = xp.asarray(grid_day["temperature"].to_numpy(zero_copy_only=False)
                            .astype(np.float64))
        r_grid = xp.asarray(grid_day["r"].to_numpy(zero_copy_only=False)
                            .astype(np.float64))
        fair_i = xp.asarray(dc["fair"])
        u_cl = xp.asarray(uni.u)
        base_act = xp.asarray(uni.base_activity * uni.intensity_mult[day_ordinal])
        sector_ok = xp.asarray(((uni.mandate_sector_mask[:, None]
                                 >> np.arange(ns)[None, :]) & 1).astype(bool))
        tok_of = xp.asarray(self.tok_of)                          # (n_instr, 3)

        day_base = xp.uint64(day_ordinal) << xp.uint64(DAY_SHIFT)
        out_chunks = []
        for i0 in range(0, len(arrivals), self.chunk):
            i1 = min(i0 + self.chunk, len(arrivals))
            out_chunks.append(self._chunk(
                xp.arange(i0, i1, dtype=xp.uint64), day_base, arrivals, asof,
                c_grid, T_grid, r_grid, V, sec_alive, mand_tok, signed,
                n_sec_alive, tok_instr, tok_sense, tok_of, fair_i, u_cl,
                base_act, sector_ok))

        cols = {k: np.concatenate([c[k] for c in out_chunks])
                for k in out_chunks[0]}
        return self._assemble(cols, alloc, asof)

    # ------------------------------------------------------------------
    def _chunk(self, ev, day_base, arrivals, asof, c_grid, T_grid, r_grid, V,
               sec_alive, mand_tok, signed, n_sec_alive, tok_instr, tok_sense,
               tok_of, fair_i, u_cl, base_act, sector_ok):
        cfg = self.cfg
        key = self.key
        Ec = len(ev)
        idx = to_numpy(ev).astype(np.int64)
        step = xp.asarray(arrivals.step_k[idx])
        sec = xp.asarray(arrivals.sector[idx])
        ts = xp.asarray(arrivals.ts_us[idx])
        gid = xp.asarray(arrivals.grid_idx[idx])
        c_t = c_grid[step]
        T_t = T_grid[step]
        r_t = r_grid[step]

        blk0 = day_base + ev * xp.uint64(MAX_LEGS * LINE_STRIDE)

        def u_at(off, n=1, leg=0):
            base = blk0 + xp.uint64(leg * LINE_STRIDE + _OFF[off])
            return uniforms(key, base, n)

        def z_at(off, n=1, leg=0):
            base = blk0 + xp.uint64(leg * LINE_STRIDE + _OFF[off])
            return normals(key, base, n)

        # -- package plan + author --------------------------------------
        u_pkg = u_at("PKG")[:, 0]
        is_switch = u_pkg < cfg.p_switch
        is_list = (~is_switch) & (u_pkg < cfg.p_switch + cfg.p_list)
        n_target = xp.where(is_switch, 2,
                            xp.where(is_list,
                                     2 + icdf_choice(u_at("NLEGS")[:, 0],
                                                     self._list_cum), 1))
        n_target = xp.minimum(n_target, MAX_LEGS)
        multi = n_target > 1

        w = base_act[None, :] * xp.exp(cfg.alpha_act * (c_t @ u_cl.T))
        w = xp.where(sector_ok.T[sec], w, 0.0)
        cl = row_categorical(u_at("CLIENT")[:, 0], w)
        ok_ev = cl >= 0
        cl_s = xp.maximum(cl, 0)
        tilt = c_t + u_cl[cl_s]

        T_eff = xp.where(multi, T_t * cfg.package_temp_mult, T_t)

        # -- leg rounds, grouped by sector -----------------------------------
        # The choice is within-sector by construction, so the softmax runs on
        # (rows-in-sector, tokens-in-sector) matrices: ~n_sectors^2 less work
        # and memory than a full-vocabulary pass, and the same draws (masked-
        # out tokens contribute zero to the inverse-CDF either way).
        toks = xp.full((Ec, MAX_LEGS), -1, dtype=xp.int64)
        lzs = xp.zeros((Ec, MAX_LEGS))
        lcs = xp.zeros((Ec, MAX_LEGS))
        ncs = xp.zeros((Ec, MAX_LEGS), dtype=xp.int64)
        pair_lg = xp.zeros((Ec, 2))          # buy/sell logits of leg-0 instrument
        u_tok = xp.stack([u_at("TOKEN", leg=r)[:, 0] for r in range(MAX_LEGS)],
                         axis=1)
        for s in range(int(sec_alive.shape[0])):
            rows = xp.nonzero(sec == s)[0]
            if len(rows) == 0:
                continue
            cols = xp.nonzero(sec_alive[s])[0]
            if len(cols) == 0:
                continue
            lg_s = ((tilt[rows] @ V[cols].T)
                    / T_eff[rows][:, None])                 # (Rs, Ns)
            cand = mand_tok[cl_s[rows]][:, cols] & ok_ev[rows][:, None]
            sense_c = tok_sense[cols]
            instr_c = tok_instr[cols]
            allowed = cand & xp.where(multi[rows][:, None],
                                      (sense_c != int(E.Sense.UNDISC))[None, :],
                                      True)
            want = xp.zeros(len(rows), dtype=xp.int64)
            live0 = ok_ev[rows]
            act = xp.nonzero(live0)[0]
            for r in range(MAX_LEGS):
                act = act[(r < n_target[rows][act])]
                if len(act) == 0:
                    break
                if r > 0:
                    prev = xp.maximum(toks[rows[act], r - 1], 0)
                    ok_prev = toks[rows[act], r - 1] >= 0
                    act = act[ok_prev]
                    if len(act) == 0:
                        break
                    prev_instr = tok_instr[xp.maximum(toks[rows[act], r - 1], 0)]
                    allowed_r = (allowed[act]
                                 & (sense_c[None, :] == want[act][:, None])
                                 & (instr_c[None, :] != prev_instr[:, None]))
                    allowed[act] = allowed_r
                ch, lz, lc, nc = masked_softmax_draw(
                    u_tok[rows[act], r], lg_s[act], allowed[act])
                got = ch >= 0
                glob = xp.where(got, cols[xp.maximum(ch, 0)], -1)
                toks[rows[act], r] = glob
                lzs[rows[act], r] = lz
                lcs[rows[act], r] = lc
                ncs[rows[act], r] = nc
                if r == 0:
                    s0 = xp.where(got, sense_c[xp.maximum(ch, 0)], 0)
                    want[act] = xp.where(is_switch[rows[act]], 1 - s0, s0)
                    # cache the signed-pair logits for the side-truth draw
                    inv = xp.full(int(tok_instr.max()) + 1, -1, dtype=xp.int64)
                    # local column index of each sector instrument's buy/sell
                    loc = xp.full(len(tok_sense), -1, dtype=xp.int64)
                    loc[cols] = xp.arange(len(cols))
                    gi = xp.maximum(glob, 0)
                    pb = loc[tok_of[tok_instr[gi], 0]]
                    ps = loc[tok_of[tok_instr[gi], 1]]
                    at = xp.arange(len(act))
                    pair_lg[rows[act], 0] = lg_s[act][at, pb] * T_eff[rows[act]] / T_t[rows[act]]
                    pair_lg[rows[act], 1] = lg_s[act][at, ps] * T_eff[rows[act]] / T_t[rows[act]]

        n_legs = (toks >= 0).sum(axis=1)
        keep = n_legs > 0
        pkg_type = xp.where(n_legs >= 2,
                            xp.where(is_switch, int(E.PackageType.SWITCH),
                                     int(E.PackageType.LIST)),
                            int(E.PackageType.SINGLE))

        # -- flatten to lines -----------------------------------------------
        leg_ix = xp.arange(MAX_LEGS)[None, :]
        line_m = (toks >= 0) & keep[:, None]
        e_l, r_l = xp.nonzero(line_m)
        order = xp.argsort(e_l * MAX_LEGS + r_l)
        e_l, r_l = e_l[order], r_l[order]
        L = len(e_l)

        tok_l = toks[e_l, r_l]
        instr_l = tok_instr[tok_l]
        sense_l = tok_sense[tok_l]
        cl_l = cl_s[e_l]
        blk_l = blk0[e_l] + r_l.astype(xp.uint64) * xp.uint64(LINE_STRIDE)

        def ul(off, n=1):
            return uniforms(key, blk_l + xp.uint64(_OFF[off]), n)

        def zl(off, n=1):
            return normals(key, blk_l + xp.uint64(_OFF[off]), n)

        # -- side truth ---------------------------------------------------------
        two = sense_l == int(E.Sense.UNDISC)
        # UNDISC only occurs on single legs (leg 0), whose signed-pair logits
        # were cached at temperature T_t during the sector pass
        p_buy = 1.0 / (1.0 + xp.exp(pair_lg[e_l, 1] - pair_lg[e_l, 0]))
        drawn = xp.where(ul("SIDE")[:, 0] < p_buy,
                         int(E.SideTrue.BUY), int(E.SideTrue.SELL))
        true_l = xp.where(two, drawn, sense_l)
        disc_l = xp.where(two, int(E.SideDisclosed.TWO_WAY), sense_l)

        # -- size ------------------------------------------------------------------
        mu = xp.asarray(self.uni.size_mu_log)[cl_l]
        qty = xp.exp(mu + cfg.size_sigma * zl("SIZE")[:, 0])
        qty = xp.clip(xp.round(qty / 25_000) * 25_000, 100_000,
                      xp.asarray(self.uni.max_line)[cl_l]).astype(xp.int64)
        qb = ((qty >= 100_000).astype(xp.int64)
              + (qty >= 1_000_000) + (qty >= 5_000_000))

        # -- our participation --------------------------------------------------------
        in_panel = xp.asarray(self.uni.our_panel)[cl_l]
        received = in_panel & (ul("RECV")[:, 0] < cfg.received_given_panel)
        ua = ul("ACTION", 2)
        passed = ua[:, 0] < cfg.pass_prob
        timed = (~passed) & (ua[:, 0] < cfg.pass_prob + cfg.timeout_prob)
        auto = ua[:, 1] < 0.7
        action = xp.where(passed, int(E.Action.PASS),
                          xp.where(timed, int(E.Action.TIMEOUT),
                                   xp.where(auto, int(E.Action.AUTOQUOTE),
                                            int(E.Action.TRADER_QUOTE))))
        we_quote = received & ~passed & ~timed

        # -- auction ---------------------------------------------------------------------
        M = self.bundle.config.max_dealers
        fair = fair_i[instr_l] + cfg.widen_per_r_bp * (r_t[e_l] - 1.0)
        n_dealers = 3 + icdf_choice(ul("NDEALERS")[:, 0], self._binom_cum)
        n_comp = xp.minimum(n_dealers - in_panel.astype(xp.int64), M - 1)
        our_px = fair + cfg.our_width_bp + cfg.sigma_quote_bp * zl("OURNOISE")[:, 0]

        slot = xp.arange(1, M)[None, :]
        comp_ok = (slot <= n_comp[:, None]) \
            & (ul("COMPVALID", M - 1) < cfg.quote_prob)
        comp_px = fair[:, None] + cfg.sigma_quote_bp * zl("COMPNOISE", M - 1)

        px = xp.full((L, M), xp.inf)
        px[:, 0] = xp.where(we_quote, our_px, xp.inf)
        px[:, 1:] = xp.where(comp_ok, comp_px, xp.inf)
        valid = xp.isfinite(px)
        n_quotes = valid.sum(axis=1)

        ordr = xp.argsort(px, axis=1)
        best_slot = ordr[:, 0]
        second_slot = ordr[:, 1]
        best = xp.take_along_axis(px, best_slot[:, None], 1)[:, 0]

        uo = ul("OUTCOME", 2)
        p_trade = 1.0 / (1.0 + xp.exp((best - fair - cfg.dnt_edge_bp)
                                      / cfg.dnt_scale_bp))
        cancelled = uo[:, 0] < cfg.cancel_prob
        expired = (~cancelled) & ((uo[:, 0] < cfg.cancel_prob + cfg.expire_prob)
                                  | (n_quotes == 0))
        traded = (~cancelled) & (~expired) & (ul("DNT")[:, 0] < p_trade)
        eo = xp.where(cancelled, int(E.EnquiryOutcome.CANCELLED),
                      xp.where(expired, int(E.EnquiryOutcome.EXPIRED),
                               xp.where(traded, int(E.EnquiryOutcome.TRADED),
                                        int(E.EnquiryOutcome.DNT))))
        winner = xp.where(traded, best_slot, -1).astype(xp.int64)
        cover = xp.where(traded & (n_quotes >= 2), second_slot, -1).astype(xp.int64)

        our = xp.where(~received, int(E.OurResult.NOT_RECEIVED),
                       xp.where(~traded, int(E.OurResult.NO_TRADE),
                                xp.where(~we_quote, int(E.OurResult.NO_QUOTE),
                                         xp.where(winner == 0, int(E.OurResult.WON),
                                                  xp.where(cover == 0,
                                                           int(E.OurResult.COVER),
                                                           int(E.OurResult.LOST))))))
        exec_sprd = xp.where(traded, xp.take_along_axis(
            xp.where(xp.isfinite(px), px, xp.nan), winner[:, None] % M, 1)[:, 0],
            xp.nan)
        cover_sprd = xp.where(traded & (cover >= 0), xp.take_along_axis(
            xp.where(xp.isfinite(px), px, xp.nan), cover[:, None] % M, 1)[:, 0],
            xp.nan)

        rank = xp.full((L, M), 255, dtype=xp.int64)
        xp.put_along_axis(rank, ordr, xp.arange(1, M + 1)[None, :]
                          .repeat(L, axis=0), axis=1)
        rank = xp.where(valid, rank, 255)
        vmask = (valid.astype(xp.int64) << xp.arange(M)[None, :]).sum(axis=1)

        ud = ul("DELAYS", 2)
        um = ul("MISC", 2)
        out_ts = ts[e_l] + ((60 + ud[:, 0] * 180) * _US).astype(xp.int64)
        resp_ts = ts[e_l] + ((5 + ud[:, 1] * 55) * _US).astype(xp.int64)

        # host transfer, one dict of numpy arrays per chunk
        def H(a, dt=None):
            a = to_numpy(a)
            return a.astype(dt) if dt else a

        return dict(
            e_local=H(e_l, np.int64), leg=H(r_l, np.int64),
            n_legs=H(n_legs[e_l], np.int64), pkg_type=H(pkg_type[e_l], np.int8),
            ts=H(ts[e_l], np.int64), gid=H(gid[e_l], np.int64),
            client=H(cl_l, np.uint32), instrument=H(instr_l, np.uint32),
            disclosed=H(disc_l, np.int8), true=H(true_l, np.int8),
            token=H(tok_l, np.uint32),
            qty=H(qty, np.int64), qb=H(qb, np.int8),
            n_dealers=H(n_dealers, np.uint8),
            anonymous=H(um[:, 0] < 0.03), platform=H((um[:, 1] * 3), np.uint8),
            in_panel=H(in_panel), received=H(received),
            action=H(action, np.int8), we_quote=H(we_quote),
            our_px=H(our_px), resp_ts=H(resp_ts), out_ts=H(out_ts),
            eo=H(eo, np.int8), our=H(our, np.int8),
            exec_sprd=H(exec_sprd), cover_sprd=H(cover_sprd),
            log_z=H(lzs[e_l, r_l]), logit=H(lcs[e_l, r_l]),
            n_cand=H(ncs[e_l, r_l], np.uint32),
            masked=H(n_sec_alive[sec[e_l]] - ncs[e_l, r_l], np.uint32),
            ctr=H(blk_l, np.uint64),
            dealer_id=H(xp.where(valid, xp.arange(M)[None, :], -1), np.int16),
            px_book=H(xp.where(valid, px, xp.nan), np.float32),
            rank=H(rank, np.uint8), vmask=H(vmask, np.uint16),
            n_quotes=H(n_quotes, np.uint8),
            winner=H(winner, np.int8), cover=H(cover, np.int8),
            our_slot=H(xp.where(we_quote, 0, -1), np.int8),
        )

    # ------------------------------------------------------------------
    def _assemble(self, c: dict, alloc: EventIdAllocator, asof: int):
        L = len(c["ts"])
        eids = alloc.take(L)
        # package id = event id of leg 0; legs are contiguous in line order
        first = np.zeros(L, dtype=np.int64)
        first[c["leg"] == 0] = 1
        pkg = eids[np.maximum(np.cumsum(first) - 1, 0)]
        recv = c["received"]

        def m(arr, mask):
            return pa.array(arr, mask=~mask)

        n = self.bundle.tables
        rfq = pa.Table.from_pydict({
            "event_id": eids, "package_id": pkg,
            "line_no": c["leg"].astype(np.uint16),
            "n_lines": c["n_legs"].astype(np.uint16),
            "package_type": c["pkg_type"],
            "ts": c["ts"], "trade_date": np.full(L, asof, np.int32),
            "grid_idx": c["gid"].astype(np.uint32),
            "response_deadline": c["ts"] + 300 * _US,
            "client_id": c["client"], "instrument_id": c["instrument"],
            "client_side_disclosed": c["disclosed"],
            "qty_par": c["qty"], "qty_bucket": c["qb"],
            "n_dealers": c["n_dealers"], "anonymous": c["anonymous"],
            "platform": c["platform"],
            "token_id": c["token"], "client_side_true": c["true"],
            "our_in_panel": c["in_panel"], "received": recv,
            "action": m(c["action"], recv),
            "quote_sprd_bp": m(c["our_px"].astype(np.float32),
                               recv & c["we_quote"]),
            "quote_px": pa.nulls(L, pa.float64()),
            "response_ts": m(c["resp_ts"], recv & c["we_quote"]),
            "enquiry_outcome": c["eo"], "our_result": c["our"],
            "outcome_ts": c["out_ts"],
            "exec_sprd_bp": m(np.nan_to_num(c["exec_sprd"]).astype(np.float32),
                              ~np.isnan(c["exec_sprd"])),
            "exec_px": pa.nulls(L, pa.float64()),
            "cover_sprd_bp": m(np.nan_to_num(c["cover_sprd"]).astype(np.float32),
                               ~np.isnan(c["cover_sprd"])),
            "cover_px": pa.nulls(L, pa.float64()),
            "rec_token_id": pa.nulls(L, pa.uint32()),
            "rec_delta": pa.nulls(L, pa.float32()),
            "rec_propensity": pa.nulls(L, pa.float32()),
            "policy_id": pa.nulls(L, pa.uint16()),
        }, schema=n["rfq_lines"])

        truth = pa.Table.from_pydict({
            "event_id": eids, "trade_date": np.full(L, asof, np.int32),
            "log_z": c["log_z"], "logit_chosen": c["logit"],
            "log_p_chosen": c["logit"] - c["log_z"],
            "n_candidates": c["n_cand"], "masked_count": c["masked"],
            "philox_key0": np.full(L, self.key[0], np.uint64),
            "philox_key1": np.full(L, self.key[1], np.uint64),
            "philox_ctr": c["ctr"],
        }, schema=n["event_truth"])

        M = self.bundle.config.max_dealers
        book = pa.Table.from_pydict({
            "event_id": eids, "trade_date": np.full(L, asof, np.int32),
            "dealer_id": _fsl(c["dealer_id"], pa.int16()),
            "px_sprd_bp": _fsl(c["px_book"], pa.float32()),
            "valid_mask": c["vmask"], "rank": _fsl(c["rank"], pa.uint8()),
            "n_quotes": c["n_quotes"],
            "winner_slot": c["winner"], "cover_slot": c["cover"],
            "our_slot": c["our_slot"],
        }, schema=n["auction_book"])
        return rfq, truth, book


def _fsl(mat: np.ndarray, typ: pa.DataType) -> pa.FixedSizeListArray:
    m = np.ascontiguousarray(mat)
    return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel(), typ), m.shape[1])
