"""Emission layer: everything that happens once an arrival exists.

Per arrival (t, sector): choose the author, then the token, then resolve the
package, size, auction and outcome. All randomness comes from the per-shard
PhiloxLedger ("event_truth" stream) with one counter block per line, and the
block coordinates are persisted in event_truth -- any single line's draws
replay in isolation.

The choice model is the agreed composition:

    author   k  ~  base_k * activity_mult_k(day) * exp(alpha <u_k, c_t>)
                   over clients whose mandate covers the sector
                   (presence-is-signal: the activity coupling)
    token    w  ~  softmax over { alive tokens in the sector } cap
                   { client's mandate mask }, logits <v_w(t), c_t + u_k> / T_t
    two-way     : the UNDISC sense competes as its own token; when chosen,
                  the TRUE side is drawn from the softmax over the two signed
                  senses of the same instrument at the same (c_t + u_k, T_t)
    packages    : switches draw an opposite-side second leg, lists draw
                  same-side legs, both at sharpened temperature (frozen,
                  low-temperature sentences)

Reference-grade implementation: python loop over events with vectorized inner
algebra -- fine at validation scale; the production path is the CuPy port of
exactly these formulas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from . import enums as E
from .intensity import ArrivalPanel
from .population import Universe, instrument_state, token_vectors
from .state import derive_stream_key
from .tables import SchemaBundle
from .vocab import TokenVocab
from .writer import EventIdAllocator, PhiloxLedger

_US = 1_000_000


class EmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmissionConfig:
    alpha_act: float = 1.0            # client-context activity coupling
    p_switch: float = 0.06
    p_list: float = 0.03
    list_len_mean: float = 4.0
    package_temp_mult: float = 0.6    # sharpened sentences
    sigma_quote_bp: float = 6.0
    our_width_bp: float = 1.5         # our systematic width: the hit-rate dial
    quote_prob: float = 0.85          # competitor responds
    pass_prob: float = 0.08
    timeout_prob: float = 0.02
    dnt_edge_bp: float = 2.0
    dnt_scale_bp: float = 6.0
    cancel_prob: float = 0.02
    expire_prob: float = 0.02
    size_sigma: float = 0.9
    widen_per_r_bp: float = 40.0      # spread widening per unit of (r - r0)
    received_given_panel: float = 0.95
    fair_idio_bp: float = 25.0


class Emitter:
    def __init__(self, uni: Universe, bundle: SchemaBundle,
                 cfg: EmissionConfig, seed_root_hex: str):
        self.uni, self.bundle, self.cfg = uni, bundle, cfg
        self.d = bundle.config.d
        self.tok_instr, self.tok_sense = uni.token_arrays()
        # (n_instr, 3): token id of each sense, for the signed-pair side draw
        self.tok_of = np.stack(
            [uni.vocab.tokens_for(np.arange(uni.n_instruments),
                                  np.full(uni.n_instruments, s)).astype(np.int64)
             for s in range(3)], axis=1)
        g = PhiloxLedger.generator(*derive_stream_key(seed_root_hex, "fair_idio"), 0)
        self.fair_idio = g.normal(0.0, cfg.fair_idio_bp, uni.n_instruments)

    # -- per-day precomputation ------------------------------------------------

    def day_context(self, asof_days: int) -> dict:
        uni = self.uni
        st = instrument_state(uni, asof_days)
        v = token_vectors(uni, asof_days, st["x"])                  # (n_tok, d)
        alive_i = (uni.issue_days <= asof_days) & (uni.maturity_days > asof_days)
        alive_t = alive_i[self.tok_instr]
        ttm = (uni.maturity_days - asof_days) / 365.25
        # client x instrument mandate mask
        K = uni.n_clients
        m = ((uni.rating[None, :] >= uni.mandate_rating[:, 0:1])
             & (uni.rating[None, :] <= uni.mandate_rating[:, 1:2])
             & (ttm[None, :] >= uni.mandate_tenor[:, 0:1])
             & (ttm[None, :] <= uni.mandate_tenor[:, 1:2])
             & (((uni.mandate_sector_mask[:, None]
                  >> uni.instr_sector[None, :]) & 1).astype(bool)))
        fair = (120.0 + 55.0 * (uni.rating - 11) + 6.0 * np.maximum(ttm, 0.25)
                + self.fair_idio)
        return dict(v=v, alive_t=alive_t, mandate=m, fair=fair, ttm=ttm)

    # -- inner draws ---------------------------------------------------------------

    @staticmethod
    def _softmax_draw(gen, v: np.ndarray, cand: np.ndarray,
                      tilt: np.ndarray, T: float):
        logits = (v[cand] @ tilt) / T
        mx = logits.max()
        log_z = mx + np.log(np.exp(logits - mx).sum())
        gumb = -np.log(-np.log(gen.random(len(cand))))
        j = int(np.argmax(logits + gumb))
        return int(cand[j]), float(logits[j]), float(log_z), len(cand)

    def _true_side(self, gen, instr: int, tilt: np.ndarray, T: float,
                   v: np.ndarray) -> int:
        pair = self.tok_of[instr, :2]                # BUY, SELL senses
        lg = (v[pair] @ tilt) / T
        p_buy = 1.0 / (1.0 + np.exp(lg[1] - lg[0]))
        return int(E.SideTrue.BUY) if gen.random() < p_buy else int(E.SideTrue.SELL)

    # -- one day --------------------------------------------------------------------

    def emit_day(self, day_ordinal: int, grid_day: pa.Table,
                 arrivals: ArrivalPanel, alloc: EventIdAllocator,
                 ledger: PhiloxLedger) -> tuple[pa.Table, pa.Table, pa.Table]:
        uni, cfg = self.uni, self.cfg
        asof = int(uni.cal.days[day_ordinal])
        dc = self.day_context(asof)
        v, alive_t, mandate = dc["v"], dc["alive_t"], dc["mandate"]

        c_grid = np.asarray(grid_day["c"].combine_chunks().flatten()
                            ).reshape(grid_day.num_rows, self.d).astype(np.float64)
        T_grid = grid_day["temperature"].to_numpy(zero_copy_only=False).astype(np.float64)
        r_grid = grid_day["r"].to_numpy(zero_copy_only=False).astype(np.float64)

        sector_ok = ((uni.mandate_sector_mask[:, None]
                      >> np.arange(len(uni.spec.sectors))[None, :]) & 1).astype(bool)
        act_mult = uni.intensity_mult[day_ordinal]

        lines: list[dict] = []
        truths: list[dict] = []
        books: list[dict] = []

        for a in range(len(arrivals)):
            k_step = int(arrivals.step_k[a])
            c_t, T_t, r_t = c_grid[k_step], float(T_grid[k_step]), float(r_grid[k_step])
            sec = int(arrivals.sector[a])
            ts = int(arrivals.ts_us[a])
            gid = int(arrivals.grid_idx[a])

            gen, pk0, pk1, pctr = ledger.spawn_generator()

            # -- author -------------------------------------------------------
            ok = sector_ok[:, sec]
            wk = uni.base_activity * act_mult * np.exp(cfg.alpha_act * (uni.u @ c_t))
            wk = np.where(ok, wk, 0.0)
            tot = wk.sum()
            if tot <= 0:
                continue
            cl = int(np.searchsorted(np.cumsum(wk / tot), gen.random()))
            tilt = c_t + uni.u[cl]

            # -- candidate set --------------------------------------------------
            sec_alive = alive_t & (uni.instr_sector[self.tok_instr] == sec)
            cand_mask = sec_alive & mandate[cl][self.tok_instr]
            n_sec_alive = int(sec_alive.sum())

            # -- package plan -----------------------------------------------------
            uroll = gen.random()
            if uroll < cfg.p_switch:
                pkg_type, n_legs = int(E.PackageType.SWITCH), 2
            elif uroll < cfg.p_switch + cfg.p_list:
                pkg_type = int(E.PackageType.LIST)
                n_legs = 2 + int(gen.poisson(cfg.list_len_mean - 2))
            else:
                pkg_type, n_legs = int(E.PackageType.SINGLE), 1
            T_pkg = T_t * (cfg.package_temp_mult if n_legs > 1 else 1.0)

            legs: list[tuple[int, float, float, int]] = []
            if n_legs == 1:
                cand = np.nonzero(cand_mask)[0]
                if not len(cand):
                    continue
                legs.append(self._softmax_draw(gen, v, cand, tilt, T_t))
            else:
                signed = cand_mask & (self.tok_sense != int(E.Sense.UNDISC))
                cand = np.nonzero(signed)[0]
                if len(cand) < n_legs:
                    continue
                first = self._softmax_draw(gen, v, cand, tilt, T_pkg)
                legs.append(first)
                used = {int(self.tok_instr[first[0]])}
                if pkg_type == int(E.PackageType.SWITCH):
                    want = (int(E.Sense.SELL)
                            if self.tok_sense[first[0]] == int(E.Sense.BUY)
                            else int(E.Sense.BUY))
                    sel = [t for t in cand if self.tok_sense[t] == want
                           and int(self.tok_instr[t]) not in used]
                else:
                    want = int(self.tok_sense[first[0]])   # one-way list
                    sel = [t for t in cand if self.tok_sense[t] == want
                           and int(self.tok_instr[t]) not in used]
                for _ in range(n_legs - 1):
                    if not sel:
                        break
                    leg = self._softmax_draw(gen, v, np.asarray(sel), tilt, T_pkg)
                    legs.append(leg)
                    used.add(int(self.tok_instr[leg[0]]))
                    sel = [t for t in sel if int(self.tok_instr[t]) not in used]
                if len(legs) < 2:
                    pkg_type, n_legs = int(E.PackageType.SINGLE), 1

            n_legs = len(legs)
            eids = alloc.take(n_legs)
            pkg_id = int(eids[0])

            for ln, (tok, logit, log_z, n_cand) in enumerate(legs):
                instr = int(self.tok_instr[tok])
                sense = int(self.tok_sense[tok])
                if sense == int(E.Sense.UNDISC):
                    disclosed = int(E.SideDisclosed.TWO_WAY)
                    true_side = self._true_side(gen, instr, tilt, T_t, v)
                else:
                    disclosed = sense
                    true_side = sense

                # -- size ------------------------------------------------------
                qty = float(np.exp(gen.normal(uni.size_mu_log[cl], cfg.size_sigma)))
                qty = int(np.clip(round(qty / 25_000) * 25_000,
                                  100_000, uni.max_line[cl]))
                edges = (100_000, 1_000_000, 5_000_000)
                qb = int(np.searchsorted(edges, qty, side="right"))

                # -- auction ----------------------------------------------------
                fair = dc["fair"][instr] + cfg.widen_per_r_bp * (r_t - 1.0)
                n_dealers = 3 + int(gen.binomial(5, 0.65))
                in_panel = bool(uni.our_panel[cl])
                received = in_panel and gen.random() < cfg.received_given_panel

                slots_id, slots_px = [], []
                our_slot = -1
                if received:
                    ar = gen.random()
                    if ar < cfg.pass_prob:
                        action, our_px = int(E.Action.PASS), None
                    elif ar < cfg.pass_prob + cfg.timeout_prob:
                        action, our_px = int(E.Action.TIMEOUT), None
                    else:
                        action = (int(E.Action.AUTOQUOTE) if gen.random() < 0.7
                                  else int(E.Action.TRADER_QUOTE))
                        our_px = fair + cfg.our_width_bp + gen.normal(0, cfg.sigma_quote_bp)
                    if our_px is not None:
                        our_slot = 0
                        slots_id.append(0)
                        slots_px.append(our_px)
                else:
                    action, our_px = None, None
                n_comp = n_dealers - (1 if in_panel else 0)
                for dd in range(n_comp):
                    if gen.random() < cfg.quote_prob:
                        slots_id.append(dd + 1)
                        slots_px.append(fair + gen.normal(0, cfg.sigma_quote_bp))

                n_quotes = len(slots_px)
                out_roll = gen.random()
                if out_roll < cfg.cancel_prob:
                    eo, winner, cover = int(E.EnquiryOutcome.CANCELLED), -1, -1
                elif out_roll < cfg.cancel_prob + cfg.expire_prob or n_quotes == 0:
                    eo, winner, cover = int(E.EnquiryOutcome.EXPIRED), -1, -1
                else:
                    px = np.asarray(slots_px)
                    order = np.argsort(px)
                    best = float(px[order[0]])
                    p_trade = 1.0 / (1.0 + np.exp((best - fair - cfg.dnt_edge_bp)
                                                  / cfg.dnt_scale_bp))
                    if gen.random() < p_trade:
                        eo = int(E.EnquiryOutcome.TRADED)
                        winner = int(order[0])
                        cover = int(order[1]) if n_quotes > 1 else -1
                    else:
                        eo, winner, cover = int(E.EnquiryOutcome.DNT), -1, -1

                if not received:
                    our = int(E.OurResult.NOT_RECEIVED)
                elif eo != int(E.EnquiryOutcome.TRADED):
                    our = int(E.OurResult.NO_TRADE)
                elif our_slot < 0:
                    our = int(E.OurResult.NO_QUOTE)
                elif winner == our_slot:
                    our = int(E.OurResult.WON)
                elif cover == our_slot:
                    our = int(E.OurResult.COVER)
                else:
                    our = int(E.OurResult.LOST)

                traded = eo == int(E.EnquiryOutcome.TRADED)
                exec_sprd = float(slots_px[winner]) if traded else None
                cover_sprd = (float(slots_px[cover]) if traded and cover >= 0
                              else None)
                outcome_ts = ts + int(gen.uniform(60, 240)) * _US
                resp_ts = (ts + int(gen.uniform(5, 60)) * _US
                           if received and our_px is not None else None)

                lines.append(dict(
                    event_id=int(eids[ln]), package_id=pkg_id, line_no=ln,
                    n_lines=n_legs, package_type=pkg_type,
                    ts=ts, trade_date=asof, grid_idx=gid,
                    response_deadline=ts + 300 * _US,
                    client_id=cl, instrument_id=instr,
                    client_side_disclosed=disclosed,
                    qty_par=qty, qty_bucket=qb,
                    n_dealers=n_dealers,
                    anonymous=bool(gen.random() < 0.03),
                    platform=int(gen.random() * 3),
                    token_id=tok, client_side_true=true_side,
                    our_in_panel=in_panel, received=received,
                    action=action,
                    quote_sprd_bp=(float(our_px) if our_px is not None else None),
                    quote_px=None, response_ts=resp_ts,
                    enquiry_outcome=eo, our_result=our, outcome_ts=outcome_ts,
                    exec_sprd_bp=exec_sprd, exec_px=None,
                    cover_sprd_bp=cover_sprd, cover_px=None,
                    rec_token_id=None, rec_delta=None, rec_propensity=None,
                    policy_id=None,
                ))
                truths.append(dict(
                    event_id=int(eids[ln]), trade_date=asof,
                    log_z=log_z, logit_chosen=logit,
                    log_p_chosen=logit - log_z,
                    n_candidates=n_cand,
                    masked_count=n_sec_alive - n_cand,
                    philox_key0=pk0, philox_key1=pk1, philox_ctr=pctr,
                ))
                md = self.bundle.config.max_dealers
                did = np.full(md, -1, np.int16)
                pxb = np.full(md, np.nan, np.float32)
                rk = np.full(md, 255, np.uint8)
                if n_quotes:
                    did[:n_quotes] = slots_id
                    pxb[:n_quotes] = slots_px
                    rk[np.argsort(np.asarray(slots_px))[:n_quotes]] = \
                        np.arange(1, n_quotes + 1, dtype=np.uint8)
                books.append(dict(
                    event_id=int(eids[ln]), trade_date=asof,
                    dealer_id=did.tolist(), px_sprd_bp=pxb.tolist(),
                    valid_mask=int((1 << n_quotes) - 1),
                    rank=rk.tolist(), n_quotes=n_quotes,
                    winner_slot=winner, cover_slot=cover, our_slot=our_slot,
                ))

        t = self.bundle.tables
        return (_rows(lines, t["rfq_lines"]), _rows(truths, t["event_truth"]),
                _rows(books, t["auction_book"]))


def _rows(rows: list[dict], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pydict(
        {n: [r[n] for r in rows] for n in schema.names}, schema=schema)
