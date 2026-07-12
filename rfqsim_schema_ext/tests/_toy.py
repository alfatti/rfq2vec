"""Hand-built canonical tape for the projection/leakage tests.

Eleven rows chosen to cover the full (disclosure x outcome x our_result)
lattice the RevelationPolicy branches on, plus two off-tape rows (not
received / not in panel) that the projection must drop.
"""
from __future__ import annotations

import datetime as dt

import pyarrow as pa

from rfqsim.schema import enums as E
from rfqsim.schema.tables import SchemaConfig, rfq_lines_schema

US = 1_000_000


def ts_us(y: int, m: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> int:
    return int(dt.datetime(y, m, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp() * US)


def date_days(y: int, m: int, d: int) -> int:
    return (dt.date(y, m, d) - dt.date(1970, 1, 1)).days


def base_row(i: int, ts: int, trade_date: int) -> dict:
    """A received, disclosed-BUY, won, traded single -- rows below override."""
    return dict(
        event_id=i, package_id=i, line_no=0, n_lines=1,
        package_type=int(E.PackageType.SINGLE),
        ts=ts, trade_date=trade_date, grid_idx=42,
        response_deadline=ts + 300 * US,
        client_id=7, instrument_id=100 + i,
        client_side_disclosed=int(E.SideDisclosed.BUY),
        qty_par=1_000_000, qty_bucket=int(E.QtyBucket.ROUND),
        n_dealers=5, anonymous=False, platform=0,
        token_id=(100 + i) * 3 + int(E.Sense.BUY),
        client_side_true=int(E.SideTrue.BUY),
        our_in_panel=True, received=True,
        action=int(E.Action.AUTOQUOTE),
        quote_sprd_bp=85.0, quote_px=99.5, response_ts=ts + 30 * US,
        enquiry_outcome=int(E.EnquiryOutcome.TRADED),
        our_result=int(E.OurResult.WON),
        outcome_ts=ts + 120 * US,
        exec_sprd_bp=84.0, exec_px=99.55,
        cover_sprd_bp=86.5, cover_px=99.40,
        rec_token_id=None, rec_delta=None, rec_propensity=None, policy_id=None,
    )


def _two_way(r: dict, true_side: int) -> dict:
    r.update(
        client_side_disclosed=int(E.SideDisclosed.TWO_WAY),
        client_side_true=true_side,
        token_id=r["instrument_id"] * 3 + int(E.Sense.UNDISC),
    )
    return r


def make_canonical(cfg: SchemaConfig) -> pa.Table:
    day = date_days(2026, 1, 5)
    t0 = ts_us(2026, 1, 5, 14)
    R = E.OurResult
    O = E.EnquiryOutcome
    rows = []

    # r0: disclosed BUY, WON, TRADED (pure default)
    rows.append(base_row(0, t0 + 0 * 60 * US, day))

    # r1: disclosed SELL, LOST, TRADED
    r = base_row(1, t0 + 1 * 60 * US, day)
    r.update(client_side_disclosed=int(E.SideDisclosed.SELL),
             client_side_true=int(E.SideTrue.SELL),
             token_id=r["instrument_id"] * 3 + int(E.Sense.SELL),
             action=int(E.Action.TRADER_QUOTE), our_result=int(R.LOST))
    rows.append(r)

    # r2 + r3: a SWITCH package of two two-ways -- WON and COVER
    r = _two_way(base_row(2, t0 + 2 * 60 * US, day), int(E.SideTrue.BUY))
    r.update(package_id=2, n_lines=2, package_type=int(E.PackageType.SWITCH))
    rows.append(r)
    r = _two_way(base_row(3, t0 + 2 * 60 * US, day), int(E.SideTrue.SELL))
    r.update(package_id=2, line_no=1, n_lines=2,
             package_type=int(E.PackageType.SWITCH), our_result=int(R.COVER))
    rows.append(r)

    # r4: two-way, LOST, TRADED
    r = _two_way(base_row(4, t0 + 4 * 60 * US, day), int(E.SideTrue.BUY))
    r.update(our_result=int(R.LOST))
    rows.append(r)

    # r5: two-way, we passed (NO_QUOTE), enquiry TRADED elsewhere
    r = _two_way(base_row(5, t0 + 5 * 60 * US, day), int(E.SideTrue.SELL))
    r.update(action=int(E.Action.PASS), quote_sprd_bp=None, quote_px=None,
             our_result=int(R.NO_QUOTE))
    rows.append(r)

    # r6: two-way, DNT
    r = _two_way(base_row(6, t0 + 6 * 60 * US, day), int(E.SideTrue.BUY))
    r.update(action=int(E.Action.TRADER_QUOTE),
             enquiry_outcome=int(O.DNT), our_result=int(R.NO_TRADE),
             exec_sprd_bp=None, exec_px=None, cover_sprd_bp=None, cover_px=None)
    rows.append(r)

    # r7: two-way, CANCELLED
    r = _two_way(base_row(7, t0 + 7 * 60 * US, day), int(E.SideTrue.SELL))
    r.update(action=int(E.Action.TRADER_QUOTE),
             enquiry_outcome=int(O.CANCELLED), our_result=int(R.NO_TRADE),
             exec_sprd_bp=None, exec_px=None, cover_sprd_bp=None, cover_px=None)
    rows.append(r)

    # r8: in panel, NOT received (dropped by projection)
    r = base_row(8, t0 + 8 * 60 * US, day)
    r.update(received=False, our_result=int(R.NOT_RECEIVED),
             action=None, quote_sprd_bp=None, quote_px=None, response_ts=None)
    rows.append(r)

    # r9: not even in panel (dropped by projection)
    r = base_row(9, t0 + 9 * 60 * US, day)
    r.update(our_in_panel=False, received=False,
             our_result=int(R.NOT_RECEIVED),
             client_side_disclosed=int(E.SideDisclosed.SELL),
             client_side_true=int(E.SideTrue.SELL),
             action=None, quote_sprd_bp=None, quote_px=None, response_ts=None)
    rows.append(r)

    # r10: disclosed BUY, EXPIRED (we timed out)
    r = base_row(10, t0 + 10 * 60 * US, day)
    r.update(action=int(E.Action.TIMEOUT),
             quote_sprd_bp=None, quote_px=None, response_ts=None,
             enquiry_outcome=int(O.EXPIRED), our_result=int(R.NO_TRADE),
             exec_sprd_bp=None, exec_px=None, cover_sprd_bp=None, cover_px=None)
    rows.append(r)

    schema = rfq_lines_schema(cfg)
    cols = {name: [row[name] for row in rows] for name in schema.names}
    return pa.Table.from_pydict(cols, schema=schema)
