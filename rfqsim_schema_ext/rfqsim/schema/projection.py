"""Observable projection of the canonical tape.

The canonical rfq_lines table is policy-free ground truth. What a desk would
actually have seen is a deterministic function of that truth and an
information-revelation policy; I keep the policy here, as a parameter, so one
oracle tape supports many observable datasets (MNAR ablations are a re-run of
this function, not of the simulator).

project_observable does four things, in order:
  1. filters to rows we received (and checks received => our_in_panel),
  2. computes the revelation fields (side_revealed, side_reveal_ts),
  3. gates the policy-gated price columns (exec_*, cover_*),
  4. drops latents/selectors and casts to the declared observable schema.

Revelation semantics for two-way (undisclosed-side) enquiries are the MNAR
resolution we agreed on: the disguise is signal, so the true side surfaces
only per policy, stamped with its own side_reveal_ts.

audit_observable is the independent check: it takes a table that claims to be
observable and raises LeakageError unless the schema matches the declared
contract exactly and the null patterns are consistent with the policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from .enums import CoverReveal, EnquiryOutcome, OurResult, SideDisclosed
from .tables import (
    RFQ_COLUMN_CLASS,
    ColumnClass,
    SchemaConfig,
    observable_rfq_schema,
    rfq_lines_schema,
    validate_rfq_classification,
)


class ProjectionError(RuntimeError):
    """The canonical input violates an invariant the projection relies on."""


class LeakageError(RuntimeError):
    """A table claiming to be observable fails the leakage audit."""


@dataclass(frozen=True)
class RevelationPolicy:
    """What the desk learns, and when, about two-way enquiries and losing
    auctions.

    Side revelation applies only to two-way enquiries (disclosed sides are
    known at ts by definition; the projection asserts disclosed == true).
    reveal_on_won = True is the physical floor -- if we traded it, we know the
    direction; turning it off is an ablation, not a realism setting.
    """
    reveal_on_won: bool = True
    reveal_on_cover: bool = True
    reveal_on_lost: bool = False
    reveal_on_no_quote: bool = False
    reveal_on_dnt: bool = False
    reveal_on_cancelled: bool = False
    reveal_on_expired: bool = False
    side_reveal_lag_us: int = 0
    cover_view: CoverReveal = CoverReveal.WINNER_ONLY
    # Post-trade transparency proxy: when True, the print (exec_*) becomes
    # visible on traded enquiries we did not win, whether or not we quoted.
    exec_px_on_loss: bool = False


DEFAULT_POLICY = RevelationPolicy()
# Minimal-information venue: side only via our own trades, no cover feedback.
STRICT_POLICY = RevelationPolicy(
    reveal_on_cover=False, cover_view=CoverReveal.NONE
)


# -- small helpers -----------------------------------------------------------

_I64 = pa.int64()
_TS = pa.timestamp("us", tz="UTC")


def _ts_to_i64(arr):
    return pc.cast(arr, _I64)


def _i64_to_ts(arr):
    # int64 -> naive us -> assume UTC. Direct int64 -> tz-aware cast is not a
    # thing pyarrow does, so I go through the naive type explicitly.
    return pc.assume_timezone(pc.cast(arr, pa.timestamp("us")), timezone="UTC")


def _and_flag(arr, flag: bool):
    return pc.and_(arr, pa.scalar(bool(flag)))


def _any_true(arr) -> bool:
    out = pc.any(arr).as_py()
    return bool(out) if out is not None else False


def _all_true(arr) -> bool:
    out = pc.all(arr).as_py()
    return bool(out) if out is not None else True  # vacuous truth on empty


# -- projection --------------------------------------------------------------

def project_observable(
    table: pa.Table,
    cfg: SchemaConfig,
    policy: RevelationPolicy = DEFAULT_POLICY,
    strict: bool = True,
) -> pa.Table:
    canon = rfq_lines_schema(cfg)
    validate_rfq_classification(canon)

    got, want = set(table.column_names), set(canon.names)
    if got != want:
        raise ProjectionError(
            f"canonical table columns do not match rfq_lines schema: "
            f"unknown={sorted(got - want)}, missing={sorted(want - got)}"
        )

    # Row filter: the observable plane is what reached our desk.
    if strict and _any_true(pc.and_(table["received"], pc.invert(table["our_in_panel"]))):
        raise ProjectionError("invariant violated: received=True on a row with our_in_panel=False")
    t = table.filter(table["received"])

    disclosed = t["client_side_disclosed"]
    true_side = t["client_side_true"]
    is_two = pc.equal(disclosed, int(SideDisclosed.TWO_WAY))

    # Disclosed sides are truthful in this model; a mismatch means the
    # generator is broken, and I want to know before it poisons a dataset.
    if strict:
        codes_match = pc.equal(disclosed, true_side)  # BUY/SELL codes aligned by design
        if not _all_true(pc.or_(is_two, codes_match)):
            raise ProjectionError("invariant violated: disclosed side != true side on a disclosed row")

    eo = t["enquiry_outcome"]
    our = t["our_result"]
    traded = pc.equal(eo, int(EnquiryOutcome.TRADED))
    won = pc.equal(our, int(OurResult.WON))
    cover = pc.equal(our, int(OurResult.COVER))
    lost = pc.equal(our, int(OurResult.LOST))
    no_quote = pc.equal(our, int(OurResult.NO_QUOTE))

    # -- side revelation for two-ways ---------------------------------------
    reveal_traded = pc.or_(
        pc.or_(_and_flag(won, policy.reveal_on_won), _and_flag(cover, policy.reveal_on_cover)),
        pc.or_(_and_flag(lost, policy.reveal_on_lost), _and_flag(no_quote, policy.reveal_on_no_quote)),
    )
    reveal = pc.and_(traded, reveal_traded)
    for outcome, flag in (
        (EnquiryOutcome.DNT, policy.reveal_on_dnt),
        (EnquiryOutcome.CANCELLED, policy.reveal_on_cancelled),
        (EnquiryOutcome.EXPIRED, policy.reveal_on_expired),
    ):
        reveal = pc.or_(reveal, _and_flag(pc.equal(eo, int(outcome)), flag))

    null_i8 = pa.scalar(None, pa.int8())
    side_revealed = pc.if_else(
        is_two,
        pc.if_else(reveal, true_side, null_i8),
        disclosed,  # BUY/SELL codes coincide with SideTrue by construction
    )

    ts_i = _ts_to_i64(t["ts"])
    reveal_i = pc.add(_ts_to_i64(t["outcome_ts"]), pa.scalar(int(policy.side_reveal_lag_us), _I64))
    null_i64 = pa.scalar(None, _I64)
    side_reveal_ts = _i64_to_ts(
        pc.if_else(is_two, pc.if_else(reveal, reveal_i, null_i64), ts_i)
    )

    # -- policy-gated price columns ------------------------------------------
    if policy.cover_view == CoverReveal.ALL_QUOTERS:
        quoted = pc.or_(pc.or_(won, cover), lost)
        cover_ok = pc.and_(traded, quoted)
    elif policy.cover_view == CoverReveal.WINNER_ONLY:
        cover_ok = pc.and_(traded, won)
    else:
        cover_ok = _and_flag(traded, False)

    exec_ok = pc.and_(traded, pc.or_(won, pa.scalar(bool(policy.exec_px_on_loss))))

    null_f32 = pa.scalar(None, pa.float32())
    null_f64 = pa.scalar(None, pa.float64())
    gated = {
        "exec_sprd_bp": pc.if_else(exec_ok, t["exec_sprd_bp"], null_f32),
        "exec_px": pc.if_else(exec_ok, t["exec_px"], null_f64),
        "cover_sprd_bp": pc.if_else(cover_ok, t["cover_sprd_bp"], null_f32),
        "cover_px": pc.if_else(cover_ok, t["cover_px"], null_f64),
    }

    # -- assemble to the declared contract ------------------------------------
    obs_schema = observable_rfq_schema(cfg)
    computed = {"side_revealed": side_revealed, "side_reveal_ts": side_reveal_ts, **gated}
    cols = [computed[name] if name in computed else t[name] for name in obs_schema.names]
    out = pa.table(dict(zip(obs_schema.names, cols)))
    return out.cast(obs_schema)


# -- audit --------------------------------------------------------------------

def _forbidden_names() -> set[str]:
    return {n for n, c in RFQ_COLUMN_CLASS.items()
            if c in (ColumnClass.LATENT, ColumnClass.SELECTOR)}


def audit_observable(
    obs: pa.Table,
    cfg: SchemaConfig,
    policy: RevelationPolicy = DEFAULT_POLICY,
) -> None:
    """Independent leakage audit of a table claiming to be observable.

    Checks (raising LeakageError with specifics on the first failure):
      1. schema equals the declared observable contract (names AND types, in order),
      2. no latent/selector column is present under any name we classify,
      3. null patterns are consistent with the policy:
         - disclosed rows always carry side_revealed == disclosed at ts,
         - two-way rows carry a side only where the policy permits,
         - exec_*/cover_* non-null only where the gate allows,
         - side_reveal_ts >= ts wherever present.
    """
    expected = observable_rfq_schema(cfg)
    if obs.schema.names != expected.names:
        raise LeakageError(
            f"observable schema name mismatch: got {obs.schema.names}, want {expected.names}"
        )
    for got_f, want_f in zip(obs.schema, expected):
        if not got_f.type.equals(want_f.type):
            raise LeakageError(f"column '{got_f.name}': type {got_f.type} != declared {want_f.type}")

    present_forbidden = _forbidden_names() & set(obs.column_names)
    if present_forbidden:
        raise LeakageError(f"latent/selector columns present in observable table: {sorted(present_forbidden)}")

    disclosed = obs["client_side_disclosed"]
    is_two = pc.equal(disclosed, int(SideDisclosed.TWO_WAY))
    side = obs["side_revealed"]
    side_ts = obs["side_reveal_ts"]

    # Disclosed rows: side present, equal to disclosure, stamped at ts.
    side_ok = pc.and_kleene(pc.is_valid(side), pc.equal(side, disclosed))
    if not _all_true(pc.or_kleene(is_two, side_ok)):
        raise LeakageError("disclosed row without matching side_revealed")
    ts_ok = pc.equal(_ts_to_i64(side_ts), _ts_to_i64(obs["ts"]))
    if not _all_true(pc.or_kleene(is_two, ts_ok)):
        raise LeakageError("disclosed row with side_reveal_ts != ts")

    # Two-way rows: revelation only where the policy permits.
    eo = obs["enquiry_outcome"]
    our = obs["our_result"]
    traded = pc.equal(eo, int(EnquiryOutcome.TRADED))
    won = pc.equal(our, int(OurResult.WON))
    cover = pc.equal(our, int(OurResult.COVER))
    lost = pc.equal(our, int(OurResult.LOST))
    no_quote = pc.equal(our, int(OurResult.NO_QUOTE))
    allowed = pc.and_(traded, pc.or_(
        pc.or_(_and_flag(won, policy.reveal_on_won), _and_flag(cover, policy.reveal_on_cover)),
        pc.or_(_and_flag(lost, policy.reveal_on_lost), _and_flag(no_quote, policy.reveal_on_no_quote)),
    ))
    for outcome, flag in (
        (EnquiryOutcome.DNT, policy.reveal_on_dnt),
        (EnquiryOutcome.CANCELLED, policy.reveal_on_cancelled),
        (EnquiryOutcome.EXPIRED, policy.reveal_on_expired),
    ):
        allowed = pc.or_(allowed, _and_flag(pc.equal(eo, int(outcome)), flag))
    if _any_true(pc.and_(is_two, pc.and_(pc.is_valid(side), pc.invert(allowed)))):
        raise LeakageError("two-way side revealed where the policy forbids it")

    # Gated columns: non-null implies the gate was open.
    if policy.cover_view == CoverReveal.ALL_QUOTERS:
        cover_gate = pc.and_(traded, pc.or_(pc.or_(won, cover), lost))
    elif policy.cover_view == CoverReveal.WINNER_ONLY:
        cover_gate = pc.and_(traded, won)
    else:
        cover_gate = _and_flag(traded, False)
    exec_gate = pc.and_(traded, pc.or_(won, pa.scalar(bool(policy.exec_px_on_loss))))
    for col, gate in (("exec_sprd_bp", exec_gate), ("exec_px", exec_gate),
                      ("cover_sprd_bp", cover_gate), ("cover_px", cover_gate)):
        if _any_true(pc.and_(pc.is_valid(obs[col]), pc.invert(gate))):
            raise LeakageError(f"{col} non-null outside its revelation gate")

    # Revelation cannot precede the enquiry.
    late = pc.and_kleene(pc.is_valid(side_ts),
                         pc.less(_ts_to_i64(side_ts), _ts_to_i64(obs["ts"])))
    if _any_true(late):
        raise LeakageError("side_reveal_ts precedes enquiry ts")
