"""Enum registries for the rfqsim data plane.

I keep every categorical column as a small signed/unsigned int in Parquet and
carry the code -> name mapping here. The same mapping is embedded, as JSON,
into the schema metadata of every file we write (see tables._meta), so a bare
parquet file remains self-describing even without this module on the path.
Nothing in the hot path ever touches a string column.
"""
from __future__ import annotations

import json
from enum import IntEnum

_REGISTRY: dict[str, type[IntEnum]] = {}


def _register(cls: type[IntEnum]) -> type[IntEnum]:
    _REGISTRY[cls.__name__] = cls
    return cls


def registry() -> dict[str, type[IntEnum]]:
    """All data enums, by class name."""
    return dict(_REGISTRY)


def registry_json() -> str:
    """The JSON blob I embed into every schema's metadata."""
    return json.dumps(
        {name: {m.name: int(m.value) for m in cls} for name, cls in sorted(_REGISTRY.items())},
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Token / side semantics.
# BUY/SELL codes are aligned across Sense, SideDisclosed and SideTrue on
# purpose: the projection casts between them without a lookup.
# ---------------------------------------------------------------------------

@_register
class Sense(IntEnum):
    """Distributional sense of a token: (CUSIP, sense) is the vocabulary.
    UNDISC is a genuine third sense (client requested a two-way market), not
    missing data -- the disguise itself keeps its own company."""
    BUY = 0
    SELL = 1
    UNDISC = 2


@_register
class SideDisclosed(IntEnum):
    """What the client disclosed on the wire."""
    BUY = 0
    SELL = 1
    TWO_WAY = 2


@_register
class SideTrue(IntEnum):
    """Ground-truth direction (oracle plane only until revealed)."""
    BUY = 0
    SELL = 1


@_register
class PackageType(IntEnum):
    """The 'sentence' type. SWITCH and LIST are single-author, frozen-context
    emissions; PT is a portfolio trade."""
    SINGLE = 0
    SWITCH = 1
    LIST = 2
    PT = 3


@_register
class QtyBucket(IntEnum):
    MICRO = 0
    ODD = 1
    ROUND = 2
    BLOCK = 3


@_register
class Action(IntEnum):
    """Our desk's response to a received enquiry."""
    AUTOQUOTE = 0
    TRADER_QUOTE = 1
    PASS = 2
    TIMEOUT = 3


@_register
class EnquiryOutcome(IntEnum):
    """Market-level fate of the enquiry -- well-defined for every row of the
    canonical tape, including enquiries we never saw."""
    TRADED = 0
    DNT = 1
    CANCELLED = 2
    EXPIRED = 3


@_register
class OurResult(IntEnum):
    """Competitive resolution from our point of view. WON/COVER/LOST/NO_QUOTE
    apply only when the enquiry TRADED; NO_TRADE covers DNT/CANCELLED/EXPIRED
    regardless of whether we quoted (our own behaviour lives in `action`);
    NOT_RECEIVED marks canonical-plane rows outside our received set."""
    WON = 0
    COVER = 1
    LOST = 2
    NO_QUOTE = 3
    NO_TRADE = 4
    NOT_RECEIVED = 5


@_register
class Archetype(IntEnum):
    REAL_MONEY = 0
    HEDGE_FUND = 1
    INSURER = 2
    PENSION = 3
    BANK = 4
    PT_DESK = 5


@_register
class Region(IntEnum):
    AMER = 0
    EMEA = 1
    APAC = 2


@_register
class Seniority(IntEnum):
    SR_SECURED = 0
    SR_UNSECURED = 1
    SR_SUBORDINATED = 2
    JR_SUBORDINATED = 3


@_register
class InstrumentEventField(IntEnum):
    """Discrete instrument state changes -- the event-driven affinity-shock
    channel. Values are stamped in event time in instrument_events."""
    RATING = 0
    TAP = 1
    INDEX_ADD = 2
    INDEX_DROP = 3
    OTR_ON = 4
    OTR_OFF = 5


@_register
class CoverReveal(IntEnum):
    """Who gets to see the cover price under the revelation policy."""
    NONE = 0
    WINNER_ONLY = 1
    ALL_QUOTERS = 2


@_register
class CheckStatus(IntEnum):
    PASS = 0
    WARN = 1
    FAIL = 2


class ColumnClass(IntEnum):
    """Classification of canonical rfq_lines columns for the observable
    projection. Deliberately NOT a data enum (never stored), so I don't
    register it.

    OBSERVABLE    survives projection verbatim
    LATENT        oracle plane only; dropped by projection
    SELECTOR      used to filter rows, then dropped
    POLICY_GATED  survives, but projection may null it per RevelationPolicy
    """
    OBSERVABLE = 0
    LATENT = 1
    SELECTOR = 2
    POLICY_GATED = 3


# ---------------------------------------------------------------------------
# Small static registries that aren't enums.
# ---------------------------------------------------------------------------

# 22-notch composite scale, 1 = AAA ... 22 = D. Stored as int8 everywhere.
RATING_NOTCHES: tuple[str, ...] = (
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
)
RATING_CODE: dict[str, int] = {name: i + 1 for i, name in enumerate(RATING_NOTCHES)}
HY_THRESHOLD: int = RATING_CODE["BB+"]  # rating >= this code => high yield

# Default sector set; the run config owns the real one, which must match the
# MMPP sector chains. Order defines sector_id and the layout of every
# fixed-size per-sector array (regime vector, lambda_sector).
DEFAULT_SECTORS: tuple[str, ...] = (
    "Energy", "Basic_Industry", "Capital_Goods", "Consumer_Cyclical",
    "Consumer_NonCyclical", "Healthcare", "Media_Telecom", "Services",
    "Technology", "Transportation", "Utilities", "Financials",
)

# Advisory only -- generation owns the real edges. I keep them here so the
# QtyBucket enum has one documented default meaning (USD par face).
QTY_BUCKET_DEFAULT_EDGES: dict[str, tuple[int, int | None]] = {
    "MICRO": (0, 100_000),
    "ODD": (100_000, 1_000_000),
    "ROUND": (1_000_000, 5_000_000),
    "BLOCK": (5_000_000, None),
}
