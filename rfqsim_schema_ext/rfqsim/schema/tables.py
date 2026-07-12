"""PyArrow schemas for the rfqsim data plane.

Design principles this module makes physical:

  1. One canonical fact table, two planes. `rfq_lines` is the oracle-complete
     tape; the observable dataset is `observable_rfq_schema`, produced only by
     projection.project_observable. Every rfq_lines column carries an explicit
     ColumnClass; an unclassified column is a build-time error, which is the
     tripwire that stops silent leakage under schema evolution.
  2. Store state, not probabilities. Nothing here materializes a propensity;
     event_truth stores (logZ, chosen logit, Philox coordinates) from which any
     probability, likelihood or uplift is exactly recomputable.
  3. Point-in-time by construction. Dimensions are event-sourced
     (instrument_events, SCD mandates/panels) plus daily compiled snapshots.
  4. The drift budget is a column. context_grid persists the realized l1 step,
     so the budget audit is a scan, not a rerun.

Schemas are parametric in the run dials (d, p, n_sectors, ...) via
SchemaConfig; build_schemas(cfg) returns the full bundle with sort keys,
partitioning and plane assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pyarrow as pa

from . import enums
from .enums import ColumnClass

SCHEMA_VERSION = "0.2.0"   # 0.2.0: added oracle_token_vectors + oracle_attribute_gain

# Canonical arrow types I reuse everywhere.
TS = pa.timestamp("us", tz="UTC")
D32 = pa.date32()


class SchemaClassificationError(RuntimeError):
    """A canonical rfq_lines column is missing from, or unknown to, the
    ColumnClass registry. This is the anti-leakage tripwire firing."""


@dataclass(frozen=True)
class SchemaConfig:
    """Run dials that shape the physical schema.

    d              context / embedding dimension (walk state, u_k, eps_w)
    p              attribute feature dimension (x_w fed to B)
    n_sectors      number of sector MMPP chains; fixes per-sector array widths
    max_dealers    auction panel width (fixed-size arrays in auction_book)
    n_norm_params  parameters of the deterministic norm schedule s_w(t)
    run_id         stamped into every file's schema metadata
    """
    d: int
    p: int
    n_sectors: int = len(enums.DEFAULT_SECTORS)
    max_dealers: int = 8
    n_norm_params: int = 4
    run_id: str = "unset"

    def __post_init__(self) -> None:
        if self.d <= 0 or self.p <= 0:
            raise ValueError("d and p must be positive")
        if not (1 <= self.n_sectors <= 32):
            raise ValueError("n_sectors must be in [1, 32] (sector_mask is u32)")
        if not (1 <= self.max_dealers <= 16):
            raise ValueError("max_dealers must be in [1, 16]")
        if self.n_norm_params <= 0:
            raise ValueError("n_norm_params must be positive")


def _f(name: str, typ: pa.DataType, nullable: bool = False, desc: str = "",
       klass: ColumnClass | None = None) -> pa.Field:
    md = {"desc": desc}
    if klass is not None:
        md["class"] = klass.name
    return pa.field(name, typ, nullable=nullable, metadata=md)


def _meta(cfg: SchemaConfig, table: str) -> dict[str, str]:
    """Schema-level metadata: every file is self-describing."""
    return {
        "rfqsim.table": table,
        "rfqsim.schema_version": SCHEMA_VERSION,
        "rfqsim.run_id": cfg.run_id,
        "rfqsim.d": str(cfg.d),
        "rfqsim.p": str(cfg.p),
        "rfqsim.n_sectors": str(cfg.n_sectors),
        "rfqsim.enums": enums.registry_json(),
    }


# ===========================================================================
# Canonical fact table: rfq_lines (line grain; the package is the sentence).
# ===========================================================================

# The classification IS the leakage contract. Every canonical column must
# appear here exactly once; validate_rfq_classification enforces bijection.
RFQ_COLUMN_CLASS: dict[str, ColumnClass] = {
    # -- identity / package ("sentence") structure -------------------------
    "event_id": ColumnClass.OBSERVABLE,
    "package_id": ColumnClass.OBSERVABLE,
    "line_no": ColumnClass.OBSERVABLE,
    "n_lines": ColumnClass.OBSERVABLE,
    "package_type": ColumnClass.OBSERVABLE,
    # -- time ---------------------------------------------------------------
    "ts": ColumnClass.OBSERVABLE,
    "trade_date": ColumnClass.OBSERVABLE,
    # grid_idx is information-free given ts (floor of in-session minute), but
    # it is a foreign key into the oracle plane; I drop it so the observable
    # plane carries no oracle-table references at all.
    "grid_idx": ColumnClass.LATENT,
    "response_deadline": ColumnClass.OBSERVABLE,
    # -- enquiry ------------------------------------------------------------
    "client_id": ColumnClass.OBSERVABLE,
    "instrument_id": ColumnClass.OBSERVABLE,
    "client_side_disclosed": ColumnClass.OBSERVABLE,
    "qty_par": ColumnClass.OBSERVABLE,
    "qty_bucket": ColumnClass.OBSERVABLE,
    "n_dealers": ColumnClass.OBSERVABLE,
    "anonymous": ColumnClass.OBSERVABLE,
    "platform": ColumnClass.OBSERVABLE,
    # -- latents ------------------------------------------------------------
    "token_id": ColumnClass.LATENT,
    "client_side_true": ColumnClass.LATENT,
    # -- panel / receipt selectors -------------------------------------------
    "our_in_panel": ColumnClass.SELECTOR,
    "received": ColumnClass.SELECTOR,
    # -- our participation ----------------------------------------------------
    "action": ColumnClass.OBSERVABLE,
    "quote_sprd_bp": ColumnClass.OBSERVABLE,
    "quote_px": ColumnClass.OBSERVABLE,
    "response_ts": ColumnClass.OBSERVABLE,
    # -- outcome --------------------------------------------------------------
    "enquiry_outcome": ColumnClass.OBSERVABLE,
    "our_result": ColumnClass.OBSERVABLE,
    "outcome_ts": ColumnClass.OBSERVABLE,
    "exec_sprd_bp": ColumnClass.POLICY_GATED,
    "exec_px": ColumnClass.POLICY_GATED,
    "cover_sprd_bp": ColumnClass.POLICY_GATED,
    "cover_px": ColumnClass.POLICY_GATED,
    # -- logged policy (our own actions: observable by construction) ----------
    "rec_token_id": ColumnClass.OBSERVABLE,
    "rec_delta": ColumnClass.OBSERVABLE,
    "rec_propensity": ColumnClass.OBSERVABLE,
    "policy_id": ColumnClass.OBSERVABLE,
}


def rfq_lines_schema(cfg: SchemaConfig) -> pa.Schema:
    C = ColumnClass
    fields = [
        _f("event_id", pa.uint64(), desc="(shard_id << 48) | counter; sort key is (ts, event_id)", klass=C.OBSERVABLE),
        _f("package_id", pa.uint64(), desc="sentence id; equals event_id of first line for singles", klass=C.OBSERVABLE),
        _f("line_no", pa.uint16(), desc="0-based line index within the package", klass=C.OBSERVABLE),
        _f("n_lines", pa.uint16(), desc="package size (PT lists can exceed 255)", klass=C.OBSERVABLE),
        _f("package_type", pa.int8(), desc="enums.PackageType", klass=C.OBSERVABLE),
        _f("ts", TS, desc="enquiry arrival", klass=C.OBSERVABLE),
        _f("trade_date", D32, desc="session date; hive partition key", klass=C.OBSERVABLE),
        _f("grid_idx", pa.uint32(), desc="FK -> context_grid; deterministic from ts", klass=C.LATENT),
        _f("response_deadline", TS, desc="platform response deadline", klass=C.OBSERVABLE),
        _f("client_id", pa.uint32(), desc="FK -> clients (the author)", klass=C.OBSERVABLE),
        _f("instrument_id", pa.uint32(), desc="FK -> instruments", klass=C.OBSERVABLE),
        _f("client_side_disclosed", pa.int8(), desc="enums.SideDisclosed as shown on the wire", klass=C.OBSERVABLE),
        _f("qty_par", pa.int64(), desc="par face, currency units", klass=C.OBSERVABLE),
        _f("qty_bucket", pa.int8(), desc="enums.QtyBucket", klass=C.OBSERVABLE),
        _f("n_dealers", pa.uint8(), desc="competition size shown by the platform", klass=C.OBSERVABLE),
        _f("anonymous", pa.bool_(), desc="anonymous protocol flag", klass=C.OBSERVABLE),
        _f("platform", pa.uint8(), desc="venue code (run-config registry)", klass=C.OBSERVABLE),
        _f("token_id", pa.uint32(), desc="FK -> token_map: the chosen (CUSIP, sense); UNDISC sense for two-ways", klass=C.LATENT),
        _f("client_side_true", pa.int8(), desc="enums.SideTrue ground truth", klass=C.LATENT),
        _f("our_in_panel", pa.bool_(), desc="we were on the client's panel for this enquiry", klass=C.SELECTOR),
        _f("received", pa.bool_(), desc="the enquiry reached our desk (implies our_in_panel)", klass=C.SELECTOR),
        _f("action", pa.int8(), nullable=True, desc="enums.Action; null iff not received", klass=C.OBSERVABLE),
        _f("quote_sprd_bp", pa.float32(), nullable=True, desc="our quote, spread space (primary)", klass=C.OBSERVABLE),
        _f("quote_px", pa.float64(), nullable=True, desc="our quote, price space; null when reference curve disabled", klass=C.OBSERVABLE),
        _f("response_ts", TS, nullable=True, desc="when we responded", klass=C.OBSERVABLE),
        _f("enquiry_outcome", pa.int8(), desc="enums.EnquiryOutcome (market-level fact)", klass=C.OBSERVABLE),
        _f("our_result", pa.int8(), desc="enums.OurResult (our POV)", klass=C.OBSERVABLE),
        _f("outcome_ts", TS, desc="terminal-state time; anchors revelation lags", klass=C.OBSERVABLE),
        _f("exec_sprd_bp", pa.float32(), nullable=True, desc="print spread; oracle always knows when TRADED", klass=C.POLICY_GATED),
        _f("exec_px", pa.float64(), nullable=True, desc="print price", klass=C.POLICY_GATED),
        _f("cover_sprd_bp", pa.float32(), nullable=True, desc="cover spread; null when <2 quotes", klass=C.POLICY_GATED),
        _f("cover_px", pa.float64(), nullable=True, desc="cover price", klass=C.POLICY_GATED),
        _f("rec_token_id", pa.uint32(), nullable=True, desc="logged recommendation (our action)", klass=C.OBSERVABLE),
        _f("rec_delta", pa.float32(), nullable=True, desc="logged additive log-odds tilt", klass=C.OBSERVABLE),
        _f("rec_propensity", pa.float32(), nullable=True, desc="logging-policy propensity (OPE-native)", klass=C.OBSERVABLE),
        _f("policy_id", pa.uint16(), nullable=True, desc="logging policy version", klass=C.OBSERVABLE),
    ]
    return pa.schema(fields, metadata=_meta(cfg, "rfq_lines"))


# Fields the projection ADDS (they exist in no canonical table; the leakage
# policy is a projection parameter, so one oracle tape supports many
# observable datasets).
def projection_added_fields() -> list[pa.Field]:
    return [
        _f("side_revealed", pa.int8(), nullable=True,
           desc="enums.SideTrue as known to us post-revelation; null = never revealed"),
        _f("side_reveal_ts", TS, nullable=True,
           desc="when the side became known to us (ts for disclosed; outcome_ts+lag for revealed two-ways)"),
    ]


def observable_rfq_schema(cfg: SchemaConfig) -> pa.Schema:
    """The declared contract of the observable plane. project_observable casts
    its output to exactly this schema; audit_observable checks against it."""
    canon = rfq_lines_schema(cfg)
    keep = [f for f in canon
            if RFQ_COLUMN_CLASS[f.name] in (ColumnClass.OBSERVABLE, ColumnClass.POLICY_GATED)]
    # Revelation fields slot after the outcome block, before the logged policy.
    idx = next(i for i, f in enumerate(keep) if f.name == "rec_token_id")
    keep[idx:idx] = projection_added_fields()
    return pa.schema(keep, metadata=_meta(cfg, "rfq_lines_obs"))


def validate_rfq_classification(schema: pa.Schema) -> None:
    """Bijection check between the canonical schema and RFQ_COLUMN_CLASS.
    Any drift -- a new column without a class, or a stale class entry --
    raises before a single row can be written."""
    names = set(schema.names)
    classed = set(RFQ_COLUMN_CLASS)
    missing = sorted(names - classed)
    stale = sorted(classed - names)
    if missing or stale:
        raise SchemaClassificationError(
            f"rfq_lines classification drift: unclassified={missing}, stale={stale}"
        )


# ===========================================================================
# Shared-plane dimensions.
# ===========================================================================

def issuers_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("issuer_id", pa.uint32()),
        _f("name", pa.string(), desc="synthetic issuer name"),
        _f("sector_id", pa.uint16(), desc="index into the run's sector registry (MMPP chains)"),
        _f("sub_sector_id", pa.uint16()),
        _f("is_financial", pa.bool_()),
    ], metadata=_meta(cfg, "issuers"))


def instruments_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("instrument_id", pa.uint32()),
        _f("cusip", pa.string(), desc="synthetic 9-char identifier"),
        _f("issuer_id", pa.uint32()),
        _f("coupon", pa.float32(), desc="pct"),
        _f("issue_date", D32),
        _f("maturity_date", D32),
        _f("amt_issued", pa.int64(), desc="par face at issue, currency units"),
        _f("rating_at_issue", pa.int8(), desc="enums.RATING_NOTCHES code"),
        _f("seniority", pa.int8(), desc="enums.Seniority"),
        _f("is_144a", pa.bool_()),
        _f("benchmark_tenor_yrs", pa.float32(), desc="benchmark tenor at issue"),
    ], metadata=_meta(cfg, "instruments"))


def instrument_events_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("instrument_id", pa.uint32()),
        _f("ts", TS, desc="event time; the affinity-shock channel is stamped here"),
        _f("field", pa.int8(), desc="enums.InstrumentEventField"),
        _f("old_value", pa.float64(), nullable=True),
        _f("new_value", pa.float64(), nullable=True),
    ], metadata=_meta(cfg, "instrument_events"))


def instrument_state_daily_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("trade_date", D32),
        _f("instrument_id", pa.uint32()),
        _f("rating", pa.int8()),
        _f("age_yrs", pa.float32()),
        _f("ttm_yrs", pa.float32()),
        _f("is_otr", pa.bool_()),
        _f("index_mask", pa.uint32(), desc="index membership bitmask"),
        _f("x", pa.list_(pa.float32(), cfg.p),
           desc="materialized x_w(t): the exact attribute vector fed to B; "
                "single source of truth for oracle and learner alike"),
    ], metadata=_meta(cfg, "instrument_state_daily"))


def clients_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("client_id", pa.uint32()),
        _f("archetype", pa.int8(), desc="enums.Archetype; desk-known client type"),
        _f("region", pa.int8(), desc="enums.Region"),
        _f("onboard_date", D32),
        _f("offboard_date", D32, nullable=True),
    ], metadata=_meta(cfg, "clients"))


def client_mandates_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("mandate_id", pa.uint32()),
        _f("client_id", pa.uint32()),
        _f("rating_min", pa.int8(), desc="best notch allowed (inclusive)"),
        _f("rating_max", pa.int8(), desc="worst notch allowed (inclusive)"),
        _f("tenor_min_yrs", pa.float32()),
        _f("tenor_max_yrs", pa.float32()),
        _f("sector_mask", pa.uint32(), desc="allowed sectors bitmask (bit i = sector_id i)"),
        _f("max_line_par", pa.int64()),
        _f("effective_from", D32),
        _f("effective_to", D32, nullable=True, desc="null = open"),
    ], metadata=_meta(cfg, "client_mandates"))


def client_dealer_panel_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("client_id", pa.uint32()),
        _f("dealer_id", pa.int16(), desc="0 = us; competitors 1..N"),
        _f("tier", pa.int8()),
        _f("effective_from", D32),
        _f("effective_to", D32, nullable=True),
    ], metadata=_meta(cfg, "client_dealer_panel"))


def token_map_schema(cfg: SchemaConfig) -> pa.Schema:
    # The vocabulary is public structure -- which token a given RFQ
    # instantiates is what's latent, and that lives in rfq_lines.token_id.
    return pa.schema([
        _f("token_id", pa.uint32()),
        _f("instrument_id", pa.uint32()),
        _f("sense", pa.int8(), desc="enums.Sense"),
        _f("born_ts", TS, desc="token birth (issuance)"),
        _f("retired_ts", TS, nullable=True, desc="maturity/call; null = live"),
    ], metadata=_meta(cfg, "token_map"))


def calendar_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("trade_date", D32),
        _f("session_open", TS),
        _f("session_close", TS),
        _f("is_half_day", pa.bool_()),
    ], metadata=_meta(cfg, "calendar"))


def primary_calendar_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("deal_id", pa.uint32()),
        _f("instrument_id", pa.uint32()),
        _f("pricing_ts", TS, desc="drives token birth and issuance-day shocks"),
        _f("deal_size", pa.int64()),
    ], metadata=_meta(cfg, "primary_calendar"))


# ===========================================================================
# Oracle plane: latent state, per-event truth, auction book.
# ===========================================================================

def context_grid_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("grid_idx", pa.uint32(), desc="global in-session 1-min step index"),
        _f("ts", TS),
        _f("trade_date", D32),
        _f("regime", pa.list_(pa.int8(), cfg.n_sectors),
           desc="per-sector CTMC state vector (sector-level bidimensional chains)"),
        _f("r", pa.float32(), desc="radial component |c|: intensity load and inverse temperature driver"),
        _f("temperature", pa.float32(), desc="realized softmax temperature"),
        _f("c", pa.list_(pa.float32(), cfg.d), desc="context vector"),
        _f("lambda_sector", pa.list_(pa.float32(), cfg.n_sectors), desc="realized per-sector arrival intensity"),
        _f("step_l1", pa.float32(), nullable=True,
           desc="realized |c_t - c_{t-1}|_1; null at session open. The drift-budget audit is a scan of this column."),
    ], metadata=_meta(cfg, "context_grid"))


def regime_path_schema(cfg: SchemaConfig) -> pa.Schema:
    # Redundant with context_grid.regime, but shaped for the existing MMPP EM
    # tooling (sojourn-level records).
    return pa.schema([
        _f("sector_id", pa.uint16()),
        _f("sojourn_no", pa.uint32()),
        _f("state", pa.int8()),
        _f("t_start", TS),
        _f("t_end", TS, nullable=True, desc="null = open at run end"),
    ], metadata=_meta(cfg, "regime_path"))


def client_activity_daily_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("trade_date", D32),
        _f("client_id", pa.uint32()),
        _f("activity", pa.float32(), desc="OU state"),
        _f("intensity_mult", pa.float32(), desc="realized multiplier"),
    ], metadata=_meta(cfg, "client_activity_daily"))


def oracle_clients_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("client_id", pa.uint32()),
        _f("u", pa.list_(pa.float32(), cfg.d), desc="affinity tilt u_k (static by design; nonstationarity lives in activity/mandates/aging/primary)"),
        _f("u_norm", pa.float32(), desc="|u_k| convenience; concentration anchor"),
    ], metadata=_meta(cfg, "oracle_clients"))


def oracle_embeddings_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("token_id", pa.uint32()),
        _f("eps", pa.list_(pa.float32(), cfg.d),
           desc="purely distributional residual; v_w(t) = B x_w(t) + eps_w. "
                "Static by hypothesis -- the eps-stability gate tests exactly this."),
        _f("norm_params", pa.list_(pa.float32(), cfg.n_norm_params),
           desc="deterministic norm schedule s_w(t) parameters; daily values are "
                "derived, never materialized (store state, not probabilities)"),
    ], metadata=_meta(cfg, "oracle_embeddings"))


def oracle_token_vectors_schema(cfg: SchemaConfig) -> pa.Schema:
    # The realized word vectors at run start (asof = calendar day 0), the one
    # thing the residual+norm oracle could not reconstruct on its own. This is
    # the ground-truth geometry the substitution / side-sense probes align a
    # trained model's embeddings against: v_w = s_w * normalize(B x_w + eps_w),
    # so directions carry substitution structure and the norm carries frequency.
    return pa.schema([
        _f("token_id", pa.uint32()),
        _f("v", pa.list_(pa.float32(), cfg.d),
           desc="realized v_(w,side) at run start (asof = calendar day 0); "
                "full vector including the norm s_w -- direction = substitution "
                "geometry, ||v|| = frequency/importance"),
    ], metadata=_meta(cfg, "oracle_token_vectors"))


def oracle_attribute_gain_schema(cfg: SchemaConfig) -> pa.Schema:
    # The attribute gain matrix B (p feature rows x d latent columns). Lets a
    # probe split the recovered geometry into the shared low-rank part driven by
    # observable bond features (B x) and the full-rank idiosyncratic residual
    # (eps) -- the crux of the small-vs-large substitution-rank test.
    return pa.schema([
        _f("feat_idx", pa.uint16(), desc="attribute feature index 0..p-1 (row of B)"),
        _f("gain", pa.list_(pa.float32(), cfg.d),
           desc="row of the attribute gain matrix B mapping feature -> latent d"),
    ], metadata=_meta(cfg, "oracle_attribute_gain"))


def event_truth_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("event_id", pa.uint64()),
        _f("trade_date", D32, desc="denormalized for partition pruning"),
        _f("log_z", pa.float64(), desc="logZ over the mandate-masked candidate set at c_t + u_k"),
        _f("logit_chosen", pa.float64(), desc="<v_w, c_t + u_k>/T for the chosen token"),
        _f("log_p_chosen", pa.float64(), desc="= logit_chosen - log_z; kept for convenience"),
        _f("n_candidates", pa.uint32()),
        _f("masked_count", pa.uint32(), desc="tokens removed by mandate mask (structural zeros)"),
        _f("philox_key0", pa.uint64()),
        _f("philox_key1", pa.uint64()),
        _f("philox_ctr", pa.uint64(), desc="counter block start; exact replay via PhiloxLedger.generator"),
    ], metadata=_meta(cfg, "event_truth"))


def auction_book_schema(cfg: SchemaConfig) -> pa.Schema:
    m = cfg.max_dealers
    return pa.schema([
        _f("event_id", pa.uint64()),
        _f("trade_date", D32),
        _f("dealer_id", pa.list_(pa.int16(), m), desc="-1 padding beyond n_quotes"),
        _f("px_sprd_bp", pa.list_(pa.float32(), m), desc="quotes in spread space; NaN padding"),
        _f("valid_mask", pa.uint16(), desc="bit i set = slot i holds a live quote"),
        _f("rank", pa.list_(pa.uint8(), m), desc="1 = best; 255 padding"),
        _f("n_quotes", pa.uint8()),
        _f("winner_slot", pa.int8(), desc="-1 if enquiry did not trade"),
        _f("cover_slot", pa.int8(), desc="-1 if <2 quotes"),
        _f("our_slot", pa.int8(), desc="-1 if we were not in the auction"),
    ], metadata=_meta(cfg, "auction_book"))


def validation_report_schema(cfg: SchemaConfig) -> pa.Schema:
    return pa.schema([
        _f("run_id", pa.string()),
        _f("check", pa.string(), desc="e.g. zipf_tail_beta, z_c_dispersion, window_sum_elevation, eps_stability, drift_budget"),
        _f("stratum", pa.string(), desc="e.g. all, liquid_decile_1, regime=stress"),
        _f("statistic", pa.string()),
        _f("value", pa.float64()),
        _f("target_lo", pa.float64(), nullable=True),
        _f("target_hi", pa.float64(), nullable=True),
        _f("status", pa.int8(), desc="enums.CheckStatus"),
        _f("details", pa.string(), nullable=True, desc="JSON blob"),
    ], metadata=_meta(cfg, "validation_report"))


# ===========================================================================
# Bundle.
# ===========================================================================

_ORACLE = "oracle"
_SHARED = "shared"
_CANONICAL = "canonical"


@dataclass(frozen=True)
class SchemaBundle:
    config: SchemaConfig
    tables: Mapping[str, pa.Schema]
    sort_keys: Mapping[str, tuple[tuple[str, str], ...]]
    month_partitioned: frozenset[str]
    plane: Mapping[str, str]

    def schema(self, table: str) -> pa.Schema:
        return self.tables[table]


def build_schemas(cfg: SchemaConfig) -> SchemaBundle:
    tables: dict[str, pa.Schema] = {
        # canonical fact + its observable projection (written by the pipeline,
        # never by hand -- see projection.project_observable)
        "rfq_lines": rfq_lines_schema(cfg),
        "rfq_lines_obs": observable_rfq_schema(cfg),
        # shared dimensions
        "issuers": issuers_schema(cfg),
        "instruments": instruments_schema(cfg),
        "instrument_events": instrument_events_schema(cfg),
        "instrument_state_daily": instrument_state_daily_schema(cfg),
        "clients": clients_schema(cfg),
        "token_map": token_map_schema(cfg),
        "calendar": calendar_schema(cfg),
        "primary_calendar": primary_calendar_schema(cfg),
        # oracle plane
        "client_mandates": client_mandates_schema(cfg),
        "client_dealer_panel": client_dealer_panel_schema(cfg),
        "context_grid": context_grid_schema(cfg),
        "regime_path": regime_path_schema(cfg),
        "client_activity_daily": client_activity_daily_schema(cfg),
        "oracle_clients": oracle_clients_schema(cfg),
        "oracle_embeddings": oracle_embeddings_schema(cfg),
        "oracle_token_vectors": oracle_token_vectors_schema(cfg),
        "oracle_attribute_gain": oracle_attribute_gain_schema(cfg),
        "event_truth": event_truth_schema(cfg),
        "auction_book": auction_book_schema(cfg),
        # metadata
        "validation_report": validation_report_schema(cfg),
    }
    validate_rfq_classification(tables["rfq_lines"])

    asc = "ascending"
    sort_keys: dict[str, tuple[tuple[str, str], ...]] = {
        "rfq_lines": (("ts", asc), ("event_id", asc)),
        "rfq_lines_obs": (("ts", asc), ("event_id", asc)),
        "event_truth": (("event_id", asc),),
        "auction_book": (("event_id", asc),),
        "context_grid": (("grid_idx", asc),),
        "instrument_state_daily": (("trade_date", asc), ("instrument_id", asc)),
        "client_activity_daily": (("trade_date", asc), ("client_id", asc)),
        "regime_path": (("sector_id", asc), ("sojourn_no", asc)),
        "token_map": (("token_id", asc),),
        "instrument_events": (("ts", asc), ("instrument_id", asc)),
    }

    month_partitioned = frozenset({
        "rfq_lines", "rfq_lines_obs", "event_truth", "auction_book", "context_grid",
        "instrument_state_daily", "client_activity_daily",
    })

    plane = {name: _SHARED for name in tables}
    plane["rfq_lines"] = _CANONICAL
    plane["rfq_lines_obs"] = "observable"
    for name in ("client_mandates", "client_dealer_panel", "context_grid",
                 "regime_path", "client_activity_daily", "oracle_clients",
                 "oracle_embeddings", "oracle_token_vectors",
                 "oracle_attribute_gain", "event_truth", "auction_book"):
        plane[name] = _ORACLE

    return SchemaBundle(
        config=cfg,
        tables=tables,
        sort_keys={k: tuple(v) for k, v in sort_keys.items()},
        month_partitioned=month_partitioned,
        plane=plane,
    )
