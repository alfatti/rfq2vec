"""rfqsim.schema -- the data-plane contract for the simulator.

Public surface:

    SchemaConfig, build_schemas, SchemaBundle      parametric schemas + bundle
    rfq_lines_schema, observable_rfq_schema        the two planes of the tape
    RFQ_COLUMN_CLASS, validate_rfq_classification  the leakage tripwire
    RevelationPolicy, project_observable           canonical -> observable
    audit_observable                               CI-grade leakage audit
    ShardWriter, EventIdAllocator, PhiloxLedger    write path + reproducibility
    RunManifest                                    the run contract
"""
from . import enums
from .enums import ColumnClass
from .backend import gpu_available, to_numpy, xp
from .emission import EmissionConfig, Emitter
from .emission_batch import BatchEmitter
from .gpurand import derive_key32, masked_softmax_draw, normals, philox4x32, uniforms
from .intensity import ArrivalPanel, sample_arrivals
from .manifest import RunManifest
from .pipeline import RunDials, generate_run, validation_battery
from .production import generate_run_production, shard_worker
from .population import (Universe, UniverseConfig, build_universe,
                         business_calendar, instrument_state, token_vectors,
                         write_instrument_state_day, write_universe)
from .projection import (DEFAULT_POLICY, STRICT_POLICY, LeakageError,
                         ProjectionError, RevelationPolicy, audit_observable,
                         project_observable)
from .state import (GridCalendar, LatentStateConfig, LatentStateEngine,
                    RegimeConfig, StateError, derive_stream_key,
                    max_window_steps, predicted_displacement,
                    realized_drift, run_and_write)
from .tables import (RFQ_COLUMN_CLASS, SCHEMA_VERSION,
                     SchemaBundle, SchemaClassificationError, SchemaConfig,
                     build_schemas, observable_rfq_schema, rfq_lines_schema,
                     validate_rfq_classification)
from .vocab import (FeatureSpec, Sentence, TokenVocab, VocabError,
                    sense_from_disclosed, sentences, tokenize_table,
                    window_pair_counts)
from .writer import (STAT_COLUMNS, EventIdAllocator, FileRecord, PhiloxLedger,
                     ShardWriter, sha256_file)

__all__ = [
    "enums", "ColumnClass",
    "SCHEMA_VERSION", "SchemaConfig", "SchemaBundle", "build_schemas",
    "rfq_lines_schema", "observable_rfq_schema",
    "RFQ_COLUMN_CLASS", "validate_rfq_classification", "SchemaClassificationError",
    "RevelationPolicy", "DEFAULT_POLICY", "STRICT_POLICY",
    "project_observable", "audit_observable", "ProjectionError", "LeakageError",
    "FeatureSpec", "TokenVocab", "VocabError", "Sentence",
    "sense_from_disclosed", "tokenize_table", "sentences", "window_pair_counts",
    "GridCalendar", "RegimeConfig", "LatentStateConfig", "LatentStateEngine",
    "StateError", "derive_stream_key", "run_and_write", "realized_drift",
    "predicted_displacement", "max_window_steps",
    "ShardWriter", "EventIdAllocator", "PhiloxLedger", "FileRecord",
    "STAT_COLUMNS", "sha256_file",
    "RunManifest",
    "UniverseConfig", "Universe", "build_universe", "business_calendar",
    "instrument_state", "token_vectors", "write_universe", "write_instrument_state_day",
    "ArrivalPanel", "sample_arrivals", "EmissionConfig", "Emitter",
    "RunDials", "generate_run", "validation_battery",
    "generate_run_production", "shard_worker", "BatchEmitter",
    "xp", "gpu_available", "to_numpy",
    "philox4x32", "uniforms", "normals", "derive_key32", "masked_softmax_draw",
]
