"""Schema integrity: the contract itself."""
import json

import pyarrow as pa
import pytest

from rfqsim.schema import enums
from rfqsim.schema.enums import ColumnClass
from rfqsim.schema.tables import (RFQ_COLUMN_CLASS, SchemaClassificationError,
                                  SchemaConfig, observable_rfq_schema,
                                  rfq_lines_schema,
                                  validate_rfq_classification)


def test_every_table_builds_and_is_self_describing(bundle, cfg):
    for name, sch in bundle.tables.items():
        empty = sch.empty_table()
        assert empty.num_rows == 0
        md = {k.decode(): v.decode() for k, v in sch.metadata.items()}
        assert md["rfqsim.table"] == name
        assert md["rfqsim.run_id"] == cfg.run_id
        # enum registry embedded and parseable in every file
        reg = json.loads(md["rfqsim.enums"])
        assert reg["OurResult"]["NO_TRADE"] == int(enums.OurResult.NO_TRADE)


def test_rfq_classification_is_a_bijection(cfg):
    validate_rfq_classification(rfq_lines_schema(cfg))  # must not raise
    hacked = rfq_lines_schema(cfg).append(pa.field("mystery", pa.int8()))
    with pytest.raises(SchemaClassificationError):
        validate_rfq_classification(hacked)


def test_observable_schema_contract(cfg):
    obs = observable_rfq_schema(cfg)
    names = set(obs.names)
    for col, klass in RFQ_COLUMN_CLASS.items():
        if klass in (ColumnClass.LATENT, ColumnClass.SELECTOR):
            assert col not in names, f"{col} must not be observable"
        else:
            assert col in names, f"{col} must be observable"
    # projection-added fields, in the outcome block
    assert "side_revealed" in names and "side_reveal_ts" in names
    assert obs.names.index("side_revealed") < obs.names.index("rec_token_id")


def test_parametric_widths(bundle, cfg):
    t = bundle.tables
    assert t["context_grid"].field("c").type.list_size == cfg.d
    assert t["context_grid"].field("regime").type.list_size == cfg.n_sectors
    assert t["context_grid"].field("lambda_sector").type.list_size == cfg.n_sectors
    assert t["oracle_clients"].field("u").type.list_size == cfg.d
    assert t["oracle_embeddings"].field("eps").type.list_size == cfg.d
    assert t["oracle_embeddings"].field("norm_params").type.list_size == cfg.n_norm_params
    assert t["instrument_state_daily"].field("x").type.list_size == cfg.p
    assert t["auction_book"].field("px_sprd_bp").type.list_size == cfg.max_dealers


def test_bundle_partitioning_and_planes(bundle):
    assert "rfq_lines" in bundle.month_partitioned
    assert "event_truth" in bundle.month_partitioned
    assert bundle.plane["rfq_lines"] == "canonical"
    assert bundle.plane["event_truth"] == "oracle"
    assert bundle.plane["token_map"] == "shared"      # vocabulary is public structure
    assert bundle.plane["client_mandates"] == "oracle"  # structural zeros stay oracle-side
    assert bundle.sort_keys["rfq_lines"] == (("ts", "ascending"), ("event_id", "ascending"))


def test_config_guards():
    with pytest.raises(ValueError):
        SchemaConfig(d=8, p=6, n_sectors=33)   # sector_mask is u32
    with pytest.raises(ValueError):
        SchemaConfig(d=0, p=6)
    with pytest.raises(ValueError):
        SchemaConfig(d=8, p=6, max_dealers=17)
