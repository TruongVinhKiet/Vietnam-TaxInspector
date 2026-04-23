import json
from pathlib import Path


def test_graph_forensic_contract_v21_schema_structure():
    schema_path = Path(__file__).resolve().parents[1] / "data" / "graph_forensic_contract_v2_1.schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    assert payload["properties"]["contract_version"]["const"] == "graph-intelligence-v2.1"
    for block in ("integrity_signals", "pricing_signals", "phoenix_signals"):
        assert block in payload["properties"]
        assert "available" in payload["properties"][block]["required"]

    node_props = payload["properties"]["nodes"]["items"]["properties"]
    assert "vat_washout_score" in node_props
    assert "industry_mismatch_exposure" in node_props
    assert "phoenix_candidate_score" in node_props

    edge_props = payload["properties"]["edges"]["items"]["properties"]
    assert "lifecycle_state" in edge_props
    assert "price_deviation_score" in edge_props
    assert "invoice_payment_match_score" in edge_props
