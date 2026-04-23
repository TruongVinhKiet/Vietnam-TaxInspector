import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.database import get_db
from app.routers import graph, monitoring


class _DummyDB:
    pass


@pytest.fixture
def client():
    def _override_get_db():
        yield _DummyDB()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_graph_endpoint_returns_enriched_model_contract(client, monkeypatch):
    companies = [
        {
            "tax_code": "0101",
            "name": "Cong ty A",
            "industry": "Manufacturing",
            "registration_date": "2024-01-01",
            "risk_score": 88.0,
            "is_active": True,
            "lat": 0.0,
            "lng": 0.0,
        },
        {
            "tax_code": "0102",
            "name": "Cong ty B",
            "industry": "Trading",
            "registration_date": "2024-01-05",
            "risk_score": 32.0,
            "is_active": True,
            "lat": 0.0,
            "lng": 0.0,
        },
    ]
    invoices = [
        {
            "seller_tax_code": "0101",
            "buyer_tax_code": "0102",
            "amount": 5000000,
            "vat_rate": 10.0,
            "date": "2025-01-01",
            "invoice_number": "INV-001",
        }
    ]

    class _FakeEngine:
        _loaded = True
        _load_error = None

        def predict(self, _companies, _invoices):
            return {
                "nodes": [
                    {
                        "id": "0101",
                        "tax_code": "0101",
                        "label": "Cong ty A",
                        "group": "shell",
                        "risk_score": 88.0,
                        "shell_probability": 0.9,
                        "is_shell": True,
                    }
                ],
                "edges": [
                    {
                        "from": "0101",
                        "to": "0102",
                        "is_circular": True,
                        "circular_probability": 0.7,
                    }
                ],
                "decision_thresholds": {
                    "node": 0.5,
                    "edge": 0.45,
                    "policy": {
                        "cold_start_degree_threshold": 5,
                        "cold_start_threshold_delta": 0.06,
                        "node_blend_alpha_gnn": 1.0,
                    },
                },
                "attention_weights": [
                    {"from": "0101", "to": "0102", "weight": 0.81},
                    {"from": "0102", "to": "0101", "weight": 0.53},
                ],
                "model_loaded": True,
                "ensemble_active": True,
            }

    monkeypatch.setattr(graph, "_extract_full_graph", lambda db, limit=200: (companies, invoices))
    monkeypatch.setattr(graph, "_get_gnn_engine", lambda: _FakeEngine())

    response = client.get("/api/graph")
    assert response.status_code == 200

    payload = response.json()
    assert payload["contract_version"] == "graph-intelligence-v1"
    assert payload["model_info"]["model_loaded"] is True
    assert payload["model_info"]["inference_mode"] == "gnn_ensemble"
    assert payload["decision_thresholds"]["node"] == pytest.approx(0.5)
    assert payload["decision_thresholds"]["edge"] == pytest.approx(0.45)
    assert payload["attention_summary"]["count"] == 2
    assert payload["query_context"]["depth"] == 2

    node = payload["nodes"][0]
    edge = payload["edges"][0]
    assert node["threshold_margin"] == pytest.approx(0.4)
    assert edge["threshold_margin"] == pytest.approx(0.25)


def test_graph_endpoint_marks_fallback_mode(client, monkeypatch):
    companies = [{"tax_code": "0101", "name": "Cong ty A"}]
    invoices = []

    class _FallbackEngine:
        _loaded = False
        _load_error = "model_artifacts_missing"

        def predict(self, _companies, _invoices):
            return {
                "nodes": [{"id": "0101", "tax_code": "0101", "label": "Cong ty A", "risk_score": 60.0}],
                "edges": [],
            }

    monkeypatch.setattr(graph, "_extract_full_graph", lambda db, limit=200: (companies, invoices))
    monkeypatch.setattr(graph, "_get_gnn_engine", lambda: _FallbackEngine())

    response = client.get("/api/graph")
    assert response.status_code == 200

    payload = response.json()
    assert payload["fallback_active"] is True
    assert payload["fallback_reason"] == "model_artifacts_missing"
    assert payload["model_info"]["inference_mode"] == "heuristic_fallback"


def test_graph_quality_summary_reports_gate_status(client, monkeypatch, tmp_path):
    serving_report = {
        "serving_path_test": {
            "node": {"f1": 0.8},
            "edge": {"f1": 1.0},
        },
        "comparison": {
            "node_pr_auc_delta": 0.0,
            "edge_pr_auc_delta": 0.0,
        },
        "acceptance_gates": {
            "overall_pass": True,
            "criteria": {
                "node_pr_auc_drop": {"pass": True, "actual": 0.0, "threshold": 0.02},
            },
        },
    }
    stress_report = {
        "stress_summary": {"worst_node_f1_delta": -0.03},
        "stress_acceptance_gates": {
            "overall_pass": True,
            "criteria": {
                "unseen_node_generalization_gap": {"pass": True, "actual": 0.08, "threshold": 0.45},
                "temporal_plus3m_edge_f1_drop": {"pass": True, "actual": 0.02, "threshold": 0.1},
                "temporal_plus3m_edge_prauc_drop": {"pass": True, "actual": 0.03, "threshold": 0.06},
            },
        },
    }
    config = {
        "amount_feature_mode": "robust",
        "model_version": "vat-gnn-test",
    }

    (tmp_path / "serving_e2e_report.json").write_text(json.dumps(serving_report), encoding="utf-8")
    (tmp_path / "stress_evaluation_report.json").write_text(json.dumps(stress_report), encoding="utf-8")
    (tmp_path / "gat_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "gat_model.pt").write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_drift_report",
        lambda lookback_days, baseline_days, min_samples: {
            "drift_detected": False,
            "drift_severity": "low",
            "drifted_features": [],
            "recommendation": "No retraining required at this time.",
            "features": {},
        },
    )

    response = client.get("/api/monitoring/graph_quality?include_criteria=true")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["status_reason"] == "all_quality_gates_passed"
    assert payload["gate_summary"]["all_pass"] is True
    assert payload["gate_summary"]["failed_gates"] == []
    assert payload["serving"]["overall_pass"] is True
    assert payload["stress"]["overall_pass"] is True
    assert payload["serving"]["criteria"]["node_pr_auc_drop"]["actual"] == pytest.approx(0.0)
    assert payload["model_info"]["amount_feature_mode"] == "robust"


def test_graph_quality_summary_handles_missing_reports(client, monkeypatch, tmp_path):
    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_drift_report",
        lambda lookback_days, baseline_days, min_samples: {
            "drift_detected": False,
            "drift_severity": "insufficient_data",
            "drifted_features": [],
            "recommendation": "Insufficient feature telemetry",
            "features": {},
        },
    )

    response = client.get("/api/monitoring/graph_quality")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "unknown"
    assert payload["status_reason"] == "quality_reports_unavailable"
    assert payload["serving"]["available"] is False
    assert payload["stress"]["available"] is False
    assert payload["gate_summary"]["all_pass"] is False
    assert payload["gate_summary"]["failed_gates"] == []


def test_graph_quality_summary_surfaces_failed_stress_gate(client, monkeypatch, tmp_path):
    serving_report = {
        "serving_path_test": {"node": {"f1": 0.9}, "edge": {"f1": 0.88}},
        "acceptance_gates": {"overall_pass": True, "criteria": {}},
    }
    stress_report = {
        "stress_summary": {"worst_node_f1_delta": -0.2},
        "stress_acceptance_gates": {
            "overall_pass": False,
            "criteria": {
                "temporal_plus3m_edge_f1_drop": {"pass": False, "actual": 0.13, "threshold": 0.1},
            },
        },
    }
    (tmp_path / "serving_e2e_report.json").write_text(json.dumps(serving_report), encoding="utf-8")
    (tmp_path / "stress_evaluation_report.json").write_text(json.dumps(stress_report), encoding="utf-8")
    (tmp_path / "gat_model.pt").write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_drift_report",
        lambda lookback_days, baseline_days, min_samples: {
            "drift_detected": False,
            "drift_severity": "low",
            "drifted_features": [],
            "recommendation": "",
            "features": {},
        },
    )

    response = client.get("/api/monitoring/graph_quality?include_criteria=true")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["status_reason"] == "stress_temporal_plus3m_edge_f1_drop_failed"
    assert payload["gate_summary"]["stress_pass"] is False
    assert payload["gate_summary"]["failed_gates"] == [
        {
            "source": "stress",
            "gate": "temporal_plus3m_edge_f1_drop",
            "actual": pytest.approx(0.13),
            "threshold": pytest.approx(0.1),
        }
    ]


def test_graph_link_prediction_returns_non_empty_for_open_chain(client, monkeypatch):
    companies = [
        {"tax_code": "A", "name": "Cong ty A"},
        {"tax_code": "B", "name": "Cong ty B"},
        {"tax_code": "C", "name": "Cong ty C"},
    ]
    invoices = [
        {
            "seller_tax_code": "A",
            "buyer_tax_code": "B",
            "amount": 1000000,
            "date": "2026-01-01",
        },
        {
            "seller_tax_code": "B",
            "buyer_tax_code": "C",
            "amount": 1200000,
            "date": "2026-01-03",
        },
    ]

    monkeypatch.setattr(graph, "_extract_full_graph", lambda db, limit=200: (companies, invoices))

    response = client.get("/api/graph/link-prediction?top_k=5")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total"] >= 1
    assert isinstance(payload["predictions"], list)

    top = payload["predictions"][0]
    assert top["prediction_score"] > 0
    assert "jaccard" in top
    assert "adamic_adar" in top


class _OwnershipResult:
    def __init__(self, rows, keys):
        self._rows = rows
        self._keys = keys

    def keys(self):
        return self._keys

    def fetchall(self):
        return self._rows


def test_ring_scoring_reports_cycle_gap_data_status(client, monkeypatch):
    companies = [
        {"tax_code": "0101000001", "name": "Cong ty A"},
        {"tax_code": "0202000002", "name": "Cong ty B"},
    ]
    invoices = [
        {
            "seller_tax_code": "0101000001",
            "buyer_tax_code": "0202000002",
            "amount": 1500000,
            "vat_rate": 10.0,
            "date": "2026-02-01",
            "invoice_number": "INV-100",
        }
    ]

    class _FakeEngine:
        def predict(self, _companies, _invoices):
            return {
                "cycles": [],
                "forensic_metrics": {
                    "circular_edge_count": 4,
                    "cycle_backed_circular_edge_count": 0,
                    "circular_edge_cycle_coverage": 0.0,
                },
            }

    monkeypatch.setattr(
        graph,
        "_extract_full_graph",
        lambda db, limit=graph.FULL_GRAPH_COMPANY_LIMIT: (companies, invoices),
    )
    monkeypatch.setattr(graph, "_get_gnn_engine", lambda: _FakeEngine())

    response = client.get("/api/graph/ring-scoring")
    assert response.status_code == 200

    payload = response.json()
    assert payload["data_status"] == "no_cycles_with_circular_edges"
    assert payload["diagnostics"]["cycle_detection_gap"] is True
    assert payload["cycles_detected"] == 0
    assert payload["circular_edge_count"] == 4
    assert payload["query_scope"]["source"] == "graph_ring_scoring"
    assert payload["snapshot_id"].startswith("snap-")


def test_ownership_reports_scope_gap_status(client, monkeypatch):
    ownership_rows = [
        (
            1,
            "1111111111",
            "2222222222",
            75.0,
            "shareholder",
            None,
            None,
            None,
            None,
            True,
        )
    ]
    ownership_keys = [
        "id",
        "parent_tax_code",
        "child_tax_code",
        "ownership_percent",
        "relationship_type",
        "person_name",
        "person_id",
        "effective_date",
        "end_date",
        "verified",
    ]

    companies = [
        {"tax_code": "1111111111", "name": "Cong ty A"},
        {"tax_code": "9999999999", "name": "Cong ty B"},
    ]
    invoices = [
        {
            "seller_tax_code": "1111111111",
            "buyer_tax_code": "9999999999",
            "amount": 2200000,
            "vat_rate": 10.0,
            "date": "2026-03-01",
            "invoice_number": "INV-200",
        }
    ]

    class _FakeOwnershipAnalyzer:
        def analyze(self, _ownership_links, _invoices):
            return {
                "summary": {
                    "total_clusters": 2,
                    "total_cross_trades": 0,
                    "total_common_controllers": 1,
                },
                "clusters": [{"cluster_id": "cluster-1"}],
                "cross_ownership_trades": [],
            }

    monkeypatch.setattr(
        _DummyDB,
        "execute",
        lambda self, statement, params=None: _OwnershipResult(ownership_rows, ownership_keys),
        raising=False,
    )
    monkeypatch.setattr(
        graph,
        "_extract_subgraph",
        lambda db, center_tax_code, depth: (companies, invoices),
    )

    import ml_engine.graph_intelligence as graph_intelligence

    monkeypatch.setattr(
        graph_intelligence,
        "OwnershipGraphAnalyzer",
        _FakeOwnershipAnalyzer,
    )

    response = client.get("/api/graph/ownership?tax_code=1234567890")
    assert response.status_code == 200

    payload = response.json()
    assert payload["data_status"] == "no_parent_child_pairs_in_invoice_scope"
    assert payload["coverage"]["ownership_nodes_in_invoice_graph_count"] == 1
    assert payload["coverage"]["ownership_pairs_in_invoice_scope"] == 0
    assert payload["query_scope"]["source"] == "graph_ownership"
    assert payload["snapshot_id"].startswith("snap-")
