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
    assert payload["gate_summary"]["all_pass"] is True
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
    assert payload["serving"]["available"] is False
    assert payload["stress"]["available"] is False
    assert payload["gate_summary"]["all_pass"] is False
