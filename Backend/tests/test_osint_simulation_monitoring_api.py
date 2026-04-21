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
from app.routers import monitoring


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


def test_osint_quality_summary_reports_health_and_gates(client, monkeypatch, tmp_path):
    osint_quality_report = {
        "model_version": "osint-classifier-v1",
        "generated_at": "2026-04-20T00:00:00Z",
        "metrics": {
            "auc": 0.74,
            "pr_auc": 0.46,
        },
        "acceptance_gates": {
            "overall_pass": True,
            "criteria": {
                "auc_min": {"pass": True, "actual": 0.74, "threshold": 0.60},
                "pr_auc_min": {"pass": True, "actual": 0.46, "threshold": 0.35},
            },
        },
        "dataset": {
            "total_samples": 640,
        },
    }
    osint_config = {
        "model_version": "osint-classifier-v1",
        "model_type": "xgboost",
    }
    osint_baseline = {
        "model_version": "osint-classifier-v1",
        "created_at": "2026-04-20T00:00:00Z",
        "features": {"n_dom_subs": {"mean": 2.1, "std": 0.7, "q0": 0.0, "q100": 8.0}},
    }

    (tmp_path / "osint_quality_report.json").write_text(json.dumps(osint_quality_report), encoding="utf-8")
    (tmp_path / "osint_config.json").write_text(json.dumps(osint_config), encoding="utf-8")
    (tmp_path / "osint_drift_baseline.json").write_text(json.dumps(osint_baseline), encoding="utf-8")
    (tmp_path / "osint_risk_model.joblib").write_text("osint-model", encoding="utf-8")

    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_osint_drift_report",
        lambda min_samples=10: {
            "drift_detected": False,
            "drift_severity": "low",
            "drifted_features": [],
            "recommendation": "No significant drift detected.",
        },
    )

    response = client.get("/api/monitoring/osint_quality?include_criteria=true")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_info"]["quality_report_available"] is True
    assert payload["model_info"]["model_available"] is True
    assert payload["model_info"]["drift_baseline_available"] is True
    assert payload["gate_summary"]["all_pass"] is True
    assert payload["performance"]["metrics"]["auc"] == pytest.approx(0.74)
    assert payload["performance"]["criteria"]["auc_min"]["pass"] is True


def test_simulation_quality_summary_marks_warning_on_drift(client, monkeypatch, tmp_path):
    simulation_quality_report = {
        "model_version": "simulation-macro-v1",
        "generated_at": "2026-04-20T00:00:00Z",
        "metrics": {
            "r2": 0.95,
            "rmse": 0.06,
            "mae": 0.03,
        },
        "acceptance_gates": {
            "overall_pass": True,
            "criteria": {
                "r2_min": {"pass": True, "actual": 0.95, "threshold": 0.90},
                "rmse_max": {"pass": True, "actual": 0.06, "threshold": 0.08},
            },
        },
        "dataset": {
            "total_samples": 20000,
        },
    }
    sim_config = {
        "model_version": "simulation-macro-v1",
        "model_type": "lightgbm",
    }
    sim_baseline = {
        "model_version": "simulation-macro-v1",
        "created_at": "2026-04-20T00:00:00Z",
        "features": {"base_rate": {"mean": 0.2, "std": 0.05, "q0": 0.01, "q100": 0.95}},
    }

    (tmp_path / "simulation_quality_report.json").write_text(json.dumps(simulation_quality_report), encoding="utf-8")
    (tmp_path / "simulation_config.json").write_text(json.dumps(sim_config), encoding="utf-8")
    (tmp_path / "simulation_drift_baseline.json").write_text(json.dumps(sim_baseline), encoding="utf-8")
    (tmp_path / "simulation_lgbm.joblib").write_text("sim-model", encoding="utf-8")

    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_simulation_drift_report",
        lambda min_samples=5: {
            "drift_detected": True,
            "drift_severity": "high",
            "drifted_features": ["base_rate", "company_count"],
            "recommendation": "Feature drift detected.",
        },
    )

    response = client.get("/api/monitoring/simulation_quality")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "warning"
    assert payload["gate_summary"]["all_pass"] is True
    assert payload["drift"]["detected"] is True
    assert payload["drift"]["severity"] == "high"


def test_osint_and_simulation_drift_endpoints_use_builder(client, monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "_build_osint_drift_report",
        lambda min_samples=10: {
            "drift_detected": False,
            "drift_severity": "low",
            "drifted_features": [],
            "features": {},
            "recommendation": "stable",
        },
    )
    monkeypatch.setattr(
        monitoring,
        "_build_simulation_drift_report",
        lambda min_samples=5: {
            "drift_detected": True,
            "drift_severity": "medium",
            "drifted_features": ["base_rate"],
            "features": {},
            "recommendation": "retrain soon",
        },
    )

    osint_response = client.get("/api/monitoring/osint_drift?min_samples=3")
    assert osint_response.status_code == 200
    assert osint_response.json()["drift_detected"] is False

    simulation_response = client.get("/api/monitoring/simulation_drift?min_samples=2")
    assert simulation_response.status_code == 200
    assert simulation_response.json()["drift_detected"] is True

    bad_response = client.get("/api/monitoring/osint_drift?min_samples=0")
    assert bad_response.status_code == 400
