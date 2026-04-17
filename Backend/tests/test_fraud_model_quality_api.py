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


def test_fraud_quality_summary_reports_calibration_and_gates(client, monkeypatch, tmp_path):
    fraud_quality_report = {
        "generated_at": "2026-04-16T00:00:00Z",
        "model_info": {
            "model_version": "fraud-hybrid-v1",
            "calibration_method": "isotonic",
        },
        "dataset": {
            "test_size": 120,
        },
        "performance": {
            "raw": {
                "auc_roc": 0.86,
                "pr_auc": 0.71,
                "brier": 0.14,
                "ece": 0.08,
            },
            "calibrated": {
                "auc_roc": 0.86,
                "pr_auc": 0.71,
                "brier": 0.12,
                "ece": 0.05,
            },
        },
        "calibration": {
            "available": True,
            "method": "isotonic",
            "brier_improvement": 0.02,
        },
        "acceptance_gates": {
            "overall_pass": True,
            "criteria": {
                "auc_roc_min": {"pass": True, "actual": 0.86, "threshold": 0.7},
                "pr_auc_min": {"pass": True, "actual": 0.71, "threshold": 0.5},
                "brier_max": {"pass": True, "actual": 0.12, "threshold": 0.3},
                "ece_max": {"pass": True, "actual": 0.05, "threshold": 0.15},
                "brier_not_worse_than_raw": {"pass": True, "actual": -0.02, "threshold": 0.02},
            },
        },
    }

    (tmp_path / "fraud_quality_report.json").write_text(json.dumps(fraud_quality_report), encoding="utf-8")
    (tmp_path / "xgboost_model.joblib").write_text("xgb", encoding="utf-8")
    (tmp_path / "isolation_forest.joblib").write_text("iso", encoding="utf-8")
    (tmp_path / "fraud_calibrator.joblib").write_text("cal", encoding="utf-8")

    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_fraud_drift_report",
        lambda lookback_days, baseline_days, min_samples: {
            "drift_detected": False,
            "drift_severity": "low",
            "drifted_features": [],
            "recommendation": "No retraining required.",
            "features": {},
        },
    )

    response = client.get("/api/monitoring/fraud_quality?include_criteria=true")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_info"]["quality_report_available"] is True
    assert payload["model_info"]["calibrator_available"] is True
    assert payload["gate_summary"]["all_pass"] is True
    assert payload["performance"]["available"] is True
    assert payload["performance"]["sample_size"] == 120
    assert payload["calibration"]["method"] == "isotonic"
    assert payload["performance"]["criteria"]["auc_roc_min"]["pass"] is True


def test_fraud_quality_summary_marks_warning_when_soft_gate_fails(client, monkeypatch, tmp_path):
    fraud_quality_report = {
        "generated_at": "2026-04-16T00:00:00Z",
        "model_info": {
            "model_version": "fraud-hybrid-v2",
            "calibration_method": "isotonic",
        },
        "dataset": {
            "test_size": 200,
        },
        "performance": {
            "raw": {
                "auc_roc": 0.89,
                "pr_auc": 0.74,
                "brier": 0.13,
                "ece": 0.07,
            },
            "calibrated": {
                "auc_roc": 0.89,
                "pr_auc": 0.74,
                "brier": 0.11,
                "ece": 0.05,
            },
        },
        "calibration": {
            "available": True,
            "method": "isotonic",
            "brier_improvement": 0.02,
        },
        "acceptance_gates": {
            "overall_pass": True,
            "core_criteria": [
                "auc_roc_min",
                "pr_auc_min",
                "brier_max",
                "ece_max",
                "brier_not_worse_than_raw",
            ],
            "soft_gate_criteria": [
                "temporal_auc_drop_max_soft",
                "slice_min_auc_soft",
            ],
            "criteria": {
                "auc_roc_min": {"pass": True, "actual": 0.89, "threshold": 0.7},
                "pr_auc_min": {"pass": True, "actual": 0.74, "threshold": 0.5},
                "brier_max": {"pass": True, "actual": 0.11, "threshold": 0.3},
                "ece_max": {"pass": True, "actual": 0.05, "threshold": 0.15},
                "brier_not_worse_than_raw": {"pass": True, "actual": -0.02, "threshold": 0.02},
                "temporal_auc_drop_max_soft": {
                    "pass": False,
                    "actual": 0.11,
                    "threshold": 0.08,
                    "soft_gate": True,
                },
                "slice_min_auc_soft": {
                    "pass": True,
                    "actual": 0.66,
                    "threshold": 0.60,
                    "soft_gate": True,
                },
            },
        },
    }

    (tmp_path / "fraud_quality_report.json").write_text(json.dumps(fraud_quality_report), encoding="utf-8")
    (tmp_path / "xgboost_model.joblib").write_text("xgb", encoding="utf-8")
    (tmp_path / "isolation_forest.joblib").write_text("iso", encoding="utf-8")
    (tmp_path / "fraud_calibrator.joblib").write_text("cal", encoding="utf-8")

    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_fraud_drift_report",
        lambda lookback_days, baseline_days, min_samples: {
            "drift_detected": False,
            "drift_severity": "low",
            "drifted_features": [],
            "recommendation": "No retraining required.",
            "features": {},
        },
    )

    response = client.get("/api/monitoring/fraud_quality?include_criteria=true")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "warning"
    assert payload["gate_summary"]["all_pass"] is True
    assert payload["gate_summary"]["soft_gate_pass"] is False
    assert "temporal_auc_drop_max_soft" in payload["gate_summary"]["soft_gate_warnings"]


def test_fraud_quality_summary_handles_missing_artifacts(client, monkeypatch, tmp_path):
    monkeypatch.setattr(monitoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        monitoring,
        "_build_fraud_drift_report",
        lambda lookback_days, baseline_days, min_samples: {
            "drift_detected": False,
            "drift_severity": "insufficient_data",
            "drifted_features": [],
            "recommendation": "Collect more telemetry.",
            "features": {},
        },
    )

    response = client.get("/api/monitoring/fraud_quality")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "unknown"
    assert payload["model_info"]["quality_report_available"] is False
    assert payload["model_info"]["calibrator_available"] is False
    assert payload["performance"]["available"] is False
    assert payload["gate_summary"]["all_pass"] is False
