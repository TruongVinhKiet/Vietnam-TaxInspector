import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.database import get_db
from app import auth
from app.routers import monitoring


class _DummyDB:
    pass


@pytest.fixture
def client():
    def _override_get_db():
        yield _DummyDB()

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[auth.get_current_user] = lambda: SimpleNamespace(
        id=1,
        badge_id="ADM001",
        role="admin",
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_normalize_kpi_policy_payload_sanitizes_fields():
    items = [
        monitoring.KPIPolicyInput(
            track_name="Audit Value",
            metric_name="Precision Top 50",
            comparator=">=",
            threshold=0.72,
            min_sample=20,
            window_days=30,
            cooldown_days=10,
            enabled=True,
            rationale="keep stable precision",
        )
    ]

    normalized = monitoring._normalize_kpi_policy_payload(items)

    assert len(normalized) == 1
    assert normalized[0]["track_name"] == "audit_value"
    assert normalized[0]["metric_name"] == "precision_top_50"
    assert normalized[0]["comparator"] == ">="
    assert normalized[0]["threshold"] == pytest.approx(0.72)
    assert normalized[0]["min_sample"] == 20


def test_normalize_kpi_policy_payload_rejects_duplicates():
    items = [
        monitoring.KPIPolicyInput(
            track_name="audit-value",
            metric_name="precision_top_50",
            comparator=">=",
            threshold=0.7,
        ),
        monitoring.KPIPolicyInput(
            track_name="audit value",
            metric_name="precision-top-50",
            comparator=">=",
            threshold=0.8,
        ),
    ]

    with pytest.raises(HTTPException) as exc:
        monitoring._normalize_kpi_policy_payload(items)

    assert exc.value.status_code == 422
    assert "trùng" in str(exc.value.detail)


class _SnapshotCursor:
    def __init__(self):
        self._rows = []
        self.inserted = []

    def execute(self, query, params=None):
        compact = " ".join(query.lower().split())
        if "from information_schema.columns" in compact:
            table_name = params[0] if params else ""
            if table_name == "kpi_metric_snapshots":
                self._rows = [("track_name",), ("metric_name",)]
            else:
                self._rows = []
            return

        if "insert into kpi_metric_snapshots" in compact:
            self.inserted.append(params)
            self._rows = []
            return

        self._rows = []

    def fetchall(self):
        return list(self._rows)


class _PolicyCursor:
    def __init__(self):
        self._rows = []
        self.policies = {}

    def execute(self, query, params=None):
        compact = " ".join(query.lower().split())

        if "from information_schema.columns" in compact:
            table_name = params[0] if params else ""
            if table_name == "kpi_trigger_policies":
                self._rows = [("track_name",), ("metric_name",)]
            else:
                self._rows = []
            return

        if compact.startswith("insert into kpi_trigger_policies"):
            (
                track_name,
                metric_name,
                comparator,
                threshold,
                min_sample,
                window_days,
                cooldown_days,
                enabled,
                rationale,
            ) = params
            self.policies[(track_name, metric_name)] = (
                track_name,
                metric_name,
                comparator,
                float(threshold),
                int(min_sample),
                int(window_days),
                int(cooldown_days),
                bool(enabled),
                rationale,
            )
            self._rows = []
            return

        if compact.startswith("select track_name, metric_name from kpi_trigger_policies"):
            self._rows = [(track, metric) for (track, metric) in sorted(self.policies.keys())]
            return

        if "from kpi_trigger_policies" in compact and "order by track_name, metric_name" in compact:
            self._rows = [
                self.policies[key]
                for key in sorted(self.policies.keys())
            ]
            return

        if compact.startswith("delete from kpi_trigger_policies"):
            key = (params[0], params[1])
            self.policies.pop(key, None)
            self._rows = []
            return

        raise AssertionError(f"Unexpected SQL in test stub: {query}")

    def fetchall(self):
        return list(self._rows)

    def close(self):
        return None


class _PolicyConnection:
    def __init__(self):
        self.cursor_obj = _PolicyCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        return None


def test_persist_kpi_snapshots_inserts_expected_rows():
    cursor = _SnapshotCursor()
    payload = {
        "readiness_score": 75.0,
        "window_days": 28,
        "generated_at": "2026-04-16T09:00:00",
        "track_status": {
            "audit_value": {
                "ready_for_split": False,
                "rules": [
                    {
                        "metric_name": "precision_top_50",
                        "actual": 0.66,
                        "sample_size": 52,
                        "comparator": ">=",
                        "threshold": 0.70,
                        "status": "fail",
                        "enabled": True,
                        "min_sample": 50,
                        "window_days": 28,
                        "passed": False,
                    },
                    {
                        "metric_name": "roi_positive_rate",
                        "actual": 0.82,
                        "sample_size": 52,
                        "comparator": ">=",
                        "threshold": 0.80,
                        "status": "pass",
                        "enabled": True,
                        "min_sample": 50,
                        "window_days": 28,
                        "passed": True,
                    },
                    {
                        "metric_name": "conversion_uplift",
                        "actual": 0.85,
                        "sample_size": 52,
                        "comparator": ">=",
                        "threshold": 0.80,
                        "status": "cooldown_active",
                        "enabled": True,
                        "min_sample": 50,
                        "window_days": 28,
                        "cooldown_days": 14,
                        "cooldown_active": True,
                        "cooldown_remaining_days": 9,
                        "cooldown_until": "2026-04-30T09:00:00",
                        "last_failed_at": "2026-04-16T09:00:00",
                        "passed": False,
                    },
                ],
            }
        },
    }

    inserted = monitoring._persist_kpi_snapshots(cursor, payload, source="manual_capture")

    assert inserted == 3
    assert len(cursor.inserted) == 3
    assert cursor.inserted[0][0] == "audit_value"
    assert cursor.inserted[0][1] == "precision_top_50"
    assert cursor.inserted[0][8] == "manual_capture"

    details = json.loads(cursor.inserted[0][9])
    assert details["ready_for_split"] is False
    assert details["readiness_score"] == pytest.approx(75.0)

    cooldown_details = json.loads(cursor.inserted[2][9])
    assert cursor.inserted[2][6] == "cooldown_active"
    assert cooldown_details["cooldown_active"] is True
    assert cooldown_details["cooldown_remaining_days"] == 9


def test_build_split_trigger_status_payload_applies_cooldown(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "_compute_intervention_effectiveness",
        lambda *_args, **_kwargs: {
            "schema_ready": True,
            "metrics": {"precision_at_top_k": 0.9},
            "summary": {"total_labels": 120, "attempted_labels": 100, "terminal_labels": 80},
        },
    )
    monkeypatch.setattr(
        monitoring,
        "_compute_audit_value_effectiveness",
        lambda *_args, **_kwargs: {
            "schema_ready": True,
            "metrics": {"precision_at_top_k": 0.9},
            "summary": {"total_labels": 120, "attempted_labels": 100, "terminal_labels": 80},
        },
    )
    monkeypatch.setattr(
        monitoring,
        "_compute_vat_refund_effectiveness",
        lambda *_args, **_kwargs: {
            "schema_ready": True,
            "metrics": {"precision_at_top_k": 0.9},
            "summary": {"total_labels": 80, "attempted_labels": 60, "terminal_labels": 40},
        },
    )
    monkeypatch.setattr(monitoring, "_compute_precision_at_k", lambda *_args, **_kwargs: (0.86, 120))
    monkeypatch.setattr(monitoring, "_compute_vat_precision_at_k", lambda *_args, **_kwargs: (0.84, 80))

    recent_failure = datetime.utcnow() - timedelta(days=2)
    monkeypatch.setattr(
        monitoring,
        "_fetch_recent_rule_failures",
        lambda *_args, **_kwargs: {("audit_value", "precision_top_50"): recent_failure},
    )

    policies = [
        {
            "track_name": "audit_value",
            "metric_name": "precision_top_50",
            "comparator": ">=",
            "threshold": 0.80,
            "min_sample": 50,
            "window_days": 28,
            "cooldown_days": 14,
            "enabled": True,
        }
    ]

    payload = monitoring._build_split_trigger_status_payload(cur=object(), policies=policies)
    audit_track = payload["track_status"]["audit_value"]
    rule = audit_track["rules"][0]

    assert payload["ready"] is False
    assert audit_track["ready_for_split"] is False
    assert rule["status"] == "cooldown_active"
    assert rule["passed"] is False
    assert rule["cooldown_active"] is True
    assert int(rule["cooldown_remaining_days"]) > 0


def test_extract_active_kpi_breaches_includes_cooldown_active_status():
    split_payload = {
        "track_status": {
            "audit_value": {
                "rules": [
                    {
                        "enabled": True,
                        "metric_name": "precision_top_50",
                        "status": "cooldown_active",
                        "actual": 0.86,
                        "threshold": 0.80,
                        "comparator": ">=",
                        "sample_size": 120,
                        "min_sample": 50,
                        "window_days": 28,
                    }
                ]
            }
        }
    }

    breaches = monitoring._extract_active_kpi_breaches(split_payload)

    assert len(breaches) == 1
    assert breaches[0]["track_name"] == "audit_value"
    assert breaches[0]["status"] == "cooldown_active"


def test_upsert_kpi_policy_endpoint_updates_rows(client, monkeypatch):
    fake_conn = _PolicyConnection()

    def _fake_connect(*_args, **_kwargs):
        return fake_conn

    monkeypatch.setattr(monitoring.psycopg2, "connect", _fake_connect)

    response = client.put(
        "/api/monitoring/kpi_policy",
        json={
            "replace_existing": False,
            "policies": [
                {
                    "track_name": "Audit Value",
                    "metric_name": "Precision Top 50",
                    "comparator": ">=",
                    "threshold": 0.73,
                    "min_sample": 50,
                    "window_days": 28,
                    "cooldown_days": 14,
                    "enabled": True,
                    "rationale": "policy test",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["updated_count"] == 1
    assert payload["replace_existing"] is False
    assert fake_conn.committed is True
    assert payload["policies"][0]["track_name"] == "audit_value"
    assert payload["policies"][0]["metric_name"] == "precision_top_50"


def test_upsert_kpi_policy_endpoint_forbids_non_admin(client):
    app.dependency_overrides[auth.get_current_user] = lambda: SimpleNamespace(
        id=2,
        badge_id="USR001",
        role="viewer",
    )

    response = client.put(
        "/api/monitoring/kpi_policy",
        json={
            "replace_existing": False,
            "policies": [
                {
                    "track_name": "Audit Value",
                    "metric_name": "Precision Top 50",
                    "comparator": ">=",
                    "threshold": 0.73,
                    "min_sample": 50,
                    "window_days": 28,
                    "cooldown_days": 14,
                    "enabled": True,
                }
            ],
        },
    )

    assert response.status_code == 403
    assert "admin" in response.json().get("detail", "").lower()


def test_split_trigger_alerts_returns_safe_payload_on_db_error(client, monkeypatch):
    def _raise_connect(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(monitoring.psycopg2, "connect", _raise_connect)

    response = client.get("/api/monitoring/split_trigger_alerts")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["alert_level"] == "critical"
    assert isinstance(body.get("alerts"), list)
    assert isinstance(body.get("track_pass_rates"), list)
    assert isinstance(body.get("readiness_summary"), dict)
    assert isinstance(body.get("snapshot_freshness"), dict)


def test_split_trigger_status_persist_snapshot_forbids_non_admin(client, monkeypatch):
    monkeypatch.setattr(
        monitoring.auth,
        "get_current_user",
        lambda request, db: SimpleNamespace(
            id=3,
            badge_id="USR002",
            role="viewer",
        ),
    )

    monkeypatch.setattr(
        monitoring,
        "get_split_trigger_status_snapshot",
        lambda **_kwargs: {
            "ready": False,
            "schema_ready": True,
            "readiness_score": 65.0,
            "track_status": {},
            "totals": {"enabled_rules": 0, "passed_rules": 0},
        },
    )

    denied_response = client.get("/api/monitoring/split_trigger_status?persist_snapshot=true")
    assert denied_response.status_code == 403
    assert "admin" in denied_response.json().get("detail", "").lower()

    allowed_response = client.get("/api/monitoring/split_trigger_status?persist_snapshot=false")
    assert allowed_response.status_code == 200
    assert allowed_response.json()["schema_ready"] is True


def test_audit_value_effectiveness_returns_safe_payload_on_db_error(client, monkeypatch):
    def _raise_connect(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(monitoring.psycopg2, "connect", _raise_connect)

    response = client.get("/api/monitoring/audit_value_effectiveness")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_ready"] is False
    assert "audit value effectiveness" in str(body.get("message") or "").lower()


def test_vat_refund_effectiveness_returns_safe_payload_on_db_error(client, monkeypatch):
    def _raise_connect(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(monitoring.psycopg2, "connect", _raise_connect)

    response = client.get("/api/monitoring/vat_refund_effectiveness")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_ready"] is False
    assert "vat refund effectiveness" in str(body.get("message") or "").lower()


def test_get_split_trigger_status_snapshot_returns_safe_payload_on_db_error(monkeypatch):
    def _raise_connect(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(monitoring.psycopg2, "connect", _raise_connect)

    payload = monitoring.get_split_trigger_status_snapshot()

    assert payload["ready"] is False
    assert payload["schema_ready"] is False
    assert payload["readiness_score"] == 0
    assert isinstance(payload.get("track_status"), dict)
    assert isinstance(payload.get("totals"), dict)
    assert "split-trigger" in str(payload.get("reason") or "")


def test_specialized_rollout_status_aggregates_reports(client, monkeypatch):
    def _fake_artifact_report(path):
        name = path.name
        if name == monitoring.AUDIT_QUALITY_REPORT_FILE:
            return {
                "exists": True,
                "updated_at": "2026-04-18T09:00:00Z",
                "payload": {
                    "acceptance_gates": {"overall_pass": True},
                    "performance": {
                        "calibrated": {
                            "auc_roc": 0.85,
                            "pr_auc": 0.88,
                            "brier": 0.15,
                            "ece": 0.02,
                        }
                    },
                },
            }
        if name == monitoring.VAT_QUALITY_REPORT_FILE:
            return {
                "exists": True,
                "updated_at": "2026-04-18T09:00:00Z",
                "payload": {
                    "acceptance_gates": {"overall_pass": True},
                    "performance": {
                        "calibrated": {
                            "auc_roc": 0.94,
                            "pr_auc": 0.96,
                            "brier": 0.09,
                            "ece": 0.03,
                        }
                    },
                },
            }
        if name == monitoring.SPECIALIZED_PILOT_REPORT_FILE:
            return {
                "exists": True,
                "updated_at": "2026-04-18T09:05:00Z",
                "payload": {
                    "tracks": {
                        "audit_value": {
                            "samples_evaluated": 200,
                            "delta_model_minus_heuristic": {"f1_delta": 0.24},
                        },
                        "vat_refund": {
                            "samples_evaluated": 200,
                            "delta_model_minus_heuristic": {"f1_delta": -0.03},
                        },
                    }
                },
            }
        if name == monitoring.SPECIALIZED_GO_NO_GO_REPORT_FILE:
            return {
                "exists": True,
                "updated_at": "2026-04-18T09:06:00Z",
                "payload": {
                    "summary": {
                        "hard_gates_pass": True,
                        "split_gate_pass": True,
                        "stability_gate_pass": False,
                    },
                    "decision": {
                        "status": "conditional_go_continue_integrated_first",
                        "go_live_phase_d": False,
                        "message": "Need more stable cycles.",
                        "recommended_actions": ["Keep integrated-first."],
                    },
                },
            }
        return {"exists": False, "updated_at": None, "payload": {}}

    monkeypatch.setattr(monitoring, "_artifact_report", _fake_artifact_report)
    monkeypatch.setattr(
        monitoring,
        "get_split_trigger_status_snapshot",
        lambda **_kwargs: {
            "ready": False,
            "schema_ready": True,
            "readiness_score": 72.0,
            "track_status": {},
            "totals": {"enabled_rules": 4, "passed_rules": 3},
        },
    )

    response = client.get("/api/monitoring/specialized_rollout_status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["rollout_status"] == "conditional_go"
    assert body["summary"]["hard_gates_pass"] is True
    assert body["summary"]["soft_gates_pass"] is False
    assert body["artifacts"]["go_no_go"]["decision_status"] == "conditional_go_continue_integrated_first"
    assert body["artifacts"]["pilot"]["audit_value"]["f1_delta_model_minus_heuristic"] == pytest.approx(0.24)
    assert isinstance(body.get("recommended_actions"), list)


def test_specialized_rollout_status_returns_safe_payload_on_error(client, monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "_build_specialized_rollout_status_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.get("/api/monitoring/specialized_rollout_status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["rollout_status"] == "error"
    assert body["phase_d_candidate"] is False
    assert "specialized rollout status" in str(body.get("error") or "").lower()
