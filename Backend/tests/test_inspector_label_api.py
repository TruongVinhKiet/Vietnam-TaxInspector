import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import auth, models
from app.database import get_db
from app.main import app


class _AssessmentRow:
    def __init__(self, assessment_id: int, tax_code: str):
        self.id = assessment_id
        self.tax_code = tax_code


class _AssessmentQuery:
    def __init__(self, row):
        self._row = row

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return self._row


class _FakeDB:
    def __init__(self, assessment_row=None):
        self.assessment_row = assessment_row
        self.added = []

    def query(self, model):
        if model is models.AIRiskAssessment:
            return _AssessmentQuery(self.assessment_row)
        raise AssertionError(f"Unexpected model query: {model}")

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        return None

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_inspector_label_requires_authentication(client):
    fake_db = _FakeDB(_AssessmentRow(assessment_id=10, tax_code="01010001"))

    def _override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = _override_get_db

    payload = {
        "tax_code": "01010001",
        "label_type": "fraud_confirmed",
        "confidence": "high",
        "assessment_id": 10,
    }
    response = client.post("/api/ai/inspector-label", json=payload)

    assert response.status_code == 401


def test_inspector_label_binds_current_user_as_inspector(client):
    fake_db = _FakeDB(_AssessmentRow(assessment_id=11, tax_code="01010001"))

    def _override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[auth.get_current_user] = lambda: SimpleNamespace(id=7, badge_id="B007")

    payload = {
        "tax_code": "01010001",
        "label_type": "fraud_confirmed",
        "confidence": "high",
        "assessment_id": 11,
        "evidence_summary": "Invoice loop pattern confirmed",
        "decision": "investigate",
    }
    response = client.post("/api/ai/inspector-label", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["tax_code"] == "01010001"
    assert body["inspector_id"] == 7
    assert body["assessment_id"] == 11

    assert len(fake_db.added) == 1
    assert fake_db.added[0].inspector_id == 7


def test_inspector_label_rejects_mismatched_assessment_tax_code(client):
    fake_db = _FakeDB(_AssessmentRow(assessment_id=12, tax_code="09999999"))

    def _override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[auth.get_current_user] = lambda: SimpleNamespace(id=9, badge_id="B009")

    payload = {
        "tax_code": "01010001",
        "label_type": "fraud_rejected",
        "confidence": "medium",
        "assessment_id": 12,
    }
    response = client.post("/api/ai/inspector-label", json=payload)

    assert response.status_code == 400
    assert "khớp" in response.json()["detail"]
    assert fake_db.added == []


def test_inspector_label_accepts_outcome_kpi_fields(client):
    fake_db = _FakeDB(_AssessmentRow(assessment_id=13, tax_code="01010001"))

    def _override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[auth.get_current_user] = lambda: SimpleNamespace(id=11, badge_id="B011")

    payload = {
        "tax_code": "01010001",
        "label_type": "fraud_confirmed",
        "confidence": "high",
        "assessment_id": 13,
        "intervention_action": "field_audit",
        "intervention_attempted": True,
        "outcome_status": "partial_recovered",
        "predicted_collection_uplift": 120000000,
        "expected_recovery": 200000000,
        "expected_net_recovery": 130000000,
        "estimated_audit_cost": 30000000,
        "actual_audit_cost": 25000000,
        "actual_audit_hours": 42.5,
        "amount_recovered": 85000000,
        "kpi_window_days": 120,
    }

    response = client.post("/api/ai/inspector-label", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["intervention_action"] == "field_audit"
    assert body["intervention_attempted"] is True
    assert body["outcome_status"] == "partial_recovered"
    assert body["kpi_window_days"] == 120

    assert len(fake_db.added) == 1
    assert fake_db.added[0].expected_recovery == 200000000
    assert fake_db.added[0].actual_audit_hours == 42.5
