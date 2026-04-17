import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import get_db
from app.main import app


class _EmptyQuery:
    def __init__(self):
        self.c = SimpleNamespace(tax_code="tax_code", latest="latest")

    def filter(self, *_args, **_kwargs):
        return self

    def distinct(self):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def join(self, *_args, **_kwargs):
        return self

    def subquery(self):
        return self

    def group_by(self, *_args, **_kwargs):
        return self

    def count(self):
        return 0

    def offset(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def all(self):
        return []


class _EmptyDB:
    def query(self, *_args, **_kwargs):
        return _EmptyQuery()

    def rollback(self):
        return None



def _override_get_db():
    yield _EmptyDB()


def test_delinquency_list_invalid_freshness_returns_422():
    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            resp = client.get("/api/delinquency?freshness=very_old")
            assert resp.status_code == 422
            body = resp.json()
            assert "freshness" in body.get("detail", "")
    finally:
        app.dependency_overrides.clear()


def test_delinquency_batch_predict_empty_candidates_returns_zero_counts():
    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            resp = client.post("/api/delinquency/predict-batch", json={"limit": 10, "refresh_existing": False})
            assert resp.status_code == 200
            body = resp.json()
            assert body["total_candidates"] == 0
            assert body["processed"] == 0
            assert body["created"] == 0
            assert body["updated"] == 0
            assert body["failed"] == 0
    finally:
        app.dependency_overrides.clear()


def test_delinquency_cache_health_invalid_threshold_returns_422():
    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            resp = client.get("/api/delinquency/health/cache?fresh_days=30&stale_days=7")
            assert resp.status_code == 422
            body = resp.json()
            assert "fresh_days" in body.get("detail", "")
    finally:
        app.dependency_overrides.clear()


def test_delinquency_cache_health_no_data_payload_shape():
    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            resp = client.get("/api/delinquency/health/cache")
            assert resp.status_code == 200
            body = resp.json()

            assert body["status"] == "no_data"
            assert body["coverage"]["companies_with_payments"] == 0
            assert body["coverage"]["companies_with_latest_prediction"] == 0
            assert "freshness" in body
            assert "sources" in body
            assert "alerts" in body
    finally:
        app.dependency_overrides.clear()
