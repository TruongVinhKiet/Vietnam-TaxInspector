import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import get_db
from app.main import app


class _DummyDB:
    pass


class _ThreadDummyDB:
    def close(self):
        return None


@pytest.fixture
def client():
    def _override_get_db():
        yield _DummyDB()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _fake_orch_response(**overrides):
    base = {
        "session_id": "sess-test-001",
        "intent": "invoice_risk",
        "intent_confidence": 0.91,
        "complexity": "medium",
        "reasoning_trace": "router->planner->tools->synthesis",
        "tools_used": ["company_risk_lookup"],
        "plan_steps": [{"tool_name": "company_risk_lookup", "tool_inputs": {}}],
        "answer": "phan tich da hoan tat",
        "summary": "tom tat",
        "citations": [],
        "recommendations": ["kiem tra bo sung"],
        "confidence": 0.88,
        "abstained": False,
        "escalation_required": False,
        "escalation_domain": "",
        "compliance_warnings": [],
        "active_tax_code": "0101234567",
        "active_tax_period": "2025Q4",
        "latency_ms": 123.4,
        "latency_breakdown": {"intent_ms": 10.0},
        "synthesis_tier": "fraud_analysis_v2",
        "policy_traces": [],
        "tool_results": {},
        "visualization_data": {"fraud": {"risk_gauge": {"score": 82}}},
        "dialogue_act": "task",
        "answer_contract": "fraud_analysis",
        "routing_decision": {"requested_domain": "fraud"},
        "focus_score": 1.0,
        "route_violation": False,
        "selected_model_bundle": ["fraud", "xai"],
        "mode_validation": {"valid": True, "model_mode": "full", "requested_domain": "fraud"},
        "mode_mismatch": False,
        "suggested_mode": None,
        "suppressed_domains": ["legal", "vat"],
        "analysis_blocks": [{"type": "summary", "title": "Tong hop", "data": {}}],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_chat_v2_includes_router_metadata_fields(client, monkeypatch):
    import ml_engine.tax_agent_orchestrator as orchestrator_module

    class _FakeOrchestrator:
        def process(self, db, session_id, message, user_id=None, top_k=5, model_mode="full", attachment_analysis=None):
            return _fake_orch_response(session_id=session_id)

    monkeypatch.setattr(orchestrator_module, "get_orchestrator", lambda: _FakeOrchestrator())

    response = client.post(
        "/api/tax-agent/chat/v2",
        json={
            "message": "Top 10 doanh nghiep rui ro",
            "session_id": "sess-api-v2-001",
            "model_mode": "full",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model_bundle"] == ["fraud", "xai"]
    assert payload["mode_validation"]["valid"] is True
    assert payload["mode_mismatch"] is False
    assert payload["suppressed_domains"] == ["legal", "vat"]


def test_chat_v2_with_file_returns_mode_mismatch_contract(client, monkeypatch):
    import app.multimodal_analysis as multimodal_module
    import ml_engine.tax_agent_orchestrator as orchestrator_module

    monkeypatch.setattr(
        multimodal_module,
        "detect_attachment_for_agent",
        lambda content, filename, content_type=None, original_filename=None: {
            "status": "ok",
            "analysis_type": "risk_csv",
            "requested_domain": "fraud",
        },
    )
    monkeypatch.setattr(
        multimodal_module,
        "analyze_attachment_for_agent",
        lambda *args, **kwargs: {"status": "should_not_be_called"},
    )

    class _FakeOrchestrator:
        def process(self, db, session_id, message, user_id=None, top_k=5, model_mode="full", attachment_analysis=None):
            assert attachment_analysis is not None
            assert attachment_analysis.get("mode_mismatch") is True
            return _fake_orch_response(
                session_id=session_id,
                mode_mismatch=True,
                suggested_mode="fraud",
                mode_validation=attachment_analysis.get("mode_validation", {}),
                answer_contract="mode_mismatch",
                analysis_blocks=[{"type": "mode_mismatch", "title": "Che do khong phu hop", "data": {}}],
            )

    monkeypatch.setattr(orchestrator_module, "get_orchestrator", lambda: _FakeOrchestrator())

    response = client.post(
        "/api/tax-agent/chat/v2/with-file",
        data={"message": "Kiem tra", "session_id": "sess-file-001", "model_mode": "legal"},
        files={"file": ("risk.csv", io.BytesIO(b"tax_code,risk_score\n0101,88\n"), "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode_mismatch"] is True
    assert payload["suggested_mode"] == "fraud"
    assert payload["mode_validation"]["reason"] == "selected_mode_does_not_match_attachment_schema"
    assert payload["analysis_blocks"][0]["type"] == "mode_mismatch"


def test_chat_v2_stream_emits_done_event_with_route_metadata(client, monkeypatch):
    import app.database as database_module
    import ml_engine.tax_agent_orchestrator as orchestrator_module

    class _FakeOrchestrator:
        def process_streaming(self, db, session_id, message, user_id=None, top_k=5, model_mode="full"):
            yield {
                "event": "done",
                "data": {
                    "selected_model_bundle": ["vat", "invoice_risk"],
                    "mode_validation": {"valid": True, "requested_domain": "vat"},
                    "mode_mismatch": False,
                    "suggested_mode": None,
                    "suppressed_domains": ["legal"],
                },
            }

    monkeypatch.setattr(orchestrator_module, "get_orchestrator", lambda: _FakeOrchestrator())
    monkeypatch.setattr(database_module, "SessionLocal", lambda: _ThreadDummyDB())

    response = client.post(
        "/api/tax-agent/chat/v2/stream",
        json={"message": "Phan tich VAT", "session_id": "sess-stream-001", "model_mode": "full"},
    )
    assert response.status_code == 200
    text = response.text
    assert "event: done" in text
    assert '"selected_model_bundle": ["vat", "invoice_risk"]' in text
    assert '"suppressed_domains": ["legal"]' in text


def test_build_v2_response_compacts_async_heavy_payload(monkeypatch):
    import app.routers.tax_agent as tax_agent_router

    heavy = [{"idx": i} for i in range(500)]
    orch = _fake_orch_response(
        tool_results={
            "_attachment_analysis": {
                "analysis_type": "vat_graph_csv",
                "graph": {
                    "nodes": heavy,
                    "edges": heavy,
                    "top_invoice_risks": heavy,
                },
                "canonical_batch_results": {"huge": heavy},
            },
            "_batch_results": {
                "assessments": heavy,
                "canonical_batch_results": {"huge": heavy},
            },
        },
        legal_workspace={"facts": ["A"]},
        simulation_workspace={"current_params": {"gdp_growth": 6.3}},
    )
    payload = tax_agent_router._build_v2_response(orch, model_mode="full", compact_async=True)
    assert len(payload.tool_results["_batch_results"]["assessments"]) <= 60
    assert "canonical_batch_results" not in payload.tool_results["_batch_results"]
    assert len(payload.attachment_analysis["graph"]["nodes"]) <= 180
    assert payload.legal_workspace["facts"] == ["A"]
    assert payload.simulation_workspace["current_params"]["gdp_growth"] == 6.3

