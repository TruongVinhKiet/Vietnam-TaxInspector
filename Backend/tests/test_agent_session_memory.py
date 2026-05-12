"""Session memory: batch/VAT snapshot persist and MST follow-up injection."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from ml_engine.tax_agent_memory import ConversationMemory
from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator


@pytest.fixture
def memory() -> ConversationMemory:
    return ConversationMemory(MagicMock())


def test_persist_upload_risk_csv_saves_batch(memory: ConversationMemory) -> None:
    orch = TaxAgentOrchestrator.__new__(TaxAgentOrchestrator)
    orch._memory = memory
    ctx = SimpleNamespace(last_batch_data=None, last_vat_snapshot=None)
    orch._persist_upload_session_memory(
        session_id="sid_risk",
        context=ctx,
        csv_attachment=None,
        nl_results={
            "_batch_results": {
                "analysis_type": "risk_csv",
                "status": "success",
                "filename": "risk.csv",
                "total": 1,
                "assessments": [{"tax_code": "0100000000", "risk_score": 42.0}],
                "by_level": {},
                "top_5": [],
            }
        },
    )
    cached = memory.get_batch_data("sid_risk")
    assert cached is not None
    assert cached["companies"][0]["tax_code"] == "0100000000"
    assert ctx.last_batch_data is not None
    assert ctx.last_batch_data["filename"] == "risk.csv"


def test_inject_followup_mst_matches_fixture_tax_code(memory: ConversationMemory) -> None:
    orch = TaxAgentOrchestrator.__new__(TaxAgentOrchestrator)
    orch._memory = memory
    csv_path = BACKEND_ROOT / "data" / "risk_data_5000_companies.csv"
    assert csv_path.is_file(), f"missing fixture {csv_path}"
    row_mst = "0200000001"
    memory.save_batch_data(
        "sid_fixture",
        {
            "filename": csv_path.name,
            "total": 1,
            "companies": [{"tax_code": row_mst, "company_name": "Fixture row", "risk_score": 77.0}],
            "by_level": {},
            "top_risky": [],
        },
    )
    ctx = SimpleNamespace(last_batch_data=None)
    nl: dict = {}
    orch._inject_session_followup_rows(
        session_id="sid_fixture",
        message=f"Phân tích chi tiết MST {row_mst}",
        context=ctx,
        nl_results=nl,
    )
    assert nl["_session_upload_row"]["tax_code"] == row_mst
    assert nl["_session_upload_row"]["row"]["company_name"] == "Fixture row"


def test_inject_vat_focus_from_snapshot(memory: ConversationMemory) -> None:
    orch = TaxAgentOrchestrator.__new__(TaxAgentOrchestrator)
    orch._memory = memory
    vat_csv = BACKEND_ROOT / "data" / "vat_invoices_15000.csv"
    assert vat_csv.is_file(), f"missing fixture {vat_csv}"
    seller = "S2900010280"
    memory.save_vat_snapshot(
        "sid_vat",
        {
            "filename": vat_csv.name,
            "batch_id": 1,
            "top_invoice_risks": [
                {
                    "seller_tax_code": seller,
                    "buyer_tax_code": "B3600010917",
                    "amount": 100.0,
                }
            ],
            "nodes": [],
            "edges": [],
        },
    )
    ctx = SimpleNamespace(last_batch_data=None, last_vat_snapshot=None)
    nl: dict = {}
    orch._inject_session_followup_rows(
        session_id="sid_vat",
        message=f"Hóa đơn liên quan MST {seller}",
        context=ctx,
        nl_results=nl,
    )
    assert "_vat_session_focus" in nl
    assert nl["_vat_session_focus"]["invoices"][0]["seller_tax_code"] == seller


def test_normalize_xlsx_produces_csv_work_name() -> None:
    pytest.importorskip("openpyxl")
    import pandas as pd
    from io import BytesIO

    from app.multimodal_analysis import normalize_agent_upload_bytes

    buf = BytesIO()
    pd.DataFrame({"tax_code": ["0100000000"], "net_profit": [100.0]}).to_excel(
        buf, index=False, engine="openpyxl"
    )
    raw = buf.getvalue()
    csv_bytes, work_name, original = normalize_agent_upload_bytes(raw, "mini_risk.xlsx")
    assert work_name.endswith(".csv")
    assert original == "mini_risk.xlsx"
    head = csv_bytes.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    assert "tax_code" in head.lower()
