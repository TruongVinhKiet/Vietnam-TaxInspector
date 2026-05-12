import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator


def _orchestrator():
    # Do not initialize heavy dependencies; we only test pure payload builders.
    return TaxAgentOrchestrator()


def test_analysis_blocks_include_required_shape_for_fraud_and_vat():
    orch = _orchestrator()
    synthesis_result = SimpleNamespace(
        summary="Tong hop ket qua",
        recommendations=["Kiem tra bo sung", "Theo doi 30 ngay"],
    )

    fraud_blocks = orch._build_analysis_blocks(
        tool_results={
            "_batch_results": {
                "total": 2,
                "by_level": {"high": 1, "medium": 1},
                "assessments": [{"tax_code": "0101", "risk_score": 86.0}],
            }
        },
        routing_decision=SimpleNamespace(answer_contract=SimpleNamespace(value="fraud_analysis")),
        synthesis_result=synthesis_result,
    )
    assert fraud_blocks
    fraud = fraud_blocks[0]
    assert {"type", "title", "summary", "metrics", "next_steps"}.issubset(fraud.keys())
    assert fraud["type"] == "fraud_analysis"
    assert {"total_records", "risk_distribution", "top_records"}.issubset(fraud["metrics"].keys())

    vat_blocks = orch._build_analysis_blocks(
        tool_results={
            "_vat_graph_batch_results": {
                "processed_rows": 12,
                "summary": {"suspect_value": 120000000},
                "graph": {
                    "rings": [{"id": "r1"}],
                    "top_invoice_risks": [{"invoice_no": "INV-001", "risk": 0.91}],
                },
            }
        },
        routing_decision=SimpleNamespace(answer_contract=SimpleNamespace(value="vat_graph")),
        synthesis_result=synthesis_result,
    )
    assert vat_blocks
    vat = vat_blocks[0]
    assert {"type", "title", "summary", "metrics", "next_steps"}.issubset(vat.keys())
    assert vat["type"] == "vat_graph"
    assert {"processed_rows", "suspect_value", "rings", "top_invoice_risks"}.issubset(vat["metrics"].keys())


def test_fraud_visualization_payload_exposes_minimum_contract_keys():
    orch = _orchestrator()
    payload = orch._build_fraud_visualization_payload(
        tool_results={
            "_batch_results": {
                "total": 2,
                "assessments": [
                    {"tax_code": "0101", "risk_score": 82, "risk_level": "high", "revenue": 1_000_000, "year": 2024},
                    {"tax_code": "0102", "risk_score": 67, "risk_level": "medium", "revenue": 750_000, "year": 2025},
                ],
                "by_level": {"high": 1, "medium": 1},
            }
        },
        sub_agent_analysis={"analytics": {"summary": "Dong thuan cao"}, "investigation": {"flags": 2}},
    )
    required = {
        "summary",
        "risk_gauge",
        "radar",
        "yearly_trend",
        "revenue_risk_scatter",
        "risk_distribution",
        "cumulative_risk",
        "top_companies",
        "case_narrative",
        "cross_model_consensus",
    }
    assert required.issubset(payload.keys())
    assert {"score", "level", "color", "confidence"}.issubset(payload["risk_gauge"].keys())


def test_vat_visualization_payload_exposes_minimum_contract_keys():
    orch = _orchestrator()
    payload = orch._build_vat_visualization_payload(
        tool_results={
            "_vat_graph_batch_results": {
                "batch_id": "batch-001",
                "row_count": 20,
                "processed_rows": 20,
                "summary": {"suspect_value": 3_500_000_000},
                "graph": {
                    "nodes": [{"id": "0101"}, {"id": "0102"}],
                    "edges": [{"source": "0101", "target": "0102"}],
                    "timeline": [{"month": "2025-01", "count": 4}],
                    "risk_bars": [{"name": "A", "risk": 0.8}],
                    "model_intelligence": {"version": "graph-intelligence-v2.1"},
                    "rings": [{"id": "ring-1"}],
                    "ownership_summary": {"links": 5},
                    "forensic_logs": [{"event": "ring_detected"}],
                    "evidence_paths": [{"from": "0101", "to": "0102"}],
                    "cross_border_signals": {"offshore_nodes": 1},
                    "top_invoice_risks": [{"invoice_no": "INV-001", "risk": 0.95}],
                },
            }
        }
    )
    required = {
        "summary",
        "graph",
        "timeline",
        "risk_bars",
        "model_intelligence",
        "ring_scoring",
        "ownership_summary",
        "forensic_logs",
        "evidence_paths",
        "cross_border_signals",
        "top_invoice_risks",
        "ocr_invoice",
    }
    assert required.issubset(payload.keys())
    assert {"nodes", "edges"}.issubset(payload["graph"].keys())


def test_simulation_workspace_payload_contract_for_macro_mode():
    orch = _orchestrator()
    ws = orch._build_simulation_workspace(
        context=SimpleNamespace(session_id="sess-sim"),
        model_mode="macro",
        message="mô phỏng thay đổi lãi suất",
        tool_results={
            "macro_scenario_simulation": {
                "parameters": {"interest_rate": 6.5},
                "recommended_parameters": {"interest_rate": 6.0},
                "parameter_ranges": {"interest_rate": [4.0, 10.0]},
                "sensitivity_top_factors": [{"name": "interest_rate", "impact": 0.72}],
                "scenario_label": "Base",
                "projection_years": 5,
            }
        },
        viz_data={"macro_kpis": {"gdp": 6.2}},
    )
    assert ws["current_params"]["interest_rate"] == 6.5
    assert ws["recommended_params"]["interest_rate"] == 6.0
    assert ws["projection_years"] == 5
    assert ws["kpis"]["gdp"] == 6.2

