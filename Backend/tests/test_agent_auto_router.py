import sys

sys.path.append("e:/TaxInspector/Backend")

from ml_engine.tax_agent_task_router import TaskRouter
from ml_engine.tax_agent_planner import TaxAgentPlanner


def test_full_top_n_uses_direct_data_contract_only():
    decision = TaskRouter().route(
        query="Top 10 doanh nghiệp rủi ro cao nhất",
        intent="top_n_query",
        model_mode="full",
    )

    assert decision.answer_contract.value == "data_table"
    assert decision.requested_domain == "fraud"
    assert decision.allowed_tools == {"top_n_risky_companies"}
    assert not decision.allow_legal
    assert "knowledge_search" in decision.suppressed_tools


def test_legal_mode_rejects_risk_csv_attachment():
    decision = TaskRouter().route(
        query="phân tích file này",
        intent="batch_analysis",
        model_mode="legal",
        has_attachment=True,
        attachment_analysis={
            "status": "detected",
            "analysis_type": "risk_csv",
            "requested_domain": "fraud",
            "filename": "risk.csv",
        },
    )

    assert decision.answer_contract.value == "mode_mismatch"
    assert decision.mode_mismatch
    assert decision.requested_domain == "fraud"
    assert decision.suggested_mode == "fraud"
    assert decision.allowed_tools == set()


def test_vat_mode_rejects_risk_csv_attachment():
    decision = TaskRouter().route(
        query="chấm điểm rủi ro doanh nghiệp",
        intent="batch_analysis",
        model_mode="vat",
        has_attachment=True,
        attachment_analysis={
            "status": "detected",
            "analysis_type": "risk_csv",
            "requested_domain": "fraud",
            "filename": "risk.csv",
        },
    )

    assert decision.mode_mismatch
    assert decision.suggested_mode == "fraud"


def test_full_risk_csv_routes_to_fraud_file_analysis_without_legal():
    decision = TaskRouter().route(
        query="phân tích file này",
        intent="batch_analysis",
        model_mode="full",
        has_attachment=True,
        attachment_analysis={
            "status": "detected",
            "analysis_type": "risk_csv",
            "requested_domain": "fraud",
            "filename": "risk.csv",
        },
    )

    assert decision.answer_contract.value == "fraud_analysis"
    assert decision.requested_domain == "fraud"
    assert decision.allowed_tools == set()
    assert not decision.allow_legal


def test_vat_legal_question_stays_in_vat_mode_but_allows_graphrag():
    decision = TaskRouter().route(
        query="căn cứ pháp lý hoàn thuế VAT",
        intent="vat_refund_risk",
        model_mode="vat",
    )

    assert decision.answer_contract.value == "legal_consultation"
    assert decision.requested_domain == "vat"
    assert not decision.mode_mismatch
    assert decision.allow_legal
    assert "knowledge_search" in decision.allowed_tools


def test_macro_mode_planner_selects_macro_forecast_only():
    plan = TaxAgentPlanner().plan(
        query="Chay mo phong vi mo voi cac tham so hien tai",
        intent="general_tax_query",
        intent_confidence=0.82,
        model_mode="macro",
    )

    assert [step.tool_name for step in plan.steps] == ["macro_forecast"]


def test_auto_macro_request_routes_to_macro_bundle_without_legal():
    decision = TaskRouter().route(
        query="Chay mo phong vi mo voi VAT 10 va GDP 6.5",
        intent="macro_forecast",
        model_mode="full",
    )

    assert decision.requested_domain == "macro"
    assert "macro_forecast" in decision.allowed_tools
    assert "knowledge_search" in decision.suppressed_tools
    assert not decision.allow_legal
