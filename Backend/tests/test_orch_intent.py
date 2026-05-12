import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def test_orchestrator_rule_based_intent_for_vat_legal_question():
    from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator

    orch = TaxAgentOrchestrator()
    message = (
        "Công ty tôi kinh doanh dịch vụ ăn uống. Theo Nghị định 72/2024/NĐ-CP, "
        "tôi có được áp dụng thuế GTGT 8% không?"
    )

    intent, confidence = orch._rule_based_intent(message)
    assert intent == "vat_refund_risk"
    assert confidence >= 0.35


def test_orchestrator_rule_based_intent_for_top_n_query():
    from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator

    orch = TaxAgentOrchestrator()
    intent, confidence = orch._rule_based_intent("Top 10 doanh nghiệp rủi ro cao nhất")
    assert intent == "top_n_query"
    assert confidence == 0.9
