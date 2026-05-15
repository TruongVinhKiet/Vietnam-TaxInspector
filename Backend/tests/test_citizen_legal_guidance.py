from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_citizen_legal import retrieve_citizen_legal_snippets
from ml_engine.tax_agent_synthesis import TaxAgentSynthesizer
from ml_engine.tax_agent_task_router import TaskRouter


def test_everyday_tax_question_routes_to_legal_consultation() -> None:
    decision = TaskRouter().route(
        query=(
            "Em thấy công ty khấu trừ thuế TNCN sai. Lương em 20 triệu, "
            "có 1 con nhỏ là người phụ thuộc, làm sao hoàn thuế?"
        ),
        intent="general_tax_query",
        model_mode="full",
    )

    assert decision.requested_domain == "legal"
    assert decision.answer_contract.value == "legal_consultation"
    assert decision.allow_legal
    assert "knowledge_search" in decision.allowed_tools


def test_citizen_legal_fallback_retrieves_relevant_tncn_guidance() -> None:
    hits = retrieve_citizen_legal_snippets(
        "Lương 20 triệu có người phụ thuộc nhưng công ty khấu trừ thuế TNCN quá cao, muốn hoàn thuế",
        top_k=3,
    )

    assert hits
    assert hits[0]["chunk_key"].startswith("citizen_tax_faq:")
    assert "TNCN" in hits[0]["title"] or "TNCN" in hits[0]["text"]
    assert hits[0]["official_letter_scope"]["binding_level"] == "guidance_not_normative"


def test_synthesizer_uses_citizen_fallback_as_legal_evidence() -> None:
    hits = retrieve_citizen_legal_snippets(
        "Em bán hàng trên Shopee và TikTok Shop có phải đóng thuế không?",
        top_k=2,
    )
    synthesizer = TaxAgentSynthesizer()
    result = synthesizer.synthesize(
        query="Em bán hàng trên Shopee và TikTok Shop có phải đóng thuế không?",
        intent="general_tax_query",
        tool_results={"knowledge_search": {"status": "success", "hits": hits}},
        answer_contract="legal_consultation",
    )

    assert result.evidence
    assert "các bước tiếp theo" in result.detailed_analysis.lower() or "bước xử lý" in result.detailed_analysis.lower()
    assert result.confidence > 0
