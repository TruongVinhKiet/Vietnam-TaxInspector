from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_synthesis import TaxAgentSynthesizer


def test_legal_synthesis_adds_verification_and_metadata():
    synthesizer = TaxAgentSynthesizer()
    result = synthesizer.synthesize(
        query="Dieu kien khau tru thue dau vao nam 2025 cho doanh nghiep?",
        intent="general_tax_query",
        tool_results={
            "knowledge_search": {
                "status": "success",
                "hits": [
                    {
                        "title": "Circular VAT",
                        "text": "Doanh nghiep duoc khau tru thue dau vao khi co hoa don hop phap va thanh toan qua ngan hang.",
                        "score": 0.91,
                        "chunk_key": "VAT_1",
                        "citation_spans": [{"start": 0, "end": 80}],
                        "effective_status": {"dominant_state": "active", "has_non_usable": False},
                        "official_letter_scope": {"has_official_letter": False, "warnings": []},
                        "authority_path": [{"display_name": "Circular VAT", "entity_type": "circular", "authority_rank": 70}],
                        "relation_path": [],
                    }
                ],
                "graph_context": {
                    "authority_path": [{"display_name": "Circular VAT", "entity_type": "circular", "authority_rank": 70}],
                    "effective_status": {"dominant_state": "active", "has_non_usable": False},
                    "official_letter_scope": {"warnings": []},
                },
            }
        },
        reasoning_trace="test",
    )

    assert result.evidence
    assert result.evidence[0].metadata["citation_spans"]
    assert result.verification["claim_count"] >= 1
    assert result.synthesis_tier in {"template", "template_verified_fallback", "llm_finetuned", "llm_base_few_shot"}
