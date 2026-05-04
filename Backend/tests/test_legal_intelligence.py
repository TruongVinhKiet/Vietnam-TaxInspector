from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_legal_intelligence import (
    LegalFaithfulnessVerifier,
    LegalKnowledgeExtractor,
    LegalSlotAnalyzer,
    canonical_doc_type,
    effective_status,
    official_letter_scope,
)


def test_doc_type_scope_and_effective_status():
    assert canonical_doc_type("cong van") == "official_letter"
    assert canonical_doc_type("Nghi dinh") == "decree"

    scope = official_letter_scope(
        doc_type="official_letter",
        title="Cong van tra loi Cong ty ABC",
        text="Tra loi theo de nghi cua Cong ty ABC",
    )
    assert scope["is_official_letter"] is True
    assert scope["binding_level"] == "guidance_not_normative"
    assert scope["warnings"]

    expired = effective_status(
        effective_from="2020-01-01",
        effective_to="2021-01-01",
        as_of=date(2026, 5, 3),
    )
    assert expired["state"] == "expired"
    assert expired["is_usable"] is False


def test_legal_extractor_finds_entities_relations_and_citations():
    extractor = LegalKnowledgeExtractor()
    result = extractor.extract_document(
        document_key="LAW_TEST",
        title="Luat thue GTGT test",
        doc_type="law",
        content=(
            "Dieu 1. Pham vi ap dung. Khoan 1 quy dinh ve khau tru thue. "
            "Van ban nay sua doi 219/2013/TT-BTC va thay the 12/2015/TT-BTC."
        ),
        chunks=[{
            "chunk_id": 10,
            "chunk_index": 0,
            "text": (
                "Dieu 1. Pham vi ap dung. Khoan 1 quy dinh ve khau tru thue. "
                "Van ban nay sua doi 219/2013/TT-BTC va thay the 12/2015/TT-BTC."
            ),
            "heading": "Dieu 1",
        }],
    )
    entity_types = {entity.entity_type for entity in result.entities}
    relation_types = {relation.relation_type for relation in result.relations}

    assert "law" in entity_types
    assert "article" in entity_types
    assert "contains" in relation_types
    assert {"amends", "replaces"} & relation_types
    assert result.citations


def test_faithfulness_verifier_flags_unsupported_claims():
    verifier = LegalFaithfulnessVerifier()
    evidence = [{
        "source_type": "legal",
        "title": "Circular A",
        "content": "Doanh nghiep duoc khau tru thue dau vao khi co hoa don hop phap va thanh toan qua ngan hang.",
    }]
    result = verifier.verify(
        answer_text=(
            "Doanh nghiep duoc khau tru thue dau vao khi co hoa don hop phap [1].\n"
            "Ho so phai nop tai so xay dung trong 5 ngay."
        ),
        evidence=evidence,
    )
    assert result["claim_count"] >= 2
    assert result["verified_claims"]
    assert result["unsupported_claims"]


def test_slot_analyzer_requests_missing_legal_facts():
    analyzer = LegalSlotAnalyzer()
    missing = analyzer.missing_slots("Dieu kien ap dung quy dinh thue", intent="general_tax_query")
    assert "tax_period_or_document_date" in missing
    assert "taxpayer_type" in missing
    assert "transaction_type" in missing
    prompt = analyzer.clarification_prompt(missing)
    assert "ky thue" in prompt
