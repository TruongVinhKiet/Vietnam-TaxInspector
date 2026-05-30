import json
import sys

sys.path.append("e:/TaxInspector/Backend")

from ml_engine.tax_agent_conversation_intelligence import ConversationIntelligence
from ml_engine.tax_agent_enhanced_intent import EnhancedIntentClassifier
from ml_engine.tax_agent_tool_contracts import CANONICAL_TOOL_NAMES
from scripts.generate_mega_agent_dataset_v4 import (
    TOOL_BLUEPRINTS,
    generate,
    load_legal_deep_data,
    make_noisy_variants,
)


def test_v4_generator_covers_real_backend_tool_surface():
    actual_tools = {spec["name"] for spec in TOOL_BLUEPRINTS}

    assert actual_tools == CANONICAL_TOOL_NAMES


def test_v4_generator_loads_deep_legal_and_citizen_topics():
    legal_data = load_legal_deep_data()
    titles = " ".join(item["q"] for item in legal_data).lower()

    assert len(legal_data) >= 80
    assert "shopee" in titles or "tiktok" in titles
    assert "người phụ thuộc" in titles


def test_v4_generator_has_no_accent_and_typo_variants():
    variants = make_noisy_variants("xin chào", __import__("random").Random(42), max_variants=8)
    normalized = {v.lower() for v in variants}

    assert "xin chao" in normalized
    assert "xn chào" in normalized or "xi chà" in normalized


def test_v4_generator_writes_valid_chatml_sample(tmp_path):
    out = tmp_path / "agent_sample.jsonl"
    summary = generate(
        total_simple=42,
        total_legal=10,
        total_smalltalk=4,
        total_clarification=4,
        seed=7,
        output_path=out,
        write_latest_alias=False,
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    text_blob = "\n".join(lines[:20])

    assert summary["total_records"] == 60
    assert len(records) == 60
    assert any(r["metadata"]["kind"] == "legal_graphrag_answer" for r in records)
    assert any(any(m["role"] == "tool" for m in r["messages"]) for r in records)
    assert "Báº" not in text_blob
    assert "Ä‘" not in text_blob


def test_runtime_handles_short_noisy_greetings():
    conv = ConversationIntelligence()

    for query in ("xn chào", "xi chà", "xin chao"):
        result = conv.process(
            message=query,
            active_tax_code=None,
            recent_turns=[],
            active_entities=[],
        )
        assert result.dialogue_act == "greeting"
        assert result.should_plan is False


def test_intent_classifier_does_not_turn_tax_question_into_smalltalk():
    classifier = EnhancedIntentClassifier()

    assert classifier._detect_dialogue_act("xi chà") == "smalltalk"
    assert classifier._detect_dialogue_act("xin chào cho hỏi thuế TNCN") is None
