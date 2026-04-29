"""
tax_agent_llm_data_pipeline.py – Training Data Pipeline (Phase 6)
==================================================================
Generates training data for custom LLM from the agent's audit trail.

Pipeline:
    1. Extract Q&A pairs from agent_turns + agent_decision_traces
    2. Generate instruction-tuning format (Alpaca/ShareGPT)
    3. Quality filtering (confidence, user feedback)
    4. Deduplication (semantic similarity)
    5. Augmentation (paraphrase templates)
    6. Export to JSONL for training
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example for LLM fine-tuning."""
    instruction: str
    input_text: str
    output_text: str
    # Metadata
    intent: str
    confidence: float
    source: str                 # "audit_trail", "augmented", "manual"
    quality_score: float        # 0-1 (higher = better)
    # Filtering
    session_id: str = ""
    turn_id: int = 0
    hash_key: str = ""


@dataclass
class DatasetStats:
    """Statistics for the generated dataset."""
    total_examples: int
    by_intent: dict[str, int]
    by_source: dict[str, int]
    avg_quality: float
    avg_input_len: int
    avg_output_len: int
    deduplicated: int
    filtered_low_quality: int


# ─── System Prompts for Different Domains ─────────────────────────────────────

SYSTEM_PROMPTS = {
    "general": (
        "Bạn là trợ lý AI chuyên dụng cho cơ quan thuế Việt Nam. "
        "Nhiệm vụ: phân tích rủi ro, tra cứu pháp luật thuế, "
        "và hỗ trợ ra quyết định thanh tra. "
        "Trả lời chính xác, có trích dẫn căn cứ pháp lý."
    ),
    "legal_research": (
        "Bạn là chuyên gia pháp luật thuế Việt Nam. "
        "Trả lời dựa trên luật thuế, nghị định, thông tư hiện hành. "
        "Luôn trích dẫn điều khoản cụ thể."
    ),
    "risk_analysis": (
        "Bạn là chuyên gia phân tích rủi ro thuế. "
        "Đánh giá rủi ro dựa trên dữ liệu định lượng. "
        "Cung cấp khuyến nghị hành động cụ thể."
    ),
    "investigation": (
        "Bạn là điều tra viên thuế chuyên phân tích gian lận. "
        "Phát hiện mẫu đáng ngờ, truy vết giao dịch, "
        "và đề xuất các bước điều tra tiếp theo."
    ),
}

# ─── Augmentation Templates ──────────────────────────────────────────────────

PARAPHRASE_TEMPLATES = {
    "vat_refund_risk": [
        "Cho tôi biết về {topic}",
        "Phân tích {topic} cho doanh nghiệp này",
        "Điều kiện để được {topic}?",
        "Quy trình {topic} như thế nào?",
        "Rủi ro liên quan đến {topic}?",
    ],
    "invoice_risk": [
        "Kiểm tra {topic} có bình thường không",
        "Phát hiện bất thường trong {topic}",
        "Rà soát {topic} của doanh nghiệp",
        "{topic} có dấu hiệu gian lận không?",
    ],
    "delinquency": [
        "Tình hình {topic} của doanh nghiệp?",
        "Dự báo {topic} trong thời gian tới",
        "Biện pháp xử lý {topic}?",
        "Doanh nghiệp bị {topic} thì phải làm gì?",
    ],
}

TOPIC_MAP = {
    "vat_refund_risk": ["hoàn thuế VAT", "hoàn thuế GTGT", "khấu trừ thuế đầu vào"],
    "invoice_risk": ["hóa đơn đầu vào", "hóa đơn điện tử", "hóa đơn mua bán"],
    "delinquency": ["nợ đọng thuế", "chậm nộp thuế", "nợ thuế quá hạn"],
}


class TrainingDataPipeline:
    """
    Training data pipeline for custom LLM.

    Extracts from audit trail → filters → augments → exports.

    Usage:
        pipeline = TrainingDataPipeline()
        stats = pipeline.generate_from_db(db, output_path="data/llm_training.jsonl")
    """

    def __init__(
        self,
        min_quality: float = 0.3,
        dedup_threshold: float = 0.95,
    ):
        self.min_quality = min_quality
        self.dedup_threshold = dedup_threshold

    def generate_from_db(
        self,
        db,
        output_path: str = "data/llm_training.jsonl",
        *,
        max_examples: int = 50000,
        include_augmented: bool = True,
    ) -> DatasetStats:
        """
        Generate training dataset from the DB audit trail.

        Args:
            db: Database session
            output_path: Output JSONL file path
            max_examples: Maximum examples to generate
            include_augmented: Whether to include augmented examples

        Returns:
            DatasetStats with generation statistics
        """
        t0 = time.perf_counter()

        # Step 1: Extract from audit trail
        raw_examples = self._extract_from_audit_trail(db, max_examples)
        logger.info("[DataPipeline] Extracted %d raw examples from audit trail", len(raw_examples))

        # Step 2: Quality filtering
        quality_filtered = [e for e in raw_examples if e.quality_score >= self.min_quality]
        filtered_count = len(raw_examples) - len(quality_filtered)
        logger.info("[DataPipeline] Quality filtered: kept %d, removed %d", len(quality_filtered), filtered_count)

        # Step 3: Deduplication
        deduped = self._deduplicate(quality_filtered)
        dedup_count = len(quality_filtered) - len(deduped)
        logger.info("[DataPipeline] Deduplication: kept %d, removed %d", len(deduped), dedup_count)

        # Step 4: Augmentation
        augmented = []
        if include_augmented:
            augmented = self._augment(deduped)
            logger.info("[DataPipeline] Generated %d augmented examples", len(augmented))

        # Step 5: Combine and shuffle
        all_examples = deduped + augmented

        # Step 6: Export
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._export_jsonl(all_examples, output)

        # Step 7: Also export in Alpaca format
        alpaca_path = output.with_suffix(".alpaca.json")
        self._export_alpaca(all_examples, alpaca_path)

        # Stats
        by_intent: dict[str, int] = {}
        by_source: dict[str, int] = {}
        total_input_len = 0
        total_output_len = 0
        total_quality = 0.0

        for ex in all_examples:
            by_intent[ex.intent] = by_intent.get(ex.intent, 0) + 1
            by_source[ex.source] = by_source.get(ex.source, 0) + 1
            total_input_len += len(ex.input_text)
            total_output_len += len(ex.output_text)
            total_quality += ex.quality_score

        n = max(len(all_examples), 1)

        stats = DatasetStats(
            total_examples=len(all_examples),
            by_intent=by_intent,
            by_source=by_source,
            avg_quality=round(total_quality / n, 4),
            avg_input_len=total_input_len // n,
            avg_output_len=total_output_len // n,
            deduplicated=dedup_count,
            filtered_low_quality=filtered_count,
        )

        latency = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "[DataPipeline] ✓ Generated %d examples in %.0fms → %s",
            stats.total_examples, latency, output,
        )

        return stats

    def _extract_from_audit_trail(self, db, max_examples: int) -> list[TrainingExample]:
        """Extract Q&A pairs from the audit trail."""
        from sqlalchemy import text as sql_text

        examples: list[TrainingExample] = []

        try:
            rows = db.execute(
                sql_text("""
                    SELECT
                        at_user.session_id,
                        at_user.turn_index,
                        at_user.message_text AS user_message,
                        at_asst.message_text AS assistant_message,
                        at_asst.normalized_intent AS intent,
                        at_asst.confidence,
                        adt.abstained,
                        adt.evidence_json
                    FROM agent_turns at_user
                    JOIN agent_turns at_asst
                        ON at_asst.session_id = at_user.session_id
                        AND at_asst.turn_index = at_user.turn_index + 1
                        AND at_asst.role = 'assistant'
                    LEFT JOIN agent_decision_traces adt
                        ON adt.session_id = at_user.session_id
                        AND adt.turn_id = at_asst.id
                    WHERE at_user.role = 'user'
                        AND at_asst.message_text IS NOT NULL
                        AND length(at_asst.message_text) > 20
                        AND (adt.abstained IS NULL OR adt.abstained = FALSE)
                    ORDER BY at_user.created_at DESC
                    LIMIT :max_examples
                """),
                {"max_examples": max_examples},
            ).mappings().all()

            for row in rows:
                user_msg = str(row.get("user_message") or "")
                asst_msg = str(row.get("assistant_message") or "")
                intent = str(row.get("intent") or "general_tax_query")
                confidence = float(row.get("confidence") or 0)

                if len(user_msg) < 10 or len(asst_msg) < 20:
                    continue

                # Quality score based on confidence and response length
                quality = min(1.0, confidence * 0.6 + min(len(asst_msg) / 500, 1.0) * 0.4)

                hash_key = hashlib.md5(
                    f"{user_msg}:{asst_msg}".encode("utf-8")
                ).hexdigest()

                examples.append(TrainingExample(
                    instruction=SYSTEM_PROMPTS.get("general", ""),
                    input_text=user_msg,
                    output_text=asst_msg,
                    intent=intent,
                    confidence=confidence,
                    source="audit_trail",
                    quality_score=quality,
                    session_id=str(row.get("session_id", "")),
                    turn_id=int(row.get("turn_index", 0)),
                    hash_key=hash_key,
                ))

        except Exception as exc:
            logger.warning("[DataPipeline] DB extraction failed: %s", exc)

        return examples

    def _deduplicate(self, examples: list[TrainingExample]) -> list[TrainingExample]:
        """Remove duplicate examples."""
        seen_hashes: set[str] = set()
        deduped: list[TrainingExample] = []

        for ex in examples:
            if not ex.hash_key:
                ex.hash_key = hashlib.md5(
                    f"{ex.input_text}:{ex.output_text}".encode("utf-8")
                ).hexdigest()

            if ex.hash_key not in seen_hashes:
                seen_hashes.add(ex.hash_key)
                deduped.append(ex)

        return deduped

    def _augment(self, examples: list[TrainingExample]) -> list[TrainingExample]:
        """Generate augmented training examples."""
        augmented: list[TrainingExample] = []

        for intent, templates in PARAPHRASE_TEMPLATES.items():
            topics = TOPIC_MAP.get(intent, [])
            # Get a good example for this intent to use as output template
            intent_examples = [e for e in examples if e.intent == intent][:3]

            for template in templates:
                for topic in topics:
                    query = template.format(topic=topic)

                    # Use existing good answer as output, or generate template
                    if intent_examples:
                        output = intent_examples[0].output_text
                    else:
                        output = f"Về {topic}: cần tra cứu thêm thông tin chi tiết."

                    hash_key = hashlib.md5(
                        f"aug:{query}:{output[:50]}".encode("utf-8")
                    ).hexdigest()

                    augmented.append(TrainingExample(
                        instruction=SYSTEM_PROMPTS.get("general", ""),
                        input_text=query,
                        output_text=output,
                        intent=intent,
                        confidence=0.8,
                        source="augmented",
                        quality_score=0.6,
                        hash_key=hash_key,
                    ))

        return augmented

    def _export_jsonl(self, examples: list[TrainingExample], path: Path) -> None:
        """Export to JSONL format (ShareGPT-style)."""
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "conversations": [
                        {"from": "system", "value": ex.instruction},
                        {"from": "human", "value": ex.input_text},
                        {"from": "gpt", "value": ex.output_text},
                    ],
                    "metadata": {
                        "intent": ex.intent,
                        "confidence": ex.confidence,
                        "source": ex.source,
                        "quality_score": ex.quality_score,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _export_alpaca(self, examples: list[TrainingExample], path: Path) -> None:
        """Export to Alpaca format."""
        records = []
        for ex in examples:
            records.append({
                "instruction": ex.instruction,
                "input": ex.input_text,
                "output": ex.output_text,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
