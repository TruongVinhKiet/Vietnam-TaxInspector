"""
tax_agent_enhanced_intent.py – Enhanced Intent Classification (Phase 4)
========================================================================
Upgrades intent classification from TF-IDF+LogReg to semantic embeddings.

Capabilities:
    1. Semantic embedding-based intent classification (e5-small)
    2. Multi-intent detection (e.g., "VAT refund + delinquency check")
    3. Named entity extraction (MST, tax_period, amounts, company names)
    4. Intent confidence calibration
    5. Context-aware re-ranking (using conversation history)
    6. Zero-shot intent detection for unseen intents

Architecture:
    Tier 1: Semantic similarity + calibrated thresholds
    Tier 2: TF-IDF+LogReg (trained, offline fallback)
    Tier 3: Keyword rules (zero-dependency fallback)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntentCandidate:
    """A candidate intent with score."""
    intent: str
    score: float
    source: str              # "semantic", "model", "keyword"
    description: str = ""


@dataclass
class MultiIntentResult:
    """Result of multi-intent classification."""
    primary_intent: str
    primary_confidence: float
    # Multi-intent support
    secondary_intents: list[IntentCandidate]
    is_multi_intent: bool
    # Entity extraction
    extracted_entities: list[dict[str, Any]]
    # Metadata
    classification_source: str   # "semantic", "model", "keyword"
    all_scores: dict[str, float]
    latency_ms: float = 0.0


# ─── Intent Definitions (Vietnamese tax domain) ──────────────────────────────

INTENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "vat_refund_risk": {
        "description": "Hoàn thuế GTGT, rủi ro hoàn thuế VAT, xin hoàn thuế",
        "exemplars": [
            "điều kiện hoàn thuế VAT cho doanh nghiệp xuất khẩu",
            "hồ sơ xin hoàn thuế GTGT cần những gì",
            "kiểm tra rủi ro hoàn thuế cho công ty này",
            "đánh giá rủi ro gian lận hoàn thuế VAT",
            "doanh nghiệp nào đủ điều kiện hoàn thuế GTGT",
        ],
        "keywords": [
            "hoàn thuế", "vat", "hồ sơ hoàn", "refund", "đề nghị hoàn", "gtgt", "xuất khẩu",
            "hoan thue", "thuế gtgt", "thuế giá trị gia tăng", "giảm thuế",
            "thuế suất 8", "thuế suất 10", "khấu trừ", "đầu vào", "đầu ra",
            "nghị định 72", "nd 72", "72/2024", "thuế suất", "phương pháp khấu trừ",
            "tỷ lệ % trên doanh thu"
        ],
    },
    "invoice_risk": {
        "description": "Rủi ro hóa đơn, hóa đơn giả, hóa đơn bất hợp pháp",
        "exemplars": [
            "kiểm tra hóa đơn đầu vào có hợp lệ không",
            "phát hiện hóa đơn giả trong hồ sơ kê khai",
            "rà soát hóa đơn điện tử của doanh nghiệp",
            "doanh nghiệp sử dụng hóa đơn bất hợp pháp",
            "tỷ lệ hóa đơn rủi ro cao bất thường",
        ],
        "keywords": [
            "hóa đơn", "invoice", "xuất hóa đơn", "mua vào", "bán ra", "HĐĐT",
            "hoa don", "hóa đơn điện tử", "máy tính tiền",
            "hóa đơn không hợp pháp", "hóa đơn bất hợp pháp", "hóa đơn giả",
            "thông tư 78", "nghị định 123"
        ],
    },
    "delinquency": {
        "description": "Nợ đọng thuế, chậm nộp, quá hạn, cưỡng chế",
        "exemplars": [
            "doanh nghiệp này có nợ thuế quá hạn không",
            "dự báo khả năng chậm nộp thuế trong 90 ngày",
            "danh sách doanh nghiệp nợ đọng thuế",
            "biện pháp cưỡng chế nợ thuế áp dụng khi nào",
            "tình hình tuân thủ thời hạn nộp thuế",
        ],
        "keywords": [
            "nợ đọng", "chậm nộp", "delinquency", "quá hạn", "thu nợ", "cưỡng chế",
            "no dong", "cham nop", "qua han", "thu no",
            "tiền chậm nộp", "tiền phạt", "nhắc nợ"
        ],
    },
    "osint_ownership": {
        "description": "Sở hữu, UBO, offshore, công ty mẹ, liên kết",
        "exemplars": [
            "ai là chủ sở hữu thực sự của doanh nghiệp này",
            "phát hiện cấu trúc offshore trong chuỗi sở hữu",
            "truy vết UBO qua các lớp công ty",
            "doanh nghiệp có liên kết với công ty nước ngoài",
            "phân tích mối quan hệ giữa các công ty trong nhóm",
        ],
        "keywords": [
            "offshore", "sở hữu", "ubo", "phoenix", "công ty mẹ", "liên kết", "cổ đông",
            "so huu", "cong ty me", "cấu trúc sở hữu", "người hưởng lợi",
            "pháp nhân nước ngoài", "singapore", "bvi", "cayman"
        ],
    },
    "transfer_pricing": {
        "description": "Chuyển giá, giao dịch liên kết, giá thị trường",
        "exemplars": [
            "kiểm tra giao dịch liên kết của doanh nghiệp FDI",
            "phân tích rủi ro chuyển giá trong chuỗi giao dịch",
            "giá chuyển giao có phù hợp với giá thị trường không",
            "tỷ suất lợi nhuận thấp bất thường so với ngành",
            "doanh nghiệp FDI lỗ liên tục nhiều năm",
        ],
        "keywords": [
            "chuyển giá", "transfer pricing", "giao dịch liên kết", "GDLK", "FDI",
            "chuyen gia", "gia giao dich lien ket", "mispricing",
            "giá giao dịch liên kết", "nghị định 132", "bên liên kết", "arm's length"
        ],
    },
    "audit_selection": {
        "description": "Lựa chọn thanh tra, kiểm tra thuế, rủi ro thanh tra",
        "exemplars": [
            "doanh nghiệp nào cần ưu tiên thanh tra",
            "xếp hạng rủi ro để chọn hồ sơ kiểm tra",
            "tiêu chí lựa chọn đối tượng thanh tra thuế",
            "đánh giá rủi ro thanh tra cho kỳ thuế này",
            "danh sách doanh nghiệp cần kiểm tra thuế",
        ],
        "keywords": [
            "thanh tra", "audit", "kiểm tra", "xếp hạng hồ sơ", "lựa chọn",
            "kiem tra", "xep hang ho so", "chọn thanh tra"
        ],
    },
    "top_n_query": {
        "description": "Truy vấn top N doanh nghiệp theo tiêu chí rủi ro",
        "exemplars": [
            "cho tôi 10 doanh nghiệp rủi ro cao nhất",
            "top 5 công ty có điểm gian lận cao nhất",
            "danh sách 20 doanh nghiệp nợ thuế nhiều nhất",
            "liệt kê 15 doanh nghiệp cần thanh tra gấp",
            "xếp hạng doanh nghiệp theo mức rủi ro",
            "những doanh nghiệp nào có rủi ro cao nhất",
        ],
        "keywords": ["top", "danh sách", "liệt kê", "cao nhất", "nhiều nhất", "xếp hạng", "bao nhiêu doanh nghiệp"],
    },
    "company_name_lookup": {
        "description": "Tra cứu doanh nghiệp theo tên công ty",
        "exemplars": [
            "phân tích công ty TNHH ABC",
            "thông tin doanh nghiệp Vinamilk",
            "kiểm tra công ty cổ phần XYZ",
            "tìm doanh nghiệp tên là ABC",
            "cho tôi thông tin về công ty Hòa Phát",
        ],
        "keywords": ["công ty", "doanh nghiệp", "CT TNHH", "CT CP", "tìm", "tên là"],
    },
    "batch_analysis": {
        "description": "Phân tích lô dữ liệu từ file CSV hoặc danh sách",
        "exemplars": [
            "phân tích rủi ro danh sách tôi gửi",
            "chấm điểm gian lận file này",
            "đánh giá rủi ro cho file CSV",
            "phân tích lô doanh nghiệp từ file",
            "xử lý file dữ liệu tôi upload",
        ],
        "keywords": ["file", "csv", "danh sách", "lô", "batch", "upload", "gửi file"],
    },
    "general_tax_query": {
        "description": "Câu hỏi chung về thuế, quy định, thủ tục",
        "exemplars": [
            "thuế suất thuế TNDN hiện hành là bao nhiêu",
            "thủ tục đăng ký mã số thuế mới",
            "thời hạn nộp tờ khai thuế quý",
            "quy định về khấu trừ thuế GTGT đầu vào",
            "hướng dẫn kê khai thuế trực tuyến",
        ],
        "keywords": [
            "thuế suất", "thủ tục", "đăng ký", "kê khai", "quy định", "hướng dẫn",
            "thuế tndn", "thuế thu nhập doanh nghiệp", "ưu đãi thuế",
            "miễn thuế", "đầu tư mở rộng", "dự án đầu tư",
            "khu công nghiệp", "luật thuế", "chính sách thuế",
            "quản lý thuế", "công văn",
            # PIT / Thuế TNCN
            "thuế tncn", "thuế thu nhập cá nhân", "lương", "tiền lương", "tiền công",
            "giảm trừ gia cảnh", "người phụ thuộc", "biểu thuế lũy tiến",
            "khấu trừ thuế tncn", "quyết toán thuế", "hoàn thuế tncn",
            "bậc thuế", "thuế lũy tiến",
            # Hộ kinh doanh / Cá nhân
            "hộ kinh doanh", "cá nhân kinh doanh", "100 triệu", "doanh thu",
            "tạp hóa", "tiệm", "quán", "cửa hàng", "freelance", "tự do",
            "bán hàng online", "shopee", "lazada", "tiktok", "facebook",
            # Cho thuê tài sản
            "cho thuê nhà", "cho thuê", "thuê tài sản", "thu nhập cho thuê",
            # Chi phí được trừ TNDN
            "chi phí được trừ", "chi phí hợp lý", "chi phí không được trừ",
            "tiếp khách", "chứng từ thanh toán", "tiền mặt", "20 triệu",
            # Hóa đơn & Xử phạt
            "hóa đơn điện tử", "xuất hóa đơn", "quên xuất hóa đơn",
            "phạt", "xử phạt", "vi phạm", "nghị định 125", "mức phạt",
            "chậm nộp tờ khai", "nộp trễ", "trễ hạn",
            # Thuế môn bài
            "thuế môn bài", "lệ phí môn bài",
        ],
    },
}


class EnhancedIntentClassifier:
    """
    Multi-tier intent classifier with semantic understanding.

    Tier 1: Semantic Similarity
        - Encode query + intent exemplars using e5-small
        - Compute cosine similarity → pick best intent
        - Multi-intent: if 2nd intent score > 0.7 * 1st intent score

    Tier 2: Trained Model (TF-IDF + LogReg)
        - Offline-trained scikit-learn model
        - Fallback when embeddings unavailable

    Tier 3: Keyword Rules
        - Zero-dependency fallback

    Usage:
        classifier = EnhancedIntentClassifier()
        classifier.load()
        result = classifier.classify("hoàn thuế VAT điều kiện gì?")
    """

    def __init__(self):
        self._embedding_engine = None
        self._intent_embeddings: dict[str, np.ndarray] = {}
        self._trained_model = None
        self._loaded = False
        self._tier: str = "keyword"

    def load(self) -> str:
        """Load the best available classification model."""
        # Try semantic tier first
        tier = self._try_load_semantic()
        if tier:
            return tier

        # Try trained model
        tier = self._try_load_trained()
        if tier:
            return tier

        # Keyword fallback
        self._tier = "keyword"
        self._loaded = True
        return "keyword"

    def _try_load_semantic(self) -> Optional[str]:
        """Load semantic embeddings for intent classification."""
        try:
            from ml_engine.tax_agent_embeddings import get_embedding_engine

            self._embedding_engine = get_embedding_engine()
            if not self._embedding_engine.is_semantic:
                logger.info("[IntentClassifier] Embedding engine not semantic, skipping")
                return None

            # Pre-compute intent exemplar embeddings
            for intent_key, intent_def in INTENT_DEFINITIONS.items():
                exemplars = intent_def.get("exemplars", [])
                if not exemplars:
                    continue

                # Embed all exemplars and average
                batch = self._embedding_engine.embed_passages_batch(exemplars)
                vecs = np.stack([e.vector for e in batch.embeddings])
                centroid = vecs.mean(axis=0)
                # Normalize
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    centroid = centroid / norm
                self._intent_embeddings[intent_key] = centroid

            self._tier = "semantic"
            self._loaded = True
            logger.info(
                "[IntentClassifier] ✓ Semantic tier loaded (%d intents)",
                len(self._intent_embeddings),
            )
            return "semantic"

        except Exception as exc:
            logger.warning("[IntentClassifier] Semantic load failed: %s", exc)
            return None

    def _try_load_trained(self) -> Optional[str]:
        """Load trained TF-IDF + LogReg model."""
        try:
            from pathlib import Path
            from ml_engine.tax_agent_intent_model import TaxAgentIntentModel

            model_dir = Path(__file__).resolve().parent.parent / "data" / "models"
            model = TaxAgentIntentModel(model_dir)
            if model.load():
                self._trained_model = model
                self._tier = "model"
                self._loaded = True
                logger.info("[IntentClassifier] ✓ Trained model loaded")
                return "model"
        except Exception as exc:
            logger.warning("[IntentClassifier] Trained model load failed: %s", exc)
        return None

    def classify(
        self,
        query: str,
        *,
        context_intents: list[str] | None = None,
        multi_intent_threshold: float = 0.70,
    ) -> MultiIntentResult:
        """
        Classify query intent with multi-intent support.

        Args:
            query: User query text
            context_intents: Previous intents (for context-aware ranking)
            multi_intent_threshold: Ratio threshold for secondary intent

        Returns:
            MultiIntentResult with primary + secondary intents
        """
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        
        # Always run keyword extraction first as a strong prior
        kw_result = self._classify_keyword(query)
        kw_primary = kw_result.primary_intent
        kw_conf = kw_result.primary_confidence
        
        if self._tier == "semantic":
            result = self._classify_semantic(query, context_intents, multi_intent_threshold)
            # Override semantic if keyword has a strong hit (coverage > 0)
            # Since kw_conf starts at 0.22, anything above 0.25 means actual hits
            if kw_conf > 0.35:
                result.primary_intent = kw_primary
                result.primary_confidence = kw_conf
                result.classification_source = "keyword_override"
        elif self._tier == "model":
            result = self._classify_trained(query)
            if kw_conf > 0.35:
                result.primary_intent = kw_primary
                result.primary_confidence = kw_conf
                result.classification_source = "keyword_override"
        else:
            result = kw_result

        result.latency_ms = (time.perf_counter() - t0) * 1000.0

        # Always extract entities regardless of tier
        result.extracted_entities = self._extract_entities(query)

        return result

    def _classify_semantic(
        self,
        query: str,
        context_intents: list[str] | None,
        multi_intent_threshold: float,
    ) -> MultiIntentResult:
        """Classify using semantic similarity."""
        query_emb = self._embedding_engine.embed_query(query)

        scores: dict[str, float] = {}
        for intent_key, centroid in self._intent_embeddings.items():
            sim = self._embedding_engine.cosine_similarity(query_emb.vector, centroid)
            scores[intent_key] = float(sim)

        # Context-aware boosting: slightly boost intents seen in context
        if context_intents:
            for ci in context_intents:
                if ci in scores:
                    scores[ci] *= 1.05  # 5% boost

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary_intent = ranked[0][0]
        primary_score = ranked[0][1]

        # Multi-intent detection
        secondary_intents: list[IntentCandidate] = []
        is_multi = False
        for intent, score in ranked[1:3]:
            if score >= primary_score * multi_intent_threshold and score > 0.3:
                secondary_intents.append(IntentCandidate(
                    intent=intent,
                    score=round(score, 4),
                    source="semantic",
                    description=INTENT_DEFINITIONS.get(intent, {}).get("description", ""),
                ))
                is_multi = True

        # Calibrate confidence (semantic similarity range is usually 0.3-0.9)
        calibrated_conf = self._calibrate_confidence(primary_score, secondary_intents)

        return MultiIntentResult(
            primary_intent=primary_intent,
            primary_confidence=round(calibrated_conf, 4),
            secondary_intents=secondary_intents,
            is_multi_intent=is_multi,
            extracted_entities=[],
            classification_source="semantic",
            all_scores={k: round(v, 4) for k, v in scores.items()},
        )

    def _classify_trained(self, query: str) -> MultiIntentResult:
        """Classify using trained model."""
        intent, conf, meta = self._trained_model.predict(query)
        conf = min(0.95, max(0.15, float(conf)))

        return MultiIntentResult(
            primary_intent=intent,
            primary_confidence=round(conf, 4),
            secondary_intents=[],
            is_multi_intent=False,
            extracted_entities=[],
            classification_source="model",
            all_scores={intent: conf},
        )

    def _classify_keyword(self, query: str) -> MultiIntentResult:
        """Classify using keyword matching."""
        query_lower = query.lower()
        scores: dict[str, float] = {}

        for intent_key, intent_def in INTENT_DEFINITIONS.items():
            keywords = intent_def.get("keywords", [])
            hit_count = sum(1 for kw in keywords if kw.lower() in query_lower)
            # Use hit_count directly rather than coverage, 
            # since one strong keyword is often a solid indicator.
            score = min(1.0, hit_count * 0.4)
            scores[intent_key] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = ranked[0] if ranked else ("general_tax_query", 0.0)
        conf = min(0.95, 0.25 + primary[1] * 0.7)

        # Only suppress general_tax_query if there were NO actual keyword hits.
        if primary[1] <= 0.0:
            primary = ("general_tax_query", 0.0)
            conf = 0.22

        return MultiIntentResult(
            primary_intent=primary[0],
            primary_confidence=round(conf, 4),
            secondary_intents=[],
            is_multi_intent=False,
            extracted_entities=[],
            classification_source="keyword",
            all_scores={k: round(v, 4) for k, v in scores.items()},
        )

    def _calibrate_confidence(
        self,
        raw_score: float,
        secondary: list[IntentCandidate],
    ) -> float:
        """
        Calibrate semantic similarity score to confidence.
        Raw similarity is typically in range [0.3, 0.9].
        Map to [0.15, 0.95] with adjustments.
        """
        # Linear mapping from [0.3, 0.9] to [0.15, 0.95]
        calibrated = (raw_score - 0.3) / 0.6 * 0.8 + 0.15
        calibrated = max(0.15, min(0.95, calibrated))

        # Reduce confidence if multi-intent (ambiguity)
        if secondary:
            calibrated *= 0.9

        return calibrated

    def _extract_entities(self, query: str) -> list[dict[str, Any]]:
        """
        Extract tax-domain entities from query.
        Enhanced version with more entity types.
        """
        entities: list[dict[str, Any]] = []

        # MST (tax code): 10 or 13 digits
        for match in re.finditer(r"\b(\d{10}(?:-\d{3})?)\b", query):
            val = match.group(1)
            if len(val.replace("-", "")) in (10, 13):
                entities.append({
                    "type": "tax_code",
                    "value": val,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Tax period: quý/tháng/năm patterns
        period_patterns = [
            (r"(?:quý|quy)\s*(\d)\s*/?\s*(\d{4})", "quarter"),
            (r"(?:tháng|thang)\s*(\d{1,2})\s*/?\s*(\d{4})", "month"),
            (r"(?:năm|nam)\s*(\d{4})", "year"),
            (r"(\d{4})[/\-]Q([1-4])", "quarter"),
        ]
        for pattern, period_type in period_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append({
                    "type": "tax_period",
                    "value": match.group(0).strip(),
                    "period_type": period_type,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Company name patterns (common Vietnamese company prefixes)
        company_patterns = [
            r"(?:công ty|CT)\s+(?:TNHH|CP|cổ phần|trách nhiệm)\s+([A-ZĐÀ-Ỹa-zà-ỹ\s]{3,40})",
            r"(?:DN|doanh nghiệp)\s+([A-ZĐÀ-Ỹa-zà-ỹ\s]{3,40})",
            r"(?:công ty|CT)\s+([A-ZĐÀ-Ỹ][A-ZĐÀ-Ỹa-zà-ỹ\s]{2,40})",
            r"(?:về|của|tên là|tên)\s+([A-ZĐÀ-Ỹ][A-ZĐÀ-Ỹa-zà-ỹ\s]{2,30}?)(?:\s*$|\s*\?)",
        ]
        for pattern in company_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                name_val = (match.group(1).strip() if match.lastindex else match.group(0).strip())
                # Filter out common false positives
                if name_val.lower() not in ("thuế", "rủi ro", "gian lận", "nợ", "phân tích"):
                    entities.append({
                        "type": "company_name",
                        "value": name_val,
                        "start": match.start(),
                        "end": match.end(),
                    })
                    break  # take first match only

        # Quantity extraction: "top 10", "cho tôi 5", "liệt kê 20"
        quantity_patterns = [
            r"(?:top|cho tôi|liệt kê|lấy|xem)\s+(\d{1,3})\b",
            r"(\d{1,3})\s+(?:doanh nghiệp|công ty|DN|CT|hồ sơ)",
        ]
        for pattern in quantity_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                val = int(match.group(1))
                if 1 <= val <= 200:
                    entities.append({
                        "type": "quantity",
                        "value": str(val),
                        "start": match.start(),
                        "end": match.end(),
                    })
                    break
            else:
                continue
            break

        # Legal document references
        doc_patterns = [
            r"(?:Luật|Nghị định|Thông tư|Quyết định)\s+(?:số\s+)?(\d+[/\-]\d+[/\-]?[\w\-]*)",
        ]
        for pattern in doc_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append({
                    "type": "legal_document",
                    "value": match.group(0).strip(),
                    "start": match.start(),
                    "end": match.end(),
                })

        return entities

    @property
    def tier(self) -> str:
        return self._tier


# ─── Singleton ────────────────────────────────────────────────────────────────

_classifier_instance: EnhancedIntentClassifier | None = None


def get_intent_classifier() -> EnhancedIntentClassifier:
    """Get or create the singleton intent classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EnhancedIntentClassifier()
        _classifier_instance.load()
    return _classifier_instance
