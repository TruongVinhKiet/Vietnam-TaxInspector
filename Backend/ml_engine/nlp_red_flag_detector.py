"""
nlp_red_flag_detector.py – NLP Red Flag Detector cho Hóa Đơn
===============================================================
Text classification trên mô tả hóa đơn để phát hiện dấu hiệu
bất thường: ngành nghề không khớp, mô tả đáng ngờ, sentiment
tiêu cực trong ghi chú thanh tra.

Capabilities:
    1. Invoice description text classifier (đáng ngờ / bình thường)
    2. Industry-transaction mismatch detection
    3. Inspector note sentiment analysis
    4. Keyword-based và semantic-based dual approach

Data Sources:
    - invoice_line_items.item_description
    - companies.industry, companies.business_type
    - inspector notes (agent_turns, collection_actions.notes)

Design:
    - Dual approach: keyword heuristic (fast) + semantic embedding (accurate)
    - CPU-first: sentence-transformers + sklearn
    - Lazy model loading
    - Thread-safe inference
    - Vietnamese text normalization

Reference:
    - Vietnamese NLP: underthesea / PhoBERT patterns
    - Text classification: TF-IDF + LogReg → Embedding + NN
"""

from __future__ import annotations

import logging
import math
import re
import time
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class RedFlagResult:
    """Kết quả phân tích red flag cho một invoice/document."""
    entity_id: str               # invoice_number hoặc tax_code
    risk_score: float = 0.0      # 0-1
    risk_level: str = "low"      # low | medium | high | critical
    flags: list[dict[str, Any]] = field(default_factory=list)
    method: str = ""             # keyword | semantic | ensemble
    confidence: float = 0.0
    processing_ms: float = 0.0


@dataclass
class IndustryMismatchResult:
    """Kết quả phát hiện mismatch ngành nghề."""
    tax_code: str
    declared_industry: str = ""
    transaction_categories: list[str] = field(default_factory=list)
    mismatch_score: float = 0.0
    mismatched_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SentimentResult:
    """Kết quả sentiment analysis cho ghi chú thanh tra."""
    text: str
    sentiment: str = "neutral"    # positive | neutral | negative | alarming
    score: float = 0.0
    keywords_found: list[str] = field(default_factory=list)


@dataclass
class RedFlagConfig:
    """Cấu hình cho Red Flag Detector."""
    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    risk_threshold_medium: float = 0.3
    risk_threshold_high: float = 0.6
    risk_threshold_critical: float = 0.8
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_items_per_invoice: int = 50
    model_dir: str = str(MODEL_DIR)


# ════════════════════════════════════════════════════════════════
#  1. Vietnamese Text Normalizer
# ════════════════════════════════════════════════════════════════

class VietnameseTextNormalizer:
    """
    Chuẩn hóa text tiếng Việt cho NLP processing.

    Xử lý: lowercase, bỏ dấu câu thừa, chuẩn hóa khoảng trắng,
    abbreviation expansion, number normalization.
    """

    # Các abbreviation thường gặp trong hóa đơn
    ABBREVIATIONS = {
        "cty": "công ty",
        "dn": "doanh nghiệp",
        "hđ": "hóa đơn",
        "dvt": "đơn vị tính",
        "sl": "số lượng",
        "đg": "đơn giá",
        "tt": "thành tiền",
        "vnd": "việt nam đồng",
        "gtgt": "giá trị gia tăng",
        "tncn": "thu nhập cá nhân",
        "tndn": "thu nhập doanh nghiệp",
        "mst": "mã số thuế",
        "stk": "số tài khoản",
        "xk": "xuất khẩu",
        "nk": "nhập khẩu",
    }

    def normalize(self, text: str) -> str:
        """Chuẩn hóa text."""
        if not text:
            return ""

        text = text.lower().strip()
        # Bỏ ký tự đặc biệt thừa
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ.,;:!?/\-]', ' ', text)
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        # Expand abbreviations
        words = text.split()
        expanded = [self.ABBREVIATIONS.get(w, w) for w in words]
        return " ".join(expanded)


# ════════════════════════════════════════════════════════════════
#  2. Keyword-based Red Flag Detector
# ════════════════════════════════════════════════════════════════

class KeywordRedFlagDetector:
    """
    Phát hiện red flags dựa trên từ khóa và patterns.

    Nhanh, không cần model, phù hợp làm first-pass filter.
    """

    # Danh mục từ khóa đáng ngờ trong mô tả hóa đơn
    SUSPICIOUS_KEYWORDS = {
        # Dịch vụ mơ hồ (thường dùng trong hóa đơn ảo)
        "tu_van": {
            "keywords": ["tư vấn", "tu van", "consulting", "dịch vụ tư vấn",
                         "phí tư vấn", "phi tu van"],
            "score": 0.3,
            "description": "Dịch vụ tư vấn mơ hồ — thường dùng trong hóa đơn khống",
        },
        "dich_vu_chung": {
            "keywords": ["dịch vụ", "phí dịch vụ", "dich vu", "service fee",
                         "phí quản lý", "chi phí khác", "chi phi khac"],
            "score": 0.25,
            "description": "Mô tả dịch vụ chung chung",
        },
        # Giao dịch tròn số (round-trip)
        "tron_so": {
            "keywords": [],  # Detected by amount pattern
            "score": 0.2,
            "description": "Giá trị tròn số bất thường",
        },
        # Hàng hóa không rõ ràng
        "hang_hoa_chung": {
            "keywords": ["hàng hóa", "hang hoa", "vật tư", "vat tu",
                         "nguyên vật liệu", "nguyen vat lieu", "phụ kiện",
                         "linh kiện", "linh kien"],
            "score": 0.15,
            "description": "Mô tả hàng hóa không cụ thể",
        },
        # Giả mạo / ảo
        "gia_mao": {
            "keywords": ["hoa don gia", "hóa đơn giả", "khống", "ảo",
                         "fake", "fictitious"],
            "score": 0.9,
            "description": "Từ khóa liên quan trực tiếp đến gian lận",
        },
        # Chuyển giá
        "chuyen_gia": {
            "keywords": ["chuyển giá", "chuyen gia", "transfer pricing",
                         "intercompany", "liên kết", "lien ket",
                         "bên liên quan", "related party"],
            "score": 0.4,
            "description": "Dấu hiệu chuyển giá / giao dịch liên kết",
        },
    }

    # Từ khóa sentiment tiêu cực trong ghi chú thanh tra
    NEGATIVE_INSPECTOR_KEYWORDS = {
        "high_concern": [
            "nghi vấn", "nghi van", "gian lận", "gian lan",
            "trốn thuế", "tron thue", "vi phạm", "vi pham",
            "bất thường", "bat thuong", "đáng ngờ", "dang ngo",
            "không hợp lệ", "khong hop le", "từ chối", "tu choi",
        ],
        "medium_concern": [
            "cần kiểm tra", "can kiem tra", "chưa rõ", "chua ro",
            "thiếu chứng từ", "thieu chung tu", "sai sót", "sai sot",
            "không khớp", "khong khop", "chênh lệch", "chenh lech",
        ],
        "low_concern": [
            "lưu ý", "luu y", "theo dõi", "theo doi",
            "cần bổ sung", "can bo sung",
        ],
    }

    def __init__(self):
        self._normalizer = VietnameseTextNormalizer()

    def detect(
        self,
        descriptions: list[str],
        invoice_id: str = "",
    ) -> RedFlagResult:
        """
        Phát hiện red flags từ danh sách mô tả hóa đơn.

        Args:
            descriptions: List mô tả line items
            invoice_id: ID hóa đơn

        Returns:
            RedFlagResult.
        """
        t0 = time.perf_counter()
        flags: list[dict] = []
        total_score = 0.0

        for desc in descriptions:
            normalized = self._normalizer.normalize(desc)
            for flag_key, flag_info in self.SUSPICIOUS_KEYWORDS.items():
                for kw in flag_info["keywords"]:
                    if kw in normalized:
                        flags.append({
                            "type": flag_key,
                            "keyword": kw,
                            "text_snippet": desc[:100],
                            "score": flag_info["score"],
                            "description": flag_info["description"],
                        })
                        total_score = max(total_score, flag_info["score"])
                        break  # Chỉ cần match 1 keyword/category

        # Risk level
        risk_level = "low"
        if total_score >= 0.8:
            risk_level = "critical"
        elif total_score >= 0.5:
            risk_level = "high"
        elif total_score >= 0.25:
            risk_level = "medium"

        elapsed = (time.perf_counter() - t0) * 1000

        return RedFlagResult(
            entity_id=invoice_id,
            risk_score=round(total_score, 4),
            risk_level=risk_level,
            flags=flags,
            method="keyword",
            confidence=0.7 if flags else 0.9,  # Keyword ít false positive
            processing_ms=round(elapsed, 1),
        )

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Phân tích sentiment ghi chú thanh tra."""
        normalized = self._normalizer.normalize(text)
        keywords_found = []

        # Check high concern first
        for kw in self.NEGATIVE_INSPECTOR_KEYWORDS["high_concern"]:
            if kw in normalized:
                keywords_found.append(kw)

        if keywords_found:
            return SentimentResult(
                text=text, sentiment="alarming",
                score=0.9, keywords_found=keywords_found,
            )

        for kw in self.NEGATIVE_INSPECTOR_KEYWORDS["medium_concern"]:
            if kw in normalized:
                keywords_found.append(kw)

        if keywords_found:
            return SentimentResult(
                text=text, sentiment="negative",
                score=0.6, keywords_found=keywords_found,
            )

        for kw in self.NEGATIVE_INSPECTOR_KEYWORDS["low_concern"]:
            if kw in normalized:
                keywords_found.append(kw)

        if keywords_found:
            return SentimentResult(
                text=text, sentiment="neutral",
                score=0.3, keywords_found=keywords_found,
            )

        return SentimentResult(text=text, sentiment="neutral", score=0.1)


# ════════════════════════════════════════════════════════════════
#  3. Semantic Red Flag Detector (Embedding-based)
# ════════════════════════════════════════════════════════════════

class SemanticRedFlagDetector:
    """
    Phát hiện red flags bằng semantic similarity.

    So sánh embedding của mô tả hóa đơn với các prototype
    patterns đáng ngờ/bình thường.
    """

    # Prototype descriptions cho từng category
    SUSPICIOUS_PROTOTYPES = [
        "phí tư vấn quản lý dự án",
        "dịch vụ tư vấn chiến lược kinh doanh",
        "chi phí thuê ngoài dịch vụ",
        "phí quản lý hành chính",
        "hoa hồng môi giới",
        "phí hỗ trợ kỹ thuật",
        "dịch vụ marketing tổng hợp",
    ]

    NORMAL_PROTOTYPES = [
        "xi măng portland PC40 loại 1",
        "thép xây dựng D12 SD295A",
        "gạch ceramic 600x600mm",
        "xăng RON 95 loại III",
        "máy tính laptop Dell Latitude 5520",
        "giấy A4 Double A 80gsm",
        "dầu nhớt Shell Rimula R4",
    ]

    def __init__(self, config: RedFlagConfig | None = None):
        self.config = config or RedFlagConfig()
        self._model = None
        self._suspicious_embeddings = None
        self._normal_embeddings = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Lazy load embedding model."""
        if self._loaded:
            return True

        with self._lock:
            if self._loaded:
                return True

            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.config.embedding_model,
                    device="cpu",
                )
                # Pre-compute prototype embeddings
                self._suspicious_embeddings = self._model.encode(
                    self.SUSPICIOUS_PROTOTYPES, normalize_embeddings=True
                )
                self._normal_embeddings = self._model.encode(
                    self.NORMAL_PROTOTYPES, normalize_embeddings=True
                )
                self._loaded = True
                logger.info("[SemanticRF] Embedding model loaded")
                return True
            except ImportError:
                logger.warning("[SemanticRF] sentence-transformers not installed")
                return False
            except Exception as exc:
                logger.warning("[SemanticRF] Load failed: %s", exc)
                return False

    def detect(
        self,
        descriptions: list[str],
        invoice_id: str = "",
    ) -> RedFlagResult:
        """
        Phát hiện red flags bằng semantic similarity.

        Args:
            descriptions: List mô tả line items
            invoice_id: ID hóa đơn

        Returns:
            RedFlagResult.
        """
        t0 = time.perf_counter()

        if not self.load() or not descriptions:
            return RedFlagResult(
                entity_id=invoice_id, method="semantic",
                confidence=0.0,
            )

        # Encode descriptions
        desc_embeddings = self._model.encode(
            descriptions[:self.config.max_items_per_invoice],
            normalize_embeddings=True,
        )

        flags = []
        max_score = 0.0

        for i, (desc, emb) in enumerate(zip(descriptions, desc_embeddings)):
            # Cosine similarity với suspicious prototypes
            sus_sim = float(np.max(emb @ self._suspicious_embeddings.T))
            norm_sim = float(np.max(emb @ self._normal_embeddings.T))

            # Score: suspicious similarity - normal similarity
            red_flag_score = max(0.0, (sus_sim - norm_sim + 0.3) / 1.3)

            if red_flag_score > 0.3:
                flags.append({
                    "type": "semantic_mismatch",
                    "text_snippet": desc[:100],
                    "suspicious_similarity": round(sus_sim, 4),
                    "normal_similarity": round(norm_sim, 4),
                    "score": round(red_flag_score, 4),
                    "description": "Mô tả gần với pattern đáng ngờ (semantic)",
                })
                max_score = max(max_score, red_flag_score)

        risk_level = "low"
        if max_score >= 0.8:
            risk_level = "critical"
        elif max_score >= 0.5:
            risk_level = "high"
        elif max_score >= 0.3:
            risk_level = "medium"

        elapsed = (time.perf_counter() - t0) * 1000

        return RedFlagResult(
            entity_id=invoice_id,
            risk_score=round(max_score, 4),
            risk_level=risk_level,
            flags=flags,
            method="semantic",
            confidence=0.85 if self._loaded else 0.0,
            processing_ms=round(elapsed, 1),
        )


# ════════════════════════════════════════════════════════════════
#  4. Industry-Transaction Mismatch Detector
# ════════════════════════════════════════════════════════════════

class IndustryMismatchDetector:
    """
    Phát hiện mismatch giữa ngành nghề đăng ký và giao dịch thực tế.

    Ví dụ: Công ty sản xuất bê tông mua "dịch vụ tư vấn IT" giá trị lớn.
    """

    # Mapping ngành nghề → từ khóa hàng hóa expected
    INDUSTRY_KEYWORDS = {
        "xây dựng": ["xi măng", "thép", "gạch", "cát", "đá", "bê tông",
                      "sắt", "nhôm", "kính", "sơn", "vữa"],
        "thực phẩm": ["thực phẩm", "đồ uống", "gạo", "thịt", "cá",
                       "rau", "trái cây", "đường", "sữa", "nước"],
        "công nghệ": ["phần mềm", "máy tính", "server", "hosting",
                       "cloud", "IT", "digital", "website"],
        "may mặc": ["vải", "sợi", "quần áo", "giày dép", "túi xách",
                     "phụ kiện thời trang"],
        "vận tải": ["xăng", "dầu", "lốp xe", "phụ tùng", "sửa chữa xe",
                     "bảo hiểm xe", "phí cầu đường"],
        "bất động sản": ["đất", "nhà", "căn hộ", "bất động sản",
                          "môi giới", "chuyển nhượng"],
    }

    def __init__(self):
        self._normalizer = VietnameseTextNormalizer()

    def detect_mismatch(
        self,
        tax_code: str,
        industry: str,
        item_descriptions: list[str],
    ) -> IndustryMismatchResult:
        """
        Kiểm tra mismatch giữa industry và transaction descriptions.

        Args:
            tax_code: Mã số thuế
            industry: Ngành nghề đăng ký
            item_descriptions: Mô tả hàng hóa/dịch vụ

        Returns:
            IndustryMismatchResult.
        """
        result = IndustryMismatchResult(
            tax_code=tax_code,
            declared_industry=industry,
        )

        normalized_industry = self._normalizer.normalize(industry)

        # Tìm industry category
        expected_keywords: list[str] = []
        for ind_key, keywords in self.INDUSTRY_KEYWORDS.items():
            if ind_key in normalized_industry:
                expected_keywords = keywords
                break

        if not expected_keywords:
            # Không match industry → không thể detect mismatch
            return result

        # Kiểm tra từng item description
        mismatched = []
        for desc in item_descriptions:
            normalized_desc = self._normalizer.normalize(desc)
            is_expected = any(kw in normalized_desc for kw in expected_keywords)

            if not is_expected and len(normalized_desc) > 5:
                mismatched.append({
                    "description": desc[:200],
                    "reason": f"Không khớp với ngành {industry}",
                })

        result.mismatched_items = mismatched
        result.mismatch_score = round(
            len(mismatched) / max(1, len(item_descriptions)), 4
        )
        result.transaction_categories = list(set(
            desc[:30] for desc in item_descriptions[:10]
        ))

        return result


# ════════════════════════════════════════════════════════════════
#  5. Main Red Flag Engine (Ensemble)
# ════════════════════════════════════════════════════════════════

class NLPRedFlagEngine:
    """
    Engine phát hiện red flags end-to-end (keyword + semantic ensemble).

    Usage:
        engine = NLPRedFlagEngine()
        result = engine.analyze_invoice("INV001", descriptions, "xây dựng")
        sentiment = engine.analyze_inspector_notes("Nghi vấn hóa đơn ảo")
    """

    def __init__(self, config: RedFlagConfig | None = None):
        self.config = config or RedFlagConfig()
        self._keyword_detector = KeywordRedFlagDetector()
        self._semantic_detector = SemanticRedFlagDetector(self.config)
        self._mismatch_detector = IndustryMismatchDetector()
        self._lock = threading.Lock()

    def analyze_invoice(
        self,
        invoice_id: str,
        descriptions: list[str],
        industry: str = "",
        tax_code: str = "",
    ) -> RedFlagResult:
        """
        Phân tích toàn diện red flags cho một hóa đơn.

        Combines keyword + semantic + industry mismatch.
        """
        t0 = time.perf_counter()

        # Keyword analysis (luôn available)
        kw_result = self._keyword_detector.detect(descriptions, invoice_id)

        # Semantic analysis (nếu model available)
        sem_result = self._semantic_detector.detect(descriptions, invoice_id)

        # Industry mismatch
        mismatch = None
        if industry and tax_code:
            mismatch = self._mismatch_detector.detect_mismatch(
                tax_code, industry, descriptions
            )

        # Ensemble scoring
        if sem_result.confidence > 0:
            # Weighted ensemble
            combined_score = (
                self.config.keyword_weight * kw_result.risk_score
                + self.config.semantic_weight * sem_result.risk_score
            )
        else:
            combined_score = kw_result.risk_score

        # Boost nếu có industry mismatch
        if mismatch and mismatch.mismatch_score > 0.5:
            combined_score = min(1.0, combined_score + 0.15)

        # Aggregate flags
        all_flags = kw_result.flags + sem_result.flags
        if mismatch and mismatch.mismatched_items:
            all_flags.append({
                "type": "industry_mismatch",
                "score": mismatch.mismatch_score,
                "description": f"Mismatch ngành nghề: {len(mismatch.mismatched_items)} items",
                "details": mismatch.mismatched_items[:5],
            })

        # Risk level
        risk_level = "low"
        if combined_score >= self.config.risk_threshold_critical:
            risk_level = "critical"
        elif combined_score >= self.config.risk_threshold_high:
            risk_level = "high"
        elif combined_score >= self.config.risk_threshold_medium:
            risk_level = "medium"

        elapsed = (time.perf_counter() - t0) * 1000

        return RedFlagResult(
            entity_id=invoice_id,
            risk_score=round(combined_score, 4),
            risk_level=risk_level,
            flags=all_flags,
            method="ensemble" if sem_result.confidence > 0 else "keyword",
            confidence=round(max(kw_result.confidence, sem_result.confidence), 4),
            processing_ms=round(elapsed, 1),
        )

    def analyze_inspector_notes(self, text: str) -> SentimentResult:
        """Phân tích sentiment ghi chú thanh tra."""
        return self._keyword_detector.analyze_sentiment(text)

    def batch_analyze(
        self,
        invoices: list[dict[str, Any]],
    ) -> list[RedFlagResult]:
        """
        Phân tích batch nhiều hóa đơn.

        Args:
            invoices: List of dicts với keys:
                - invoice_number, descriptions, industry, tax_code

        Returns:
            List of RedFlagResult.
        """
        results = []
        for inv in invoices:
            result = self.analyze_invoice(
                invoice_id=inv.get("invoice_number", ""),
                descriptions=inv.get("descriptions", []),
                industry=inv.get("industry", ""),
                tax_code=inv.get("tax_code", ""),
            )
            results.append(result)

        # Sort by risk score descending
        results.sort(key=lambda r: r.risk_score, reverse=True)

        logger.info(
            "[RedFlag] Batch analyzed %d invoices: %d high/critical",
            len(results),
            sum(1 for r in results if r.risk_level in ("high", "critical")),
        )

        return results


# ════════════════════════════════════════════════════════════════
#  Singleton
# ════════════════════════════════════════════════════════════════

_red_flag_engine: NLPRedFlagEngine | None = None


def get_red_flag_engine() -> NLPRedFlagEngine:
    """Singleton cho NLPRedFlagEngine."""
    global _red_flag_engine
    if _red_flag_engine is None:
        _red_flag_engine = NLPRedFlagEngine()
    return _red_flag_engine
