"""
entity_resolution_model.py – Siamese Network cho Deduplicate Doanh Nghiệp
============================================================================
Entity resolution / record linkage cho doanh nghiệp Việt Nam:
    phát hiện cùng một chủ thể hoạt động dưới nhiều tên / MST khác nhau.

Capabilities:
    1. Vietnamese company name normalization (dấu, abbreviations, pháp nhân)
    2. Siamese bi-encoder cho fuzzy matching (name + address + representative)
    3. Blocking strategy cho scalability (TF-IDF sparse → dense refinement)
    4. Integration với ownership_links graph (entity_alias_edges, phoenix_candidates)

Data Sources:
    - entity_identities: legal_name, normalized_name, address, representative
    - companies: tax_code, company_name, industry
    - ownership_links: đại diện pháp luật, cổ đông

Design:
    - CPU-first: sentence-transformers bi-encoder, sklearn blocking
    - Blocking → Candidate Pairs → Scoring → Clustering
    - Thread-safe inference pipeline
    - Configurable similarity thresholds

Reference:
    - Mudgal et al., "Deep Learning for Entity Matching", SIGMOD 2018
    - Thirumuruganathan et al., "Deep Learning Approaches for Entity Resolution"
"""

from __future__ import annotations

import logging
import re
import time
import threading
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
class EntityRecord:
    """Một bản ghi entity (doanh nghiệp)."""
    tax_code: str
    legal_name: str = ""
    normalized_name: str = ""
    address: str = ""
    representative_name: str = ""
    representative_id: str = ""
    phone: str = ""
    email: str = ""
    industry: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Kết quả matching giữa hai entities."""
    src_tax_code: str
    dst_tax_code: str
    overall_score: float = 0.0
    name_similarity: float = 0.0
    address_similarity: float = 0.0
    representative_similarity: float = 0.0
    phone_match: bool = False
    email_match: bool = False
    match_type: str = "fuzzy"      # exact | fuzzy | semantic
    evidence: list[str] = field(default_factory=list)
    confidence: str = "low"        # low | medium | high


@dataclass
class ResolutionConfig:
    """Cấu hình entity resolution."""
    name_weight: float = 0.45
    address_weight: float = 0.25
    representative_weight: float = 0.20
    contact_weight: float = 0.10
    match_threshold_high: float = 0.85
    match_threshold_medium: float = 0.65
    match_threshold_low: float = 0.45
    blocking_top_k: int = 50
    max_candidates_per_entity: int = 20
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_dir: str = str(MODEL_DIR)


# ════════════════════════════════════════════════════════════════
#  1. Vietnamese Company Name Normalizer
# ════════════════════════════════════════════════════════════════

class CompanyNameNormalizer:
    """
    Chuẩn hóa tên doanh nghiệp Việt Nam.

    Xử lý:
        - Bỏ prefix pháp nhân (CÔNG TY TNHH, CTY CP, ...)
        - Lowercase + bỏ dấu câu
        - Chuẩn hóa khoảng trắng
        - Xử lý abbreviations
        - Unicode normalization (NFC)
    """

    # Prefix pháp nhân cần bỏ
    LEGAL_PREFIXES = [
        r"công ty tnhh(?: mtv)?",
        r"cong ty tnhh(?: mtv)?",
        r"công ty cổ phần",
        r"cong ty co phan",
        r"công ty trách nhiệm hữu hạn(?: một thành viên)?",
        r"cong ty trach nhiem huu han(?: mot thanh vien)?",
        r"công ty hợp danh",
        r"cong ty hop danh",
        r"doanh nghiệp tư nhân",
        r"doanh nghiep tu nhan",
        r"chi nhánh",
        r"chi nhanh",
        r"văn phòng đại diện",
        r"van phong dai dien",
        r"cty tnhh",
        r"cty cp",
        r"cty",
        r"ctcp",
        r"tnhh",
        r"dntn",
    ]

    # Abbreviation mapping
    ABBREVIATIONS = {
        "xd": "xây dựng",
        "tm": "thương mại",
        "dv": "dịch vụ",
        "sx": "sản xuất",
        "kdv": "kinh doanh",
        "cn": "chi nhánh",
        "vpdd": "văn phòng đại diện",
        "dt": "đầu tư",
        "ptrien": "phát triển",
        "qlda": "quản lý dự án",
        "vlxd": "vật liệu xây dựng",
    }

    def __init__(self):
        self._prefix_pattern = re.compile(
            r"^(" + "|".join(self.LEGAL_PREFIXES) + r")\s*",
            re.IGNORECASE
        )

    def normalize(self, name: str) -> str:
        """Chuẩn hóa tên doanh nghiệp."""
        if not name:
            return ""

        import unicodedata
        # Unicode NFC normalization
        name = unicodedata.normalize("NFC", name)

        # Lowercase
        name = name.lower().strip()

        # Bỏ ký tự đặc biệt (giữ chữ cái VN + số)
        name = re.sub(
            r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
            ' ', name
        )

        # Bỏ prefix pháp nhân
        name = self._prefix_pattern.sub("", name)

        # Chuẩn hóa khoảng trắng
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def normalize_address(self, address: str) -> str:
        """Chuẩn hóa địa chỉ."""
        if not address:
            return ""

        addr = address.lower().strip()
        # Bỏ prefix thừa
        addr = re.sub(r'^(số|so|đ/c|dia chi|địa chỉ)\s*[:.]?\s*', '', addr)
        addr = re.sub(r'\s+', ' ', addr).strip()
        return addr

    def normalize_representative(self, name: str) -> str:
        """Chuẩn hóa tên đại diện pháp luật."""
        if not name:
            return ""
        import unicodedata
        name = unicodedata.normalize("NFC", name)
        name = name.lower().strip()
        # Bỏ chức danh
        name = re.sub(
            r'^(ông|bà|mr\.?|mrs\.?|ms\.?|giám đốc|tổng giám đốc|chủ tịch)\s*',
            '', name, flags=re.IGNORECASE
        )
        name = re.sub(r'\s+', ' ', name).strip()
        return name


# ════════════════════════════════════════════════════════════════
#  2. String Similarity Functions
# ════════════════════════════════════════════════════════════════

class StringSimilarity:
    """Các hàm tính similarity cho text matching."""

    @staticmethod
    def jaccard(a: str, b: str) -> float:
        """Jaccard similarity trên character n-grams."""
        if not a or not b:
            return 0.0
        a_ngrams = set(a[i:i+3] for i in range(len(a) - 2))
        b_ngrams = set(b[i:i+3] for i in range(len(b) - 2))
        if not a_ngrams or not b_ngrams:
            return 1.0 if a == b else 0.0
        intersection = len(a_ngrams & b_ngrams)
        union = len(a_ngrams | b_ngrams)
        return intersection / max(1, union)

    @staticmethod
    def levenshtein_ratio(a: str, b: str) -> float:
        """Normalized Levenshtein similarity (1 - distance/max_len)."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        n, m = len(a), len(b)
        if n > m:
            a, b = b, a
            n, m = m, n

        prev = list(range(n + 1))
        for j in range(1, m + 1):
            curr = [j] + [0] * n
            for i in range(1, n + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                curr[i] = min(curr[i-1] + 1, prev[i] + 1, prev[i-1] + cost)
            prev = curr

        distance = prev[n]
        return 1.0 - distance / max(n, m)

    @staticmethod
    def token_overlap(a: str, b: str) -> float:
        """Token-level overlap ratio."""
        if not a or not b:
            return 0.0
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        intersection = len(tokens_a & tokens_b)
        min_len = min(len(tokens_a), len(tokens_b))
        return intersection / max(1, min_len)


# ════════════════════════════════════════════════════════════════
#  3. Blocking Strategy (Candidate Generation)
# ════════════════════════════════════════════════════════════════

class BlockingStrategy:
    """
    Blocking strategy cho scalability.

    Thay vì so sánh O(N²), dùng TF-IDF + cosine để tìm
    top-K candidates cho mỗi entity trước khi scoring chi tiết.
    """

    def __init__(self, top_k: int = 50):
        self.top_k = top_k
        self._vectorizer = None
        self._tfidf_matrix = None
        self._entity_ids: list[str] = []

    def build_index(self, entities: list[EntityRecord]) -> None:
        """Build TF-IDF index từ normalized names."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = []
            self._entity_ids = []
            for entity in entities:
                # Combine name + address for better blocking
                text = f"{entity.normalized_name} {entity.address}"
                texts.append(text if text.strip() else "unknown")
                self._entity_ids.append(entity.tax_code)

            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=10000,
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)

            logger.info("[Blocking] Built index for %d entities", len(entities))

        except ImportError:
            logger.warning("[Blocking] sklearn not available")

    def find_candidates(
        self, entity: EntityRecord, exclude_self: bool = True
    ) -> list[tuple[str, float]]:
        """
        Tìm top-K candidates cho một entity.

        Returns:
            List of (tax_code, tfidf_similarity) tuples.
        """
        if self._vectorizer is None or self._tfidf_matrix is None:
            return []

        text = f"{entity.normalized_name} {entity.address}"
        if not text.strip():
            return []

        query_vec = self._vectorizer.transform([text])

        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Sort and filter
        top_indices = np.argsort(similarities)[::-1]
        candidates = []

        for idx in top_indices:
            if len(candidates) >= self.top_k:
                break
            tax_code = self._entity_ids[idx]
            if exclude_self and tax_code == entity.tax_code:
                continue
            sim = float(similarities[idx])
            if sim > 0.05:  # Minimum similarity threshold
                candidates.append((tax_code, round(sim, 4)))

        return candidates


# ════════════════════════════════════════════════════════════════
#  4. Siamese Bi-Encoder
# ════════════════════════════════════════════════════════════════

class SiameseBiEncoder:
    """
    Siamese bi-encoder cho entity matching.

    Dùng pre-trained sentence-transformers để encode entity attributes,
    so sánh bằng cosine similarity trong embedding space.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
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
                self._model = SentenceTransformer(self._model_name, device="cpu")
                self._loaded = True
                logger.info("[SiameseEncoder] Model loaded: %s", self._model_name)
                return True
            except ImportError:
                logger.warning("[SiameseEncoder] sentence-transformers not installed")
                return False
            except Exception as exc:
                logger.warning("[SiameseEncoder] Load failed: %s", exc)
                return False

    def similarity(self, text_a: str, text_b: str) -> float:
        """Tính semantic similarity giữa hai text."""
        if not self.load() or not text_a or not text_b:
            return 0.0

        embeddings = self._model.encode(
            [text_a, text_b], normalize_embeddings=True
        )
        return float(np.dot(embeddings[0], embeddings[1]))

    def batch_similarity(
        self,
        queries: list[str],
        candidates: list[str],
    ) -> np.ndarray:
        """
        Tính similarity matrix giữa queries và candidates.

        Returns:
            (len(queries), len(candidates)) numpy array.
        """
        if not self.load() or not queries or not candidates:
            return np.zeros((len(queries), len(candidates)))

        q_emb = self._model.encode(queries, normalize_embeddings=True)
        c_emb = self._model.encode(candidates, normalize_embeddings=True)
        return q_emb @ c_emb.T


# ════════════════════════════════════════════════════════════════
#  5. Entity Resolution Engine
# ════════════════════════════════════════════════════════════════

class EntityResolutionEngine:
    """
    Engine entity resolution end-to-end.

    Pipeline: Normalize → Block → Score → Cluster → Output

    Usage:
        engine = EntityResolutionEngine()
        engine.build_index(entities)
        matches = engine.find_duplicates(query_entity)
        all_matches = engine.resolve_all(entities)
    """

    def __init__(self, config: ResolutionConfig | None = None):
        self.config = config or ResolutionConfig()
        self._normalizer = CompanyNameNormalizer()
        self._string_sim = StringSimilarity()
        self._blocker = BlockingStrategy(top_k=self.config.blocking_top_k)
        self._encoder = SiameseBiEncoder(self.config.embedding_model)
        self._entities: dict[str, EntityRecord] = {}
        self._lock = threading.Lock()

    def normalize_entity(self, entity: EntityRecord) -> EntityRecord:
        """Chuẩn hóa tất cả fields của entity."""
        entity.normalized_name = self._normalizer.normalize(entity.legal_name)
        entity.address = self._normalizer.normalize_address(entity.address)
        entity.representative_name = self._normalizer.normalize_representative(
            entity.representative_name
        )
        return entity

    def build_index(self, entities: list[EntityRecord]) -> None:
        """Build index từ danh sách entities."""
        # Normalize all
        for entity in entities:
            self.normalize_entity(entity)
            self._entities[entity.tax_code] = entity

        # Build blocking index
        self._blocker.build_index(entities)

        logger.info("[ER] Built index for %d entities", len(entities))

    def find_duplicates(
        self,
        entity: EntityRecord,
        top_k: int | None = None,
    ) -> list[MatchResult]:
        """
        Tìm duplicates cho một entity.

        Args:
            entity: Entity cần tìm duplicates
            top_k: Số lượng candidates tối đa

        Returns:
            List of MatchResult sorted by score descending.
        """
        top_k = top_k or self.config.max_candidates_per_entity
        entity = self.normalize_entity(entity)

        # Step 1: Blocking — tìm candidates
        candidates = self._blocker.find_candidates(entity)

        # Step 2: Detailed scoring
        matches: list[MatchResult] = []
        for cand_tax_code, block_score in candidates[:top_k]:
            cand_entity = self._entities.get(cand_tax_code)
            if not cand_entity:
                continue

            match = self._score_pair(entity, cand_entity)

            if match.overall_score >= self.config.match_threshold_low:
                matches.append(match)

        # Sort by score
        matches.sort(key=lambda m: m.overall_score, reverse=True)

        return matches

    def resolve_all(
        self,
        entities: list[EntityRecord] | None = None,
    ) -> list[list[MatchResult]]:
        """
        Chạy entity resolution trên toàn bộ dataset.

        Returns:
            List of match groups (mỗi group = list of MatchResult).
        """
        t0 = time.perf_counter()

        if entities:
            self.build_index(entities)

        all_matches: list[list[MatchResult]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for tax_code, entity in self._entities.items():
            matches = self.find_duplicates(entity)

            # Deduplicate pairs
            new_matches = []
            for match in matches:
                pair = tuple(sorted([match.src_tax_code, match.dst_tax_code]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    new_matches.append(match)

            if new_matches:
                all_matches.append(new_matches)

        elapsed = time.perf_counter() - t0
        total_pairs = sum(len(group) for group in all_matches)

        logger.info(
            "[ER] Resolved %d entities → %d match groups (%d pairs) in %.1fs",
            len(self._entities), len(all_matches), total_pairs, elapsed,
        )

        return all_matches

    def export_to_db_format(
        self, matches: list[list[MatchResult]]
    ) -> list[dict[str, Any]]:
        """
        Export matches sang format entity_alias_edges table.

        Returns:
            List of dicts ready for DB insertion.
        """
        rows = []
        for group in matches:
            for match in group:
                edge_type = "name_sim"
                if match.representative_similarity > 0.8:
                    edge_type = "rep_sim"
                elif match.address_similarity > 0.8:
                    edge_type = "address_sim"
                elif match.phone_match or match.email_match:
                    edge_type = "phone_sim"

                rows.append({
                    "src_tax_code": match.src_tax_code,
                    "dst_tax_code": match.dst_tax_code,
                    "edge_type": edge_type,
                    "score": round(match.overall_score, 4),
                    "evidence_json": {
                        "name_sim": match.name_similarity,
                        "address_sim": match.address_similarity,
                        "rep_sim": match.representative_similarity,
                        "phone_match": match.phone_match,
                        "email_match": match.email_match,
                        "evidence": match.evidence,
                    },
                })

        return rows

    # ─── Internal Scoring ────────────────────────────────────────

    def _score_pair(
        self, entity_a: EntityRecord, entity_b: EntityRecord
    ) -> MatchResult:
        """Tính điểm matching chi tiết giữa hai entities."""
        match = MatchResult(
            src_tax_code=entity_a.tax_code,
            dst_tax_code=entity_b.tax_code,
        )

        evidence = []

        # 1. Name similarity (string + semantic)
        name_string_sim = max(
            self._string_sim.jaccard(
                entity_a.normalized_name, entity_b.normalized_name
            ),
            self._string_sim.levenshtein_ratio(
                entity_a.normalized_name, entity_b.normalized_name
            ),
            self._string_sim.token_overlap(
                entity_a.normalized_name, entity_b.normalized_name
            ),
        )

        # Semantic similarity (nếu encoder available)
        name_semantic_sim = self._encoder.similarity(
            entity_a.normalized_name, entity_b.normalized_name
        )

        match.name_similarity = round(
            max(name_string_sim, name_semantic_sim * 0.9), 4
        )
        if match.name_similarity > 0.7:
            evidence.append(f"Tên tương tự: {match.name_similarity:.2f}")

        # 2. Address similarity
        if entity_a.address and entity_b.address:
            match.address_similarity = round(max(
                self._string_sim.jaccard(entity_a.address, entity_b.address),
                self._string_sim.token_overlap(entity_a.address, entity_b.address),
            ), 4)
            if match.address_similarity > 0.7:
                evidence.append(f"Địa chỉ tương tự: {match.address_similarity:.2f}")

        # 3. Representative similarity
        if entity_a.representative_name and entity_b.representative_name:
            match.representative_similarity = round(
                self._string_sim.levenshtein_ratio(
                    entity_a.representative_name,
                    entity_b.representative_name,
                ), 4
            )
            if match.representative_similarity > 0.8:
                evidence.append("Cùng đại diện pháp luật")

        # Same representative ID = very strong signal
        if (entity_a.representative_id and entity_b.representative_id
                and entity_a.representative_id == entity_b.representative_id):
            match.representative_similarity = 1.0
            evidence.append(f"Cùng CCCD đại diện: {entity_a.representative_id}")

        # 4. Contact matching
        if entity_a.phone and entity_b.phone:
            clean_a = re.sub(r'\D', '', entity_a.phone)
            clean_b = re.sub(r'\D', '', entity_b.phone)
            if clean_a and clean_b and clean_a == clean_b:
                match.phone_match = True
                evidence.append("Cùng số điện thoại")

        if entity_a.email and entity_b.email:
            if entity_a.email.lower() == entity_b.email.lower():
                match.email_match = True
                evidence.append("Cùng email")

        # Overall score (weighted)
        contact_score = 0.0
        if match.phone_match:
            contact_score += 0.5
        if match.email_match:
            contact_score += 0.5

        match.overall_score = round(
            self.config.name_weight * match.name_similarity
            + self.config.address_weight * match.address_similarity
            + self.config.representative_weight * match.representative_similarity
            + self.config.contact_weight * contact_score,
            4
        )

        # Confidence level
        if match.overall_score >= self.config.match_threshold_high:
            match.confidence = "high"
        elif match.overall_score >= self.config.match_threshold_medium:
            match.confidence = "medium"
        else:
            match.confidence = "low"

        # Match type
        if match.name_similarity > 0.95:
            match.match_type = "near_exact"
        elif name_semantic_sim > name_string_sim:
            match.match_type = "semantic"
        else:
            match.match_type = "fuzzy"

        match.evidence = evidence

        return match


# ════════════════════════════════════════════════════════════════
#  Singleton
# ════════════════════════════════════════════════════════════════

_er_engine: EntityResolutionEngine | None = None


def get_entity_resolution_engine() -> EntityResolutionEngine:
    """Singleton cho EntityResolutionEngine."""
    global _er_engine
    if _er_engine is None:
        _er_engine = EntityResolutionEngine()
    return _er_engine
