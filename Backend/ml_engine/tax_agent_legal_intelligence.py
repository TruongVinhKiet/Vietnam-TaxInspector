"""
Legal intelligence utilities for TaxInspector GraphRAG and Tax Agent.

This module is intentionally deterministic and local-first. It provides:
- Vietnamese tax-law entity/relation extraction for KG ingestion.
- Effective-date and official-letter scope reasoning.
- Lightweight citation faithfulness checks for legal synthesis.
- Legal slot analysis for clarification-first consulting.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


AUTHORITY_RANKS: dict[str, int] = {
    "constitution": 100,
    "law": 90,
    "ordinance": 85,
    "decree": 80,
    "resolution": 75,
    "circular": 70,
    "joint_circular": 68,
    "decision": 60,
    "article": 50,
    "clause": 45,
    "official_letter": 40,
    "guideline": 30,
    "concept": 35,
}


_DOC_TYPE_ALIASES = {
    "luat": "law",
    "law": "law",
    "bo luat": "law",
    "nghi dinh": "decree",
    "nghi dinh": "decree",
    "decree": "decree",
    "nd": "decree",
    "nd-cp": "decree",
    "thong tu": "circular",
    "circular": "circular",
    "tt": "circular",
    "tt-btc": "circular",
    "nghi quyet": "resolution",
    "resolution": "resolution",
    "quyet dinh": "decision",
    "decision": "decision",
    "cong van": "official_letter",
    "official letter": "official_letter",
    "official_letter": "official_letter",
    "cv": "official_letter",
    "huong dan": "guideline",
    "guideline": "guideline",
    "article": "article",
    "dieu": "article",
    "clause": "clause",
    "khoan": "clause",
}


def strip_accents(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def canonical_doc_type(value: str | None) -> str:
    raw = strip_accents(str(value or "")).lower()
    raw = raw.replace("_", " ").replace("-", " ").strip()
    raw_compact = raw.replace(" ", "-")
    if raw in _DOC_TYPE_ALIASES:
        return _DOC_TYPE_ALIASES[raw]
    if raw_compact in _DOC_TYPE_ALIASES:
        return _DOC_TYPE_ALIASES[raw_compact]
    if "cong van" in raw or re.search(r"\bcv\b", raw):
        return "official_letter"
    if "thong tu" in raw or re.search(r"\btt\b", raw):
        return "circular"
    if "nghi dinh" in raw or re.search(r"\bnd\b", raw):
        return "decree"
    if "quyet dinh" in raw:
        return "decision"
    if "luat" in raw:
        return "law"
    return "concept" if not raw else raw.replace(" ", "_")[:60]


def authority_rank(doc_type: str | None) -> int:
    return AUTHORITY_RANKS.get(canonical_doc_type(doc_type), 35)


def slugify_key(value: str) -> str:
    raw = strip_accents(value or "").upper()
    raw = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
    return raw[:180] or "UNKNOWN"


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    text = str(value)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except Exception:
            pass
    return None


def effective_status(
    *,
    effective_from: Any = None,
    effective_to: Any = None,
    status: str | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    as_of = as_of or date.today()
    start = parse_date(effective_from)
    end = parse_date(effective_to)
    raw_status = str(status or "active").lower()

    if raw_status in {"repealed", "deleted", "inactive"}:
        state = "repealed"
    elif raw_status in {"amended", "superseded"}:
        state = "amended"
    elif start and as_of < start:
        state = "pending"
    elif end and as_of > end:
        state = "expired"
    else:
        state = "active"

    return {
        "state": state,
        "as_of": as_of.isoformat(),
        "effective_from": start.isoformat() if start else None,
        "effective_to": end.isoformat() if end else None,
        "source_status": raw_status,
        "is_usable": state in {"active", "amended"},
    }


def official_letter_scope(
    *,
    doc_type: str | None,
    title: str = "",
    text: str = "",
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = canonical_doc_type(doc_type)
    attrs = attributes or {}
    if canonical != "official_letter":
        return {
            "is_official_letter": False,
            "binding_level": "normative",
            "scope": "general",
            "warnings": [],
        }

    haystack = strip_accents(f"{title} {text} {attrs}").lower()
    scope = "case_specific"
    if any(token in haystack for token in ["huong dan chung", "toan nganh", "toan quoc"]):
        scope = "administrative_guidance"
    elif any(token in haystack for token in ["tra loi", "theo de nghi", "truong hop cua", "cong ty"]):
        scope = "case_specific"

    return {
        "is_official_letter": True,
        "binding_level": "guidance_not_normative",
        "scope": scope,
        "warnings": [
            "Official letters are guidance for a concrete case or administrative interpretation; do not treat them as higher-authority normative law.",
        ],
    }


@dataclass
class ExtractedLegalEntity:
    entity_key: str
    entity_type: str
    display_name: str
    description: str = ""
    authority_rank: int = 35
    effective_from: str | None = None
    effective_to: str | None = None
    status: str = "active"
    chunk_ids: list[int] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedLegalRelation:
    source_key: str
    target_key: str
    relation_type: str
    weight: float = 0.7
    confidence: float = 0.65
    evidence_text: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalExtractionResult:
    entities: list[ExtractedLegalEntity]
    relations: list[ExtractedLegalRelation]
    citations: list[dict[str, Any]]


class LegalKnowledgeExtractor:
    """Rule-based Legal NER + relation extractor for Vietnamese tax documents."""

    ARTICLE_RE = re.compile(r"\b(?:Điều|Dieu)\s+(\d+[a-zA-Z]?)\.?", re.IGNORECASE)
    CLAUSE_RE = re.compile(r"\b(?:Khoản|Khoan)\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
    POINT_RE = re.compile(r"\b(?:Điểm|Diem)\s+([a-zA-Z])\b", re.IGNORECASE)
    DOC_NUMBER_RE = re.compile(
        r"\b(\d{1,4}[/\-](?:19|20)\d{2}[/\-](?:QH\d+|NĐ-CP|ND-CP|TT-BTC|TT|BTC|QĐ-TCT|QD-TCT|QĐ-BTC|QD-BTC|CV|TCT|NQ-CP|NQ|UBTVQH)[A-Z0-9\-]*)\b",
        re.IGNORECASE,
    )
    EFFECTIVE_RE = re.compile(
        r"(?:có hiệu lực|co hieu luc|hiệu lực|hieu luc)[^0-9]{0,40}(\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2})",
        re.IGNORECASE,
    )
    DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2})\b")

    def extract_document(
        self,
        *,
        document_key: str,
        title: str,
        doc_type: str,
        content: str,
        chunks: list[dict[str, Any]] | None = None,
        effective_from: str | None = None,
        effective_to: str | None = None,
        authority: str | None = None,
    ) -> LegalExtractionResult:
        canonical = canonical_doc_type(doc_type)
        doc_key = document_key or f"DOC_{slugify_key(title)}"
        chunk_payloads = chunks or [{"chunk_id": None, "chunk_index": 0, "text": content, "heading": title}]
        chunk_ids = [int(c["chunk_id"]) for c in chunk_payloads if c.get("chunk_id")]

        inferred_from = effective_from or self._infer_effective_from(f"{title}\n{content}")
        doc_scope = official_letter_scope(doc_type=canonical, title=title, text=content[:1200])
        doc_entity = ExtractedLegalEntity(
            entity_key=doc_key,
            entity_type=canonical,
            display_name=title,
            description=(content or "")[:500],
            authority_rank=authority_rank(canonical),
            effective_from=inferred_from,
            effective_to=effective_to,
            chunk_ids=chunk_ids,
            attributes={
                "authority": authority,
                "source": "legal_kg_extractor",
                "official_letter_scope": doc_scope,
                "aliases": [self._doc_key_from_reference(title)],
            },
        )

        entities: dict[str, ExtractedLegalEntity] = {doc_key: doc_entity}
        relations: list[ExtractedLegalRelation] = []
        citations: list[dict[str, Any]] = []

        for chunk in chunk_payloads:
            chunk_id = int(chunk["chunk_id"]) if chunk.get("chunk_id") else None
            text = str(chunk.get("text") or "")
            heading = str(chunk.get("heading") or "")
            article_matches = list(self.ARTICLE_RE.finditer(f"{heading}\n{text}"))
            article_key = None
            if article_matches:
                article_no = article_matches[0].group(1)
                article_key = f"{doc_key}:article:{article_no}"
                entities[article_key] = ExtractedLegalEntity(
                    entity_key=article_key,
                    entity_type="article",
                    display_name=f"Dieu {article_no} - {title}",
                    description=text[:500],
                    authority_rank=AUTHORITY_RANKS["article"],
                    effective_from=inferred_from,
                    effective_to=effective_to,
                    chunk_ids=[chunk_id] if chunk_id else [],
                    attributes={"article_number": article_no, "parent_document": doc_key},
                )
                relations.append(ExtractedLegalRelation(
                    source_key=doc_key,
                    target_key=article_key,
                    relation_type="contains",
                    weight=0.95,
                    confidence=0.9,
                    evidence_text=heading or text[:240],
                ))

            parent_key = article_key or doc_key
            for match in self.CLAUSE_RE.finditer(text):
                clause_no = match.group(1)
                clause_key = f"{parent_key}:clause:{clause_no}"
                if clause_key not in entities:
                    entities[clause_key] = ExtractedLegalEntity(
                        entity_key=clause_key,
                        entity_type="clause",
                        display_name=f"Khoan {clause_no} - {title}",
                        description=text[max(0, match.start() - 80):match.end() + 240],
                        authority_rank=AUTHORITY_RANKS["clause"],
                        effective_from=inferred_from,
                        effective_to=effective_to,
                        chunk_ids=[chunk_id] if chunk_id else [],
                        attributes={"clause_number": clause_no, "parent": parent_key},
                    )
                    relations.append(ExtractedLegalRelation(
                        source_key=parent_key,
                        target_key=clause_key,
                        relation_type="contains",
                        weight=0.9,
                        confidence=0.85,
                        evidence_text=text[max(0, match.start() - 80):match.end() + 180],
                    ))

            for ref_match in self.DOC_NUMBER_RE.finditer(text):
                ref_text = ref_match.group(1)
                target_key = self._doc_key_from_reference(ref_text)
                if target_key and target_key != doc_key:
                    ref_type = self._infer_doc_type_from_reference(ref_text)
                    entities.setdefault(target_key, ExtractedLegalEntity(
                        entity_key=target_key,
                        entity_type=ref_type,
                        display_name=ref_text,
                        description=f"Referenced document {ref_text}",
                        authority_rank=authority_rank(ref_type),
                        attributes={"source": "reference_stub", "reference_text": ref_text},
                    ))
                    rel_type = self._infer_relation_type(text, ref_match.start())
                    relations.append(ExtractedLegalRelation(
                        source_key=parent_key,
                        target_key=target_key,
                        relation_type=rel_type,
                        weight={"amends": 0.9, "supplements": 0.85, "replaces": 0.9, "interprets": 0.82}.get(rel_type, 0.7),
                        confidence=0.75 if rel_type != "cites" else 0.62,
                        evidence_text=text[max(0, ref_match.start() - 160):ref_match.end() + 160],
                    ))
                    citations.append({
                        "chunk_id": chunk_id,
                        "legal_reference": ref_text,
                        "citation_text": text[max(0, ref_match.start() - 80):ref_match.end() + 160],
                        "confidence": 0.75,
                    })

        return LegalExtractionResult(
            entities=list(entities.values()),
            relations=relations,
            citations=citations,
        )

    def _infer_effective_from(self, text: str) -> str | None:
        match = self.EFFECTIVE_RE.search(text or "")
        if not match:
            return None
        dt = parse_date(match.group(1))
        return dt.isoformat() if dt else None

    def _infer_doc_type_from_reference(self, ref: str) -> str:
        normalized = strip_accents(ref).upper()
        if "QH" in normalized:
            return "law"
        if "ND" in normalized or "NĐ" in ref.upper():
            return "decree"
        if "TT" in normalized:
            return "circular"
        if "QD" in normalized or "QĐ" in ref.upper():
            return "decision"
        if "CV" in normalized or "TCT" in normalized:
            return "official_letter"
        if "NQ" in normalized:
            return "resolution"
        return "concept"

    def _doc_key_from_reference(self, ref: str) -> str:
        clean = slugify_key(ref)
        if not clean:
            return ""
        return f"VB_{clean}"

    def _infer_relation_type(self, text: str, pos: int) -> str:
        window = strip_accents(text[max(0, pos - 180):pos + 180]).lower()
        if any(token in window for token in ["thay the", "bai bo", "het hieu luc"]):
            return "replaces"
        if "sua doi" in window:
            return "amends"
        if "bo sung" in window:
            return "supplements"
        if any(token in window for token in ["huong dan", "giai thich", "can cu"]):
            return "interprets"
        if any(token in window for token in ["quy dinh tai", "theo"]):
            return "cites"
        return "cites"


class LegalSlotAnalyzer:
    """Identify missing facts for professional legal consultation."""

    TAX_PERIOD_RE = re.compile(r"\b(20\d{2}|q[1-4]|qu[yý]\s*[1-4]|th[aá]ng\s*\d{1,2}|k[yỳ]\s*thu[eế])\b", re.IGNORECASE)
    DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b")
    TAXPAYER_RE = re.compile(r"\b(doanh nghi[eệ]p|h[oộ]\s+kinh doanh|c[aá]\s+nh[aâ]n|t[oổ]\s+ch[uứ]c|nh[aà]\s+th[aầ]u|xu[aấ]t kh[aẩ]u)\b", re.IGNORECASE)
    TRANSACTION_RE = re.compile(r"\b(h[oó]a\s+d[oơ]n|xu[aấ]t\s+kh[aẩ]u|nh[aậ]p\s+kh[aẩ]u|ho[aà]n\s+thu[eế]|kh[aấ]u\s+tr[uừ]|chuy[eể]n\s+nh[uượ]ng|giao\s+d[iị]ch)\b", re.IGNORECASE)

    def missing_slots(self, query: str, *, intent: str = "general_tax_query") -> list[str]:
        normalized = query or ""
        if not self._looks_like_applied_legal_question(normalized, intent):
            return []
        missing = []
        if not self.TAX_PERIOD_RE.search(normalized) and not self.DATE_RE.search(normalized):
            missing.append("tax_period_or_document_date")
        if not self.TAXPAYER_RE.search(normalized):
            missing.append("taxpayer_type")
        if not self.TRANSACTION_RE.search(normalized):
            missing.append("transaction_type")
        return missing

    def _looks_like_applied_legal_question(self, query: str, intent: str) -> bool:
        q = strip_accents(query).lower()
        if intent in {"vat_refund_risk", "invoice_risk", "transfer_pricing", "audit_selection"}:
            return True
        triggers = ["co duoc", "ap dung", "truong hop", "can lam gi", "dieu kien", "xet ho so", "cong van"]
        return any(t in q for t in triggers)

    def clarification_prompt(self, missing: list[str]) -> str:
        labels = {
            "tax_period_or_document_date": "ky thue hoac ngay phat sinh chung tu",
            "taxpayer_type": "loai nguoi nop thue/doanh nghiep",
            "transaction_type": "loai giao dich hoac nghiep vu thue",
        }
        items = [labels.get(slot, slot) for slot in missing]
        return (
            "De tu van phap ly chinh xac, toi can bo sung: "
            + "; ".join(items)
            + ". Vui long cung cap cac thong tin nay de toi doi chieu van ban co hieu luc va tra loi co can cu."
        )


class LegalFaithfulnessVerifier:
    """Lightweight claim-to-evidence verifier for grounded legal answers."""

    SENTENCE_RE = re.compile(r"(?<=[.!?。])\s+|\n+-\s+|\n+")

    def verify(
        self,
        *,
        answer_text: str,
        evidence: list[dict[str, Any]],
        min_overlap: float = 0.18,
    ) -> dict[str, Any]:
        legal_evidence = [ev for ev in evidence if str(ev.get("source_type", "legal")) == "legal" or ev.get("doc_type")]
        evidence_texts = [
            str(ev.get("content") or ev.get("text") or "") + " " + str(ev.get("title") or "")
            for ev in legal_evidence
        ]
        claims = self._extract_claims(answer_text)
        verified = []
        unsupported = []
        for claim in claims:
            best = self._best_support(claim, evidence_texts)
            has_inline_citation = bool(re.search(r"\[\d+\]", claim))
            supported = best["score"] >= min_overlap or (has_inline_citation and best["score"] >= min_overlap * 0.7)
            payload = {"claim": claim[:400], "support_score": round(best["score"], 4), "evidence_index": best["index"]}
            if supported:
                verified.append(payload)
            else:
                unsupported.append(payload)

        coverage = len(verified) / max(1, len(claims))
        citation_markers = len(re.findall(r"\[\d+\]", answer_text or ""))
        faithfulness = round(0.65 * coverage + 0.35 * min(1.0, citation_markers / max(1, len(claims))), 4)
        return {
            "claim_count": len(claims),
            "verified_claims": verified,
            "unsupported_claims": unsupported,
            "faithfulness_score": faithfulness,
            "citation_marker_count": citation_markers,
            "requires_abstain": bool(claims and legal_evidence and faithfulness < 0.35),
            "status": "pass" if faithfulness >= 0.55 or not claims else "review",
        }

    def _extract_claims(self, answer_text: str) -> list[str]:
        raw_parts = [p.strip(" -\t\r") for p in self.SENTENCE_RE.split(answer_text or "")]
        claims = []
        for part in raw_parts:
            if len(part) < 28:
                continue
            lowered = strip_accents(part).lower()
            if lowered.startswith(("do tin cay", "cong cu", "tier", "khuyen nghi", "can cu phap ly")):
                continue
            claims.append(part)
        return claims[:24]

    def _best_support(self, claim: str, evidence_texts: list[str]) -> dict[str, Any]:
        claim_tokens = self._tokens(claim)
        if not claim_tokens:
            return {"score": 0.0, "index": -1}
        best_score = 0.0
        best_index = -1
        for idx, text in enumerate(evidence_texts):
            ev_tokens = self._tokens(text)
            if not ev_tokens:
                continue
            overlap = len(claim_tokens & ev_tokens) / max(1, min(len(claim_tokens), 24))
            if overlap > best_score:
                best_score = overlap
                best_index = idx
        return {"score": min(1.0, best_score), "index": best_index}

    def _tokens(self, text: str) -> set[str]:
        normalized = strip_accents(text).lower()
        tokens = set(re.findall(r"[a-z0-9]{3,}", normalized))
        stop = {
            "the", "and", "for", "voi", "cac", "cua", "khi", "thi", "nay", "neu",
            "duoc", "phai", "can", "theo", "trong", "ngoai", "khong", "hoac",
        }
        return {tok for tok in tokens if tok not in stop}
