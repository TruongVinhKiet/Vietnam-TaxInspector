"""
legal.py – Legal Library API Router
=====================================
Serves legal document data from the knowledge_documents + knowledge_document_versions
tables to the frontend Legal Library page.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter(prefix="/api/legal", tags=["Legal"])

# ── Vietnamese doc-type display mapping ──────────────────────────────
DOC_TYPE_MAP = {
    "law":       "Luật",
    "decree":    "Nghị định",
    "circular":  "Thông tư",
    "decision":  "Quyết định",
    "resolution":"Nghị quyết",
}

# ── Agency → stamp info mapping ──────────────────────────────────────
AGENCY_MAP = {
    "quoc hoi":       {"agency": "QUỐC HỘI",     "subAgency": "",              "signerTitle": "CHỦ TỊCH QUỐC HỘI"},
    "chinh phu":      {"agency": "CHÍNH PHỦ",     "subAgency": "",              "signerTitle": "TM. CHÍNH PHỦ\nTHỦ TƯỚNG"},
    "bo tai chinh":   {"agency": "BỘ TÀI CHÍNH",  "subAgency": "",              "signerTitle": "KT. BỘ TRƯỞNG\nTHỨ TRƯỞNG"},
    "tong cuc thue":  {"agency": "BỘ TÀI CHÍNH",  "subAgency": "TỔNG CỤC THUẾ","signerTitle": "TỔNG CỤC TRƯỞNG"},
}


def _normalize_ascii(s: str) -> str:
    """Remove diacritics for lookup (very simplified)."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _resolve_agency(authority: str | None, doc_type: str | None) -> dict[str, str]:
    """Resolve agency display info from authority text or doc_type."""
    if authority:
        key = _normalize_ascii(authority)
        for pattern, info in AGENCY_MAP.items():
            if pattern in key:
                return info
    # Fallback based on doc_type
    dt = (doc_type or "").lower()
    if dt == "law":
        return AGENCY_MAP["quoc hoi"]
    if dt == "decree":
        return AGENCY_MAP["chinh phu"]
    if dt in ("circular", "decision"):
        return AGENCY_MAP["bo tai chinh"]
    return {"agency": "CƠ QUAN BAN HÀNH", "subAgency": "", "signerTitle": "TM. CƠ QUAN BAN HÀNH"}


def _format_number(document_key: str, doc_type: str) -> str:
    """Best-effort readable number from document_key.
    
    Examples:
        LUAT_38_2019  →  38/2019/QH14
        ND_126_2020   →  126/2020/NĐ-CP
        TT_80_2021    →  80/2021/TT-BTC
    """
    # Try to extract numbers from the key
    nums = re.findall(r'\d+', document_key)
    if len(nums) >= 2:
        number, year = nums[0], nums[1]
        suffix_map = {
            "law": "QH14",
            "decree": "NĐ-CP",
            "circular": "TT-BTC",
            "decision": "QĐ-TCT",
        }
        suffix = suffix_map.get((doc_type or "").lower(), "")
        return f"{number}/{year}/{suffix}" if suffix else f"{number}/{year}"
    return document_key.replace("_", "/")


def _format_date(effective_from) -> str:
    """Format effective_from date to Vietnamese style."""
    if effective_from:
        try:
            from datetime import date as dt_date
            if isinstance(effective_from, str):
                effective_from = dt_date.fromisoformat(effective_from)
            return f"Hà Nội, ngày {effective_from.day:02d} tháng {effective_from.month:02d} năm {effective_from.year}"
        except Exception:
            pass
    return ""


@router.get("/documents")
def get_legal_documents(
    doc_type: str | None = None,
    db: Session = Depends(get_db),
):
    """Fetch legal documents for the Legal Library UI.

    Optional query param ``doc_type`` filters by doc_type (law, decree, circular, decision).
    """
    query = """
        SELECT 
            kd.id,
            kd.document_key, 
            kd.title, 
            kd.doc_type, 
            kd.authority,
            kd.effective_from,
            kd.metadata_json,
            kdv.raw_text
        FROM knowledge_documents kd
        LEFT JOIN knowledge_document_versions kdv ON kd.id = kdv.document_id
        WHERE kd.status = 'active'
    """
    params: dict[str, Any] = {}

    if doc_type:
        query += " AND LOWER(kd.doc_type) = :doc_type"
        params["doc_type"] = doc_type.lower()

    query += " ORDER BY kd.created_at DESC LIMIT 60"

    rows = db.execute(text(query), params).mappings().all()

    results = []
    for row in rows:
        dt = (row["doc_type"] or "unknown").lower()
        display_type = DOC_TYPE_MAP.get(dt, (row["doc_type"] or "VĂN BẢN").upper())
        agency_info = _resolve_agency(row["authority"], row["doc_type"])

        raw_text = row["raw_text"] or "(Nội dung đang được cập nhật)"
        meta = row["metadata_json"] or {}

        results.append({
            "id":          row["document_key"],
            "number":      _format_number(row["document_key"], row["doc_type"]),
            "type":        display_type,
            "title":       row["title"],
            "agency":      agency_info["agency"],
            "subAgency":   agency_info["subAgency"],
            "date":        _format_date(row["effective_from"]),
            "signerTitle": agency_info["signerTitle"],
            "signerName":  "Đã ký",
            "content":     raw_text,
            "tags":        [display_type, "Nghiệp vụ thuế"],
            "effective_status": meta.get("effective_status", "Còn hiệu lực"),
            "official_letter_scope": meta.get("official_letter_scope", "Toàn quốc"),
            "authority_path": meta.get("authority_path", "Bộ Tài chính > Tổng cục Thuế"),
        })

    return {"documents": results, "total": len(results)}


@router.get("/documents/{document_key}")
def get_legal_document_detail(document_key: str, db: Session = Depends(get_db)):
    """Fetch a single legal document with full content for the A4 viewer."""
    row = db.execute(
        text("""
            SELECT
                kd.document_key,
                kd.title,
                kd.doc_type,
                kd.authority,
                kd.effective_from,
                kd.metadata_json,
                kdv.raw_text
            FROM knowledge_documents kd
            LEFT JOIN knowledge_document_versions kdv ON kd.id = kdv.document_id
            WHERE kd.document_key = :key AND kd.status = 'active'
            LIMIT 1
        """),
        {"key": document_key},
    ).mappings().first()

    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Không tìm thấy văn bản")

    dt = (row["doc_type"] or "unknown").lower()
    display_type = DOC_TYPE_MAP.get(dt, (row["doc_type"] or "VĂN BẢN").upper())
    agency_info = _resolve_agency(row["authority"], row["doc_type"])
    meta = row["metadata_json"] or {}

    return {
        "id":          row["document_key"],
        "number":      _format_number(row["document_key"], row["doc_type"]),
        "type":        display_type,
        "title":       row["title"],
        "agency":      agency_info["agency"],
        "subAgency":   agency_info["subAgency"],
        "date":        _format_date(row["effective_from"]),
        "signerTitle": agency_info["signerTitle"],
        "signerName":  "Đã ký",
        "content":     row["raw_text"] or "(Nội dung đang được cập nhật)",
        "tags":        [display_type, "Nghiệp vụ thuế"],
        "effective_status": meta.get("effective_status", "Còn hiệu lực"),
        "official_letter_scope": meta.get("official_letter_scope", "Toàn quốc"),
        "authority_path": meta.get("authority_path", "Bộ Tài chính > Tổng cục Thuế"),
    }
