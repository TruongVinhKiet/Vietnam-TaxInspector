"""Audit and expand the legal knowledge base for TaxInspector GraphRAG.

The script first inspects the running database if it is reachable. When the DB
is not available in a dev machine, it falls back to repository seed scripts and
the citizen legal snippet corpus. It then writes an expansion plan and, with
``--ingest``, can insert curated local documents through the existing ingestor.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
for _path in (BACKEND_DIR, REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


SEED_PATHS = [
    BACKEND_DIR / "app" / "scripts" / "seed_minimal_tax_knowledge.py",
    BACKEND_DIR / "app" / "scripts" / "seed_real_tax_knowledge.py",
    BACKEND_DIR / "app" / "scripts" / "seed_complex_tax_knowledge.py",
    BACKEND_DIR / "scripts" / "ingest_missing_legal_docs.py",
]


@dataclass
class CuratedLegalDocument:
    key: str
    title: str
    doc_type: str
    authority: str
    authority_rank: int
    effective_from: str
    topic: str
    content: str
    relations: list[dict[str, str]]


CURATED_EXPANSION_DOCS: list[CuratedLegalDocument] = [
    CuratedLegalDocument(
        key="ND_123_2020_NDCP_EINVOICE_DETAIL",
        title="Nghi dinh 123/2020/ND-CP - hoa don, chung tu dien tu",
        doc_type="decree",
        authority="Chinh phu",
        authority_rank=60,
        effective_from="2022-07-01",
        topic="einvoice",
        content=(
            "Nghi dinh 123/2020/ND-CP quy dinh ve hoa don, chung tu. Noi dung can duoc "
            "uu tien trong RAG gom thoi diem lap hoa don, xu ly hoa don sai sot, hoa don "
            "thay the/dieu chinh, hoa don dien tu co ma cua co quan thue va trach nhiem "
            "luu tru chung tu phuc vu kiem tra sau nay."
        ),
        relations=[{"target": "VB_78_2021_TT_BTC", "type": "implemented_by"}],
    ),
    CuratedLegalDocument(
        key="TT_78_2021_TT_BTC_EINVOICE_GUIDE",
        title="Thong tu 78/2021/TT-BTC - huong dan hoa don dien tu",
        doc_type="circular",
        authority="Bo Tai chinh",
        authority_rank=50,
        effective_from="2021-07-01",
        topic="einvoice",
        content=(
            "Thong tu 78/2021/TT-BTC huong dan trien khai hoa don dien tu, bao gom ky hieu "
            "mau so hoa don, xu ly sai sot, chuyen du lieu hoa don den co quan thue va cach "
            "lap hoa don dieu chinh hoac thay the khi thong tin giao dich khong dung."
        ),
        relations=[{"target": "ND_123_2020_NDCP_EINVOICE_DETAIL", "type": "implements"}],
    ),
    CuratedLegalDocument(
        key="ND_125_2020_NDCP_PENALTY_FAQ",
        title="Nghi dinh 125/2020/ND-CP - xu phat vi pham ve thue va hoa don",
        doc_type="decree",
        authority="Chinh phu",
        authority_rank=60,
        effective_from="2020-12-05",
        topic="penalty",
        content=(
            "Nghi dinh 125/2020/ND-CP la can cu chinh cho cac cau hoi ve nop to khai tre, "
            "khai sai, cham nop tien thue, su dung hoa don khong hop phap va tinh tiet giam "
            "nhe khi nguoi nop thue tu phat hien, tu khac phuc truoc khi bi kiem tra."
        ),
        relations=[{"target": "LUAT_38_2019", "type": "implements"}],
    ),
    CuratedLegalDocument(
        key="TT_40_2021_TT_BTC_HOUSEHOLD",
        title="Thong tu 40/2021/TT-BTC - thue ho, ca nhan kinh doanh",
        doc_type="circular",
        authority="Bo Tai chinh",
        authority_rank=50,
        effective_from="2021-08-01",
        topic="household_business",
        content=(
            "Thong tu 40/2021/TT-BTC huong dan VAT, TNCN va quan ly thue doi voi ho kinh "
            "doanh, ca nhan kinh doanh, cho thue tai san, ban hang qua san thuong mai dien tu "
            "va phuong phap khai thue theo tung lan phat sinh hoac theo ky."
        ),
        relations=[{"target": "LUAT_38_2019", "type": "implements"}],
    ),
    CuratedLegalDocument(
        key="TT_100_2021_TT_BTC_PLATFORM",
        title="Thong tu 100/2021/TT-BTC - sua doi quy dinh san TMĐT",
        doc_type="circular",
        authority="Bo Tai chinh",
        authority_rank=50,
        effective_from="2022-01-01",
        topic="ecommerce",
        content=(
            "Thong tu 100/2021/TT-BTC bo sung nghia vu cung cap thong tin va co che khai "
            "thue thay, nop thue thay trong mot so truong hop ca nhan kinh doanh qua san "
            "thuong mai dien tu, dong thoi lam ro nguong doanh thu cho thue tai san."
        ),
        relations=[{"target": "TT_40_2021_TT_BTC_HOUSEHOLD", "type": "amends"}],
    ),
    CuratedLegalDocument(
        key="NQ_954_2020_UBTVQH14_DEPENDENT",
        title="Nghi quyet 954/2020/UBTVQH14 - muc giam tru gia canh",
        doc_type="resolution",
        authority="Uy ban Thuong vu Quoc hoi",
        authority_rank=75,
        effective_from="2020-07-01",
        topic="pit",
        content=(
            "Nghi quyet 954/2020/UBTVQH14 quy dinh muc giam tru ban than 11 trieu dong/thang "
            "va giam tru nguoi phu thuoc 4,4 trieu dong/thang, la can cu thuong gap khi tu "
            "van quyet toan thue TNCN, luong thang 13 va hoan thue."
        ),
        relations=[{"target": "TT_111_2013_TT_BTC", "type": "supplements"}],
    ),
    CuratedLegalDocument(
        key="TT_111_2013_TT_BTC_PIT_SALARY",
        title="Thong tu 111/2013/TT-BTC - thue TNCN tu tien luong",
        doc_type="circular",
        authority="Bo Tai chinh",
        authority_rank=50,
        effective_from="2013-10-01",
        topic="pit",
        content=(
            "Thong tu 111/2013/TT-BTC huong dan bieu thue luy tien tung phan, thu nhap tinh "
            "thue, giam tru gia canh, khau tru 10% voi hop dong duoi 3 thang va chung tu "
            "khau tru thue TNCN."
        ),
        relations=[{"target": "NQ_954_2020_UBTVQH14_DEPENDENT", "type": "related_to"}],
    ),
    CuratedLegalDocument(
        key="TT_96_2015_TT_BTC_DEDUCTIBLE_COSTS",
        title="Thong tu 96/2015/TT-BTC - chi phi duoc tru va thanh toan khong dung tien mat",
        doc_type="circular",
        authority="Bo Tai chinh",
        authority_rank=50,
        effective_from="2015-08-06",
        topic="cit",
        content=(
            "Thong tu 96/2015/TT-BTC huong dan chi phi duoc tru khi tinh thue TNDN, dieu kien "
            "hoa don chung tu hop phap va thanh toan khong dung tien mat voi hoa don tu "
            "nguong quy dinh, thuong dung cho cau hoi mua hang tien mat tren 20 trieu."
        ),
        relations=[{"target": "TT_78_2014_TT_BTC", "type": "amends"}],
    ),
    CuratedLegalDocument(
        key="TT_103_2014_TT_BTC_FCT",
        title="Thong tu 103/2014/TT-BTC - thue nha thau nuoc ngoai",
        doc_type="circular",
        authority="Bo Tai chinh",
        authority_rank=50,
        effective_from="2014-10-01",
        topic="fct",
        content=(
            "Thong tu 103/2014/TT-BTC huong dan thue nha thau nuoc ngoai, nghia vu khau tru "
            "nop thay VAT va TNDN/TNCN khi ben Viet Nam thanh toan dich vu cho to chuc, ca "
            "nhan nuoc ngoai khong co co so thuong tru tai Viet Nam."
        ),
        relations=[{"target": "LUAT_38_2019", "type": "related_to"}],
    ),
    CuratedLegalDocument(
        key="ND_132_2020_NDCP_TRANSFER_PRICING",
        title="Nghi dinh 132/2020/ND-CP - giao dich lien ket",
        doc_type="decree",
        authority="Chinh phu",
        authority_rank=60,
        effective_from="2020-12-20",
        topic="transfer_pricing",
        content=(
            "Nghi dinh 132/2020/ND-CP quy dinh ve quan ly thue doi voi doanh nghiep co giao "
            "dich lien ket, ho so xac dinh gia giao dich lien ket, nguyen tac doc lap va tran "
            "chi phi lai vay thuan duoc tru."
        ),
        relations=[{"target": "LUAT_38_2019", "type": "implements"}],
    ),
    CuratedLegalDocument(
        key="ND_64_2024_NDCP_TAX_EXTENSION",
        title="Nghi dinh 64/2024/ND-CP - gia han nop thue nam 2024",
        doc_type="decree",
        authority="Chinh phu",
        authority_rank=60,
        effective_from="2024-06-17",
        topic="extension",
        content=(
            "Nghi dinh 64/2024/ND-CP quy dinh gia han thoi han nop thue GTGT, TNDN tam nop, "
            "TNCN cua ho kinh doanh va tien thue dat cho cac doi tuong bi anh huong, khong "
            "dong nghia voi mien thue."
        ),
        relations=[{"target": "LUAT_38_2019", "type": "implements"}],
    ),
    CuratedLegalDocument(
        key="LUAT_GTGT_48_2024_QH15_TRANSITION",
        title="Luat Thue GTGT 48/2024/QH15 - diem moi tu 2025",
        doc_type="law",
        authority="Quoc hoi",
        authority_rank=90,
        effective_from="2025-07-01",
        topic="vat_2025",
        content=(
            "Luat Thue GTGT 48/2024/QH15 co hieu luc tu 01/07/2025. Khi tu van can tach "
            "ro quy dinh dang ap dung truoc ngay hieu luc va quy dinh moi sau ngay hieu luc, "
            "dac biet voi nguong doanh thu ho, ca nhan kinh doanh va nhom hang hoa dich vu."
        ),
        relations=[{"target": "LUAT_GTGT_13_2008", "type": "replaces"}],
    ),
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _db_audit() -> dict[str, Any]:
    try:
        from sqlalchemy import text
        from app.database import SessionLocal
    except Exception as exc:
        return {"available": False, "error": f"import_failed: {exc}"}

    try:
        db = SessionLocal()
        try:
            docs = db.execute(
                text(
                    """
                    SELECT doc_type, count(*) AS n
                    FROM knowledge_documents
                    GROUP BY doc_type
                    ORDER BY doc_type
                    """
                )
            ).mappings().all()
            chunks = db.execute(text("SELECT count(*) AS n FROM knowledge_chunks")).scalar() or 0
            entities = db.execute(text("SELECT count(*) AS n FROM kg_entities")).scalar() or 0
            relations = db.execute(text("SELECT count(*) AS n FROM kg_relations")).scalar() or 0
            recent = db.execute(
                text(
                    """
                    SELECT document_key, title, doc_type
                    FROM knowledge_documents
                    ORDER BY id DESC
                    LIMIT 12
                    """
                )
            ).mappings().all()
            return {
                "available": True,
                "documents_by_type": {str(r["doc_type"]): int(r["n"]) for r in docs},
                "document_count": int(sum(int(r["n"]) for r in docs)),
                "chunk_count": int(chunks),
                "kg_entity_count": int(entities),
                "kg_relation_count": int(relations),
                "recent_documents": [dict(r) for r in recent],
            }
        finally:
            db.close()
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _seed_audit() -> dict[str, Any]:
    documents = []
    for path in SEED_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        keys = re.findall(r"['\"](?:key|document_key)['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
        titles = re.findall(r"['\"]title['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
        documents.append(
            {
                "path": str(path.relative_to(REPO_DIR)),
                "key_count": len(set(keys)),
                "sample_keys": sorted(set(keys))[:12],
                "title_count": len(titles),
            }
        )
    try:
        from ml_engine.tax_agent_citizen_legal import SNIPPETS

        snippet_count = len(SNIPPETS)
        snippet_topics = sorted({s.key for s in SNIPPETS})[:30]
    except Exception:
        snippet_count = 0
        snippet_topics = []
    return {
        "seed_files": documents,
        "seed_document_count_estimate": int(sum(item["key_count"] for item in documents)),
        "citizen_snippet_count": snippet_count,
        "citizen_snippet_sample": snippet_topics,
    }


def audit_legal_knowledge_base() -> dict[str, Any]:
    db = _db_audit()
    seeds = _seed_audit()
    db_docs = int(db.get("document_count") or 0)
    db_chunks = int(db.get("chunk_count") or 0)
    seed_docs = int(seeds.get("seed_document_count_estimate") or 0)
    coverage_score = min(1.0, 0.5 * min(1.0, max(db_docs, seed_docs) / 35.0) + 0.5 * min(1.0, max(db_chunks, seed_docs * 8) / 300.0))
    if not db.get("available"):
        recommendation = "database_unavailable_use_seed_and_curated_expansion"
    elif db_docs < 24 or db_chunks < 180:
        recommendation = "ingest_curated_expansion_before_reranker_tuning"
    else:
        recommendation = "kb_sufficient_prioritize_reranker_and_grounding_eval"
    return {
        "generated_at": _now_iso(),
        "database": db,
        "repository_seeds": seeds,
        "coverage_score": round(float(coverage_score), 4),
        "recommended_path": recommendation,
        "curated_expansion_count": len(CURATED_EXPANSION_DOCS),
        "curated_topics": sorted({doc.topic for doc in CURATED_EXPANSION_DOCS}),
    }


def write_audit_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "legal_kb_audit.json"
    md_path = out_dir / "legal_kb_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    db = report["database"]
    seeds = report["repository_seeds"]
    lines = [
        "# Legal Knowledge Base Audit",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- DB available: `{db.get('available')}`",
        f"- DB documents: `{db.get('document_count', 0)}`",
        f"- DB chunks: `{db.get('chunk_count', 0)}`",
        f"- KG entities/relations: `{db.get('kg_entity_count', 0)}` / `{db.get('kg_relation_count', 0)}`",
        f"- Repository seed estimate: `{seeds.get('seed_document_count_estimate', 0)}`",
        f"- Citizen snippets: `{seeds.get('citizen_snippet_count', 0)}`",
        f"- Coverage score: `{report['coverage_score']}`",
        f"- Recommended path: `{report['recommended_path']}`",
        "",
        "## Curated Expansion Topics",
        "",
        *[f"- `{topic}`" for topic in report["curated_topics"]],
    ]
    if not db.get("available"):
        lines.extend(["", "## DB Error", "", f"```text\n{db.get('error')}\n```"])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def export_curated_documents(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "legal_kb_curated_expansion.json"
    path.write_text(
        json.dumps([asdict(doc) for doc in CURATED_EXPANSION_DOCS], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def ingest_curated_documents() -> dict[str, Any]:
    try:
        from app.scripts.ingest_tax_knowledge import ingest_document
    except Exception as exc:
        return {"status": "error", "error": f"ingestor_import_failed: {exc}", "ingested": 0}

    ingested = 0
    errors = []
    for doc in CURATED_EXPANSION_DOCS:
        try:
            ingest_document(
                document_key=doc.key,
                title=doc.title,
                doc_type=doc.doc_type,
                authority=doc.authority,
                source_uri=f"local://curated/{doc.key}",
                version_tag="curated-v1",
                content=doc.content,
                metadata={
                    "authority_rank": doc.authority_rank,
                    "effective_from": doc.effective_from,
                    "topic": doc.topic,
                    "relations": doc.relations,
                    "curated": True,
                },
            )
            ingested += 1
        except Exception as exc:
            errors.append({"key": doc.key, "error": str(exc)})
    return {"status": "ok" if not errors else "partial", "ingested": ingested, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and optionally expand the TaxInspector legal KB")
    parser.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "reports")
    parser.add_argument("--ingest", action="store_true", help="Insert curated expansion docs into the configured DB")
    args = parser.parse_args()

    report = audit_legal_knowledge_base()
    json_path, md_path = write_audit_report(report, args.out_dir)
    curated_path = export_curated_documents(args.out_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] wrote {curated_path}")
    if args.ingest:
        ingest_result = ingest_curated_documents()
        ingest_path = args.out_dir / "legal_kb_ingest_result.json"
        ingest_path.write_text(json.dumps(ingest_result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {ingest_path}")
    print(json.dumps({"recommended_path": report["recommended_path"], "coverage_score": report["coverage_score"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
