from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from . import models


UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "data" / "uploads" / "multimodal"
TAX_CODE_RE = re.compile(r"^\d{10}$")
ALLOWED_DOCUMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
ALLOWED_CSV_EXTENSIONS = {".csv"}

RISK_REQUIRED_COLUMNS = {
    "tax_code",
    "year",
    "revenue",
    "total_expenses",
    "net_profit",
    "vat_input",
    "vat_output",
}
VAT_REQUIRED_COLUMNS = {
    "seller_tax_code",
    "buyer_tax_code",
    "amount",
    "vat_rate",
    "date",
}
VAT_OPTIONAL_COLUMNS = {
    "invoice_number",
    "seller_name",
    "buyer_name",
    "seller_industry",
    "buyer_industry",
    "goods_category",
    "payment_status",
    "is_adjustment",
    "quantity",
    "unit_price",
    "item_description",
}

_fraud_pipeline = None
_gnn_engine = None
_invoice_risk_scorer = None


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.bin").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:180] or "upload.bin"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return value


def _now_utc() -> datetime:
    return datetime.utcnow()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save_upload(
    db: Session,
    *,
    content: bytes,
    filename: str,
    content_type: str | None,
    source: str,
    batch_type: str,
) -> models.AnalysisUpload:
    digest = _sha256(content)
    safe_name = _safe_filename(filename)
    target_dir = UPLOAD_ROOT / batch_type
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{digest[:16]}_{safe_name}"
    file_path = target_dir / stored_name
    file_path.write_bytes(content)

    upload = models.AnalysisUpload(
        source=source,
        batch_type=batch_type,
        original_filename=filename or safe_name,
        stored_filename=stored_name,
        file_path=str(file_path),
        content_type=content_type,
        file_size_bytes=len(content),
        sha256=digest,
        status="received",
        metadata_json={},
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def _mark_upload(
    db: Session,
    upload: models.AnalysisUpload | None,
    *,
    status: str,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if upload is None:
        return
    upload.status = status
    upload.processed_at = _now_utc()
    if metadata is not None:
        upload.metadata_json = _jsonable(metadata)
    if error:
        upload.error_message = error
    db.commit()


def _csv_headers(content: bytes) -> list[str]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return [str(h or "").strip() for h in (reader.fieldnames or [])]


def detect_csv_schema(content: bytes) -> dict[str, Any]:
    headers = _csv_headers(content)
    normalized = {h.strip().lower() for h in headers}
    risk_missing = sorted(RISK_REQUIRED_COLUMNS - normalized)
    vat_missing = sorted(VAT_REQUIRED_COLUMNS - normalized)
    detected = "unknown_csv"
    if not vat_missing:
        detected = "vat_graph_csv"
    elif not risk_missing:
        detected = "risk_scoring_csv"
    return {
        "detected_schema": detected,
        "headers": headers,
        "risk_missing": risk_missing,
        "vat_missing": vat_missing,
        "vat_optional_present": sorted(VAT_OPTIONAL_COLUMNS.intersection(normalized)),
    }


def _normalize_tax_code(value: Any) -> str:
    raw = str(value or "").strip()
    raw = re.sub(r"^(\d+)\.0+$", r"\1", raw)
    raw = re.sub(r"\D+", "", raw)
    return raw


def _parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    raw = str(value).strip()
    if raw == "":
        return default
    raw = raw.replace(" ", "")
    if "." in raw and "," in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        parts = raw.split(",")
        raw = raw.replace(",", ".") if len(parts[-1]) <= 2 else raw.replace(",", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "co", "adjustment", "credit_note"}


def _parse_date(value: Any) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def _read_csv_rows(content: bytes) -> list[dict[str, Any]]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        rows.append({str(k or "").strip().lower(): v for k, v in raw.items()})
    return rows


def _get_fraud_pipeline():
    global _fraud_pipeline
    if _fraud_pipeline is None:
        from ml_engine.pipeline import TaxFraudPipeline

        _fraud_pipeline = TaxFraudPipeline()
        _fraud_pipeline.load_models()
    return _fraud_pipeline


def _get_gnn_engine():
    global _gnn_engine
    if _gnn_engine is None:
        from ml_engine.gnn_model import GNNInference

        _gnn_engine = GNNInference()
        try:
            _gnn_engine.load()
        except Exception:
            _gnn_engine._loaded = False
            _gnn_engine._load_error = "load_failed"
    return _gnn_engine


def _get_invoice_risk_scorer():
    global _invoice_risk_scorer
    if _invoice_risk_scorer is None:
        from ml_engine.invoice_risk_model import InvoiceRiskScorer

        _invoice_risk_scorer = InvoiceRiskScorer()
    return _invoice_risk_scorer


def analyze_risk_csv_inline(db: Session, *, content: bytes, filename: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    schema = detect_csv_schema(content)
    if schema["detected_schema"] != "risk_scoring_csv":
        return {
            "status": "error",
            "analysis_type": "risk_csv",
            "filename": filename,
            "detected_schema": schema,
            "error": "CSV does not match the risk scoring contract.",
        }

    try:
        frame = pd.read_csv(io.BytesIO(content), dtype={"tax_code": "string"}, low_memory=False)
        pipeline = _get_fraud_pipeline()
        result = pipeline.predict_batch(frame)
        assessments = sorted(result.get("assessments", []), key=lambda row: float(row.get("risk_score") or 0.0), reverse=True)
        by_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for row in assessments:
            level = str(row.get("risk_level") or "low")
            by_level[level] = by_level.get(level, 0) + 1
        return {
            "status": "success",
            "analysis_type": "risk_csv",
            "filename": filename,
            "detected_schema": schema,
            "total": int(result.get("total_companies") or len(assessments)),
            "by_level": by_level,
            "top_5": _jsonable(assessments[:5]),
            "assessments": _jsonable(assessments[:50]),
            "statistics": _jsonable(result.get("statistics") or {}),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
    except Exception as exc:
        return {
            "status": "error",
            "analysis_type": "risk_csv",
            "filename": filename,
            "detected_schema": schema,
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }


def _upsert_company(db: Session, *, tax_code: str, name: str, industry: str) -> models.Company:
    company = db.query(models.Company).filter(models.Company.tax_code == tax_code).first()
    if company is None:
        company = models.Company(
            tax_code=tax_code,
            name=name or f"Company {tax_code}",
            industry=industry or "Unknown",
            is_active=True,
        )
        db.add(company)
    else:
        if name and (not company.name or company.name == tax_code):
            company.name = name
        if industry and not company.industry:
            company.industry = industry
    return company


def _edge_risk_level(score: float | None) -> str:
    score = float(score or 0.0)
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _build_vat_summary(graph_result: dict[str, Any], row_count: int, warnings: list[str]) -> dict[str, Any]:
    edges = graph_result.get("edges") if isinstance(graph_result.get("edges"), list) else []
    nodes = graph_result.get("nodes") if isinstance(graph_result.get("nodes"), list) else []
    top_edges = sorted(edges, key=lambda e: float(e.get("circular_probability") or 0.0), reverse=True)[:10]
    top_nodes = sorted(nodes, key=lambda n: float(n.get("risk_score") or 0.0), reverse=True)[:10]
    return {
        "row_count": row_count,
        "companies": len(nodes),
        "invoices": len(edges),
        "cycles": len(graph_result.get("cycles") or []),
        "total_suspicious_amount": graph_result.get("total_suspicious_amount", 0),
        "total_suspicious_invoices": graph_result.get("total_suspicious_invoices", 0),
        "model_loaded": bool(graph_result.get("model_loaded")),
        "warnings": warnings[:25],
        "top_edges": _jsonable(top_edges),
        "top_nodes": _jsonable(top_nodes),
    }


def analyze_vat_csv_upload(
    db: Session,
    *,
    content: bytes,
    filename: str,
    content_type: str | None = None,
    source: str = "vat_graph_csv",
    persist: bool = True,
    analysis_depth: str = "standard",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    schema = detect_csv_schema(content)
    if schema["detected_schema"] != "vat_graph_csv":
        raise ValueError(f"CSV does not match VAT graph contract. Missing: {schema.get('vat_missing')}")

    upload = _save_upload(
        db,
        content=content,
        filename=filename,
        content_type=content_type,
        source=source,
        batch_type="vat_graph_csv",
    )
    batch = models.VatGraphAnalysisBatch(
        upload_id=upload.id,
        filename=filename,
        detected_schema=schema["detected_schema"],
        status="processing",
        started_at=_now_utc(),
        warnings=[],
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    warnings: list[str] = []
    processed_invoices: list[dict[str, Any]] = []
    companies_by_tax_code: dict[str, dict[str, Any]] = {}

    try:
        rows = _read_csv_rows(content)
        batch.total_rows = len(rows)
        db.commit()

        if not rows:
            raise ValueError("CSV is empty.")

        for idx, row in enumerate(rows, start=1):
            seller_tc = _normalize_tax_code(row.get("seller_tax_code"))
            buyer_tc = _normalize_tax_code(row.get("buyer_tax_code"))
            invoice_date = _parse_date(row.get("date"))
            amount = _parse_float(row.get("amount"))
            vat_rate = _parse_float(row.get("vat_rate"), 10.0)
            if not TAX_CODE_RE.fullmatch(seller_tc) or not TAX_CODE_RE.fullmatch(buyer_tc):
                warnings.append(f"row {idx}: invalid seller/buyer tax code")
                continue
            if invoice_date is None:
                warnings.append(f"row {idx}: invalid invoice date")
                continue
            if amount <= 0:
                warnings.append(f"row {idx}: amount must be positive")
                continue

            invoice_number = str(row.get("invoice_number") or "").strip()
            if not invoice_number:
                row_hash = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
                invoice_number = f"VAT-{batch.id}-{idx}-{row_hash}"
            invoice_number = invoice_number[:50]

            seller_name = str(row.get("seller_name") or "").strip() or f"Company {seller_tc}"
            buyer_name = str(row.get("buyer_name") or "").strip() or f"Company {buyer_tc}"
            seller_industry = str(row.get("seller_industry") or "").strip() or "Unknown"
            buyer_industry = str(row.get("buyer_industry") or "").strip() or "Unknown"

            if persist:
                _upsert_company(db, tax_code=seller_tc, name=seller_name, industry=seller_industry)
                _upsert_company(db, tax_code=buyer_tc, name=buyer_name, industry=buyer_industry)
                db.flush()

                invoice = db.query(models.Invoice).filter(models.Invoice.invoice_number == invoice_number).first()
                if invoice is None:
                    invoice = models.Invoice(invoice_number=invoice_number)
                    db.add(invoice)
                invoice.seller_tax_code = seller_tc
                invoice.buyer_tax_code = buyer_tc
                invoice.amount = amount
                invoice.vat_rate = vat_rate
                invoice.date = invoice_date
                invoice.payment_status = str(row.get("payment_status") or "unknown").strip() or "unknown"
                invoice.goods_category = str(row.get("goods_category") or "").strip() or None
                invoice.is_adjustment = _parse_bool(row.get("is_adjustment"))
                db.flush()

                if row.get("item_description") or row.get("quantity") or row.get("unit_price"):
                    existing_line = (
                        db.query(models.InvoiceLineItem)
                        .filter(models.InvoiceLineItem.invoice_id == invoice.id)
                        .first()
                    )
                    if existing_line is None:
                        qty = _parse_float(row.get("quantity"), 1.0)
                        unit_price = _parse_float(row.get("unit_price"), amount)
                        line_amount = amount if amount > 0 else qty * unit_price
                        db.add(models.InvoiceLineItem(
                            invoice_id=invoice.id,
                            item_description=str(row.get("item_description") or "").strip() or None,
                            quantity=qty,
                            unit_price=unit_price,
                            line_amount=line_amount,
                            vat_amount=line_amount * vat_rate / 100.0,
                        ))

            companies_by_tax_code[seller_tc] = {
                "tax_code": seller_tc,
                "name": seller_name,
                "industry": seller_industry,
                "registration_date": None,
                "risk_score": 0.0,
                "is_active": True,
                "lat": 0.0,
                "lng": 0.0,
            }
            companies_by_tax_code[buyer_tc] = {
                "tax_code": buyer_tc,
                "name": buyer_name,
                "industry": buyer_industry,
                "registration_date": None,
                "risk_score": 0.0,
                "is_active": True,
                "lat": 0.0,
                "lng": 0.0,
            }
            processed_invoices.append({
                "invoice_number": invoice_number,
                "seller_tax_code": seller_tc,
                "buyer_tax_code": buyer_tc,
                "amount": amount,
                "vat_rate": vat_rate,
                "date": invoice_date.isoformat(),
                "payment_status": str(row.get("payment_status") or "unknown").strip() or "unknown",
                "goods_category": str(row.get("goods_category") or "").strip(),
                "is_adjustment": _parse_bool(row.get("is_adjustment")),
            })

        if not processed_invoices:
            raise ValueError("No valid VAT invoice rows were found.")

        companies = list(companies_by_tax_code.values())
        engine = _get_gnn_engine()
        graph_result = engine.predict(companies, processed_invoices, [])
        graph_result["batch_id"] = batch.id
        graph_result["analysis_depth"] = analysis_depth
        graph_result["detected_schema"] = schema

        edge_map = {
            str(edge.get("invoice_number") or ""): edge
            for edge in graph_result.get("edges", [])
            if isinstance(edge, dict)
        }
        scorer = _get_invoice_risk_scorer()
        top_invoice_risks: list[dict[str, Any]] = []
        for inv in processed_invoices:
            edge = edge_map.get(inv["invoice_number"], {})
            edge_score = round(float(edge.get("circular_probability") or 0.0) * 100.0, 2)
            risk_result = scorer.score(inv, {
                "same_day_pair_count": sum(
                    1 for other in processed_invoices
                    if other["seller_tax_code"] == inv["seller_tax_code"] and other["date"] == inv["date"]
                ),
                "linked_invoice_ids": [inv["invoice_number"]],
            })
            combined_score = max(edge_score, float(risk_result.risk_score or 0.0))
            level = _edge_risk_level(combined_score)
            signals = {
                "edge": edge,
                "invoice_risk": asdict(risk_result),
            }
            db.add(models.VatGraphBatchResult(
                batch_id=batch.id,
                invoice_number=inv["invoice_number"],
                seller_tax_code=inv["seller_tax_code"],
                buyer_tax_code=inv["buyer_tax_code"],
                amount=inv["amount"],
                vat_rate=inv["vat_rate"],
                invoice_date=_parse_date(inv["date"]) or date.today(),
                edge_risk_score=combined_score,
                edge_risk_level=level,
                signals=_jsonable(signals),
            ))
            if persist:
                db.add(models.InvoiceRiskPrediction(
                    invoice_number=inv["invoice_number"],
                    as_of_date=date.today(),
                    model_version=risk_result.model_version,
                    risk_score=float(risk_result.risk_score),
                    risk_level=risk_result.risk_level,
                    reason_codes=risk_result.reason_codes,
                    explanations=_jsonable(risk_result.explanations),
                    linked_invoice_ids=risk_result.linked_invoice_ids,
                ))
            top_invoice_risks.append({
                "invoice_number": inv["invoice_number"],
                "seller_tax_code": inv["seller_tax_code"],
                "buyer_tax_code": inv["buyer_tax_code"],
                "amount": inv["amount"],
                "edge_risk_score": combined_score,
                "risk_level": level,
                "reason_codes": risk_result.reason_codes,
            })

        top_invoice_risks.sort(key=lambda row: float(row.get("edge_risk_score") or 0.0), reverse=True)
        graph_result["top_invoice_risks"] = top_invoice_risks[:20]
        summary = _build_vat_summary(graph_result, len(processed_invoices), warnings)

        batch.status = "done"
        batch.processed_rows = len(processed_invoices)
        batch.warnings = warnings
        batch.result_summary = _jsonable(summary)
        batch.result_json = _jsonable(graph_result)
        batch.completed_at = _now_utc()
        _mark_upload(db, upload, status="completed", metadata={"batch_id": batch.id, "summary": summary})
        db.commit()

        return {
            "status": "done",
            "analysis_type": "vat_graph_csv",
            "batch_id": batch.id,
            "upload_id": upload.id,
            "filename": filename,
            "detected_schema": schema,
            "row_count": len(rows),
            "processed_rows": len(processed_invoices),
            "warnings": warnings[:25],
            "summary": summary,
            "graph": _jsonable(graph_result),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
    except Exception as exc:
        batch.status = "failed"
        batch.error_message = str(exc)
        batch.completed_at = _now_utc()
        batch.warnings = warnings
        _mark_upload(db, upload, status="failed", metadata={"batch_id": batch.id, "warnings": warnings}, error=str(exc))
        db.commit()
        raise


def get_vat_batch_status(db: Session, batch_id: int) -> dict[str, Any]:
    batch = db.query(models.VatGraphAnalysisBatch).filter(models.VatGraphAnalysisBatch.id == batch_id).first()
    if batch is None:
        raise LookupError("VAT graph batch not found.")
    progress = 0.0
    if batch.total_rows:
        progress = round(float(batch.processed_rows or 0) / float(batch.total_rows) * 100.0, 1)
    return {
        "batch_id": batch.id,
        "upload_id": batch.upload_id,
        "filename": batch.filename,
        "detected_schema": batch.detected_schema,
        "status": batch.status,
        "total_rows": batch.total_rows,
        "processed_rows": batch.processed_rows,
        "progress": progress,
        "warnings": batch.warnings or [],
        "error_message": batch.error_message,
        "created_at": _jsonable(batch.created_at),
        "completed_at": _jsonable(batch.completed_at),
    }


def get_vat_batch_results(db: Session, batch_id: int) -> dict[str, Any]:
    batch = db.query(models.VatGraphAnalysisBatch).filter(models.VatGraphAnalysisBatch.id == batch_id).first()
    if batch is None:
        raise LookupError("VAT graph batch not found.")
    rows = (
        db.query(models.VatGraphBatchResult)
        .filter(models.VatGraphBatchResult.batch_id == batch_id)
        .order_by(models.VatGraphBatchResult.edge_risk_score.desc().nullslast())
        .limit(500)
        .all()
    )
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "summary": _jsonable(batch.result_summary or {}),
        "graph": _jsonable(batch.result_json or {}),
        "results": [
            {
                "invoice_number": r.invoice_number,
                "seller_tax_code": r.seller_tax_code,
                "buyer_tax_code": r.buyer_tax_code,
                "amount": float(r.amount or 0.0),
                "vat_rate": float(r.vat_rate or 0.0),
                "invoice_date": _jsonable(r.invoice_date),
                "edge_risk_score": r.edge_risk_score,
                "edge_risk_level": r.edge_risk_level,
                "signals": r.signals or {},
            }
            for r in rows
        ],
    }


def _document_confidence(payload: dict[str, Any]) -> float:
    fields = payload.get("invoice_fields") or {}
    field_conf = _parse_float(fields.get("confidence"), 0.0)
    ocr_results = payload.get("ocr_results") or []
    page_scores = [_parse_float(page.get("confidence"), 0.0) for page in ocr_results if isinstance(page, dict)]
    if field_conf > 0:
        return round(field_conf, 4)
    if page_scores:
        return round(sum(page_scores) / len(page_scores), 4)
    return 0.0


def _invoice_from_ocr_fields(fields: dict[str, Any]) -> dict[str, Any]:
    amount = _parse_float(fields.get("grand_total"), 0.0) or _parse_float(fields.get("total_amount"), 0.0)
    return {
        "invoice_number": str(fields.get("invoice_number") or f"OCR-{uuid.uuid4().hex[:10]}"),
        "seller_tax_code": _normalize_tax_code(fields.get("seller_tax_code")),
        "buyer_tax_code": _normalize_tax_code(fields.get("buyer_tax_code")),
        "amount": amount,
        "vat_rate": _parse_float(fields.get("vat_rate"), 10.0),
        "payment_status": "unknown",
        "is_adjustment": False,
    }


def _graph_linkage_candidates(db: Session, fields: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for role, code in (("seller", fields.get("seller_tax_code")), ("buyer", fields.get("buyer_tax_code"))):
        tax_code = _normalize_tax_code(code)
        if not TAX_CODE_RE.fullmatch(tax_code):
            continue
        company = db.query(models.Company).filter(models.Company.tax_code == tax_code).first()
        candidates.append({
            "role": role,
            "tax_code": tax_code,
            "exists_in_graph": company is not None,
            "company_name": company.name if company else None,
            "industry": company.industry if company else None,
            "risk_score": float(company.risk_score or 0.0) if company else None,
        })
    return candidates


def analyze_invoice_document_upload(
    db: Session,
    *,
    content: bytes,
    filename: str,
    content_type: str | None = None,
    source: str = "ocr_invoice",
) -> dict[str, Any]:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise ValueError("Only PNG, JPG, JPEG, and PDF files are supported for OCR.")

    t0 = time.perf_counter()
    upload = _save_upload(
        db,
        content=content,
        filename=filename,
        content_type=content_type,
        source=source,
        batch_type="ocr_invoice",
    )

    try:
        from dataclasses import asdict as dc_asdict
        from ml_engine.document_ocr_engine import get_ocr_engine, DocumentResult as _DocResult

        doc_result = get_ocr_engine().process_bytes(content, filename=filename)
        # DocumentResult is a dataclass – convert to plain dict safely
        if is_dataclass(doc_result):
            payload = _jsonable(dc_asdict(doc_result))
        elif isinstance(doc_result, dict):
            payload = _jsonable(doc_result)
        else:
            payload = _jsonable({"status": "error", "errors": ["Unknown result type"]})

        fields = payload.get("invoice_fields") or {}
        confidence = _document_confidence(payload)
        invoice_payload = _invoice_from_ocr_fields(fields)
        risk = _get_invoice_risk_scorer().score(invoice_payload, {})
        linkage = _graph_linkage_candidates(db, fields)
        warnings: list[str] = []
        if confidence < 0.5:
            warnings.append("low_ocr_confidence")
        if not invoice_payload.get("seller_tax_code") or not invoice_payload.get("buyer_tax_code"):
            warnings.append("missing_counterparty_tax_code")

        # Determine OCR backend used
        ocr_backend = payload.get("_ocr_backend", "DocumentOCREngine")

        response = {
            "status": payload.get("status") or "success",
            "analysis_type": "ocr_invoice",
            "upload_id": upload.id,
            "filename": filename,
            "file_size_bytes": len(content),
            "processing_time_ms": payload.get("total_processing_ms"),
            "confidence": confidence,
            "engine": ocr_backend,
            "full_text": payload.get("full_text", ""),
            "text_preview": str(payload.get("full_text") or "")[:1200],
            "extracted_fields": fields,
            "tables": payload.get("tables") or [],
            "tables_ai_detected": sum(
                1 for t in (payload.get("tables") or [])
                if t.get("extraction_method") == "table_transformer"
            ),
            "table_extraction_method": payload.get("table_extraction_method", "none"),
            "ocr_pages": payload.get("ocr_results") or [],
            "invoice_risk": _jsonable(asdict(risk)),
            "graph_linkage_candidates": linkage,
            "warnings": warnings + list(payload.get("errors") or []),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        _mark_upload(db, upload, status="completed", metadata=response)
        return response
    except Exception as exc:
        _mark_upload(db, upload, status="failed", metadata={"filename": filename}, error=str(exc))
        raise


def analyze_attachment_for_agent(
    db: Session,
    *,
    content: bytes,
    filename: str,
    content_type: str | None,
    model_mode: str,
) -> dict[str, Any]:
    ext = Path(filename or "").suffix.lower()
    if ext in ALLOWED_CSV_EXTENSIONS:
        schema = detect_csv_schema(content)
        mode = str(model_mode or "").lower()
        if schema["detected_schema"] == "vat_graph_csv":
            return analyze_vat_csv_upload(
                db,
                content=content,
                filename=filename,
                content_type=content_type,
                source="agent_attachment",
                persist=True,
            )
        if schema["detected_schema"] == "risk_scoring_csv":
            return analyze_risk_csv_inline(db, content=content, filename=filename)
        if mode == "vat":
            return analyze_vat_csv_upload(
                db,
                content=content,
                filename=filename,
                content_type=content_type,
                source="agent_attachment",
                persist=True,
            )
        return analyze_risk_csv_inline(db, content=content, filename=filename)
    if ext in ALLOWED_DOCUMENT_EXTENSIONS:
        return analyze_invoice_document_upload(
            db,
            content=content,
            filename=filename,
            content_type=content_type,
            source="agent_attachment",
        )
    raise ValueError("Unsupported attachment type. Use CSV, PNG, JPG, JPEG, or PDF.")
