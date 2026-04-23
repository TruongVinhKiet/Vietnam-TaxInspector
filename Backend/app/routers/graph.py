"""
graph.py – VAT Network Graph API (GNN-powered)
================================================
Endpoints:
    GET  /api/graph?tax_code=...  → Full subgraph analysis with GNN inference
    GET  /api/graph/search?q=...  → Search companies for graph exploration
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Any
import os
import hashlib
import json
import re
from datetime import datetime, timezone

from ..database import get_db
from ..observability import get_structured_logger, log_event
from . import monitoring as monitoring_router

router = APIRouter(prefix="/api", tags=["VAT Invoice Graph (GNN)"])
logger = get_structured_logger("taxinspector.graph")
TAX_CODE_PATTERN = re.compile(r"^\d{10}$")

# Lazy-loaded GNN inference engine
_gnn_engine = None

SUBGRAPH_MAX_NODES = 150
SUBGRAPH_INVOICE_LIMIT = 500
FULL_GRAPH_COMPANY_LIMIT = 200
OWNERSHIP_LINK_QUERY_LIMIT = 500
GRAPH_COMPAT_SCHEMA_VERSION = "graph-forensic-compat-v2"
GRAPH_CONTEXT_SOURCE = "graph_forensic_shared"
GRAPH_FORENSIC_FEATURE_FLAG = os.getenv("GRAPH_FORENSIC_COMPAT_V2", "1").strip().lower() not in {"0", "false", "off", "no"}
COUNTRY_CONFIDENCE_MIN = 0.55
HIGH_RISK_COUNTRY_KEYS = {
    "cayman islands",
    "british virgin islands",
    "bvi",
    "panama",
    "seychelles",
}


def _ensure_numeric_tax_code(tax_code: Optional[str]) -> Optional[str]:
    if tax_code is None:
        return None
    normalized = str(tax_code).strip()
    if not TAX_CODE_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=422,
            detail="tax_code phải gồm đúng 10 chữ số (ví dụ: 0101234567).",
        )
    return normalized


def _table_exists(db: Session, table_name: str) -> bool:
    try:
        exists = db.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = :table_name
                )
                """
            ),
            {"table_name": table_name},
        ).scalar()
        return bool(exists)
    except Exception:
        db.rollback()
        return False


def _collect_seed_tax_codes_from_ownership(db: Session, tax_code: str, limit: int = 40) -> list[str]:
    """
    Return VAT-graph seed tax codes connected via ownership relations.

    This bridges offshore centers to domestic entities so graph analysis does not
    collapse to a single isolated offshore proxy node.
    """
    if limit <= 0:
        return []

    identifiers: list[str] = [tax_code]
    if _table_exists(db, "offshore_entities"):
        try:
            rows = db.execute(
                text(
                    """
                    SELECT entity_code, proxy_tax_code
                    FROM offshore_entities
                    WHERE proxy_tax_code = :tax_code OR entity_code = :tax_code
                    LIMIT 20
                    """
                ),
                {"tax_code": tax_code},
            ).fetchall()
            for row in rows:
                for raw in row:
                    value = str(raw or "").strip()
                    if value and value not in identifiers:
                        identifiers.append(value)
        except Exception:
            db.rollback()

    results: list[str] = []
    seen: set[str] = set()
    per_identifier_limit = max(1, min(20, int(limit)))

    for identifier in identifiers:
        try:
            outward_rows = db.execute(
                text(
                    """
                    SELECT DISTINCT child_tax_code
                    FROM ownership_links
                    WHERE parent_tax_code = :identifier
                      AND child_tax_code ~ '^\\d{10}$'
                    LIMIT :limit
                    """
                ),
                {"identifier": identifier, "limit": per_identifier_limit},
            ).fetchall()
            inward_rows = db.execute(
                text(
                    """
                    SELECT DISTINCT parent_tax_code
                    FROM ownership_links
                    WHERE child_tax_code = :identifier
                      AND parent_tax_code ~ '^\\d{10}$'
                    LIMIT :limit
                    """
                ),
                {"identifier": identifier, "limit": per_identifier_limit},
            ).fetchall()
        except Exception:
            db.rollback()
            continue

        for row in [*outward_rows, *inward_rows]:
            code = str(row[0] or "").strip() if row else ""
            if not TAX_CODE_PATTERN.fullmatch(code):
                continue
            if code == tax_code or code in seen:
                continue

            seen.add(code)
            results.append(code)
            if len(results) >= limit:
                return results

    return results


def _get_gnn_engine():
    """Lazy-load the GNN inference engine (singleton)."""
    global _gnn_engine
    if _gnn_engine is None:
        from ml_engine.gnn_model import GNNInference
        _gnn_engine = GNNInference()
        try:
            _gnn_engine.load()
            if _gnn_engine._loaded:
                _gnn_engine._load_error = None
                log_event(logger, "info", "graph_gnn_engine_ready", model_loaded=True)
            else:
                _gnn_engine._load_error = "model_artifacts_missing"
                log_event(
                    logger,
                    "warning",
                    "graph_gnn_engine_fallback",
                    reason="model_artifacts_missing",
                )
        except (FileNotFoundError, OSError) as e:
            _gnn_engine._loaded = False
            _gnn_engine._load_error = "artifact_io_error"
            log_event(
                logger,
                "error",
                "graph_gnn_engine_load_failed",
                error_type=type(e).__name__,
                reason="artifact_io_error",
                error=str(e),
            )
        except RuntimeError as e:
            _gnn_engine._loaded = False
            _gnn_engine._load_error = "runtime_error"
            log_event(
                logger,
                "error",
                "graph_gnn_engine_load_failed",
                error_type=type(e).__name__,
                reason="runtime_error",
                error=str(e),
            )
        except Exception as e:
            _gnn_engine._loaded = False
            _gnn_engine._load_error = "initialization_error"
            log_event(
                logger,
                "error",
                "graph_gnn_engine_load_failed",
                error_type=type(e).__name__,
                reason="initialization_error",
                error=str(e),
            )
    return _gnn_engine


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_country(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    aliases = {
        "vn": "Vietnam",
        "viet nam": "Vietnam",
        "vietnam": "Vietnam",
        "việt nam": "Vietnam",
        "hong kong sar": "Hong Kong",
        "cayman": "Cayman Islands",
        "british virgin islands (bvi)": "British Virgin Islands",
    }
    return aliases.get(lowered, raw)


def _is_high_risk_country(country: str) -> bool:
    return str(country or "").strip().lower() in HIGH_RISK_COUNTRY_KEYS


def _extract_company_geo_profiles(companies: list[dict]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for company in companies:
        if not isinstance(company, dict):
            continue
        tax_code = str(company.get("tax_code", "")).strip()
        if not tax_code:
            continue

        country = _normalize_country(company.get("country_inferred"))
        is_within_vietnam_raw = company.get("is_within_vietnam")
        is_within_vietnam = bool(is_within_vietnam_raw) if is_within_vietnam_raw is not None else None
        if not country and is_within_vietnam is True:
            country = "Vietnam"

        confidence = max(0.0, min(1.0, _to_float(company.get("confidence_country"), 0.0)))
        method = str(company.get("geocoding_method") or "").strip().lower() or "unknown"
        lat = _to_float(company.get("lat"), 0.0)
        lng = _to_float(company.get("lng"), 0.0)
        has_geom = not (abs(lat) < 1e-9 and abs(lng) < 1e-9)

        profiles[tax_code] = {
            "tax_code": tax_code,
            "country": country,
            "confidence": confidence,
            "is_within_vietnam": is_within_vietnam,
            "geocoding_method": method,
            "has_geom": has_geom,
            "lat": lat,
            "lng": lng,
            "industry": str(company.get("industry") or "").strip(),
        }
    return profiles


def _build_cross_border_signals(
    companies: list[dict],
    invoices: list[dict],
    ownership_links: Optional[list[dict]] = None,
    rings: Optional[list[dict]] = None,
) -> dict[str, Any]:
    profiles = _extract_company_geo_profiles(companies)
    total_companies = len(companies)
    with_country = 0
    unknown_country = 0
    low_confidence = 0
    invalid_or_missing_geom = 0
    within_vietnam = 0
    foreign_companies = 0
    offshore_proxy_companies = 0
    country_company_counts: dict[str, int] = {}

    for profile in profiles.values():
        tax_code = str(profile.get("tax_code") or "")
        country = str(profile.get("country") or "")
        confidence = _to_float(profile.get("confidence"), 0.0)
        method = str(profile.get("geocoding_method") or "")
        is_within_vietnam = profile.get("is_within_vietnam") is True
        industry = str(profile.get("industry") or "")

        if tax_code.startswith("99") or industry.lower() == "offshore entity":
            offshore_proxy_companies += 1

        if country:
            with_country += 1
            country_company_counts[country] = country_company_counts.get(country, 0) + 1
            if country != "Vietnam":
                foreign_companies += 1
        else:
            unknown_country += 1

        if confidence < COUNTRY_CONFIDENCE_MIN:
            low_confidence += 1
        if method in {"missing_geom", "invalid_geom", "unknown"} or profile.get("has_geom") is False:
            invalid_or_missing_geom += 1
        if is_within_vietnam:
            within_vietnam += 1

    cross_border_invoice_count = 0
    invoice_country_coverage_count = 0
    country_pair_counts: dict[str, int] = {}
    for inv in invoices:
        if not isinstance(inv, dict):
            continue
        seller = str(inv.get("seller_tax_code", inv.get("from", ""))).strip()
        buyer = str(inv.get("buyer_tax_code", inv.get("to", ""))).strip()
        seller_country = str((profiles.get(seller) or {}).get("country") or "")
        buyer_country = str((profiles.get(buyer) or {}).get("country") or "")
        if not seller_country or not buyer_country:
            continue
        invoice_country_coverage_count += 1
        if seller_country != buyer_country:
            cross_border_invoice_count += 1
            pair_key = f"{seller_country}->{buyer_country}"
            country_pair_counts[pair_key] = country_pair_counts.get(pair_key, 0) + 1

    high_risk_countries = sorted(
        [country for country in country_company_counts.keys() if _is_high_risk_country(country)]
    )

    cross_border_ratio = (
        float(cross_border_invoice_count) / float(max(1, invoice_country_coverage_count))
    )
    invoice_coverage = float(invoice_country_coverage_count) / float(max(1, len(invoices)))

    risk_score = min(
        100.0,
        (cross_border_ratio * 55.0)
        + (min(5, len(high_risk_countries)) * 8.0)
        + (min(10, foreign_companies) * 2.5)
        + (min(10, offshore_proxy_companies) * 1.5),
    )
    if risk_score >= 70.0:
        risk_level = "high"
    elif risk_score >= 35.0:
        risk_level = "medium"
    else:
        risk_level = "low"

    top_country_exposures = sorted(
        [
            {
                "country": country,
                "companies": count,
                "high_risk": _is_high_risk_country(country),
            }
            for country, count in country_company_counts.items()
            if country and country != "Vietnam"
        ],
        key=lambda row: (row["companies"], row["high_risk"]),
        reverse=True,
    )[:6]

    top_country_pairs = sorted(
        [
            {"pair": pair, "invoice_count": count}
            for pair, count in country_pair_counts.items()
        ],
        key=lambda row: row["invoice_count"],
        reverse=True,
    )[:6]

    return {
        "available": bool(total_companies > 0),
        "risk_level": risk_level,
        "risk_score": round(float(risk_score), 2),
        "scope_companies_total": int(total_companies),
        "scope_invoices_total": int(len(invoices)),
        "companies_with_country": int(with_country),
        "companies_unknown_country": int(unknown_country),
        "companies_with_low_confidence": int(low_confidence),
        "companies_invalid_or_missing_geom": int(invalid_or_missing_geom),
        "companies_within_vietnam": int(within_vietnam),
        "companies_outside_vietnam": int(foreign_companies),
        "offshore_proxy_companies": int(offshore_proxy_companies),
        "cross_border_invoice_count": int(cross_border_invoice_count),
        "cross_border_invoice_ratio": round(float(cross_border_ratio), 4),
        "invoice_country_coverage": round(float(invoice_coverage), 4),
        "high_risk_country_exposure": {
            "count": int(len(high_risk_countries)),
            "countries": high_risk_countries,
        },
        "country_company_distribution": top_country_exposures,
        "cross_border_country_pairs": top_country_pairs,
        "ownership_links_count": int(len(ownership_links or [])),
        "rings_count": int(len(rings or [])),
    }


def _classify_zero_semantics(
    data_status: str,
    metric_name: str,
    metric_value: Any,
    coverage_value: Optional[Any] = None,
) -> dict[str, Any]:
    status = str(data_status or "unknown").strip().lower()
    value = _to_float(metric_value, 0.0)
    coverage = None if coverage_value is None else _to_float(coverage_value, 0.0)

    if value > 0.0:
        meaning = "observed_non_zero"
        ambiguous = False
    elif status in {
        "no_invoice_context",
        "no_ownership_links",
        "ownership_outside_invoice_scope",
        "request_failed",
        "http_error",
        "invalid_payload",
    }:
        meaning = "zero_due_to_missing_coverage"
        ambiguous = False
    elif coverage is not None and coverage <= 0.0:
        meaning = "zero_with_low_coverage"
        ambiguous = True
    else:
        meaning = "true_zero_observed"
        ambiguous = False

    return {
        "metric": metric_name,
        "value": value,
        "coverage": coverage,
        "data_status": status,
        "semantic_meaning": meaning,
        "ambiguous": bool(ambiguous),
    }


def _build_forensic_provenance(
    endpoint: str,
    snapshot_id: str,
    query_scope: dict[str, Any],
    companies: list[dict],
    invoices: list[dict],
    extra_counts: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    records = {
        "companies": int(len(companies)),
        "invoices": int(len(invoices)),
    }
    for key, value in (extra_counts or {}).items():
        records[str(key)] = _to_int(value, 0)

    source_tables = ["companies", "invoices"]
    if records.get("ownership_links", 0) > 0:
        source_tables.append("ownership_links")

    return {
        "schema_version": GRAPH_COMPAT_SCHEMA_VERSION,
        "endpoint": endpoint,
        "generated_at": _utc_now_iso(),
        "snapshot_id": snapshot_id,
        "shared_context_id": str(query_scope.get("shared_context_id") or snapshot_id),
        "query_scope_source": str(query_scope.get("source") or endpoint),
        "source_tables": source_tables,
        "record_counts": records,
        "scope_limits": {
            "company_row_limit": query_scope.get("company_row_limit"),
            "invoice_row_limit": query_scope.get("invoice_row_limit"),
            "ownership_links_limit": query_scope.get("ownership_links_limit"),
        },
    }


def _build_kpi_snapshot(
    snapshot_id: str,
    query_scope: dict[str, Any],
    ring_metrics: Optional[dict[str, Any]],
    ownership_metrics: Optional[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "scope": {
            "company_rows_returned": _to_int(query_scope.get("company_rows_returned"), 0),
            "invoice_rows_returned": _to_int(query_scope.get("invoice_rows_returned"), 0),
            "graph_mode": str(query_scope.get("graph_mode") or "unknown"),
        },
        "ring_metrics": ring_metrics or {"available": False},
        "ownership_metrics": ownership_metrics or {"available": False},
        "provenance": {
            "endpoint": provenance.get("endpoint"),
            "generated_at": provenance.get("generated_at"),
            "source_tables": provenance.get("source_tables"),
        },
    }


def _build_compatibility_diagnostics(
    endpoint: str,
    data_status: str,
    query_scope: dict[str, Any],
    kpi_snapshot: dict[str, Any],
    zero_semantics: dict[str, Any],
    cross_border_signals: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "snapshot_id": bool(kpi_snapshot.get("snapshot_id")),
        "scope": isinstance(kpi_snapshot.get("scope"), dict),
        "ring_metrics": isinstance(kpi_snapshot.get("ring_metrics"), dict),
        "ownership_metrics": isinstance(kpi_snapshot.get("ownership_metrics"), dict),
        "provenance": isinstance(kpi_snapshot.get("provenance"), dict),
        "shared_context_id": bool(query_scope.get("shared_context_id")),
    }

    total_required = max(1, len(required))
    passed_required = sum(1 for value in required.values() if value)
    completion_ratio = round((float(passed_required) / float(total_required)) * 100.0, 2)

    if completion_ratio >= 100.0:
        status = "pass"
    elif completion_ratio >= 80.0:
        status = "partial"
    else:
        status = "fail"

    return {
        "schema_version": GRAPH_COMPAT_SCHEMA_VERSION,
        "endpoint": endpoint,
        "status": status,
        "completion_ratio": completion_ratio,
        "required_fields": required,
        "data_status": str(data_status or "unknown"),
        "zero_semantics": zero_semantics,
        "cross_border_risk_level": cross_border_signals.get("risk_level"),
        "cross_border_risk_score": cross_border_signals.get("risk_score"),
        "feature_flag_enabled": bool(GRAPH_FORENSIC_FEATURE_FLAG),
    }


def _append_cross_border_logs(base_logs: Any, cross_border_signals: dict[str, Any]) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    if isinstance(base_logs, list):
        logs.extend([log for log in base_logs if isinstance(log, dict)])

    if not cross_border_signals:
        return logs

    now_ts = _utc_now_iso()
    risk_level = str(cross_border_signals.get("risk_level") or "low").lower()
    cross_border_count = _to_int(cross_border_signals.get("cross_border_invoice_count"), 0)
    unknown_country = _to_int(cross_border_signals.get("companies_unknown_country"), 0)

    if cross_border_count > 0:
        logs.append(
            {
                "timestamp": now_ts,
                "severity": "high" if risk_level == "high" else "medium",
                "title": f"Tín hiệu xuyên biên giới: {cross_border_count} giao dịch khác quốc gia",
                "description": "Phân tích geospatial phát hiện luồng hóa đơn giữa các pháp nhân thuộc quốc gia khác nhau trong cùng snapshot.",
            }
        )

    if unknown_country > 0:
        logs.append(
            {
                "timestamp": now_ts,
                "severity": "medium",
                "title": f"Thiếu phủ geospatial trên {unknown_country} pháp nhân",
                "description": "Một phần pháp nhân chưa đủ dữ liệu tọa độ/country inference; cần backfill để tránh hiểu sai zero coverage.",
            }
        )

    return logs[:120]


def _build_evidence_chains(
    evidence_paths: Any,
    company_geo_profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(evidence_paths, list):
        return []

    chains: list[dict[str, Any]] = []
    for path in evidence_paths:
        if not isinstance(path, dict):
            continue

        hops_raw = path.get("hops") if isinstance(path.get("hops"), list) else []
        companies_raw = path.get("companies") if isinstance(path.get("companies"), list) else []

        hops_enriched = []
        probs: list[float] = []
        cross_border_hops = 0
        weighted_cross_border = 0.0
        for hop in hops_raw:
            if not isinstance(hop, dict):
                continue
            from_tc = str(hop.get("from") or "").strip()
            to_tc = str(hop.get("to") or "").strip()
            from_geo = company_geo_profiles.get(from_tc, {})
            to_geo = company_geo_profiles.get(to_tc, {})
            from_country = str(from_geo.get("country") or "")
            to_country = str(to_geo.get("country") or "")
            cross_border = bool(from_country and to_country and from_country != to_country)
            if cross_border:
                cross_border_hops += 1
                hop_weight = 1.0
                if _is_high_risk_country(from_country) or _is_high_risk_country(to_country):
                    hop_weight = 1.35
                weighted_cross_border += hop_weight

            prob = max(0.0, min(1.0, _to_float(hop.get("fraud_probability"), 0.0)))
            probs.append(prob)

            hops_enriched.append(
                {
                    **hop,
                    "cross_border": cross_border,
                    "narrative": (
                        f"{from_tc} -> {to_tc}: {from_country or 'Unknown'}"
                        f" to {to_country or 'Unknown'}"
                        + (" (cross-border)" if cross_border else " (domestic/unknown)")
                    ),
                    "provenance": {
                        "from_country": from_country or None,
                        "to_country": to_country or None,
                        "from_confidence": from_geo.get("confidence"),
                        "to_confidence": to_geo.get("confidence"),
                        "from_geocoding_method": from_geo.get("geocoding_method"),
                        "to_geocoding_method": to_geo.get("geocoding_method"),
                    },
                }
            )

        hop_count = max(1, len(hops_enriched))
        company_count = max(1, len(companies_raw))
        unique_companies = len({str(code or "") for code in companies_raw if str(code or "").strip()})
        repeat_factor = max(0.0, float(company_count - unique_companies) / float(company_count))

        circularity_score = round(min(100.0, (sum(probs) / float(max(1, len(probs)))) * 100.0), 2)
        ownership_score = round(min(100.0, repeat_factor * 100.0), 2)
        cross_border_score = round(min(100.0, (weighted_cross_border / float(hop_count)) * 60.0), 2)
        chain_score = round(
            (0.5 * circularity_score) + (0.2 * ownership_score) + (0.3 * cross_border_score),
            2,
        )

        if chain_score >= 70.0:
            risk_level = "critical"
        elif chain_score >= 50.0:
            risk_level = "high"
        elif chain_score >= 30.0:
            risk_level = "medium"
        else:
            risk_level = "low"

        chains.append(
            {
                "path_id": path.get("path_id"),
                "summary": path.get("summary"),
                "risk_level": risk_level,
                "chain_score": chain_score,
                "scoring_breakdown": {
                    "circularity": circularity_score,
                    "ownership": ownership_score,
                    "cross_border": cross_border_score,
                },
                "companies": companies_raw,
                "hops": hops_enriched,
                "cross_border_hops": int(cross_border_hops),
                "narrative": (
                    f"Chuoi {path.get('path_id', 'unknown')}: circularity={circularity_score:.1f}, "
                    f"ownership={ownership_score:.1f}, cross-border={cross_border_score:.1f}."
                ),
            }
        )

    return chains


def _resolve_graph_context(
    db: Session,
    tax_code: Optional[str],
    depth: Optional[int],
    source: str,
    ownership_links_limit: Optional[int] = None,
) -> dict[str, Any]:
    if tax_code:
        resolved_depth = int(depth) if depth is not None else 2
        companies, invoices, ownership_links = _extract_subgraph(db, tax_code, resolved_depth)
    else:
        resolved_depth = None
        companies, invoices, ownership_links = _extract_full_graph(db, limit=FULL_GRAPH_COMPANY_LIMIT)

    snapshot_id = _build_snapshot_id(
        tax_code=tax_code,
        depth=resolved_depth,
        companies=companies,
        invoices=invoices,
        source=GRAPH_CONTEXT_SOURCE,
    )
    query_scope = _build_query_scope(
        tax_code=tax_code,
        depth=resolved_depth,
        companies=companies,
        invoices=invoices,
        source=source,
        ownership_links_limit=ownership_links_limit,
    )
    query_scope["shared_context_id"] = snapshot_id
    query_scope["compatibility_schema"] = GRAPH_COMPAT_SCHEMA_VERSION

    return {
        "tax_code": tax_code,
        "depth": resolved_depth,
        "companies": companies,
        "invoices": invoices,
        "ownership_links": ownership_links,
        "snapshot_id": snapshot_id,
        "query_scope": query_scope,
    }


def _sanitize_policy(policy: Any) -> dict:
    if not isinstance(policy, dict):
        return {}
    return {
        "cold_start_degree_threshold": int(_to_float(policy.get("cold_start_degree_threshold", 0), 0)),
        "cold_start_threshold_delta": round(_to_float(policy.get("cold_start_threshold_delta", 0.0), 0.0), 4),
        "node_blend_alpha_gnn": round(_to_float(policy.get("node_blend_alpha_gnn", 0.0), 0.0), 4),
    }


def _build_snapshot_id(
    tax_code: Optional[str],
    depth: Optional[int],
    companies: list[dict],
    invoices: list[dict],
    source: str,
) -> str:
    company_codes = sorted(
        {
            str(company.get("tax_code", "")).strip()
            for company in companies
            if isinstance(company, dict) and str(company.get("tax_code", "")).strip()
        }
    )
    edge_keys = sorted(
        {
            (
                f"{str(inv.get('seller_tax_code', inv.get('from', ''))).strip()}"
                f"->{str(inv.get('buyer_tax_code', inv.get('to', ''))).strip()}"
                f"|{str(inv.get('invoice_number', '')).strip()}"
            )
            for inv in invoices
            if isinstance(inv, dict)
        }
    )

    seed_payload = {
        "source": str(source),
        "tax_code": str(tax_code) if tax_code else "__all__",
        "depth": int(depth) if depth is not None else None,
        "company_count": len(companies),
        "invoice_count": len(invoices),
        "company_sample": company_codes[:40],
        "edge_sample": edge_keys[:80],
    }
    digest = hashlib.sha1(
        json.dumps(seed_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"snap-{digest}"


def _build_query_scope(
    tax_code: Optional[str],
    depth: Optional[int],
    companies: list[dict],
    invoices: list[dict],
    source: str,
    ownership_links_limit: Optional[int] = None,
) -> dict[str, Any]:
    focused_mode = tax_code is not None
    company_limit = SUBGRAPH_MAX_NODES if focused_mode else FULL_GRAPH_COMPANY_LIMIT
    invoice_limit = SUBGRAPH_INVOICE_LIMIT if focused_mode else None
    invoice_rows = len(invoices)

    return {
        "source": source,
        "tax_code": tax_code,
        "depth": int(depth) if depth is not None else None,
        "graph_mode": "focused_subgraph" if focused_mode else "full_graph",
        "company_row_limit": int(company_limit),
        "company_rows_returned": len(companies),
        "company_row_limit_hit": bool(len(companies) >= company_limit),
        "invoice_row_limit": int(invoice_limit) if invoice_limit is not None else None,
        "invoice_rows_returned": invoice_rows,
        "invoice_row_limit_hit": bool(invoice_limit is not None and invoice_rows >= invoice_limit),
        "ownership_links_limit": int(ownership_links_limit) if ownership_links_limit is not None else None,
    }


def _enrich_graph_result(result: dict, engine: Any, tax_code: Optional[str], depth: int) -> dict:
    model_loaded = bool(result.get("model_loaded", getattr(engine, "_loaded", False)))
    ensemble_active = bool(result.get("ensemble_active", False))
    fallback_active = bool(result.get("fallback_active", not model_loaded))
    if fallback_active and not result.get("fallback_reason"):
        result["fallback_reason"] = getattr(engine, "_load_error", "model_unavailable")

    thresholds_raw = result.get("decision_thresholds")
    if isinstance(thresholds_raw, dict):
        node_threshold = _to_float(thresholds_raw.get("node", 0.5), 0.5)
        edge_threshold = _to_float(thresholds_raw.get("edge", 0.5), 0.5)
        policy = _sanitize_policy(thresholds_raw.get("policy"))
    else:
        node_threshold = 0.5
        edge_threshold = 0.5
        policy = {}

    decision_thresholds = {
        "node": round(node_threshold, 4),
        "edge": round(edge_threshold, 4),
        "policy": policy,
    }
    result["decision_thresholds"] = decision_thresholds

    nodes = result.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            shell_probability = _to_float(
                node.get("shell_probability"),
                _to_float(node.get("risk_score", 0.0), 0.0) / 100.0,
            )
            node_threshold_local = _to_float(node.get("decision_threshold", node_threshold), node_threshold)
            node["shell_probability"] = round(shell_probability, 4)
            node["decision_threshold"] = round(node_threshold_local, 4)
            node["threshold_margin"] = round(shell_probability - node_threshold_local, 4)

    edges = result.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            circular_probability = _to_float(edge.get("circular_probability"), 0.0)
            edge["circular_probability"] = round(circular_probability, 4)
            edge["decision_threshold"] = round(edge_threshold, 4)
            edge["threshold_margin"] = round(circular_probability - edge_threshold, 4)

    attention_weights = result.get("attention_weights")
    attention_top = []
    if isinstance(attention_weights, list):
        normalized_attention = []
        for item in attention_weights:
            if not isinstance(item, dict):
                continue
            normalized_attention.append(
                {
                    "from": str(item.get("from", "")),
                    "to": str(item.get("to", "")),
                    "weight": round(_to_float(item.get("weight", 0.0), 0.0), 4),
                }
            )
        attention_top = sorted(normalized_attention, key=lambda row: row["weight"], reverse=True)[:5]
        attention_count = len(normalized_attention)
    else:
        attention_count = 0

    result["attention_summary"] = {
        "count": attention_count,
        "top_edges": attention_top,
    }

    result["query_context"] = {
        "tax_code": tax_code,
        "depth": int(depth),
    }

    result["contract_version"] = "graph-intelligence-v1"
    result["model_info"] = {
        "contract_version": "graph-intelligence-v1",
        "inference_mode": "gnn_ensemble" if model_loaded else "heuristic_fallback",
        "model_loaded": model_loaded,
        "ensemble_active": ensemble_active,
        "fallback_active": fallback_active,
        "fallback_reason": result.get("fallback_reason"),
        "decision_thresholds": decision_thresholds,
    }

    return result


def _build_split_trigger_status_context(snapshot_source: str = "graph_main") -> dict[str, Any]:
    payload = monitoring_router.get_split_trigger_status_snapshot(
        persist_snapshot=False,
        snapshot_source=snapshot_source,
    )
    if isinstance(payload, dict):
        return payload
    return {
        "ready": False,
        "schema_ready": False,
        "readiness_score": 0,
        "reason": "Không thể tải split-trigger status.",
        "track_status": {},
        "totals": {"enabled_rules": 0, "passed_rules": 0},
        "generated_at": "",
    }


@router.get("/graph")
def get_vat_invoice_graph(
    tax_code: Optional[str] = Query(None, description="Tax code tâm điểm để dựng subgraph"),
    depth: int = Query(2, ge=1, le=4, description="Độ sâu truy vết (1-4 bước)"),
    db: Session = Depends(get_db),
):
    """
    Phân tích đồ thị mạng lưới mua bán hóa đơn với GNN.
    
    - Nếu tax_code được cung cấp: trích xuất subgraph xung quanh công ty đó
    - Nếu không: trả về toàn bộ mạng lưới (giới hạn 200 nodes)
    """
    tax_code = _ensure_numeric_tax_code(tax_code)

    try:
        ctx = _resolve_graph_context(
            db=db,
            tax_code=tax_code,
            depth=depth,
            source="graph_main",
        )
        companies = ctx["companies"]
        invoices = ctx["invoices"]
        snapshot_id = ctx["snapshot_id"]
        query_scope = ctx["query_scope"]
    except SQLAlchemyError as e:
        log_event(
            logger,
            "error",
            "graph_data_extraction_failed",
            error_type=type(e).__name__,
            tax_code=tax_code,
            depth=depth,
        )
        raise HTTPException(status_code=500, detail="Lỗi truy vấn dữ liệu đồ thị.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_data_extraction_unexpected_error",
            error_type=type(e).__name__,
            tax_code=tax_code,
            depth=depth,
        )
        raise HTTPException(status_code=500, detail=f"Lỗi dựng subgraph: {str(e)}")

    if not companies:
        raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu cho mã số thuế này.")

    # Run GNN inference
    try:
        engine = _get_gnn_engine()
    except (ImportError, ModuleNotFoundError) as e:
        log_event(
            logger,
            "error",
            "graph_gnn_dependency_error",
            error_type=type(e).__name__,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Không thể khởi tạo GNN engine do lỗi phụ thuộc.")

    try:
        result = engine.predict(companies, invoices, ctx.get("ownership_links", []))
    except RuntimeError as e:
        log_event(
            logger,
            "error",
            "graph_gnn_inference_runtime_error",
            error_type=type(e).__name__,
            error=str(e),
            tax_code=tax_code,
        )
        raise HTTPException(status_code=500, detail="Lỗi runtime khi suy luận GNN.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_gnn_inference_error",
            error_type=type(e).__name__,
            error=str(e),
            tax_code=tax_code,
        )
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích đồ thị: {str(e)}")

    if isinstance(result, dict):
        fallback_active = not bool(getattr(engine, "_loaded", False))
        result.setdefault("fallback_active", fallback_active)
        if fallback_active:
            result.setdefault("fallback_reason", getattr(engine, "_load_error", "model_unavailable"))
            log_event(
                logger,
                "warning",
                "graph_response_fallback_active",
                tax_code=tax_code,
                reason=result.get("fallback_reason"),
            )

        result = _enrich_graph_result(result, engine=engine, tax_code=tax_code, depth=depth)
        result["snapshot_id"] = snapshot_id
        result["query_scope"] = query_scope
        query_scope["shared_context_id"] = snapshot_id
        query_scope["compatibility_schema"] = GRAPH_COMPAT_SCHEMA_VERSION
        data_status = "ok" if invoices else "no_invoice_context"
        result["data_status"] = data_status
        result["split_trigger_status"] = _build_split_trigger_status_context(
            snapshot_source="graph_main",
        )

        # ── Forensic Compatibility v2 ──
        if GRAPH_FORENSIC_FEATURE_FLAG:
            cross_border_signals = _build_cross_border_signals(companies, invoices)
            result["cross_border_signals"] = cross_border_signals

            forensic_metrics = result.get("forensic_metrics", {}) if isinstance(result.get("forensic_metrics"), dict) else {}
            circular_edge_count = _to_int(forensic_metrics.get("circular_edge_count"), 0)
            zero_semantics = _classify_zero_semantics(
                data_status=data_status,
                metric_name="circular_edge_count",
                metric_value=circular_edge_count,
                coverage_value=forensic_metrics.get("circular_edge_cycle_coverage"),
            )

            provenance = _build_forensic_provenance(
                endpoint="graph_main",
                snapshot_id=snapshot_id,
                query_scope=query_scope,
                companies=companies,
                invoices=invoices,
            )
            result["forensic_provenance"] = provenance

            kpi_snapshot = _build_kpi_snapshot(
                snapshot_id=snapshot_id,
                query_scope=query_scope,
                ring_metrics=None,
                ownership_metrics=None,
                provenance=provenance,
            )

            diagnostics = _build_compatibility_diagnostics(
                endpoint="graph_main",
                data_status=data_status,
                query_scope=query_scope,
                kpi_snapshot=kpi_snapshot,
                zero_semantics=zero_semantics,
                cross_border_signals=cross_border_signals,
            )
            result["compatibility_diagnostics"] = diagnostics

            base_logs = result.get("logs") if isinstance(result.get("logs"), list) else []
            result["logs"] = _append_cross_border_logs(base_logs, cross_border_signals)

            geo_profiles = _extract_company_geo_profiles(companies)
            base_evidence = result.get("evidence_paths") if isinstance(result.get("evidence_paths"), list) else []
            result["evidence_chains"] = _build_evidence_chains(base_evidence, geo_profiles)

    return result


@router.get("/graph/search")
def search_companies_for_graph(
    q: str = Query(..., min_length=2, description="Từ khoá tìm kiếm (mã thuế hoặc tên)"),
    db: Session = Depends(get_db),
):
    """Tìm kiếm công ty để dựng đồ thị mạng lưới."""
    try:
        result = db.execute(text("""
            SELECT tax_code, name, industry, risk_score
            FROM companies
            WHERE tax_code ILIKE :q OR name ILIKE :q
            ORDER BY risk_score DESC
            LIMIT 10
        """), {"q": f"%{q}%"})
        
        rows = result.fetchall()
        return {
            "results": [
                {
                    "tax_code": r[0],
                    "name": r[1],
                    "industry": r[2],
                    "risk_score": float(r[3]) if r[3] else 0.0,
                }
                for r in rows
            ]
        }
    except SQLAlchemyError as e:
        log_event(
            logger,
            "error",
            "graph_company_search_failed",
            error_type=type(e).__name__,
            query=q,
        )
        raise HTTPException(status_code=500, detail="Lỗi truy vấn tìm kiếm doanh nghiệp.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_company_search_unexpected_error",
            error_type=type(e).__name__,
            query=q,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/companies")
def get_all_companies(db: Session = Depends(get_db)):
    """Lấy danh sách các doanh nghiệp cho bảng tổng hợp (limit 500)."""
    try:
        result = db.execute(text("""
            SELECT tax_code, name, industry, is_active, risk_score
            FROM companies
            ORDER BY risk_score DESC
            LIMIT 500
        """))
        
        rows = result.fetchall()
        return {
            "results": [
                {
                    "tax_code": r[0],
                    "name": r[1],
                    "industry": r[2],
                    "is_active": bool(r[3]),
                    "risk_score": float(r[4]) if r[4] else 0.0,
                }
                for r in rows
            ]
        }
    except SQLAlchemyError as e:
        log_event(
            logger,
            "error",
            "graph_companies_list_failed",
            error_type=type(e).__name__,
        )
        raise HTTPException(status_code=500, detail="Lỗi truy vấn danh sách doanh nghiệp.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_companies_list_unexpected_error",
            error_type=type(e).__name__,
        )
        raise HTTPException(status_code=500, detail=str(e))


def _extract_subgraph(db: Session, center_tax_code: str, depth: int, max_nodes: int = SUBGRAPH_MAX_NODES) -> tuple:
    """
    Extract a subgraph centered on a company, expanding outward by `depth` hops.
    Returns (companies, invoices) lists.
    """
    # BFS-expand from center
    visited_codes = {center_tax_code}
    frontier = {center_tax_code}

    # Seed traversal with ownership-linked domestic entities for offshore centers.
    ownership_seed_codes = _collect_seed_tax_codes_from_ownership(
        db,
        center_tax_code,
        limit=max(0, max_nodes - 1),
    )
    for code in ownership_seed_codes:
        if len(visited_codes) >= max_nodes:
            break
        if code in visited_codes:
            continue
        visited_codes.add(code)
        frontier.add(code)

    for current_depth in range(depth):
        if not frontier or len(visited_codes) >= max_nodes:
            break
        placeholders = ",".join([f":tc{i}" for i in range(len(frontier))])
        params = {f"tc{i}": tc for i, tc in enumerate(frontier)}

        # Lấy các giao dịch mà trong đó source/target nằm trong tập frontier
        result = db.execute(text(f"""
            SELECT DISTINCT seller_tax_code, buyer_tax_code 
            FROM invoices
            WHERE seller_tax_code IN ({placeholders}) 
               OR buyer_tax_code IN ({placeholders})
        """), params)

        new_frontier = set()
        for row in result.fetchall():
            # Mở rộng nếu chưa chạm max_nodes
            for tc in [row[0], row[1]]:
                if tc not in visited_codes:
                    if len(visited_codes) < max_nodes:
                        new_frontier.add(tc)
                        visited_codes.add(tc)
                    else:
                        break
            if len(visited_codes) >= max_nodes:
                break
        frontier = new_frontier

    if not visited_codes:
        return [], []

    # Lấy thông tin các doanh nghiệp (Company)
    placeholders = ",".join([f":tc{i}" for i in range(len(visited_codes))])
    params = {f"tc{i}": tc for i, tc in enumerate(visited_codes)}

    has_geom = True
    try:
        db.execute(text("SELECT geom FROM companies LIMIT 1"))
    except Exception:
        has_geom = False
        db.rollback()

    if has_geom:
        comp_result = db.execute(text(f"""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   COALESCE(ST_Y(geom), 0) as lat, COALESCE(ST_X(geom), 0) as lng,
                   country_inferred, confidence_country, is_within_vietnam, geocoding_method
            FROM companies
            WHERE tax_code IN ({placeholders})
        """), params)
    else:
        comp_result = db.execute(text(f"""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   0.0 as lat, 0.0 as lng,
                   country_inferred, confidence_country, is_within_vietnam, geocoding_method
            FROM companies
            WHERE tax_code IN ({placeholders})
        """), params)

    columns = [desc[0] for desc in comp_result.cursor.description] if hasattr(comp_result, 'cursor') else \
              [col for col in comp_result.keys()]
    companies = [dict(zip(columns, row)) for row in comp_result.fetchall()]

    # Lấy các hóa đơn giữa những doanh nghiệp nằm trong sub-graph
    # Giới hạn số lượng invoice để tránh crash DOM frontend (max 500 đường nối)
    inv_result = db.execute(text(f"""
        SELECT seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number
        FROM invoices
        WHERE seller_tax_code IN ({placeholders})
          AND buyer_tax_code IN ({placeholders})
        ORDER BY amount DESC
        LIMIT {SUBGRAPH_INVOICE_LIMIT}
    """), params)

    inv_columns = [col for col in inv_result.keys()]
    invoices = [dict(zip(inv_columns, row)) for row in inv_result.fetchall()]

    # Extract ownership_links within the subgraph boundary
    ownership_result = db.execute(text(f"""
        SELECT parent_tax_code, child_tax_code, ownership_percent, relationship_type, person_id
        FROM ownership_links
        WHERE parent_tax_code IN ({placeholders})
           OR child_tax_code IN ({placeholders})
    """), params)
    
    own_columns = [col for col in ownership_result.keys()]
    ownership_links = [dict(zip(own_columns, row)) for row in ownership_result.fetchall()]

    return companies, invoices, ownership_links


def _extract_full_graph(db: Session, limit: int = FULL_GRAPH_COMPANY_LIMIT) -> tuple:
    """Extract the full graph (limited to top N companies by risk & activity)."""
    has_geom = True
    try:
        db.execute(text("SELECT geom FROM companies LIMIT 1"))
    except Exception:
        has_geom = False
        db.rollback()

    if has_geom:
        comp_result = db.execute(text("""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   COALESCE(ST_Y(geom), 0) as lat, COALESCE(ST_X(geom), 0) as lng,
                   country_inferred, confidence_country, is_within_vietnam, geocoding_method
            FROM companies
            ORDER BY risk_score DESC
            LIMIT :limit
        """), {"limit": limit})
    else:
        comp_result = db.execute(text("""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   0.0 as lat, 0.0 as lng,
                   NULL as country_inferred, 0.0 as confidence_country, NULL as is_within_vietnam, NULL as geocoding_method
            FROM companies
            ORDER BY risk_score DESC
            LIMIT :limit
        """), {"limit": limit})

    columns = [col for col in comp_result.keys()]
    companies = [dict(zip(columns, row)) for row in comp_result.fetchall()]

    if not companies:
        return [], []

    tax_codes = [c["tax_code"] for c in companies]
    placeholders = ",".join([f":tc{i}" for i in range(len(tax_codes))])
    params = {f"tc{i}": tc for i, tc in enumerate(tax_codes)}

    inv_result = db.execute(text(f"""
        SELECT seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number
        FROM invoices
        WHERE seller_tax_code IN ({placeholders})
          AND buyer_tax_code IN ({placeholders})
        ORDER BY date
    """), params)

    inv_columns = [col for col in inv_result.keys()]
    invoices = [dict(zip(inv_columns, row)) for row in inv_result.fetchall()]

    # Extract ownership_links within the subgraph boundary
    ownership_result = db.execute(text(f"""
        SELECT parent_tax_code, child_tax_code, ownership_percent, relationship_type, person_id
        FROM ownership_links
        WHERE parent_tax_code IN ({placeholders})
           OR child_tax_code IN ({placeholders})
    """), params)
    
    own_columns = [col for col in ownership_result.keys()]
    ownership_links = [dict(zip(own_columns, row)) for row in ownership_result.fetchall()]

    return companies, invoices, ownership_links


# ════════════════════════════════════════════════════════════════
#  Graph Intelligence 2.0 (Program B) – Flagship Endpoints
# ════════════════════════════════════════════════════════════════

@router.get("/graph/motifs")
def detect_graph_motifs(
    tax_code: Optional[str] = Query(None, description="Tax code tâm điểm"),
    depth: int = Query(2, ge=1, le=4),
    db: Session = Depends(get_db),
):
    """
    Motif Detection: Identify suspicious transaction patterns in the VAT invoice graph.
    Detects: triangles (carousel fraud), stars (shell hubs), chains (layering),
    fan-out/fan-in patterns.
    """
    tax_code = _ensure_numeric_tax_code(tax_code)

    try:
        if tax_code:
            companies, invoices, ownership_links = _extract_subgraph(db, tax_code, depth)
        else:
            companies, invoices, ownership_links = _extract_full_graph(db, limit=200)
    except Exception as e:
        log_event(logger, "error", "graph_motif_data_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu: {str(e)}")

    if not invoices:
        return {
            "status": "no_data",
            "message": "Không có dữ liệu giao dịch để phân tích motif.",
            "motifs": {},
            "summary": {},
        }

    try:
        from ml_engine.graph_intelligence import MotifDetector
        detector = MotifDetector()
        result = detector.detect_all(companies, invoices)

        log_event(
            logger, "info", "graph_motif_detection_complete",
            triangles=result["summary"]["total_triangles"],
            stars=result["summary"]["total_stars"],
            chains=result["summary"]["total_chains"],
        )
        return result

    except Exception as e:
        log_event(logger, "error", "graph_motif_detection_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích motif: {str(e)}")


@router.get("/graph/link-prediction")
def predict_graph_links(
    tax_code: Optional[str] = Query(None),
    depth: int = Query(2, ge=1, le=4),
    top_k: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Link Prediction: Predict likely future fraudulent connections between companies.
    Uses Jaccard coefficient + Adamic-Adar index + topology risk.
    """
    tax_code = _ensure_numeric_tax_code(tax_code)

    try:
        if tax_code:
            companies, invoices, ownership_links = _extract_subgraph(db, tax_code, depth)
        else:
            companies, invoices, ownership_links = _extract_full_graph(db, limit=200)
    except Exception as e:
        log_event(logger, "error", "graph_link_pred_data_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu: {str(e)}")

    if not invoices:
        return {"predictions": [], "total": 0, "message": "Không đủ dữ liệu giao dịch."}

    try:
        from ml_engine.graph_intelligence import LinkPredictor
        predictor = LinkPredictor()
        predictions = predictor.predict_new_links(companies, invoices, top_k=top_k)

        log_event(
            logger, "info", "graph_link_prediction_complete",
            predicted_links=len(predictions),
        )
        return {
            "predictions": predictions,
            "total": len(predictions),
            "query_context": {"tax_code": tax_code, "depth": depth, "top_k": top_k},
        }

    except Exception as e:
        log_event(logger, "error", "graph_link_prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán liên kết: {str(e)}")


@router.get("/graph/ownership")
def analyze_ownership_network(
    tax_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Ownership Graph Analysis: Detect shell company networks through ownership relationships.
    Identifies common controllers, ownership chains, and cross-ownership trades.
    """
    tax_code = _ensure_numeric_tax_code(tax_code)
    ownership_depth = 2 if tax_code else None

    try:
        # Load ownership links
        if tax_code:
            ownership_result = db.execute(text("""
                SELECT ol.id, ol.parent_tax_code, ol.child_tax_code, ol.ownership_percent,
                       ol.relationship_type, ol.person_name, ol.person_id,
                       ol.effective_date, ol.end_date, ol.verified
                FROM ownership_links ol
                WHERE ol.parent_tax_code = :tc OR ol.child_tax_code = :tc
                   OR ol.parent_tax_code IN (
                       SELECT child_tax_code FROM ownership_links WHERE parent_tax_code = :tc
                   )
                   OR ol.child_tax_code IN (
                       SELECT parent_tax_code FROM ownership_links WHERE child_tax_code = :tc
                   )
                LIMIT :limit
            """), {"tc": tax_code, "limit": OWNERSHIP_LINK_QUERY_LIMIT})
        else:
            ownership_result = db.execute(text("""
                SELECT id, parent_tax_code, child_tax_code, ownership_percent,
                       relationship_type, person_name, person_id,
                       effective_date, end_date, verified
                FROM ownership_links
                ORDER BY ownership_percent DESC
                LIMIT :limit
            """), {"limit": OWNERSHIP_LINK_QUERY_LIMIT})

        ownership_cols = [col for col in ownership_result.keys()]
        ownership_links = [dict(zip(ownership_cols, row)) for row in ownership_result.fetchall()]

        # Load relevant invoices via shared context for cross-ownership trade detection
        ctx = _resolve_graph_context(db, tax_code, ownership_depth or 2, "graph_ownership")
        companies_for_scope = ctx["companies"]
        invoices = ctx["invoices"]
        snapshot_id = ctx["snapshot_id"]
        query_scope = ctx["query_scope"]

    except Exception as e:
        log_event(logger, "error", "graph_ownership_data_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu sở hữu: {str(e)}")

    ownership_nodes = set()
    ownership_edges: list[tuple[str, str]] = []
    for link in ownership_links:
        if not isinstance(link, dict):
            continue
        parent = str(link.get("parent_tax_code", "")).strip()
        child = str(link.get("child_tax_code", "")).strip()
        if parent:
            ownership_nodes.add(parent)
        if child:
            ownership_nodes.add(child)
        if parent and child:
            ownership_edges.append((parent, child))

    invoice_nodes = set()
    for inv in invoices:
        if not isinstance(inv, dict):
            continue
        seller = str(inv.get("seller_tax_code", inv.get("from", ""))).strip()
        buyer = str(inv.get("buyer_tax_code", inv.get("to", ""))).strip()
        if seller:
            invoice_nodes.add(seller)
        if buyer:
            invoice_nodes.add(buyer)

    ownership_nodes_in_invoice_graph_count = len(ownership_nodes.intersection(invoice_nodes))
    ownership_invoice_node_coverage = round(
        float(ownership_nodes_in_invoice_graph_count) / float(max(1, len(ownership_nodes))),
        4,
    )
    ownership_pairs_in_invoice_scope = sum(
        1 for parent, child in ownership_edges if parent in invoice_nodes and child in invoice_nodes
    )

    try:
        from ml_engine.graph_intelligence import OwnershipGraphAnalyzer
        analyzer = OwnershipGraphAnalyzer()
        result = analyzer.analyze(ownership_links, invoices)

        summary = result.get("summary", {}) if isinstance(result, dict) else {}
        total_clusters = int(summary.get("total_clusters", 0) or 0)
        total_cross_trades = int(summary.get("total_cross_trades", 0) or 0)

        if not ownership_links:
            data_status = "no_ownership_links"
        elif not invoice_nodes:
            data_status = "no_invoice_context"
        elif ownership_nodes_in_invoice_graph_count == 0:
            data_status = "ownership_outside_invoice_scope"
        elif total_clusters > 0 and total_cross_trades == 0 and ownership_pairs_in_invoice_scope == 0:
            data_status = "no_parent_child_pairs_in_invoice_scope"
        elif total_clusters > 0 and total_cross_trades == 0:
            data_status = "no_related_party_trades_found"
        else:
            data_status = "ok"

        result["data_status"] = data_status
        result["snapshot_id"] = snapshot_id
        result["query_scope"] = query_scope
        result["coverage"] = {
            "ownership_nodes": len(ownership_nodes),
            "invoice_nodes": len(invoice_nodes),
            "ownership_nodes_in_invoice_graph_count": ownership_nodes_in_invoice_graph_count,
            "ownership_invoice_node_coverage": ownership_invoice_node_coverage,
            "ownership_pairs": len(ownership_edges),
            "ownership_pairs_in_invoice_scope": ownership_pairs_in_invoice_scope,
        }

        # ── Forensic Compatibility v2 ──
        if GRAPH_FORENSIC_FEATURE_FLAG:
            cross_border_signals = _build_cross_border_signals(companies_for_scope, invoices)
            result["cross_border_signals"] = cross_border_signals

            provenance = _build_forensic_provenance(
                endpoint="graph_ownership",
                snapshot_id=snapshot_id,
                query_scope=query_scope,
                companies=companies_for_scope,
                invoices=invoices,
            )
            result["forensic_provenance"] = provenance

            ownership_metrics = {
                "available": True,
                "total_clusters": total_clusters,
                "total_cross_trades": total_cross_trades,
                "coverage": ownership_invoice_node_coverage,
            }
            kpi_snapshot = _build_kpi_snapshot(
                snapshot_id=snapshot_id,
                query_scope=query_scope,
                ring_metrics=None,
                ownership_metrics=ownership_metrics,
                provenance=provenance,
            )

            diagnostics_compat = _build_compatibility_diagnostics(
                endpoint="graph_ownership",
                data_status=data_status,
                query_scope=query_scope,
                kpi_snapshot=kpi_snapshot,
                zero_semantics=None,
                cross_border_signals=cross_border_signals,
            )
            result["compatibility_diagnostics"] = diagnostics_compat

        log_event(
            logger, "info", "graph_ownership_analysis_complete",
            clusters=total_clusters,
            cross_trades=total_cross_trades,
            data_status=data_status,
            coverage=ownership_invoice_node_coverage,
        )
        return result

    except Exception as e:
        log_event(logger, "error", "graph_ownership_analysis_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích sở hữu: {str(e)}")


@router.get("/graph/ring-scoring")
def score_transaction_rings(
    tax_code: Optional[str] = Query(None),
    depth: int = Query(2, ge=1, le=4),
    db: Session = Depends(get_db),
):
    """
    Ring Scoring: Score circular transaction rings by severity.
    Multi-factor analysis: amount, speed, and complexity.
    """
    tax_code = _ensure_numeric_tax_code(tax_code)

    try:
        ctx = _resolve_graph_context(db, tax_code, depth, "graph_ring_scoring")
        companies = ctx["companies"]
        invoices = ctx["invoices"]
        snapshot_id = ctx["snapshot_id"]
        query_scope = ctx["query_scope"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu: {str(e)}")

    if not invoices:
        return {
            "rings": [],
            "total": 0,
            "critical_count": 0,
            "cycles_detected": 0,
            "rings_returned": 0,
            "truncated": False,
            "circular_edge_count": 0,
            "cycle_backed_circular_edge_count": 0,
            "circular_edge_cycle_coverage": 0.0,
            "snapshot_id": snapshot_id,
            "query_scope": query_scope,
            "data_status": "no_invoice_context",
            "diagnostics": {
                "cycle_detection_gap": False,
                "cycle_to_ring_ratio": 0.0,
            },
            "query_context": {"tax_code": tax_code, "depth": depth},
        }

    # Use existing GNN engine to get cycles
    forensic_metrics = {}
    try:
        engine = _get_gnn_engine()
        result = engine.predict(companies, invoices, ctx.get("ownership_links", []))
        cycles = result.get("cycles", [])
        if isinstance(result, dict) and isinstance(result.get("forensic_metrics"), dict):
            forensic_metrics = result.get("forensic_metrics", {})
    except Exception:
        cycles = []

    try:
        from ml_engine.graph_intelligence import RingScorer
        scorer = RingScorer()
        scored_rings = scorer.score_rings(cycles, invoices)
        total_cycles_detected = len(cycles)
        total_rings_returned = len(scored_rings)
        critical_count = sum(1 for r in scored_rings if r.get("risk_level") == "critical")
        circular_edge_count = int(forensic_metrics.get("circular_edge_count", 0) or 0)
        cycle_backed_circular_edge_count = int(
            forensic_metrics.get("cycle_backed_circular_edge_count", 0) or 0
        )
        circular_edge_cycle_coverage = float(
            forensic_metrics.get("circular_edge_cycle_coverage", 0.0) or 0.0
        )

        if total_cycles_detected == 0 and circular_edge_count > 0:
            data_status = "no_cycles_with_circular_edges"
        elif total_cycles_detected == 0:
            data_status = "no_cycles_detected"
        elif total_rings_returned == 0:
            data_status = "cycles_detected_but_no_rings_scored"
        elif total_cycles_detected > total_rings_returned:
            data_status = "partial_ring_output"
        else:
            data_status = "ok"

        cycle_to_ring_ratio = round(
            float(total_rings_returned) / float(max(1, total_cycles_detected)),
            4,
        )

        log_event(
            logger, "info", "graph_ring_scoring_complete",
            rings_scored=total_rings_returned,
            cycles_detected=total_cycles_detected,
            data_status=data_status,
        )

        ring_result = {
            "rings": scored_rings,
            "total": total_rings_returned,
            "critical_count": critical_count,
            "cycles_detected": total_cycles_detected,
            "rings_returned": total_rings_returned,
            "truncated": total_cycles_detected > total_rings_returned,
            "circular_edge_count": circular_edge_count,
            "cycle_backed_circular_edge_count": cycle_backed_circular_edge_count,
            "circular_edge_cycle_coverage": circular_edge_cycle_coverage,
            "snapshot_id": snapshot_id,
            "query_scope": query_scope,
            "data_status": data_status,
            "diagnostics": {
                "cycle_detection_gap": bool(circular_edge_count > 0 and total_cycles_detected == 0),
                "cycle_to_ring_ratio": cycle_to_ring_ratio,
            },
            "query_context": {"tax_code": tax_code, "depth": depth},
        }

        # ── Forensic Compatibility v2 ──
        if GRAPH_FORENSIC_FEATURE_FLAG:
            cross_border_signals = _build_cross_border_signals(companies, invoices, rings=scored_rings)
            ring_result["cross_border_signals"] = cross_border_signals

            zero_semantics = _classify_zero_semantics(
                data_status=data_status,
                metric_name="circular_edge_count",
                metric_value=circular_edge_count,
                coverage_value=circular_edge_cycle_coverage,
            )

            provenance = _build_forensic_provenance(
                endpoint="graph_ring_scoring",
                snapshot_id=snapshot_id,
                query_scope=query_scope,
                companies=companies,
                invoices=invoices,
            )
            ring_result["forensic_provenance"] = provenance

            ring_metrics = {
                "available": True,
                "total": total_rings_returned,
                "critical_count": critical_count,
                "cycles_detected": total_cycles_detected,
            }
            kpi_snapshot = _build_kpi_snapshot(
                snapshot_id=snapshot_id,
                query_scope=query_scope,
                ring_metrics=ring_metrics,
                ownership_metrics=None,
                provenance=provenance,
            )

            diagnostics_compat = _build_compatibility_diagnostics(
                endpoint="graph_ring_scoring",
                data_status=data_status,
                query_scope=query_scope,
                kpi_snapshot=kpi_snapshot,
                zero_semantics=zero_semantics,
                cross_border_signals=cross_border_signals,
            )
            ring_result["compatibility_diagnostics"] = diagnostics_compat

        return ring_result

    except Exception as e:
        log_event(logger, "error", "graph_ring_scoring_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi chấm điểm vòng lặp: {str(e)}")

