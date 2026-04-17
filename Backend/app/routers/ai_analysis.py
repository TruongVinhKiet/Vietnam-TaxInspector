"""
ai_analysis.py – API Router cho Hệ thống AI Chấm điểm Rủi ro
================================================================
4 Endpoints:
    1. POST /api/ai/single-query/{tax_code}  – Tra cứu đơn lẻ real-time
    2. POST /api/ai/batch-upload              – Upload CSV & bắt đầu phân tích lô
    3. GET  /api/ai/batch-status/{batch_id}   – Kiểm tra tiến độ batch
    4. GET  /api/ai/batch-results/{batch_id}  – Lấy kết quả đầy đủ batch
    5. POST /api/ai/what-if/{tax_code}        – Mô phỏng What-If

Enhancements:
    - Single query tự lưu kết quả vào cache để What-If hoạt động ngay
    - Stale-cache detection: tái phân tích khi dữ liệu tài chính mới hơn cache
    - Hỗ trợ tra cứu bằng Tên doanh nghiệp ngoài Mã số thuế
"""

import os
import uuid
import threading
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from sqlalchemy import text, func, and_

from ..database import get_db
from .. import models, schemas

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml_engine.pipeline import TaxFraudPipeline
from ml_engine.feature_engineering import TaxFeatureEngineer

router = APIRouter(prefix="/api/ai", tags=["AI Risk Analysis"])

# ---- Upload directory ----
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

WHATIF_ALLOWED_FIELDS = {"revenue", "total_expenses"}
WHATIF_MIN_PCT = -80.0
WHATIF_MAX_PCT = 250.0
WHATIF_HEATMAP_DEFAULT_REVENUE_STEPS = [-30, -20, -10, 0, 10, 20, 30]
WHATIF_HEATMAP_DEFAULT_EXPENSE_STEPS = [30, 20, 10, 0, -10, -20, -30]
WHATIF_HEATMAP_MAX_POINTS = 225

# ---- Singleton pipeline ----
_pipeline = None


def get_pipeline() -> TaxFraudPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TaxFraudPipeline()
        _pipeline.load_models()
    return _pipeline


def _resolve_serving_model_version(explicit_version: Optional[str] = None) -> str:
    if explicit_version:
        return str(explicit_version)

    try:
        metadata = get_pipeline().get_serving_metadata()
        version = metadata.get("model_version") if isinstance(metadata, dict) else None
        if version:
            return str(version)
    except Exception:
        pass

    return "fraud-hybrid-legacy"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _normalize_yearly_history(raw_history: Any) -> list[dict]:
    if not isinstance(raw_history, list):
        return []

    dedup_by_year: dict[int, dict] = {}
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        year = _to_int(item.get("year"))
        if year is None or year <= 0:
            continue
        dedup_by_year[year] = {
            "year": year,
            "revenue": round(_to_float(item.get("revenue"), 0.0), 2),
            "total_expenses": round(_to_float(item.get("total_expenses"), 0.0), 2),
        }

    return [dedup_by_year[y] for y in sorted(dedup_by_year.keys())]


def _build_yearly_history_from_tax_returns(tax_returns: list[Any]) -> list[dict]:
    if not tax_returns:
        return []

    yearly: dict[int, dict] = {}
    for tr in tax_returns:
        filing_date = getattr(tr, "filing_date", None)
        year = filing_date.year if filing_date else None
        if year is None:
            continue
        bucket = yearly.setdefault(year, {"year": year, "revenue": 0.0, "total_expenses": 0.0})
        bucket["revenue"] += _to_float(getattr(tr, "revenue", 0.0), 0.0)
        bucket["total_expenses"] += _to_float(getattr(tr, "expenses", 0.0), 0.0)

    return [
        {
            "year": year,
            "revenue": round(values["revenue"], 2),
            "total_expenses": round(values["total_expenses"], 2),
        }
        for year, values in sorted(yearly.items(), key=lambda row: row[0])
    ]


def _build_yearly_history_from_assessment_rows(assessment_rows: list[Any]) -> list[dict]:
    if not assessment_rows:
        return []

    # assessment_rows is expected newest-first; setdefault keeps freshest value for same year.
    merged_by_year: dict[int, dict] = {}
    for assessment in assessment_rows:
        nested_history = _normalize_yearly_history(getattr(assessment, "yearly_history", None))
        for point in nested_history:
            merged_by_year.setdefault(point["year"], point)

        row_year = _to_int(getattr(assessment, "year", None))
        if row_year is None or row_year <= 0:
            continue
        merged_by_year.setdefault(
            row_year,
            {
                "year": row_year,
                "revenue": round(_to_float(getattr(assessment, "revenue", 0.0), 0.0), 2),
                "total_expenses": round(_to_float(getattr(assessment, "total_expenses", 0.0), 0.0), 2),
            },
        )

    return [merged_by_year[y] for y in sorted(merged_by_year.keys())]


def _resolve_yearly_history(
    cached_yearly_history: Any,
    tax_returns: list[Any],
    assessment_rows: Optional[list[Any]] = None,
) -> tuple[list[dict], str]:
    normalized_cache = _normalize_yearly_history(cached_yearly_history)
    if normalized_cache:
        return normalized_cache, "cache"

    from_tax_returns = _build_yearly_history_from_tax_returns(tax_returns)
    if from_tax_returns:
        return from_tax_returns, "tax_returns"

    from_assessments = _build_yearly_history_from_assessment_rows(assessment_rows or [])
    if from_assessments:
        return from_assessments, "assessment_history"

    return [], "unavailable"


def _empty_feature_analytics_payload() -> dict[str, Any]:
    return {
        "yearly_feature_scores": [],
        "previous_year_features": None,
        "feature_deltas": {},
    }


def _build_company_rows_from_yearly_history(
    yearly_history: list[dict],
    tax_code: str,
    company_name: str,
    industry: str,
) -> list[dict]:
    normalized = _normalize_yearly_history(yearly_history)
    rows: list[dict] = []

    for point in normalized:
        revenue = _to_float(point.get("revenue"), 0.0)
        total_expenses = _to_float(point.get("total_expenses"), 0.0)
        cost_of_goods = total_expenses * 0.75
        rows.append({
            "tax_code": tax_code,
            "company_name": company_name,
            "industry": industry,
            "year": int(point.get("year", 0)),
            "revenue": revenue,
            "total_expenses": total_expenses,
            "net_profit": revenue - total_expenses,
            "cost_of_goods": cost_of_goods,
            "operating_expenses": total_expenses * 0.25,
            "vat_output": revenue * 0.10,
            "vat_input": cost_of_goods * 0.10,
            "industry_avg_profit_margin": 0.08,
        })

    return rows


def _extract_feature_analytics_from_rows(company_rows: list[dict]) -> dict[str, Any]:
    if not company_rows:
        return _empty_feature_analytics_payload()

    try:
        result = get_pipeline().predict_single(company_rows)
    except Exception as exc:
        print(f"[WARN] Failed to derive single-query feature analytics: {exc}")
        return _empty_feature_analytics_payload()

    yearly_feature_scores = result.get("yearly_feature_scores")
    previous_year_features = result.get("previous_year_features")
    feature_deltas = result.get("feature_deltas")

    return {
        "yearly_feature_scores": yearly_feature_scores if isinstance(yearly_feature_scores, list) else [],
        "previous_year_features": previous_year_features if isinstance(previous_year_features, dict) else None,
        "feature_deltas": feature_deltas if isinstance(feature_deltas, dict) else {},
    }


def _normalize_yearly_feature_scores(raw_scores: Any) -> list[dict[str, float | int]]:
    if not isinstance(raw_scores, list):
        return []

    dedup_by_year: dict[int, dict[str, float | int]] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        year = _to_int(item.get("year"))
        if year is None or year <= 0:
            continue
        dedup_by_year[year] = {
            "year": year,
            "risk_score": round(_to_float(item.get("risk_score"), 0.0), 2),
            "f1_divergence": round(_to_float(item.get("f1_divergence"), 0.0), 4),
            "f2_ratio_limit": round(_to_float(item.get("f2_ratio_limit"), 0.0), 4),
            "f3_vat_structure": round(_to_float(item.get("f3_vat_structure"), 0.0), 4),
            "f4_peer_comparison": round(_to_float(item.get("f4_peer_comparison"), 0.0), 4),
        }

    return [dedup_by_year[year] for year in sorted(dedup_by_year.keys())]


def _classify_risk_tier(risk_score: Any) -> Literal["low", "medium", "high", "critical"]:
    score = _to_float(risk_score, 0.0)
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _compute_profit_margin(revenue: Any, total_expenses: Any) -> Optional[float]:
    rev = _to_float(revenue, 0.0)
    if rev <= 0:
        return None
    exp = _to_float(total_expenses, 0.0)
    return round(((rev - exp) / rev) * 100.0, 2)


def _build_single_risk_tier_sankey(yearly_feature_scores: Any) -> dict[str, Any]:
    rows = _normalize_yearly_feature_scores(yearly_feature_scores)
    if not rows:
        return {"nodes": [], "links": []}

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    previous_node_name: Optional[str] = None

    for row in rows:
        tier = _classify_risk_tier(row.get("risk_score", 0.0))
        node_name = f"{row['year']}:{tier}"
        nodes.append({
            "name": node_name,
            "year": int(row["year"]),
            "tier": tier,
            "risk_score": round(_to_float(row.get("risk_score"), 0.0), 2),
        })
        if previous_node_name:
            links.append({
                "source": previous_node_name,
                "target": node_name,
                "value": 1,
            })
        previous_node_name = node_name

    return {
        "nodes": nodes,
        "links": links,
    }


def _build_single_cumulative_risk_curve(yearly_feature_scores: Any) -> dict[str, Any]:
    rows = _normalize_yearly_feature_scores(yearly_feature_scores)
    if not rows:
        return {
            "points": [],
            "total_periods": 0,
            "total_risk": 0.0,
            "top_10pct_risk_share": 0.0,
            "top_20pct_risk_share": 0.0,
        }

    sorted_rows = sorted(rows, key=lambda item: _to_float(item.get("risk_score"), 0.0), reverse=True)
    total_periods = len(sorted_rows)
    total_risk = sum(max(0.0, _to_float(item.get("risk_score"), 0.0)) for item in sorted_rows)
    denominator = total_risk if total_risk > 0 else float(total_periods)

    points: list[dict[str, Any]] = []
    cumulative_risk = 0.0
    for idx, row in enumerate(sorted_rows, start=1):
        current_risk = max(0.0, _to_float(row.get("risk_score"), 0.0))
        cumulative_risk += current_risk
        points.append({
            "year": int(row["year"]),
            "period_count": idx,
            "percent_periods": round((idx / total_periods) * 100.0, 2),
            "percent_risk": round((cumulative_risk / denominator) * 100.0, 2) if denominator > 0 else 0.0,
        })

    top_10_count = max(1, math.ceil(total_periods * 0.10))
    top_20_count = max(1, math.ceil(total_periods * 0.20))

    return {
        "points": points,
        "total_periods": total_periods,
        "total_risk": round(total_risk, 2),
        "top_10pct_risk_share": points[top_10_count - 1]["percent_risk"] if points else 0.0,
        "top_20pct_risk_share": points[top_20_count - 1]["percent_risk"] if points else 0.0,
    }


def _find_margin_bin_index(value: float, edges: list[float]) -> Optional[int]:
    if len(edges) < 2:
        return None
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        right_inclusive = idx == len(edges) - 2
        if (value >= left and value < right) or (right_inclusive and value <= right):
            return idx
    return None


def _build_single_margin_distribution(
    db: Session,
    industry: Optional[str],
    company_margin: Optional[float],
    target_tax_code: Optional[str] = None,
) -> dict[str, Any]:
    default_payload = {
        "available": False,
        "industry": industry,
        "sample_size": 0,
        "company_margin": company_margin,
        "percentile": None,
        "mean_margin": None,
        "median_margin": None,
        "company_bin_index": None,
        "bins": [],
    }

    if company_margin is None or not math.isfinite(company_margin):
        return default_payload

    try:
        latest_subquery = (
            db.query(
                models.AIRiskAssessment.tax_code.label("tax_code"),
                func.max(models.AIRiskAssessment.created_at).label("latest_created_at"),
            )
            .filter(models.AIRiskAssessment.tax_code.isnot(None))
        )
        if industry:
            latest_subquery = latest_subquery.filter(models.AIRiskAssessment.industry == industry)
        latest_subquery = latest_subquery.group_by(models.AIRiskAssessment.tax_code).subquery()

        latest_rows_query = (
            db.query(models.AIRiskAssessment)
            .join(
                latest_subquery,
                and_(
                    models.AIRiskAssessment.tax_code == latest_subquery.c.tax_code,
                    models.AIRiskAssessment.created_at == latest_subquery.c.latest_created_at,
                ),
            )
        )
        if industry:
            latest_rows_query = latest_rows_query.filter(models.AIRiskAssessment.industry == industry)

        assessment_rows = latest_rows_query.limit(5000).all()
    except Exception as exc:
        print(f"[WARN] Failed to build margin distribution cohort: {exc}")
        return default_payload

    normalized_target_tax_code = str(target_tax_code).strip() if target_tax_code is not None else ""
    target_in_sample = False
    margins: list[float] = []
    for row in assessment_rows:
        margin = _compute_profit_margin(getattr(row, "revenue", 0.0), getattr(row, "total_expenses", 0.0))
        if margin is not None and math.isfinite(margin):
            margins.append(margin)
        if normalized_target_tax_code and str(getattr(row, "tax_code", "")).strip() == normalized_target_tax_code:
            target_in_sample = True

    if not target_in_sample:
        margins.append(float(company_margin))
    margins = sorted(margins)
    sample_size = len(margins)
    if sample_size == 0:
        return default_payload

    mean_margin = sum(margins) / sample_size
    if sample_size % 2 == 1:
        median_margin = margins[sample_size // 2]
    else:
        median_margin = (margins[(sample_size // 2) - 1] + margins[sample_size // 2]) / 2.0

    percentile = (sum(1 for value in margins if value <= company_margin) / sample_size) * 100.0

    edges = [-60.0, -40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0]
    if company_margin < edges[0]:
        edges.insert(0, float(math.floor(company_margin / 10.0) * 10.0))
    if company_margin > edges[-1]:
        edges.append(float(math.ceil(company_margin / 10.0) * 10.0))

    counts = [0] * (len(edges) - 1)
    for value in margins:
        bin_index = _find_margin_bin_index(value, edges)
        if bin_index is not None:
            counts[bin_index] += 1

    bins = []
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        bins.append({
            "start": left,
            "end": right,
            "label": f"{left:.0f}% đến {right:.0f}%",
            "count": counts[idx],
        })

    company_bin_index = _find_margin_bin_index(float(company_margin), edges)

    return {
        "available": sample_size >= 3,
        "industry": industry,
        "sample_size": sample_size,
        "company_margin": round(float(company_margin), 2),
        "percentile": round(percentile, 2),
        "mean_margin": round(mean_margin, 2),
        "median_margin": round(median_margin, 2),
        "company_bin_index": company_bin_index,
        "bins": bins,
    }


def _build_single_red_flags_timeline(yearly_feature_scores: Any) -> dict[str, Any]:
    rows = _normalize_yearly_feature_scores(yearly_feature_scores)
    if not rows:
        return {"year_points": [], "flags": []}

    flag_rules = [
        {
            "flag_id": "risk_high",
            "label": "Tổng điểm rủi ro cao",
            "severity": "high",
            "check": lambda row: _to_float(row.get("risk_score"), 0.0) >= 60.0,
        },
        {
            "flag_id": "f1_divergence",
            "label": "F1 lệch pha tăng trưởng",
            "severity": "high",
            "check": lambda row: _to_float(row.get("f1_divergence"), 0.0) > 0.30,
        },
        {
            "flag_id": "f2_ratio_limit",
            "label": "F2 vượt ngưỡng tỷ lệ",
            "severity": "medium",
            "check": lambda row: _to_float(row.get("f2_ratio_limit"), 0.0) > 1.20,
        },
        {
            "flag_id": "f3_vat_structure",
            "label": "F3 bất thường VAT",
            "severity": "medium",
            "check": lambda row: _to_float(row.get("f3_vat_structure"), 0.0) > 0.55,
        },
        {
            "flag_id": "f4_peer_comparison",
            "label": "F4 lệch chuẩn ngành",
            "severity": "medium",
            "check": lambda row: _to_float(row.get("f4_peer_comparison"), 0.0) > 0.25,
        },
    ]

    yearly_points: list[dict[str, Any]] = []
    aggregated_flags: dict[str, dict[str, Any]] = {}

    for row in rows:
        year = int(row["year"])
        triggered_ids: list[str] = []
        for rule in flag_rules:
            if not rule["check"](row):
                continue
            flag_id = str(rule["flag_id"])
            triggered_ids.append(flag_id)

            bucket = aggregated_flags.setdefault(
                flag_id,
                {
                    "flag_id": flag_id,
                    "label": str(rule["label"]),
                    "severity": str(rule["severity"]),
                    "years": [],
                },
            )
            bucket["years"].append(year)

        yearly_points.append({
            "year": year,
            "flag_count": len(triggered_ids),
            "flag_ids": triggered_ids,
        })

    flags: list[dict[str, Any]] = []
    for item in aggregated_flags.values():
        years = sorted(set(_to_int(year) for year in item.get("years", []) if _to_int(year) is not None))
        if not years:
            continue
        flags.append({
            "flag_id": item["flag_id"],
            "label": item["label"],
            "severity": item["severity"],
            "first_year": years[0],
            "last_year": years[-1],
            "trigger_count": len(years),
            "years": years,
        })

    flags.sort(key=lambda item: (-item["trigger_count"], item["first_year"]))
    return {
        "year_points": yearly_points,
        "flags": flags,
    }


def _build_single_extended_charts_payload(
    db: Session,
    yearly_feature_scores: Any,
    tax_code: Optional[str],
    industry: Optional[str],
    revenue: Any,
    total_expenses: Any,
) -> dict[str, Any]:
    company_margin = _compute_profit_margin(revenue, total_expenses)
    return {
        "single_risk_tier_sankey": _build_single_risk_tier_sankey(yearly_feature_scores),
        "single_cumulative_risk_curve": _build_single_cumulative_risk_curve(yearly_feature_scores),
        "single_margin_distribution": _build_single_margin_distribution(
            db=db,
            industry=industry,
            company_margin=company_margin,
            target_tax_code=tax_code,
        ),
        "single_red_flags_timeline": _build_single_red_flags_timeline(yearly_feature_scores),
    }


def _append_fraud_inference_metric(result_payload: dict, source: str, endpoint: str = "single_query") -> None:
    """Best-effort feature telemetry for fraud drift monitoring."""
    try:
        metrics_file = LOG_DIR / "ml_metrics.jsonl"
        labels = {
            "endpoint": endpoint,
            "source": source,
            "f1_divergence": _to_float(result_payload.get("f1_divergence"), 0.0),
            "f2_ratio_limit": _to_float(result_payload.get("f2_ratio_limit"), 0.0),
            "f3_vat_structure": _to_float(result_payload.get("f3_vat_structure"), 0.0),
            "f4_peer_comparison": _to_float(result_payload.get("f4_peer_comparison"), 0.0),
            "anomaly_score": _to_float(result_payload.get("anomaly_score"), 0.0),
            "revenue": _to_float(result_payload.get("revenue"), 0.0),
            "total_expenses": _to_float(result_payload.get("total_expenses"), 0.0),
            "model_confidence": _to_float(result_payload.get("model_confidence"), 0.0),
            "model_version": str(result_payload.get("model_version") or "fraud-hybrid-legacy"),
        }
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric": "fraud_inference_features",
            "value": _to_float(result_payload.get("risk_score"), 0.0),
            "labels": labels,
        }
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[WARN] Failed to log fraud inference telemetry: {exc}")


def _validate_whatif_adjustments(raw_adjustments: Any) -> dict[str, float]:
    if not isinstance(raw_adjustments, dict):
        raise ValueError("Payload điều chỉnh phải là object JSON.")

    if not raw_adjustments:
        raise ValueError("Vui lòng cung cấp ít nhất một điều chỉnh (revenue hoặc total_expenses).")

    normalized: dict[str, float] = {}
    unsupported = [k for k in raw_adjustments.keys() if k not in WHATIF_ALLOWED_FIELDS]
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise ValueError(f"Chỉ hỗ trợ điều chỉnh các trường: revenue, total_expenses. Trường không hợp lệ: {joined}")

    for field_name, raw_value in raw_adjustments.items():
        try:
            pct = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Giá trị điều chỉnh cho '{field_name}' phải là số phần trăm hợp lệ.")

        if not math.isfinite(pct):
            raise ValueError(f"Giá trị điều chỉnh cho '{field_name}' phải là số hữu hạn.")

        if pct < WHATIF_MIN_PCT or pct > WHATIF_MAX_PCT:
            raise ValueError(
                f"Điều chỉnh '{field_name}' phải nằm trong khoảng {WHATIF_MIN_PCT:.0f}% đến {WHATIF_MAX_PCT:.0f}%"
            )

        normalized[field_name] = round(pct, 2)

    return normalized


def _normalize_whatif_step_list(
    raw_steps: Any,
    *,
    field_name: str,
    default_steps: list[float],
) -> list[float]:
    if raw_steps is None:
        return [float(v) for v in default_steps]

    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"Danh sach '{field_name}_steps' phai la mang JSON khong rong.")

    normalized: list[float] = []
    seen: set[float] = set()

    for raw_value in raw_steps:
        try:
            pct = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Moi gia tri trong '{field_name}_steps' phai la so phan tram hop le.")

        if not math.isfinite(pct):
            raise ValueError(f"Moi gia tri trong '{field_name}_steps' phai la so huu han.")

        if pct < WHATIF_MIN_PCT or pct > WHATIF_MAX_PCT:
            raise ValueError(
                f"Gia tri trong '{field_name}_steps' phai nam trong khoang {WHATIF_MIN_PCT:.0f}% den {WHATIF_MAX_PCT:.0f}%"
            )

        pct_rounded = round(pct, 2)
        if pct_rounded in seen:
            continue
        seen.add(pct_rounded)
        normalized.append(pct_rounded)

    if not normalized:
        raise ValueError(f"Danh sach '{field_name}_steps' khong chua gia tri hop le nao.")

    return normalized


def _get_latest_cached_assessment(tax_code: str, db: Session) -> Optional[models.AIRiskAssessment]:
    return (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.tax_code == tax_code)
        .order_by(models.AIRiskAssessment.created_at.desc())
        .first()
    )


def _build_whatif_base_data_from_assessment(cached: models.AIRiskAssessment) -> dict[str, Any]:
    revenue = float(cached.revenue or 0)
    total_expenses = float(cached.total_expenses or 0)
    cost_of_goods = total_expenses * 0.75

    return {
        "tax_code": cached.tax_code,
        "company_name": cached.company_name or "",
        "industry": cached.industry or "",
        "year": cached.year,
        "revenue": revenue,
        "total_expenses": total_expenses,
        "net_profit": revenue - total_expenses,
        "cost_of_goods": cost_of_goods,
        "operating_expenses": total_expenses * 0.25,
        "vat_output": revenue * 0.10,
        "vat_input": cost_of_goods * 0.10,
        "industry_avg_profit_margin": 0.08,
    }


def _build_whatif_heatmap_values(
    *,
    pipeline: TaxFraudPipeline,
    base_data: dict[str, Any],
    original_risk_score: float,
    revenue_steps: list[float],
    expense_steps: list[float],
) -> list[list[float]]:
    values: list[list[float]] = []

    for y_idx, expense_pct in enumerate(expense_steps):
        for x_idx, revenue_pct in enumerate(revenue_steps):
            adjustments: dict[str, float] = {}
            if revenue_pct != 0:
                adjustments["revenue"] = revenue_pct
            if expense_pct != 0:
                adjustments["total_expenses"] = expense_pct

            simulated = pipeline.predict_whatif(base_data, adjustments)
            simulated_risk = round(_to_float(simulated.get("simulated_risk_score"), original_risk_score), 2)
            delta_risk = round(simulated_risk - original_risk_score, 2)

            values.append([
                x_idx,
                y_idx,
                simulated_risk,
                delta_risk,
            ])

    return values


# ==================================================================
# 1. SINGLE QUERY (Real-time)
# ==================================================================
@router.post("/single-query/{tax_code}", response_model=schemas.RiskAssessmentDetail)
def single_query(tax_code: str, db: Session = Depends(get_db)):
    """
    Chế độ 1: Tra cứu đơn lẻ.
    Tìm dữ liệu 3 năm gần nhất trong DB, chạy AI pipeline real-time.
    Nếu không có trong DB, trả về lỗi kèm gợi ý upload CSV.
    Hỗ trợ tra cứu bằng Tên DN: nếu tax_code không phải là số thuần,
    hệ thống sẽ tìm DN theo tên (ILIKE) rồi lấy MST.
    """
    resolved_tax_code = tax_code
    company = None

    # --- Hỗ trợ tra cứu bằng tên doanh nghiệp ---
    if not tax_code.replace("-", "").isdigit():
        # Treat input as company name search
        company = db.query(models.Company).filter(
            models.Company.name.ilike(f"%{tax_code}%")
        ).first()
        if company:
            resolved_tax_code = company.tax_code
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy doanh nghiệp tên '{tax_code}'. "
                       "Hãy nhập MST chính xác hoặc upload CSV."
            )
    else:
        company = db.query(models.Company).filter(
            models.Company.tax_code == resolved_tax_code
        ).first()

    # Check if we have tax returns for this company
    tax_returns = (
        db.query(models.TaxReturn)
        .filter(models.TaxReturn.tax_code == resolved_tax_code)
        .order_by(models.TaxReturn.filing_date.desc())
        .limit(120)  # keep a wider history window for yearly trend rebuilds
        .all()
    )

    # Check if we have a cached risk assessment
    cached_assessment = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.tax_code == resolved_tax_code)
        .order_by(models.AIRiskAssessment.created_at.desc())
        .first()
    )

    # --- Stale-cache detection ---
    # Two conditions invalidate cache:
    # 1) Newer financial data exists than cached assessment
    # 2) Cache is older than 30 days (ensures periodic model re-evaluation)
    cache_is_stale = False
    if cached_assessment:
        # Condition 1: newer filing data
        if tax_returns:
            newest_filing = max(
                (tr.filing_date for tr in tax_returns if tr.filing_date),
                default=None,
            )
            if newest_filing and cached_assessment.created_at:
                cache_created = cached_assessment.created_at
                from datetime import date as date_type
                filing_dt = datetime.combine(newest_filing, datetime.min.time()) \
                    if isinstance(newest_filing, date_type) else newest_filing
                cache_dt = cache_created.replace(tzinfo=None) \
                    if hasattr(cache_created, 'tzinfo') and cache_created.tzinfo else cache_created
                if filing_dt > cache_dt:
                    cache_is_stale = True

        # Condition 2: cache age > 30 days
        if not cache_is_stale and cached_assessment.created_at:
            from datetime import timedelta
            cache_dt = cached_assessment.created_at.replace(tzinfo=None) \
                if hasattr(cached_assessment.created_at, 'tzinfo') and cached_assessment.created_at.tzinfo \
                else cached_assessment.created_at
            if (datetime.utcnow() - cache_dt) > timedelta(days=30):
                cache_is_stale = True

    if cached_assessment and not cache_is_stale:
        yearly_history, history_source = _resolve_yearly_history(
            cached_yearly_history=cached_assessment.yearly_history,
            tax_returns=tax_returns,
            assessment_rows=[],
        )
        # If cache has sparse history, prefer richer fallback sources when available.
        tax_return_years = {
            tr.filing_date.year
            for tr in tax_returns
            if getattr(tr, "filing_date", None)
        }
        if len(yearly_history) <= 1 and len(tax_return_years) > len(yearly_history):
            rebuilt_from_tax_returns = _build_yearly_history_from_tax_returns(tax_returns)
            if len(rebuilt_from_tax_returns) > len(yearly_history):
                yearly_history = rebuilt_from_tax_returns
                history_source = "tax_returns"

        if len(yearly_history) <= 1:
            assessment_history = (
                db.query(models.AIRiskAssessment)
                .filter(models.AIRiskAssessment.tax_code == resolved_tax_code)
                .order_by(models.AIRiskAssessment.created_at.desc())
                .limit(60)
                .all()
            )
            rebuilt_history, rebuilt_source = _resolve_yearly_history(
                cached_yearly_history=cached_assessment.yearly_history,
                tax_returns=tax_returns,
                assessment_rows=assessment_history,
            )
            if len(rebuilt_history) > len(yearly_history):
                yearly_history = rebuilt_history
                history_source = rebuilt_source

        # Model confidence: read real value from DB column if available,
        # fallback to approximation for legacy records without the column
        if cached_assessment.model_confidence is not None:
            cached_confidence = cached_assessment.model_confidence
        else:
            cached_risk = cached_assessment.risk_score or 0
            fraud_prob_approx = cached_risk / 100.0
            cached_confidence = round(max(fraud_prob_approx, 1 - fraud_prob_approx) * 100, 1)

        feature_analytics = _empty_feature_analytics_payload()
        if yearly_history:
            feature_analytics = _extract_feature_analytics_from_rows(
                _build_company_rows_from_yearly_history(
                    yearly_history=yearly_history,
                    tax_code=cached_assessment.tax_code or resolved_tax_code,
                    company_name=cached_assessment.company_name or (company.name if company else ""),
                    industry=cached_assessment.industry or (company.industry if company else ""),
                )
            )

        extended_charts = _build_single_extended_charts_payload(
            db=db,
            yearly_feature_scores=feature_analytics["yearly_feature_scores"],
            tax_code=cached_assessment.tax_code or resolved_tax_code,
            industry=cached_assessment.industry or (company.industry if company else None),
            revenue=cached_assessment.revenue,
            total_expenses=cached_assessment.total_expenses,
        )

        # Return cached result
        response_payload = {
            "tax_code": cached_assessment.tax_code,
            "company_name": cached_assessment.company_name or (company.name if company else ""),
            "industry": cached_assessment.industry or (company.industry if company else ""),
            "year": cached_assessment.year,
            "revenue": float(cached_assessment.revenue or 0),
            "total_expenses": float(cached_assessment.total_expenses or 0),
            "f1_divergence": cached_assessment.f1_divergence,
            "f2_ratio_limit": cached_assessment.f2_ratio_limit,
            "f3_vat_structure": cached_assessment.f3_vat_structure,
            "f4_peer_comparison": cached_assessment.f4_peer_comparison,
            "anomaly_score": cached_assessment.anomaly_score,
            "model_confidence": cached_confidence,
            "model_version": _resolve_serving_model_version(cached_assessment.model_version),
            "risk_score": cached_assessment.risk_score,
            "risk_level": cached_assessment.risk_level,
            "red_flags": cached_assessment.red_flags or [],
            "shap_explanation": cached_assessment.shap_explanation or [],
            "yearly_feature_scores": feature_analytics["yearly_feature_scores"],
            "previous_year_features": feature_analytics["previous_year_features"],
            "feature_deltas": feature_analytics["feature_deltas"],
            "single_risk_tier_sankey": extended_charts["single_risk_tier_sankey"],
            "single_cumulative_risk_curve": extended_charts["single_cumulative_risk_curve"],
            "single_margin_distribution": extended_charts["single_margin_distribution"],
            "single_red_flags_timeline": extended_charts["single_red_flags_timeline"],
            "history_source": history_source,
            "history_year_count": len(yearly_history),
            "yearly_history": yearly_history,
            "source": "cached",
        }
        _append_fraud_inference_metric(response_payload, source="cached")
        return response_payload

    # If we have tax_returns data, build financial data for pipeline
    if tax_returns and len(tax_returns) >= 1:
        import pandas as pd

        # Aggregate quarterly data into yearly
        yearly_data = {}
        for tr in tax_returns:
            year = tr.filing_date.year if tr.filing_date else 2024
            if year not in yearly_data:
                yearly_data[year] = {
                    "tax_code": resolved_tax_code,
                    "company_name": company.name if company else "",
                    "industry": company.industry if company else "",
                    "year": year,
                    "revenue": 0,
                    "cost_of_goods": 0,
                    "operating_expenses": 0,
                    "total_expenses": 0,
                    "net_profit": 0,
                    "vat_input": 0,
                    "vat_output": 0,
                    "industry_avg_profit_margin": 0.08,
                }
            yd = yearly_data[year]
            yd["revenue"] += float(tr.revenue or 0)
            yd["total_expenses"] += float(tr.expenses or 0)
            yd["net_profit"] = yd["revenue"] - yd["total_expenses"]
            yd["cost_of_goods"] = yd["total_expenses"] * 0.75
            yd["operating_expenses"] = yd["total_expenses"] * 0.25
            yd["vat_output"] = yd["revenue"] * 0.10
            yd["vat_input"] = yd["cost_of_goods"] * 0.10

        company_data = list(yearly_data.values())
        pipeline = get_pipeline()
        try:
            result = pipeline.predict_single(company_data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        result["source"] = "realtime"
        result["model_version"] = _resolve_serving_model_version(result.get("model_version"))

        # Build yearly history for trend chart + cache persistence
        yearly_history_for_cache = sorted([
            {"year": d["year"], "revenue": d["revenue"], "total_expenses": d["total_expenses"]}
            for d in company_data
        ], key=lambda x: x["year"])
        yearly_history_for_cache = _normalize_yearly_history(yearly_history_for_cache)
        history_source = "tax_returns_aggregation" if yearly_history_for_cache else "unavailable"

        if not yearly_history_for_cache:
            assessment_history = (
                db.query(models.AIRiskAssessment)
                .filter(models.AIRiskAssessment.tax_code == resolved_tax_code)
                .order_by(models.AIRiskAssessment.created_at.desc())
                .limit(60)
                .all()
            )
            yearly_history_for_cache = _build_yearly_history_from_assessment_rows(assessment_history)
            if yearly_history_for_cache:
                history_source = "assessment_history"

        # --- Save result into cache (AIRiskAssessment) so What-If works immediately ---
        try:
            new_assessment = models.AIRiskAssessment(
                batch_id=None,
                tax_code=result.get("tax_code", resolved_tax_code),
                company_name=result.get("company_name"),
                industry=result.get("industry"),
                year=result.get("year"),
                revenue=result.get("revenue"),
                total_expenses=result.get("total_expenses"),
                f1_divergence=result.get("f1_divergence"),
                f2_ratio_limit=result.get("f2_ratio_limit"),
                f3_vat_structure=result.get("f3_vat_structure"),
                f4_peer_comparison=result.get("f4_peer_comparison"),
                anomaly_score=result.get("anomaly_score"),
                model_confidence=result.get("model_confidence"),
                model_version=result.get("model_version"),
                risk_score=result.get("risk_score", 0),
                risk_level=result.get("risk_level", "low"),
                red_flags=result.get("red_flags"),
                shap_explanation=result.get("shap_explanation"),
                yearly_history=yearly_history_for_cache,
            )
            db.add(new_assessment)
            db.commit()
        except Exception as cache_err:
            db.rollback()
            print(f"[WARN] Failed to cache realtime assessment: {cache_err}")

        result["yearly_history"] = yearly_history_for_cache
        result["history_source"] = history_source
        result["history_year_count"] = len(yearly_history_for_cache)
        result.setdefault("yearly_feature_scores", [])
        result.setdefault("previous_year_features", None)
        result.setdefault("feature_deltas", {})

        result.update(
            _build_single_extended_charts_payload(
                db=db,
                yearly_feature_scores=result.get("yearly_feature_scores"),
                tax_code=result.get("tax_code", resolved_tax_code),
                industry=result.get("industry"),
                revenue=result.get("revenue"),
                total_expenses=result.get("total_expenses"),
            )
        )

        _append_fraud_inference_metric(result, source="realtime")

        return result

    # No data at all - return informative error
    if not company:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy doanh nghiệp MST {resolved_tax_code}. "
                   "Hãy upload file CSV để nhập dữ liệu tài chính."
        )

    # Company exists but no financial data
    raise HTTPException(
        status_code=404,
        detail=f"DN {company.name} (MST: {resolved_tax_code}) chưa có dữ liệu tài chính. "
               "Hãy upload file CSV chứa báo cáo tài chính."
    )


# ==================================================================
# 1.1 COMPANY DIRECTORY (All DB vs Assessed)
# ==================================================================
@router.get("/companies", response_model=schemas.RiskCompanyListResponse)
def list_companies(
    mode: Literal["all", "assessed"] = Query("all"),
    q: Optional[str] = Query(None, min_length=1, max_length=120),
    industry: Optional[str] = Query(None, min_length=1, max_length=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=5, le=100),
    sort_by: Literal["risk_score", "name", "tax_code", "updated_at"] = Query("risk_score"),
    sort_order: Literal["asc", "desc"] = Query("desc"),
    db: Session = Depends(get_db),
):
    base_join = """
        FROM companies c
        LEFT JOIN (
            SELECT
                tax_code,
                MAX(created_at) AS latest_assessment_at,
                COUNT(*) AS assessment_count,
                (ARRAY_AGG(risk_score ORDER BY created_at DESC NULLS LAST))[1] AS latest_risk_score
            FROM ai_risk_assessments
            GROUP BY tax_code
        ) aa ON aa.tax_code = c.tax_code
    """

    where_parts = ["1=1"]
    params: dict[str, Any] = {}

    normalized_q = q.strip() if q else ""
    if normalized_q:
        params["q"] = f"%{normalized_q}%"
        where_parts.append("(c.tax_code ILIKE :q OR c.name ILIKE :q)")

    normalized_industry = industry.strip() if industry else ""
    if normalized_industry:
        params["industry"] = normalized_industry
        where_parts.append("c.industry = :industry")

    if mode == "assessed":
        where_parts.append("COALESCE(aa.assessment_count, 0) > 0")

    where_clause = " AND ".join(where_parts)

    sort_map = {
        "risk_score": "COALESCE(aa.latest_risk_score, c.risk_score, 0)",
        "name": "c.name",
        "tax_code": "c.tax_code",
        "updated_at": "aa.latest_assessment_at",
    }
    sort_expr = sort_map.get(sort_by, sort_map["risk_score"])
    direction = "ASC" if sort_order == "asc" else "DESC"

    count_sql = f"""
        SELECT COUNT(*)
        {base_join}
        WHERE {where_clause}
    """

    total = int(db.execute(text(count_sql), params).scalar() or 0)
    total_pages = max(1, (total + page_size - 1) // page_size)
    current_page = min(page, total_pages)
    offset = (current_page - 1) * page_size

    data_sql = f"""
        SELECT
            c.tax_code,
            c.name,
            c.industry,
            c.is_active,
            COALESCE(aa.latest_risk_score, c.risk_score, 0) AS display_risk_score,
            aa.latest_risk_score,
            COALESCE(aa.assessment_count, 0) AS assessment_count,
            aa.latest_assessment_at
        {base_join}
        WHERE {where_clause}
        ORDER BY {sort_expr} {direction} NULLS LAST, c.tax_code ASC
        LIMIT :limit OFFSET :offset
    """

    rows = db.execute(
        text(data_sql),
        {
            **params,
            "limit": page_size,
            "offset": offset,
        },
    ).fetchall()

    results = []
    for row in rows:
        latest_assessment_at = row[7].isoformat() if row[7] else None
        assessment_count = int(row[6] or 0)
        results.append(
            {
                "tax_code": row[0],
                "name": row[1],
                "industry": row[2],
                "is_active": bool(row[3]),
                "risk_score": round(_to_float(row[4], 0.0), 2),
                "latest_risk_score": round(_to_float(row[5], 0.0), 2) if row[5] is not None else None,
                "assessment_count": assessment_count,
                "latest_assessment_at": latest_assessment_at,
                "assessed": assessment_count > 0,
            }
        )

    return {
        "mode": mode,
        "page": current_page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "results": results,
    }


# ==================================================================
# 2. BATCH UPLOAD
# ==================================================================
@router.post("/batch-upload")
async def batch_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Chế độ 2: Upload file CSV để phân tích lô.
    - Lưu file vào disk
    - Tạo batch record
    - Khởi động background task (Celery hoặc thread đồng bộ)
    - Trả về batch_id cho Frontend polling
    """
    # Validate file
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file CSV (.csv)")

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    save_filename = f"{file_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_filename

    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > 200:  # 200MB limit
        raise HTTPException(status_code=400, detail="File quá lớn (tối đa 200MB)")

    with open(save_path, "wb") as f:
        f.write(content)

    # Create batch record in DB
    batch = models.AIAnalysisBatch(
        filename=file.filename,
        file_path=str(save_path),
        status="pending",
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    # Try Celery first, fallback to synchronous thread
    try:
        from app.worker import CELERY_AVAILABLE
        from app.tasks import analyze_batch_csv_task, run_batch_analysis

        if CELERY_AVAILABLE:
            # Dispatch to Celery worker
            analyze_batch_csv_task.delay(str(save_path), batch.id)
        else:
            # Fallback: run in background thread
            thread = threading.Thread(
                target=run_batch_analysis,
                args=(str(save_path), batch.id),
                daemon=True,
            )
            thread.start()
    except Exception:
        # Ultimate fallback: background thread
        from app.tasks import run_batch_analysis
        thread = threading.Thread(
            target=run_batch_analysis,
            args=(str(save_path), batch.id),
            daemon=True,
        )
        thread.start()

    return {
        "batch_id": batch.id,
        "filename": file.filename,
        "file_size_mb": round(file_size_mb, 2),
        "status": "pending",
        "message": "File đã được nhận. Hệ thống AI đang xử lý...",
    }


# ==================================================================
# 3. BATCH STATUS (for polling)
# ==================================================================
@router.get("/batch-status/{batch_id}")
def batch_status(batch_id: int, db: Session = Depends(get_db)):
    """
    Kiểm tra tiến độ xử lý batch.
    Frontend dùng setInterval() để gọi endpoint này mỗi 1-2 giây.
    """
    batch = db.query(models.AIAnalysisBatch).filter(
        models.AIAnalysisBatch.id == batch_id
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch không tồn tại")

    progress = 0
    if batch.total_rows and batch.total_rows > 0:
        progress = round((batch.processed_rows or 0) / batch.total_rows * 100, 1)

    return {
        "batch_id": batch.id,
        "filename": batch.filename,
        "status": batch.status,
        "total_rows": batch.total_rows,
        "processed_rows": batch.processed_rows,
        "progress_percent": progress,
        "error_message": batch.error_message,
        "created_at": str(batch.created_at) if batch.created_at else None,
        "started_at": str(batch.started_at) if batch.started_at else None,
        "completed_at": str(batch.completed_at) if batch.completed_at else None,
    }


# ==================================================================
# 4. BATCH RESULTS (full dashboard data)
# ==================================================================
@router.get("/batch-results/{batch_id}")
def batch_results(batch_id: int, db: Session = Depends(get_db)):
    """
    Lấy kết quả đầy đủ batch: thống kê, top 50, scatter data...
    Chỉ trả về khi status == 'done'.
    """
    batch = db.query(models.AIAnalysisBatch).filter(
        models.AIAnalysisBatch.id == batch_id
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch không tồn tại")

    if batch.status != "done":
        raise HTTPException(
            status_code=202,
            detail=f"Batch đang ở trạng thái: {batch.status}. Vui lòng đợi xử lý hoàn tất."
        )

    # Get individual assessments for the data table
    assessments = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.batch_id == batch_id)
        .order_by(models.AIRiskAssessment.risk_score.desc())
        .all()
    )

    assessment_list = []
    for a in assessments:
        assessment_list.append({
            "tax_code": a.tax_code,
            "company_name": a.company_name,
            "industry": a.industry,
            "year": a.year,
            "revenue": float(a.revenue or 0),
            "total_expenses": float(a.total_expenses or 0),
            "f1_divergence": a.f1_divergence,
            "f2_ratio_limit": a.f2_ratio_limit,
            "f3_vat_structure": a.f3_vat_structure,
            "f4_peer_comparison": a.f4_peer_comparison,
            "anomaly_score": a.anomaly_score,
            "model_version": a.model_version,
            "risk_score": a.risk_score,
            "risk_level": a.risk_level,
            "red_flags": a.red_flags or [],
        })

    return {
        "batch_id": batch.id,
        "filename": batch.filename,
        "status": batch.status,
        "total_records": batch.total_rows or 0,
        "total_companies": len(assessment_list),
        "statistics": batch.result_summary or {},
        "assessments": assessment_list,
    }


# ==================================================================
# 5. WHAT-IF SIMULATION (Scenario Analysis)
# ==================================================================
@router.post("/what-if/{tax_code}")
def what_if_simulation(tax_code: str, adjustments: dict, db: Session = Depends(get_db)):
    """
    Chế độ 5: Mô phỏng Tình huống (What-If Analysis).
    Nhận điều chỉnh % cho các chỉ số tài chính, chạy lại AI pipeline
    và trả về điểm rủi ro mới để so sánh.
    
    Body JSON example:
    {
        "revenue": -20,          // Giảm doanh thu 20%
        "total_expenses": 30     // Tăng chi phí 30%
    }
    """
    # Find cached assessment
    cached = _get_latest_cached_assessment(tax_code, db)

    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"Chưa có dữ liệu phân tích cho MST {tax_code}. Hãy phân tích AI trước."
        )

    try:
        normalized_adjustments = _validate_whatif_adjustments(adjustments)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Build base data from cached assessment
    base_data = _build_whatif_base_data_from_assessment(cached)

    pipeline = get_pipeline()
    try:
        result = pipeline.predict_whatif(base_data, normalized_adjustments)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result["applied_adjustments"] = normalized_adjustments
    result["original_risk_score"] = cached.risk_score
    result["original_risk_level"] = cached.risk_level
    result["delta_risk"] = round(result["simulated_risk_score"] - cached.risk_score, 2)

    return result


@router.post("/what-if-grid/{tax_code}")
def what_if_grid(tax_code: str, payload: Optional[dict] = None, db: Session = Depends(get_db)):
    """
    Tạo ma trận What-If cho sensitivity heatmap từ pipeline backend (chính xác theo mô hình).

    Body JSON optional:
    {
        "revenue_steps": [-30, -20, -10, 0, 10, 20, 30],
        "expense_steps": [30, 20, 10, 0, -10, -20, -30]
    }
    """
    cached = _get_latest_cached_assessment(tax_code, db)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"Chưa có dữ liệu phân tích cho MST {tax_code}. Hãy phân tích AI trước."
        )

    body = payload if isinstance(payload, dict) else {}

    try:
        revenue_steps = _normalize_whatif_step_list(
            body.get("revenue_steps"),
            field_name="revenue",
            default_steps=WHATIF_HEATMAP_DEFAULT_REVENUE_STEPS,
        )
        expense_steps = _normalize_whatif_step_list(
            body.get("expense_steps"),
            field_name="expense",
            default_steps=WHATIF_HEATMAP_DEFAULT_EXPENSE_STEPS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    grid_points = len(revenue_steps) * len(expense_steps)
    if grid_points > WHATIF_HEATMAP_MAX_POINTS:
        raise HTTPException(
            status_code=422,
            detail=f"So o heatmap vuot gioi han ({grid_points} > {WHATIF_HEATMAP_MAX_POINTS}). Giam so luong buoc revenue/expense."
        )

    original_risk_score = round(_to_float(cached.risk_score, 0.0), 2)
    base_data = _build_whatif_base_data_from_assessment(cached)
    pipeline = get_pipeline()

    try:
        values = _build_whatif_heatmap_values(
            pipeline=pipeline,
            base_data=base_data,
            original_risk_score=original_risk_score,
            revenue_steps=revenue_steps,
            expense_steps=expense_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    simulated_scores = [row[2] for row in values]
    min_score = round(min(simulated_scores), 2) if simulated_scores else original_risk_score
    max_score = round(max(simulated_scores), 2) if simulated_scores else original_risk_score

    return {
        "tax_code": cached.tax_code,
        "source": "what_if_backend",
        "original_risk_score": original_risk_score,
        "original_risk_level": cached.risk_level,
        "revenue_steps": revenue_steps,
        "expense_steps": expense_steps,
        "values": values,
        "grid_size": {
            "rows": len(expense_steps),
            "cols": len(revenue_steps),
            "points": grid_points,
        },
        "summary": {
            "min_simulated_risk": min_score,
            "max_simulated_risk": max_score,
            "max_delta_risk": round(max_score - original_risk_score, 2),
            "min_delta_risk": round(min_score - original_risk_score, 2),
        },
    }
