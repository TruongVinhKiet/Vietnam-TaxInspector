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
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from sqlalchemy import text, func, and_

from ..database import get_db
from .. import auth, models, schemas
from . import monitoring as monitoring_router

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

SPECIALIZED_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

AUDIT_VALUE_FEATURE_COLUMNS = [
    "risk_score",
    "anomaly_score",
    "model_confidence",
    "revenue",
    "total_expenses",
    "margin_gap",
    "expense_to_revenue_ratio",
    "red_flag_count",
    "high_red_flag_count",
    "f1_divergence",
    "f2_ratio_limit",
    "f3_vat_structure",
    "f4_peer_comparison",
]

VAT_REFUND_FEATURE_COLUMNS = [
    "risk_score",
    "anomaly_score",
    "f2_ratio_limit",
    "f3_vat_structure",
    "revenue",
    "total_expenses",
    "vat_input_output_ratio",
    "estimated_refund_gap",
    "vat_flag_count",
    "expense_to_revenue_ratio",
    "f1_divergence",
    "f4_peer_comparison",
]

VAT_KEYWORDS = ("vat", "hoa don", "hoan thue", "invoice", "input", "output")

SPECIALIZED_MODEL_SPECS = {
    "audit_value": {
        "model_path": SPECIALIZED_MODEL_DIR / "audit_value_model.joblib",
        "calibrator_path": SPECIALIZED_MODEL_DIR / "audit_value_calibrator.joblib",
        "meta_path": SPECIALIZED_MODEL_DIR / "audit_value_model_meta.json",
        "default_features": AUDIT_VALUE_FEATURE_COLUMNS,
        "default_model_version": "audit-value-heuristic",
    },
    "vat_refund": {
        "model_path": SPECIALIZED_MODEL_DIR / "vat_refund_model.joblib",
        "calibrator_path": SPECIALIZED_MODEL_DIR / "vat_refund_calibrator.joblib",
        "meta_path": SPECIALIZED_MODEL_DIR / "vat_refund_model_meta.json",
        "default_features": VAT_REFUND_FEATURE_COLUMNS,
        "default_model_version": "vat-refund-heuristic",
    },
}

_specialized_model_cache: dict[str, Optional[dict[str, Any]]] = {}

REAL_LABEL_ORIGINS = {"manual_inspector", "field_verified", "imported_casework"}
BLOCKED_LABEL_ORIGINS = {"bootstrap_generated", "auto_seed"}
KNOWN_LABEL_ORIGINS = REAL_LABEL_ORIGINS | BLOCKED_LABEL_ORIGINS

# ---- Singleton pipeline ----
_pipeline = None


def get_pipeline() -> TaxFraudPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TaxFraudPipeline()
        _pipeline.load_models()
    return _pipeline


def _load_specialized_track_bundle(track_name: str) -> Optional[dict[str, Any]]:
    if track_name in _specialized_model_cache:
        return _specialized_model_cache[track_name]

    spec = SPECIALIZED_MODEL_SPECS.get(track_name)
    if not spec:
        _specialized_model_cache[track_name] = None
        return None

    model_path = spec["model_path"]
    if not model_path.exists():
        _specialized_model_cache[track_name] = None
        return None

    try:
        model_obj = joblib.load(model_path)
    except Exception as exc:
        print(f"[WARN] Failed to load {track_name} model artifact {model_path}: {exc}")
        _specialized_model_cache[track_name] = None
        return None

    calibrator = None
    calibrator_path = spec["calibrator_path"]
    if calibrator_path.exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception as exc:
            print(f"[WARN] Failed to load {track_name} calibrator artifact {calibrator_path}: {exc}")
            calibrator = None

    metadata: dict[str, Any] = {}
    meta_path = spec["meta_path"]
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                raw_meta = json.load(handle)
                if isinstance(raw_meta, dict):
                    metadata = raw_meta
        except Exception as exc:
            print(f"[WARN] Failed to read {track_name} metadata {meta_path}: {exc}")

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        feature_columns = list(spec["default_features"])
    feature_columns = [str(item) for item in feature_columns if str(item).strip()]
    if not feature_columns:
        feature_columns = list(spec["default_features"])

    bundle = {
        "model": model_obj,
        "calibrator": calibrator,
        "feature_columns": feature_columns,
        "model_version": str(metadata.get("model_version") or spec["default_model_version"]),
    }
    _specialized_model_cache[track_name] = bundle
    return bundle


def _predict_specialized_probability(track_name: str, feature_map: dict[str, float]) -> Optional[dict[str, Any]]:
    bundle = _load_specialized_track_bundle(track_name)
    if not bundle:
        return None

    feature_columns = bundle.get("feature_columns") or []
    vector = np.asarray(
        [[_to_float(feature_map.get(name), 0.0) for name in feature_columns]],
        dtype=float,
    )
    vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)

    model_obj = bundle.get("model")
    if model_obj is None:
        return None

    try:
        if hasattr(model_obj, "predict_proba"):
            proba = np.asarray(model_obj.predict_proba(vector), dtype=float)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                raw_probability = float(proba[0, 1])
            elif proba.ndim == 2 and proba.shape[1] == 1:
                raw_probability = float(proba[0, 0])
            else:
                raw_probability = float(np.ravel(proba)[0])
        elif hasattr(model_obj, "decision_function"):
            decision = float(np.ravel(model_obj.decision_function(vector))[0])
            raw_probability = 1.0 / (1.0 + math.exp(-decision))
        else:
            raw_probability = float(np.ravel(model_obj.predict(vector))[0])
    except Exception as exc:
        print(f"[WARN] Specialized model inference failed for {track_name}: {exc}")
        return None

    raw_probability = float(np.clip(raw_probability, 0.0, 1.0))
    probability = raw_probability
    calibrator = bundle.get("calibrator")
    if calibrator is not None and hasattr(calibrator, "predict"):
        try:
            calibrated = np.asarray(calibrator.predict(np.asarray([raw_probability], dtype=float)), dtype=float)
            if len(calibrated) > 0:
                probability = float(np.clip(calibrated[0], 0.0, 1.0))
        except Exception as exc:
            print(f"[WARN] Specialized calibrator inference failed for {track_name}: {exc}")

    return {
        "probability": probability,
        "raw_probability": raw_probability,
        "model_version": str(bundle.get("model_version") or "unknown"),
        "feature_columns": feature_columns,
    }


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


def _coerce_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        payload = value.strip()
        if not payload:
            return []
        try:
            return json.loads(payload)
        except Exception:
            return []
    return []


def _count_high_severity_red_flags(red_flags: Any) -> int:
    rows = red_flags if isinstance(red_flags, list) else _coerce_json_like(red_flags)
    if not isinstance(rows, list):
        return 0

    count = 0
    for flag in rows:
        if not isinstance(flag, dict):
            continue
        severity = str(flag.get("severity") or "").strip().lower()
        if severity in {"high", "critical"}:
            count += 1
    return count


def _count_vat_related_red_flags(red_flags: Any) -> int:
    rows = red_flags if isinstance(red_flags, list) else _coerce_json_like(red_flags)
    if not isinstance(rows, list):
        return 0

    count = 0
    for flag in rows:
        if not isinstance(flag, dict):
            continue
        blob = " ".join(
            str(flag.get(field, ""))
            for field in ("feature", "title", "reason", "description", "actual_value")
        ).lower()
        if any(keyword in blob for keyword in VAT_KEYWORDS):
            count += 1
    return count


def _compute_vat_proxy_metrics(revenue: Any, total_expenses: Any) -> tuple[float, float, float, float]:
    rev = max(0.0, _to_float(revenue, 0.0))
    exp = max(0.0, _to_float(total_expenses, 0.0))
    cost_of_goods = exp * 0.75
    vat_output = rev * 0.10
    vat_input = cost_of_goods * 0.10
    vat_ratio = (vat_input / vat_output) if vat_output > 0 else (2.0 if vat_input > 0 else 0.0)
    refund_gap = vat_input - vat_output
    return float(vat_ratio), float(refund_gap), float(vat_input), float(vat_output)


def _compute_risk_trend_delta(yearly_feature_scores: Any) -> float:
    normalized = _normalize_yearly_feature_scores(yearly_feature_scores)
    if len(normalized) < 2:
        return 0.0
    latest = _to_float(normalized[-1].get("risk_score"), 0.0)
    previous = _to_float(normalized[-2].get("risk_score"), 0.0)
    return latest - previous


def _compute_f3_trend_delta(yearly_feature_scores: Any) -> float:
    normalized = _normalize_yearly_feature_scores(yearly_feature_scores)
    if len(normalized) < 2:
        return 0.0
    latest = _to_float(normalized[-1].get("f3_vat_structure"), 0.0)
    previous = _to_float(normalized[-2].get("f3_vat_structure"), 0.0)
    return latest - previous


def _build_audit_model_feature_map(
    *,
    risk_score: Any,
    anomaly_score: Any,
    model_confidence: Any,
    red_flags: Any,
    revenue: Any,
    total_expenses: Any,
    f1_divergence: Any,
    f2_ratio_limit: Any,
    f3_vat_structure: Any,
    f4_peer_comparison: Any,
) -> dict[str, float]:
    rev = max(0.0, _to_float(revenue, 0.0))
    exp = max(0.0, _to_float(total_expenses, 0.0))
    margin_gap = max(0.0, exp - rev)
    expense_ratio = (exp / rev) if rev > 0 else (2.0 if exp > 0 else 0.0)

    rows = red_flags if isinstance(red_flags, list) else _coerce_json_like(red_flags)
    red_flag_count = float(len(rows)) if isinstance(rows, list) else 0.0
    high_red_flag_count = float(_count_high_severity_red_flags(rows))

    return {
        "risk_score": max(0.0, min(100.0, _to_float(risk_score, 0.0))),
        "anomaly_score": max(0.0, _to_float(anomaly_score, 0.0)),
        "model_confidence": max(0.0, min(100.0, _to_float(model_confidence, 0.0))),
        "revenue": rev,
        "total_expenses": exp,
        "margin_gap": margin_gap,
        "expense_to_revenue_ratio": max(0.0, expense_ratio),
        "red_flag_count": red_flag_count,
        "high_red_flag_count": high_red_flag_count,
        "f1_divergence": max(0.0, _to_float(f1_divergence, 0.0)),
        "f2_ratio_limit": max(0.0, _to_float(f2_ratio_limit, 0.0)),
        "f3_vat_structure": max(0.0, _to_float(f3_vat_structure, 0.0)),
        "f4_peer_comparison": max(0.0, _to_float(f4_peer_comparison, 0.0)),
    }


def _build_vat_refund_model_feature_map(
    *,
    risk_score: Any,
    anomaly_score: Any,
    f1_divergence: Any,
    f2_ratio_limit: Any,
    f3_vat_structure: Any,
    f4_peer_comparison: Any,
    revenue: Any,
    total_expenses: Any,
    red_flags: Any,
) -> dict[str, float]:
    rev = max(0.0, _to_float(revenue, 0.0))
    exp = max(0.0, _to_float(total_expenses, 0.0))
    vat_ratio, refund_gap, _, _ = _compute_vat_proxy_metrics(rev, exp)
    expense_ratio = (exp / rev) if rev > 0 else (2.0 if exp > 0 else 0.0)
    vat_flag_count = float(_count_vat_related_red_flags(red_flags))

    return {
        "risk_score": max(0.0, min(100.0, _to_float(risk_score, 0.0))),
        "anomaly_score": max(0.0, _to_float(anomaly_score, 0.0)),
        "f2_ratio_limit": max(0.0, _to_float(f2_ratio_limit, 0.0)),
        "f3_vat_structure": max(0.0, _to_float(f3_vat_structure, 0.0)),
        "revenue": rev,
        "total_expenses": exp,
        "vat_input_output_ratio": max(0.0, vat_ratio),
        "estimated_refund_gap": max(0.0, refund_gap),
        "vat_flag_count": vat_flag_count,
        "expense_to_revenue_ratio": max(0.0, expense_ratio),
        "f1_divergence": max(0.0, _to_float(f1_divergence, 0.0)),
        "f4_peer_comparison": max(0.0, _to_float(f4_peer_comparison, 0.0)),
    }


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


def _build_decision_intelligence_payload(
    *,
    risk_score: Any,
    risk_level: Any,
    anomaly_score: Any,
    model_confidence: Any,
    red_flags: Any,
    yearly_feature_scores: Any,
    f1_divergence: Any,
    f2_ratio_limit: Any,
    f3_vat_structure: Any,
    f4_peer_comparison: Any,
) -> dict[str, Any]:
    score = max(0.0, min(100.0, _to_float(risk_score, 0.0)))
    anomaly = max(0.0, _to_float(anomaly_score, 0.0))
    confidence = _to_float(model_confidence, 0.0)
    level = str(risk_level or _classify_risk_tier(score)).lower().strip()

    level_labels = {
        "critical": "Rui ro rat cao",
        "high": "Rui ro cao",
        "medium": "Rui ro trung binh",
        "low": "An toan",
    }
    level_label = level_labels.get(level, "Khong xac dinh")

    red_flag_items = red_flags if isinstance(red_flags, list) else []
    high_red_flag_count = 0
    for flag in red_flag_items:
        if not isinstance(flag, dict):
            continue
        severity = str(flag.get("severity") or "").lower().strip()
        if severity in {"high", "critical"}:
            high_red_flag_count += 1

    normalized_yearly_scores = _normalize_yearly_feature_scores(yearly_feature_scores)
    trend_delta = 0.0
    if len(normalized_yearly_scores) >= 2:
        latest = _to_float(normalized_yearly_scores[-1].get("risk_score"), 0.0)
        previous = _to_float(normalized_yearly_scores[-2].get("risk_score"), 0.0)
        trend_delta = latest - previous

    signal_entries: list[dict[str, Any]] = []

    def _add_signal(key: str, label: str, value: float, severity: str, summary: str) -> None:
        signal_entries.append({
            "key": key,
            "label": label,
            "value": round(_to_float(value, 0.0), 4),
            "severity": severity,
            "summary": summary,
        })

    # Core score signal is always surfaced so inspectors can compare policy threshold.
    if score >= 60:
        score_severity = "high"
    elif score >= 40:
        score_severity = "medium"
    else:
        score_severity = "low"
    _add_signal(
        "risk_score",
        "Tong diem rui ro",
        score,
        score_severity,
        f"Diem tong hop hien tai {score:.1f}/100.",
    )

    if anomaly >= 0.85:
        _add_signal(
            "anomaly_score",
            "Do di biet",
            anomaly,
            "high",
            "Do di biet tai chinh vuot nguong canh bao cao.",
        )
    elif anomaly >= 0.60:
        _add_signal(
            "anomaly_score",
            "Do di biet",
            anomaly,
            "medium",
            "Do di biet tai chinh dang o muc can theo doi.",
        )

    f1 = _to_float(f1_divergence, 0.0)
    if f1 >= 0.30:
        _add_signal("f1_divergence", "F1 lech pha tang truong", f1, "high", "Doanh thu va chi phi bi lech pha ro ret.")
    elif f1 >= 0.20:
        _add_signal("f1_divergence", "F1 lech pha tang truong", f1, "medium", "Lech pha tang truong dang vuot muc thong thuong.")

    f2 = _to_float(f2_ratio_limit, 0.0)
    if f2 >= 1.20:
        _add_signal("f2_ratio_limit", "F2 vuot nguong ty le", f2, "high", "Ty le chi phi/doanh thu vuot nguong quy tac.")
    elif f2 >= 1.05:
        _add_signal("f2_ratio_limit", "F2 vuot nguong ty le", f2, "medium", "Ty le chi phi/doanh thu can giam sat bo sung.")

    f3 = _to_float(f3_vat_structure, 0.0)
    if f3 >= 0.70:
        _add_signal("f3_vat_structure", "F3 cau truc VAT", f3, "high", "Cau truc VAT co dau hieu bat thuong manh.")
    elif f3 >= 0.55:
        _add_signal("f3_vat_structure", "F3 cau truc VAT", f3, "medium", "Cau truc VAT lech khoi phan bo ky vong.")

    f4 = _to_float(f4_peer_comparison, 0.0)
    if f4 >= 0.40:
        _add_signal("f4_peer_comparison", "F4 lech chuan nganh", f4, "high", "Bien loi nhuan va hanh vi khai thue lech xa nhom dong nganh.")
    elif f4 >= 0.25:
        _add_signal("f4_peer_comparison", "F4 lech chuan nganh", f4, "medium", "Doanh nghiep dang lech xu huong nganh.")

    if trend_delta >= 15.0:
        _add_signal(
            "risk_trend",
            "Xu huong diem rui ro",
            trend_delta,
            "high",
            "Diem rui ro tang manh so voi ky gan nhat.",
        )
    elif trend_delta >= 8.0:
        _add_signal(
            "risk_trend",
            "Xu huong diem rui ro",
            trend_delta,
            "medium",
            "Diem rui ro co xu huong tang ro trong chuoi nam.",
        )

    if red_flag_items:
        red_flag_severity = "high" if high_red_flag_count > 0 or len(red_flag_items) >= 3 else "medium"
        _add_signal(
            "red_flags",
            "Cum canh bao",
            float(len(red_flag_items)),
            red_flag_severity,
            f"Phat hien {len(red_flag_items)} red flag trong ket qua giai thich.",
        )

    priority_raw = (
        score * 0.62
        + min(20.0, high_red_flag_count * 9.0)
        + min(12.0, len(red_flag_items) * 2.0)
        + min(14.0, max(0.0, trend_delta) * 0.9)
        + min(15.0, max(0.0, (anomaly - 0.5) * 35.0))
    )
    if 0.0 < confidence < 55.0:
        priority_raw += 5.0
    priority_score = int(max(0, min(100, round(priority_raw))))

    recommended_action = "periodic_monitoring"
    action_label = "Theo doi dinh ky"
    action_deadline_days = 60
    should_escalate = False

    if score >= 80.0 or high_red_flag_count >= 2 or anomaly >= 0.90 or priority_score >= 85:
        recommended_action = "urgent_audit"
        action_label = "Thanh tra dot xuat"
        action_deadline_days = 7
        should_escalate = True
    elif score >= 60.0 or high_red_flag_count >= 1 or priority_score >= 65:
        recommended_action = "targeted_review"
        action_label = "Kiem tra chuyen sau"
        action_deadline_days = 15
        should_escalate = priority_score >= 75
    elif score >= 40.0 or red_flag_items or priority_score >= 45:
        recommended_action = "enhanced_monitoring"
        action_label = "Giam sat tang cuong"
        action_deadline_days = 30

    rationale_parts = [f"Diem rui ro hien tai {score:.1f}/100 ({level_label})."]
    if red_flag_items:
        rationale_parts.append(f"Co {len(red_flag_items)} red flag dang kich hoat.")
    if high_red_flag_count > 0:
        rationale_parts.append(f"Trong do {high_red_flag_count} red flag muc cao.")
    if trend_delta >= 8.0:
        rationale_parts.append(f"Xu huong diem rui ro tang {trend_delta:.1f} diem so voi ky truoc.")
    if anomaly >= 0.70:
        rationale_parts.append("Do di biet tai chinh o muc canh bao.")
    if 0.0 < confidence < 55.0:
        rationale_parts.append("Do tin cay mo hinh thap, can bo sung bang chung nghiep vu.")

    action_steps = {
        "urgent_audit": [
            "Khoi tao ho so thanh tra dot xuat trong 24h.",
            "Yeu cau doanh nghiep giai trinh hoa don VAT dau vao dau ra trong 7 ngay.",
            "Doi soat cheo du lieu giao dich voi nhom doi tac lien quan.",
        ],
        "targeted_review": [
            "Mo dot kiem tra chuyen de theo nhom chi so F1-F4 vuot nguong.",
            "Uu tien doi soat to khai VAT va dieu chinh bo sung trong 3 ky gan nhat.",
            "Danh gia canh bao moi sau khi cap nhat bo chung tu giai trinh.",
        ],
        "enhanced_monitoring": [
            "Dua vao danh sach giam sat tang cuong theo chu ky thang.",
            "Theo doi bien dong doanh thu-chi phi va cap nhat canh bao What-If.",
            "Kich hoat kiem tra bo sung neu diem rui ro vuot 60.",
        ],
        "periodic_monitoring": [
            "Duy tri theo doi dinh ky theo chu ky quy.",
            "Cap nhat du lieu tai chinh moi truoc ky danh gia tiep theo.",
            "Tu dong nang muc canh bao neu xuat hien red flag moi.",
        ],
    }

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    signal_entries.sort(
        key=lambda item: (
            -severity_rank.get(str(item.get("severity") or "low"), 1),
            -abs(_to_float(item.get("value"), 0.0)),
        )
    )

    return {
        "recommended_action": recommended_action,
        "action_label": action_label,
        "action_deadline_days": action_deadline_days,
        "priority_score": priority_score,
        "rationale": " ".join(rationale_parts),
        "next_steps": action_steps.get(recommended_action, action_steps["periodic_monitoring"]),
        "top_signals": signal_entries[:4],
        "should_escalate": should_escalate,
    }


def _build_intervention_uplift_payload(
    *,
    risk_score: Any,
    anomaly_score: Any,
    red_flags: Any,
    yearly_feature_scores: Any,
    model_confidence: Any,
    revenue: Any,
    total_expenses: Any,
) -> dict[str, Any]:
    score = max(0.0, min(100.0, _to_float(risk_score, 0.0)))
    anomaly = max(0.0, _to_float(anomaly_score, 0.0))
    confidence_raw = max(0.0, min(100.0, _to_float(model_confidence, 0.0)))
    rev = max(0.0, _to_float(revenue, 0.0))
    exp = max(0.0, _to_float(total_expenses, 0.0))

    red_flag_items = red_flags if isinstance(red_flags, list) else []
    high_red_flag_count = 0
    for flag in red_flag_items:
        if not isinstance(flag, dict):
            continue
        severity = str(flag.get("severity") or "").lower().strip()
        if severity in {"high", "critical"}:
            high_red_flag_count += 1

    normalized_yearly_scores = _normalize_yearly_feature_scores(yearly_feature_scores)
    trend_delta = 0.0
    if len(normalized_yearly_scores) >= 2:
        latest = _to_float(normalized_yearly_scores[-1].get("risk_score"), 0.0)
        previous = _to_float(normalized_yearly_scores[-2].get("risk_score"), 0.0)
        trend_delta = latest - previous

    priority_raw = (
        score * 0.60
        + min(20.0, high_red_flag_count * 8.0)
        + min(12.0, len(red_flag_items) * 2.0)
        + min(10.0, max(0.0, trend_delta) * 0.8)
        + min(15.0, max(0.0, (anomaly - 0.55) * 35.0))
    )
    if 0.0 < confidence_raw < 55.0:
        priority_raw += 5.0
    priority_score = int(max(0, min(100, round(priority_raw))))

    if score >= 82.0 or high_red_flag_count >= 2 or anomaly >= 0.90 or priority_score >= 86:
        action = "escalated_enforcement"
    elif score >= 65.0 or high_red_flag_count >= 1 or priority_score >= 68:
        action = "field_audit"
    elif score >= 45.0 or len(red_flag_items) >= 1 or priority_score >= 50:
        action = "structured_outreach"
    elif score >= 25.0 or anomaly >= 0.60:
        action = "auto_reminder"
    else:
        action = "monitor"

    impact_ratio_map = {
        "monitor": 0.05,
        "auto_reminder": 0.12,
        "structured_outreach": 0.20,
        "field_audit": 0.30,
        "escalated_enforcement": 0.38,
    }
    impact_ratio = impact_ratio_map.get(action, 0.12)

    expected_risk_reduction_pp = round(score * impact_ratio, 1)

    estimated_penalty_exposure = max(0.0, (exp - rev) * 0.03) + high_red_flag_count * 2_500_000 + len(red_flag_items) * 800_000
    expected_penalty_saving = round(estimated_penalty_exposure * min(0.9, impact_ratio + 0.15), 2)

    estimated_collection_base = max(0.0, rev * 0.015 + exp * 0.008)
    expected_collection_uplift = round(estimated_collection_base * impact_ratio, 2)

    if confidence_raw >= 80.0:
        confidence = "high"
    elif confidence_raw >= 55.0:
        confidence = "medium"
    else:
        confidence = "low"

    if confidence == "low" and len(normalized_yearly_scores) >= 4:
        confidence = "medium"

    next_steps_map = {
        "monitor": [
            "Duy tri giam sat dinh ky theo chu ky quy.",
            "Tu dong canh bao khi score vuot nguong 45 trong ky moi.",
        ],
        "auto_reminder": [
            "Gui nhac han giai trinh cho doanh nghiep trong 72h.",
            "Rang buoc doi soat bo sung neu anomaly tiep tuc tang.",
        ],
        "structured_outreach": [
            "Mo phien lam viec co cau truc voi doanh nghiep theo checklist F1-F4.",
            "Doi chieu to khai VAT va bao cao tai chinh cua 3 ky gan nhat.",
        ],
        "field_audit": [
            "Phan cong to nghiep vu kiem tra tai cho theo ho so uu tien.",
            "Yeu cau bo chung tu giai trinh va doi soat nguon hoa don lien quan.",
        ],
        "escalated_enforcement": [
            "Kich hoat quy trinh xu ly cuong che theo tham quyen.",
            "Phoi hop lien don vi de thu hoi nghia vu ton dong trong 30 ngay.",
        ],
    }

    rationale_parts = [f"Risk score hien tai {score:.1f}/100."]
    if high_red_flag_count > 0:
        rationale_parts.append(f"Ghi nhan {high_red_flag_count} red flag muc cao.")
    if trend_delta >= 8.0:
        rationale_parts.append(f"Xu huong diem rui ro tang {trend_delta:.1f} diem.")
    if anomaly >= 0.70:
        rationale_parts.append("Anomaly score o muc can can thiep nghiep vu.")
    if 0.0 < confidence_raw < 55.0:
        rationale_parts.append("Do tin cay mo hinh thap, can bo sung bang chung thu cong.")

    return {
        "recommended_action": action,
        "priority_score": priority_score,
        "expected_risk_reduction_pp": expected_risk_reduction_pp,
        "expected_penalty_saving": expected_penalty_saving,
        "expected_collection_uplift": expected_collection_uplift,
        "confidence": confidence,
        "rationale": " ".join(rationale_parts),
        "next_steps": next_steps_map.get(action, next_steps_map["monitor"]),
    }


def _harmonize_decision_intelligence_with_intervention(
    decision_payload: Any,
    intervention_payload: Any,
) -> dict[str, Any]:
    decision = dict(decision_payload) if isinstance(decision_payload, dict) else {}
    intervention = dict(intervention_payload) if isinstance(intervention_payload, dict) else {}
    if not intervention:
        return decision

    action = str(intervention.get("recommended_action") or "monitor").strip().lower()
    priority_score = int(max(0, min(100, round(_to_float(intervention.get("priority_score"), _to_float(decision.get("priority_score"), 0))))))
    rationale = str(intervention.get("rationale") or decision.get("rationale") or "").strip()

    mapping = {
        "monitor": {
            "recommended_action": "periodic_monitoring",
            "action_label": "Theo doi dinh ky",
            "action_deadline_days": 60,
            "should_escalate": False,
            "severity": "low",
        },
        "auto_reminder": {
            "recommended_action": "enhanced_monitoring",
            "action_label": "Nhac han tu dong",
            "action_deadline_days": 30,
            "should_escalate": False,
            "severity": "medium",
        },
        "structured_outreach": {
            "recommended_action": "enhanced_monitoring",
            "action_label": "Can thiep co cau truc",
            "action_deadline_days": 21,
            "should_escalate": False,
            "severity": "medium",
        },
        "field_audit": {
            "recommended_action": "targeted_review",
            "action_label": "Kiem tra tai cho",
            "action_deadline_days": 14,
            "should_escalate": True,
            "severity": "high",
        },
        "escalated_enforcement": {
            "recommended_action": "urgent_audit",
            "action_label": "Xu ly cuong che nang cao",
            "action_deadline_days": 7,
            "should_escalate": True,
            "severity": "high",
        },
    }
    selected = mapping.get(action, mapping["monitor"])

    harmonized = {
        **decision,
        "recommended_action": selected["recommended_action"],
        "action_label": selected["action_label"],
        "action_deadline_days": selected["action_deadline_days"],
        "priority_score": priority_score,
        "rationale": rationale,
        "next_steps": intervention.get("next_steps") if isinstance(intervention.get("next_steps"), list) and intervention.get("next_steps") else decision.get("next_steps", []),
        "should_escalate": bool(selected["should_escalate"] or priority_score >= 75),
    }

    existing_signals = harmonized.get("top_signals") if isinstance(harmonized.get("top_signals"), list) else []
    has_intervention_signal = any(
        isinstance(signal, dict) and str(signal.get("key") or "") == "intervention_uplift"
        for signal in existing_signals
    )
    if not has_intervention_signal:
        existing_signals.append(
            {
                "key": "intervention_uplift",
                "label": "Intervention/Uplift",
                "value": float(priority_score),
                "severity": selected["severity"],
                "summary": rationale or "Intervention action da duoc dong bo voi Decision Intelligence.",
            }
        )
    harmonized["top_signals"] = existing_signals[:4]
    return harmonized


def _build_vat_refund_signals_payload(
    *,
    risk_score: Any,
    anomaly_score: Any,
    red_flags: Any,
    yearly_feature_scores: Any,
    f2_ratio_limit: Any,
    f3_vat_structure: Any,
    revenue: Any,
    total_expenses: Any,
    f1_divergence: Any = None,
    f4_peer_comparison: Any = None,
) -> dict[str, Any]:
    score = max(0.0, min(100.0, _to_float(risk_score, 0.0)))
    anomaly = max(0.0, _to_float(anomaly_score, 0.0))
    f2 = max(0.0, _to_float(f2_ratio_limit, 0.0))
    f3 = max(0.0, _to_float(f3_vat_structure, 0.0))
    rev = max(0.0, _to_float(revenue, 0.0))
    exp = max(0.0, _to_float(total_expenses, 0.0))

    vat_ratio_raw, refund_gap_raw, vat_input, vat_output = _compute_vat_proxy_metrics(rev, exp)
    vat_ratio = round(vat_ratio_raw, 4)
    refund_gap = round(refund_gap_raw, 2)

    f3_trend_delta = _compute_f3_trend_delta(yearly_feature_scores)

    red_flag_items = red_flags if isinstance(red_flags, list) else _coerce_json_like(red_flags)
    if not isinstance(red_flag_items, list):
        red_flag_items = []
    vat_flag_count = _count_vat_related_red_flags(red_flag_items)

    indicators: list[dict[str, Any]] = []

    def _add_indicator(key: str, label: str, value: float, severity: str, summary: str) -> None:
        indicators.append({
            "key": key,
            "label": label,
            "value": round(_to_float(value, 0.0), 4),
            "severity": severity,
            "summary": summary,
        })

    if f3 >= 0.70:
        _add_indicator(
            "f3_vat_structure",
            "Cau truc VAT bat thuong",
            f3,
            "high",
            "F3 vuot nguong canh bao cao, can doi soat lai bo khau tru VAT.",
        )
    elif f3 >= 0.55:
        _add_indicator(
            "f3_vat_structure",
            "Cau truc VAT bat thuong",
            f3,
            "medium",
            "F3 dang vuot nguong theo doi trong danh gia hoan thue.",
        )

    if vat_ratio >= 1.15:
        _add_indicator(
            "vat_input_output_ratio",
            "Ty le VAT dau vao/dau ra",
            vat_ratio,
            "high",
            "VAT dau vao vuot dang ke VAT dau ra, co dau hieu ap luc hoan thue.",
        )
    elif vat_ratio >= 1.0:
        _add_indicator(
            "vat_input_output_ratio",
            "Ty le VAT dau vao/dau ra",
            vat_ratio,
            "medium",
            "VAT dau vao dang tiep can nguong VAT dau ra, can theo doi ky tiep theo.",
        )

    if refund_gap > 0:
        gap_ratio = refund_gap / (vat_output + 1.0)
        if gap_ratio >= 0.30:
            _add_indicator(
                "estimated_refund_gap",
                "Chenh lech VAT dau vao-dau ra",
                refund_gap,
                "high",
                "Chenh lech VAT duong lon, can doi chieu chung tu dau vao cho ho so hoan.",
            )
        elif gap_ratio >= 0.10:
            _add_indicator(
                "estimated_refund_gap",
                "Chenh lech VAT dau vao-dau ra",
                refund_gap,
                "medium",
                "Chenh lech VAT duong dang mo rong, can bo sung doi soat hoa don.",
            )
    else:
        gap_ratio = 0.0

    if vat_flag_count > 0:
        _add_indicator(
            "vat_related_flags",
            "Cum red-flag lien quan VAT",
            float(vat_flag_count),
            "high" if vat_flag_count >= 2 else "medium",
            f"Phat hien {vat_flag_count} chi bao lien quan hoa don/VAT trong danh sach red flags.",
        )

    if anomaly >= 0.90:
        _add_indicator(
            "anomaly_score",
            "Do di biet tai chinh",
            anomaly,
            "high",
            "Do di biet tai chinh rat cao, can xac minh bo khau tru VAT theo giao dich.",
        )
    elif anomaly >= 0.75:
        _add_indicator(
            "anomaly_score",
            "Do di biet tai chinh",
            anomaly,
            "medium",
            "Do di biet tai chinh vuot nguong theo doi cho nhom hoan thue.",
        )

    if f3_trend_delta >= 0.08:
        _add_indicator(
            "f3_trend",
            "Xu huong F3 VAT",
            f3_trend_delta,
            "medium" if f3_trend_delta < 0.15 else "high",
            "F3 VAT co xu huong tang, can theo doi lien tuc cac ky khai thue.",
        )

    priority_raw = (
        f3 * 45.0
        + min(25.0, max(0.0, (vat_ratio - 1.0) * 60.0))
        + min(15.0, max(0.0, gap_ratio) * 10.0)
        + min(20.0, vat_flag_count * 8.0)
        + min(10.0, max(0.0, anomaly - 0.6) * 30.0)
        + min(8.0, max(0.0, f3_trend_delta) * 40.0)
        + (6.0 if score >= 60.0 else 0.0)
        + (5.0 if f2 >= 1.20 else 0.0)
    )

    specialized_prediction = _predict_specialized_probability(
        "vat_refund",
        _build_vat_refund_model_feature_map(
            risk_score=score,
            anomaly_score=anomaly,
            f1_divergence=f1_divergence,
            f2_ratio_limit=f2,
            f3_vat_structure=f3,
            f4_peer_comparison=f4_peer_comparison,
            revenue=rev,
            total_expenses=exp,
            red_flags=red_flag_items,
        ),
    )
    model_probability: Optional[float] = None
    model_version = None
    if specialized_prediction:
        model_probability = float(np.clip(_to_float(specialized_prediction.get("probability"), 0.0), 0.0, 1.0))
        model_version = str(specialized_prediction.get("model_version") or "vat-refund-specialized")
        priority_raw = (priority_raw * 0.65) + ((model_probability * 100.0) * 0.35)
        _add_indicator(
            "vat_model_probability",
            "Xac suat mo hinh VAT",
            model_probability,
            "high" if model_probability >= 0.72 else ("medium" if model_probability >= 0.48 else "low"),
            "Xac suat can can thiep hoan thue do mo hinh VAT chuyen biet du bao.",
        )

    priority_score = int(max(0, min(100, round(priority_raw))))

    queue = "monitor"
    level = "low"
    has_signal = False
    if (
        priority_score >= 80
        or (f3 >= 0.70 and vat_ratio >= 1.20)
        or vat_flag_count >= 2
        or (model_probability is not None and model_probability >= 0.80)
    ):
        queue = "priority_refund_audit"
        level = "critical"
        has_signal = True
    elif (
        priority_score >= 60
        or (f3 >= 0.65 and vat_ratio >= 1.05)
        or (model_probability is not None and model_probability >= 0.62)
    ):
        queue = "priority_refund_audit"
        level = "high"
        has_signal = True
    elif (
        priority_score >= 40
        or f3 >= 0.55
        or vat_ratio >= 1.0
        or (model_probability is not None and model_probability >= 0.45)
    ):
        queue = "refund_watchlist"
        level = "medium"
        has_signal = True

    rationale_parts = []
    if has_signal:
        rationale_parts.append(f"Muc uu tien hoan thue VAT duoc xep {level.upper()} ({priority_score}/100).")
    if f3 >= 0.55:
        rationale_parts.append(f"Chi so F3 VAT = {f3:.2f} vuot nguong theo doi.")
    if vat_ratio >= 1.0:
        rationale_parts.append(f"Ty le VAT dau vao/dau ra = {vat_ratio:.2f}.")
    if refund_gap > 0:
        rationale_parts.append(f"Chenh lech VAT dau vao-dau ra uoc tinh {refund_gap:,.0f}.")
    if vat_flag_count > 0:
        rationale_parts.append(f"Co {vat_flag_count} red-flag lien quan hoa don/VAT.")
    if f3_trend_delta >= 0.08:
        rationale_parts.append(f"F3 VAT tang {f3_trend_delta:.2f} so voi ky truoc.")
    if model_probability is not None:
        rationale_parts.append(
            f"Mo hinh VAT ({model_version}) du bao xac suat can can thiep {model_probability * 100.0:.1f}%"
        )
    if not rationale_parts:
        rationale_parts.append("Chua ghi nhan tin hieu hoan thue VAT bat thuong trong ky hien tai.")

    recommended_checks_map = {
        "priority_refund_audit": [
            "Doi chieu hoa don VAT dau vao theo chuoi nha cung cap truoc khi phe duyet hoan.",
            "Kiem tra tinh hop le bo chung tu khau tru VAT cua 3 ky gan nhat.",
            "Rang buoc quy trinh phe duyet bo sung bang xac minh giao dich ganh VAT cao.",
        ],
        "refund_watchlist": [
            "Dua ho so vao danh sach theo doi hoan thue VAT theo ky thang/quy.",
            "Yeu cau bo sung tai lieu giai trinh cho cac hoa don dau vao co gia tri lon.",
            "Canh bao tu dong neu F3 VAT tiep tuc tang trong ky tiep theo.",
        ],
        "monitor": [
            "Duy tri giam sat thuong xuyen, chua can kich hoat quy trinh kiem tra chuyen de.",
            "Cap nhat du lieu VAT ky moi de danh gia lai xu huong hoan thue.",
        ],
    }

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    indicators.sort(
        key=lambda item: (
            -severity_rank.get(str(item.get("severity") or "low"), 1),
            -abs(_to_float(item.get("value"), 0.0)),
        )
    )

    return {
        "has_signal": has_signal,
        "queue": queue,
        "level": level,
        "score": priority_score,
        "rationale": " ".join(rationale_parts),
        "vat_input_output_ratio": round(vat_ratio, 4) if (vat_input > 0 or vat_output > 0) else None,
        "estimated_refund_gap": round(max(0.0, refund_gap), 2),
        "recommended_checks": recommended_checks_map.get(queue, recommended_checks_map["monitor"]),
        "indicators": indicators[:5],
    }


def _build_audit_value_payload(
    *,
    risk_score: Any,
    anomaly_score: Any,
    model_confidence: Any,
    red_flags: Any,
    yearly_feature_scores: Any,
    revenue: Any,
    total_expenses: Any,
    f1_divergence: Any = None,
    f2_ratio_limit: Any = None,
    f3_vat_structure: Any = None,
    f4_peer_comparison: Any = None,
) -> dict[str, Any]:
    score = max(0.0, min(100.0, _to_float(risk_score, 0.0)))
    anomaly = max(0.0, _to_float(anomaly_score, 0.0))
    confidence_raw = max(0.0, min(100.0, _to_float(model_confidence, 0.0)))
    rev = max(0.0, _to_float(revenue, 0.0))
    exp = max(0.0, _to_float(total_expenses, 0.0))

    red_flag_items = red_flags if isinstance(red_flags, list) else _coerce_json_like(red_flags)
    if not isinstance(red_flag_items, list):
        red_flag_items = []
    high_red_flag_count = _count_high_severity_red_flags(red_flag_items)

    normalized_yearly_scores = _normalize_yearly_feature_scores(yearly_feature_scores)
    trend_delta = _compute_risk_trend_delta(normalized_yearly_scores)

    margin_gap = max(0.0, exp - rev)
    exposure_base = (
        margin_gap * 0.08
        + rev * 0.012
        + len(red_flag_items) * 900_000.0
        + high_red_flag_count * 2_200_000.0
        + max(0.0, anomaly - 0.55) * 25_000_000.0
        + max(0.0, trend_delta) * 450_000.0
    )

    recoverability_raw = (
        0.18
        + (score / 100.0) * 0.45
        + min(0.15, max(0.0, anomaly - 0.5) * 0.45)
        + min(0.14, high_red_flag_count * 0.04)
        + (0.05 if trend_delta >= 10.0 else 0.0)
    )
    if 0.0 < confidence_raw < 55.0:
        recoverability_raw -= 0.06

    specialized_prediction = _predict_specialized_probability(
        "audit_value",
        _build_audit_model_feature_map(
            risk_score=score,
            anomaly_score=anomaly,
            model_confidence=confidence_raw,
            red_flags=red_flag_items,
            revenue=rev,
            total_expenses=exp,
            f1_divergence=f1_divergence,
            f2_ratio_limit=f2_ratio_limit,
            f3_vat_structure=f3_vat_structure,
            f4_peer_comparison=f4_peer_comparison,
        ),
    )

    model_probability: Optional[float] = None
    model_version = None
    if specialized_prediction:
        model_probability = float(np.clip(_to_float(specialized_prediction.get("probability"), 0.0), 0.0, 1.0))
        model_version = str(specialized_prediction.get("model_version") or "audit-value-specialized")
        recoverability_raw = (recoverability_raw * 0.55) + (model_probability * 0.45)

    # Only let specialized probability fully drive lane escalation when contextual
    # audit features are available (or baseline risk is not trivially low).
    specialized_model_override_enabled = model_probability is not None and (
        any(value is not None for value in (f1_divergence, f2_ratio_limit, f3_vat_structure, f4_peer_comparison))
        or high_red_flag_count > 0
        or score >= 35.0
    )

    recoverability_ratio = max(0.08, min(0.86, recoverability_raw))
    estimated_recovery = round(max(0.0, exposure_base) * recoverability_ratio, 2)

    audit_hours_estimate = round(
        max(
            8.0,
            min(
                120.0,
                14.0 + score * 0.62 + high_red_flag_count * 4.5 + min(18.0, max(0.0, trend_delta) * 0.7),
            ),
        ),
        1,
    )
    estimated_audit_cost = round(audit_hours_estimate * 650_000.0, 2)
    expected_net_recovery = round(max(0.0, estimated_recovery - estimated_audit_cost), 2)

    net_million = expected_net_recovery / 1_000_000.0
    priority_raw = (
        score * 0.55
        + min(24.0, net_million * 0.75)
        + min(12.0, high_red_flag_count * 3.0)
        + (5.0 if recoverability_ratio >= 0.60 else 0.0)
    )
    if model_probability is not None:
        priority_raw = (priority_raw * 0.70) + ((model_probability * 100.0) * 0.30)

    priority_score = int(max(0, min(100, round(priority_raw))))

    if (
        expected_net_recovery >= 2_000_000_000.0
        or priority_score >= 80
        or (specialized_model_override_enabled and model_probability >= 0.80)
    ):
        recommended_lane = "priority_audit"
    elif (
        expected_net_recovery >= 800_000_000.0
        or priority_score >= 62
        or (specialized_model_override_enabled and model_probability >= 0.62)
    ):
        recommended_lane = "targeted_audit"
    elif (
        expected_net_recovery >= 200_000_000.0
        or priority_score >= 45
        or (specialized_model_override_enabled and model_probability >= 0.45)
    ):
        recommended_lane = "desk_review"
    else:
        recommended_lane = "monitor"

    confidence_signal = confidence_raw
    if model_probability is not None:
        confidence_signal = max(confidence_raw, abs(model_probability - 0.5) * 200.0)

    if confidence_signal >= 80.0:
        confidence = "high"
    elif confidence_signal >= 55.0:
        confidence = "medium"
    else:
        confidence = "low"
    if confidence == "low" and len(normalized_yearly_scores) >= 4:
        confidence = "medium"

    rationale_parts = [
        f"Gia tri truy thu ky vong uoc tinh {estimated_recovery:,.0f} VND.",
        f"Net recovery sau chi phi uoc tinh {expected_net_recovery:,.0f} VND.",
        f"Risk score hien tai {score:.1f}/100 voi recoverability {recoverability_ratio * 100:.1f}%.",
    ]
    if high_red_flag_count > 0:
        rationale_parts.append(f"Phat hien {high_red_flag_count} red flag muc cao can uu tien ho so.")
    if trend_delta >= 8.0:
        rationale_parts.append(f"Xu huong risk tang {trend_delta:.1f} diem so voi ky truoc.")
    if model_probability is not None:
        rationale_parts.append(
            f"Mo hinh Audit ({model_version}) uoc tinh recoverability {model_probability * 100.0:.1f}%."
        )

    drivers = [
        {
            "key": "expected_recovery",
            "label": "Gia tri truy thu uoc tinh",
            "value": estimated_recovery,
            "impact": "high" if estimated_recovery >= 1_000_000_000.0 else "medium",
            "summary": "Tong gia tri co the truy thu neu mo thanh tra ho so nay.",
        },
        {
            "key": "recoverability_ratio",
            "label": "Ty le thu hoi",
            "value": round(recoverability_ratio, 4),
            "impact": "high" if recoverability_ratio >= 0.60 else "medium",
            "summary": "Uoc tinh kha nang bien rui ro thanh gia tri thu hoi thuc te.",
        },
        {
            "key": "audit_cost",
            "label": "Chi phi thanh tra",
            "value": estimated_audit_cost,
            "impact": "low" if estimated_audit_cost <= 60_000_000.0 else "medium",
            "summary": "Nguon luc can bo uoc tinh de hoan tat mot chu ky xu ly.",
        },
        {
            "key": "risk_momentum",
            "label": "Dong luc rui ro",
            "value": round(trend_delta, 2),
            "impact": "high" if trend_delta >= 12.0 else ("medium" if trend_delta >= 4.0 else "low"),
            "summary": "Toc do tang risk score theo chuoi nam phan tich.",
        },
    ]

    if model_probability is not None:
        drivers.append(
            {
                "key": "audit_model_probability",
                "label": "Xac suat mo hinh Audit",
                "value": round(model_probability, 4),
                "impact": "high" if model_probability >= 0.70 else ("medium" if model_probability >= 0.45 else "low"),
                "summary": "Xac suat positive net recovery tu mo hinh Audit Value chuyen biet.",
            }
        )

    return {
        "estimated_recovery": estimated_recovery,
        "expected_net_recovery": expected_net_recovery,
        "recoverability_ratio": round(recoverability_ratio, 4),
        "audit_hours_estimate": audit_hours_estimate,
        "estimated_audit_cost": estimated_audit_cost,
        "priority_score": priority_score,
        "recommended_lane": recommended_lane,
        "confidence": confidence,
        "rationale": " ".join(rationale_parts),
        "drivers": drivers,
    }


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


def _build_split_trigger_status_context(snapshot_source: str) -> dict[str, Any]:
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
        "generated_at": datetime.utcnow().isoformat(),
    }


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
        response_payload["decision_intelligence"] = _build_decision_intelligence_payload(
            risk_score=response_payload.get("risk_score"),
            risk_level=response_payload.get("risk_level"),
            anomaly_score=response_payload.get("anomaly_score"),
            model_confidence=response_payload.get("model_confidence"),
            red_flags=response_payload.get("red_flags"),
            yearly_feature_scores=response_payload.get("yearly_feature_scores"),
            f1_divergence=response_payload.get("f1_divergence"),
            f2_ratio_limit=response_payload.get("f2_ratio_limit"),
            f3_vat_structure=response_payload.get("f3_vat_structure"),
            f4_peer_comparison=response_payload.get("f4_peer_comparison"),
        )
        response_payload["intervention_uplift"] = _build_intervention_uplift_payload(
            risk_score=response_payload.get("risk_score"),
            anomaly_score=response_payload.get("anomaly_score"),
            red_flags=response_payload.get("red_flags"),
            yearly_feature_scores=response_payload.get("yearly_feature_scores"),
            model_confidence=response_payload.get("model_confidence"),
            revenue=response_payload.get("revenue"),
            total_expenses=response_payload.get("total_expenses"),
        )
        response_payload["decision_intelligence"] = _harmonize_decision_intelligence_with_intervention(
            response_payload.get("decision_intelligence"),
            response_payload.get("intervention_uplift"),
        )
        response_payload["vat_refund_signals"] = _build_vat_refund_signals_payload(
            risk_score=response_payload.get("risk_score"),
            anomaly_score=response_payload.get("anomaly_score"),
            red_flags=response_payload.get("red_flags"),
            yearly_feature_scores=response_payload.get("yearly_feature_scores"),
            f2_ratio_limit=response_payload.get("f2_ratio_limit"),
            f3_vat_structure=response_payload.get("f3_vat_structure"),
            revenue=response_payload.get("revenue"),
            total_expenses=response_payload.get("total_expenses"),
            f1_divergence=response_payload.get("f1_divergence"),
            f4_peer_comparison=response_payload.get("f4_peer_comparison"),
        )
        response_payload["audit_value"] = _build_audit_value_payload(
            risk_score=response_payload.get("risk_score"),
            anomaly_score=response_payload.get("anomaly_score"),
            model_confidence=response_payload.get("model_confidence"),
            red_flags=response_payload.get("red_flags"),
            yearly_feature_scores=response_payload.get("yearly_feature_scores"),
            revenue=response_payload.get("revenue"),
            total_expenses=response_payload.get("total_expenses"),
            f1_divergence=response_payload.get("f1_divergence"),
            f2_ratio_limit=response_payload.get("f2_ratio_limit"),
            f3_vat_structure=response_payload.get("f3_vat_structure"),
            f4_peer_comparison=response_payload.get("f4_peer_comparison"),
        )
        response_payload["split_trigger_status"] = _build_split_trigger_status_context(
            snapshot_source="ai_single_query_cached",
        )
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

        result["decision_intelligence"] = _build_decision_intelligence_payload(
            risk_score=result.get("risk_score"),
            risk_level=result.get("risk_level"),
            anomaly_score=result.get("anomaly_score"),
            model_confidence=result.get("model_confidence"),
            red_flags=result.get("red_flags"),
            yearly_feature_scores=result.get("yearly_feature_scores"),
            f1_divergence=result.get("f1_divergence"),
            f2_ratio_limit=result.get("f2_ratio_limit"),
            f3_vat_structure=result.get("f3_vat_structure"),
            f4_peer_comparison=result.get("f4_peer_comparison"),
        )
        result["intervention_uplift"] = _build_intervention_uplift_payload(
            risk_score=result.get("risk_score"),
            anomaly_score=result.get("anomaly_score"),
            red_flags=result.get("red_flags"),
            yearly_feature_scores=result.get("yearly_feature_scores"),
            model_confidence=result.get("model_confidence"),
            revenue=result.get("revenue"),
            total_expenses=result.get("total_expenses"),
        )
        result["decision_intelligence"] = _harmonize_decision_intelligence_with_intervention(
            result.get("decision_intelligence"),
            result.get("intervention_uplift"),
        )
        result["vat_refund_signals"] = _build_vat_refund_signals_payload(
            risk_score=result.get("risk_score"),
            anomaly_score=result.get("anomaly_score"),
            red_flags=result.get("red_flags"),
            yearly_feature_scores=result.get("yearly_feature_scores"),
            f2_ratio_limit=result.get("f2_ratio_limit"),
            f3_vat_structure=result.get("f3_vat_structure"),
            revenue=result.get("revenue"),
            total_expenses=result.get("total_expenses"),
            f1_divergence=result.get("f1_divergence"),
            f4_peer_comparison=result.get("f4_peer_comparison"),
        )
        result["audit_value"] = _build_audit_value_payload(
            risk_score=result.get("risk_score"),
            anomaly_score=result.get("anomaly_score"),
            model_confidence=result.get("model_confidence"),
            red_flags=result.get("red_flags"),
            yearly_feature_scores=result.get("yearly_feature_scores"),
            revenue=result.get("revenue"),
            total_expenses=result.get("total_expenses"),
            f1_divergence=result.get("f1_divergence"),
            f2_ratio_limit=result.get("f2_ratio_limit"),
            f3_vat_structure=result.get("f3_vat_structure"),
            f4_peer_comparison=result.get("f4_peer_comparison"),
        )
        result["split_trigger_status"] = _build_split_trigger_status_context(
            snapshot_source="ai_single_query_realtime",
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
        display_risk_score = round(_to_float(row[4], 0.0), 2)
        intervention = _build_intervention_uplift_payload(
            risk_score=display_risk_score,
            anomaly_score=0.0,
            red_flags=[],
            yearly_feature_scores=[],
            model_confidence=65.0 if assessment_count > 0 else 0.0,
            revenue=0.0,
            total_expenses=0.0,
        )
        results.append(
            {
                "tax_code": row[0],
                "name": row[1],
                "industry": row[2],
                "is_active": bool(row[3]),
                "risk_score": display_risk_score,
                "latest_risk_score": round(_to_float(row[5], 0.0), 2) if row[5] is not None else None,
                "intervention_action": intervention.get("recommended_action"),
                "intervention_priority": int(intervention.get("priority_score") or 0),
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


# ════════════════════════════════════════════════════════════════
#  Flagship: Multi-Scenario What-If Comparison (Program C)
#  Investigator Decision Intelligence
# ════════════════════════════════════════════════════════════════

@router.post("/multi-scenario/{tax_code}", response_model=schemas.MultiScenarioResponse)
def multi_scenario_comparison(
    tax_code: str,
    request: schemas.MultiScenarioRequest,
    db: Session = Depends(get_db),
):
    """
    Program C – Multi-Scenario What-If: Compare up to 10 scenarios simultaneously.
    Each scenario adjusts financial parameters by percentage and simulates risk score.
    Returns comparative analysis with confidence intervals and recommended actions.
    """
    # Resolve company
    company = db.query(models.Company).filter(
        models.Company.tax_code == tax_code
    ).first()
    if not company:
        raise HTTPException(status_code=404, detail="Không tìm thấy doanh nghiệp.")

    # Get baseline assessment
    cached = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.tax_code == tax_code)
        .order_by(models.AIRiskAssessment.created_at.desc())
        .first()
    )

    if not cached:
        raise HTTPException(
            status_code=404,
            detail="Chưa có đánh giá rủi ro cơ sở. Vui lòng chạy phân tích AI trước.",
        )

    baseline_score = float(cached.risk_score or 0)
    baseline_level = cached.risk_level or "low"
    base_revenue = float(cached.revenue or 0)
    base_expenses = float(cached.total_expenses or 0)

    if base_revenue == 0 and base_expenses == 0:
        raise HTTPException(
            status_code=422,
            detail="Dữ liệu tài chính cơ sở = 0, không thể mô phỏng.",
        )

    # Load pipeline
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải model: {exc}",
        )

    base_data = {
        "tax_code": tax_code,
        "company_name": company.name or "",
        "industry": cached.industry or company.industry or "Unknown",
        "year": cached.year or 2024,
        "revenue": base_revenue,
        "total_expenses": base_expenses,
        "net_profit": base_revenue - base_expenses,
        "vat_output": base_revenue * 0.10,
        "vat_input": base_expenses * 0.75 * 0.10,
        "industry_avg_profit_margin": 0.08,
    }

    scenario_results = []
    for scenario in request.scenarios:
        adjusted = base_data.copy()

        for field, pct_change in scenario.adjustments.items():
            if field in adjusted and isinstance(adjusted[field], (int, float)):
                clamped_pct = max(-80.0, min(250.0, pct_change))
                adjusted[field] = adjusted[field] * (1 + clamped_pct / 100)

        # Recalculate derived fields
        adjusted["net_profit"] = adjusted["revenue"] - adjusted["total_expenses"]
        adjusted["vat_output"] = adjusted["revenue"] * 0.10
        adjusted["vat_input"] = adjusted["total_expenses"] * 0.75 * 0.10

        # Run prediction
        try:
            result = pipeline.predict_single([adjusted])
            sim_score = float(result.get("risk_score", 0))
        except Exception:
            sim_score = baseline_score

        # Classify risk level
        if sim_score >= 80:
            sim_level = "critical"
        elif sim_score >= 60:
            sim_level = "high"
        elif sim_score >= 40:
            sim_level = "medium"
        else:
            sim_level = "low"

        delta = round(sim_score - baseline_score, 2)

        # Confidence interval (heuristic: ±5% of score)
        ci_margin = max(2.0, sim_score * 0.05)
        ci_low = round(max(0, sim_score - ci_margin), 2)
        ci_high = round(min(100, sim_score + ci_margin), 2)

        # Recommended action
        if sim_level == "critical":
            action = "Khẩn cấp: Lập tức kiểm tra & thu hồi thuế"
        elif sim_level == "high":
            action = "Ưu tiên cao: Lên kế hoạch thanh tra trong 30 ngày"
        elif sim_level == "medium":
            action = "Theo dõi: Đưa vào danh sách giám sát quý"
        else:
            action = "Bình thường: Không cần hành động đặc biệt"

        # Simulated feature values
        sim_features = {}
        for field in ("revenue", "total_expenses", "net_profit", "vat_output", "vat_input"):
            sim_features[field] = round(adjusted.get(field, 0), 2)

        scenario_results.append(schemas.ScenarioResult(
            name=scenario.name,
            adjustments=scenario.adjustments,
            simulated_risk_score=round(sim_score, 2),
            risk_level=sim_level,
            delta_risk=delta,
            confidence_low=ci_low,
            confidence_high=ci_high,
            simulated_features=sim_features,
            recommended_action=action,
        ))

    # Determine best/worst
    best = min(scenario_results, key=lambda s: s.simulated_risk_score) if scenario_results else None
    worst = max(scenario_results, key=lambda s: s.simulated_risk_score) if scenario_results else None

    return schemas.MultiScenarioResponse(
        tax_code=tax_code,
        company_name=company.name or "",
        baseline_risk_score=baseline_score,
        baseline_risk_level=baseline_level,
        scenarios=scenario_results,
        best_scenario=best.name if best else None,
        worst_scenario=worst.name if worst else None,
    )


@router.post("/inspector-label")
def submit_inspector_label(
    label: schemas.InspectorLabelCreate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submit a ground-truth label from an inspector.
    Used for model retraining and false positive reduction.
    """
    new_label = _build_inspector_label_entity(
        label=label,
        current_user=current_user,
        db=db,
    )
    db.add(new_label)
    db.commit()
    db.refresh(new_label)

    return schemas.InspectorLabelResponse.model_validate(new_label)


def _normalize_label_origin(raw_origin: Optional[str]) -> str:
    normalized = str(raw_origin or "manual_inspector").strip().lower()
    if normalized in KNOWN_LABEL_ORIGINS:
        return normalized
    return "manual_inspector"


def _contains_synthetic_marker(raw_text: Optional[str]) -> bool:
    text = str(raw_text or "").strip().lower()
    if not text:
        return False

    for marker in getattr(monitoring_router, "SYNTHETIC_LABEL_MARKERS", ()):  # defensive reuse
        marker_text = str(marker or "").strip().lower()
        if marker_text and marker_text in text:
            return True
    return False


def _build_inspector_label_entity(
    *,
    label: schemas.InspectorLabelCreate,
    current_user: models.User,
    db: Session,
) -> models.InspectorLabel:
    linked_assessment_model_version: Optional[str] = None
    if label.assessment_id is not None:
        assessment = (
            db.query(models.AIRiskAssessment)
            .filter(models.AIRiskAssessment.id == label.assessment_id)
            .first()
        )
        if assessment is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy assessment_id tương ứng.")
        if assessment.tax_code and assessment.tax_code != label.tax_code:
            raise HTTPException(
                status_code=400,
                detail="assessment_id không khớp với tax_code.",
            )
        raw_assessment_model_version = getattr(assessment, "model_version", None)
        linked_assessment_model_version = str(raw_assessment_model_version).strip()[:80] if raw_assessment_model_version else None

    resolved_label_origin = _normalize_label_origin(label.label_origin)
    if resolved_label_origin in BLOCKED_LABEL_ORIGINS:
        raise HTTPException(
            status_code=400,
            detail="API ghi nhãn thực địa không chấp nhận label_origin synthetic/bootstrap.",
        )

    evidence_summary = str(label.evidence_summary or "").strip()
    if len(evidence_summary) < 12:
        raise HTTPException(
            status_code=400,
            detail="Nhãn thực địa cần evidence_summary tối thiểu 12 ký tự.",
        )
    if _contains_synthetic_marker(evidence_summary):
        raise HTTPException(
            status_code=400,
            detail="evidence_summary chứa marker synthetic/bootstrap nên bị từ chối.",
        )

    provided_model_version = str(label.model_version or "").strip()[:80] or None
    resolved_model_version = provided_model_version or linked_assessment_model_version

    resolved_intervention_attempted = bool(label.intervention_attempted or label.decision)
    resolved_outcome_status = label.outcome_status
    if resolved_outcome_status is None:
        if (label.amount_recovered or 0) > 0:
            resolved_outcome_status = "recovered"
        elif resolved_intervention_attempted:
            resolved_outcome_status = "in_progress"
        else:
            resolved_outcome_status = "pending"

    terminal_status = {"recovered", "partial_recovered", "unrecoverable", "dismissed"}
    resolved_outcome_recorded_at = label.outcome_recorded_at
    if resolved_outcome_recorded_at is None and resolved_outcome_status in terminal_status:
        resolved_outcome_recorded_at = datetime.utcnow()

    return models.InspectorLabel(
        tax_code=label.tax_code,
        inspector_id=current_user.id,
        label_type=label.label_type,
        confidence=label.confidence,
        label_origin=resolved_label_origin,
        assessment_id=label.assessment_id,
        model_version=resolved_model_version,
        evidence_summary=evidence_summary,
        decision=label.decision,
        decision_date=label.decision_date,
        tax_period=label.tax_period,
        amount_recovered=label.amount_recovered,
        intervention_action=label.intervention_action,
        intervention_attempted=resolved_intervention_attempted,
        outcome_status=resolved_outcome_status,
        predicted_collection_uplift=label.predicted_collection_uplift,
        expected_recovery=label.expected_recovery,
        expected_net_recovery=label.expected_net_recovery,
        estimated_audit_cost=label.estimated_audit_cost,
        actual_audit_cost=label.actual_audit_cost,
        actual_audit_hours=label.actual_audit_hours,
        outcome_recorded_at=resolved_outcome_recorded_at,
        kpi_window_days=label.kpi_window_days,
    )


@router.post("/inspector-labels/bulk", response_model=schemas.InspectorLabelBulkResult)
def submit_inspector_labels_bulk(
    payload: schemas.InspectorLabelBulkCreate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """
    Bulk ingest real inspector labels.

    strict_mode=true  -> reject all when any row is invalid
    strict_mode=false -> accept valid rows and return per-row errors
    """
    errors: list[dict[str, Any]] = []

    if payload.strict_mode:
        rows: list[models.InspectorLabel] = []
        for idx, item in enumerate(payload.labels):
            try:
                rows.append(
                    _build_inspector_label_entity(
                        label=item,
                        current_user=current_user,
                        db=db,
                    )
                )
            except HTTPException as exc:
                errors.append(
                    {
                        "index": idx,
                        "tax_code": item.tax_code,
                        "detail": exc.detail,
                    }
                )

        if errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Bulk ingest bị từ chối do có dòng không hợp lệ.",
                    "errors": errors,
                },
            )

        db.add_all(rows)
        db.commit()

        created_ids: list[int] = []
        for row in rows:
            db.refresh(row)
            if isinstance(row.id, int):
                created_ids.append(row.id)

        return schemas.InspectorLabelBulkResult(
            inserted=len(created_ids),
            rejected=0,
            created_ids=created_ids,
            errors=[],
        )

    created_ids: list[int] = []
    for idx, item in enumerate(payload.labels):
        try:
            row = _build_inspector_label_entity(
                label=item,
                current_user=current_user,
                db=db,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            if isinstance(row.id, int):
                created_ids.append(row.id)
        except HTTPException as exc:
            db.rollback()
            errors.append(
                {
                    "index": idx,
                    "tax_code": item.tax_code,
                    "detail": exc.detail,
                }
            )
        except Exception as exc:
            db.rollback()
            errors.append(
                {
                    "index": idx,
                    "tax_code": item.tax_code,
                    "detail": f"Lỗi hệ thống khi lưu label: {str(exc)}",
                }
            )

    return schemas.InspectorLabelBulkResult(
        inserted=len(created_ids),
        rejected=len(errors),
        created_ids=created_ids,
        errors=errors,
    )


@router.get("/inspector-labels/{tax_code}")
def get_inspector_labels(
    tax_code: str,
    db: Session = Depends(get_db),
):
    """Get all inspector labels for a specific company."""
    labels = (
        db.query(models.InspectorLabel)
        .filter(models.InspectorLabel.tax_code == tax_code)
        .order_by(models.InspectorLabel.created_at.desc())
        .limit(100)
        .all()
    )

    return {
        "tax_code": tax_code,
        "labels": [schemas.InspectorLabelResponse.model_validate(l) for l in labels],
        "total": len(labels),
    }

