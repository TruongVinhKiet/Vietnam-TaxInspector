"""
delinquency.py – Delinquency Prediction Router (REAL – Phase 0.1)
==================================================================
Replaces the old mock endpoint with real database queries and ML model
serving. Connects to:
    - tax_payments table (actual payment dates)
    - delinquency_predictions table (cached model predictions)
    - Temporal Compliance Intelligence model (Program A – Phase 1A)

Endpoints:
    GET  /api/delinquency           – List delinquency predictions (paginated, supports freshness filter)
    GET  /api/delinquency/{tax_code} – Single company delinquency detail
    POST /api/delinquency/predict-batch – Batch refresh/cache delinquency predictions
    GET  /api/delinquency/health/cache – Cache health summary for freshness/coverage monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from typing import Optional, Any
from datetime import date, datetime, timedelta
from collections import Counter
import re

from ..database import get_db
from .. import models, schemas
from . import monitoring as monitoring_router
from ..risk_utils import classify_delinquency_cluster
from ..observability import get_structured_logger, log_event

router = APIRouter(prefix="/api", tags=["Delinquency Prediction"])

logger = get_structured_logger("taxinspector.delinquency")

_HEALTH_ALERT_CODES = {
    "low_coverage",
    "partial_coverage",
    "high_stale_ratio",
    "stale_ratio_warning",
}
_HEALTH_ALERT_COOLDOWN = timedelta(minutes=5)
_health_alert_state = {
    "signature": None,
    "emitted_at": None,
}

_delinquency_pipeline = None

_TAX_CODE_10_PATTERN = re.compile(r"^\d{10}$")
_TAX_CODE_9_PATTERN = re.compile(r"^\d{9}$")


def _normalize_tax_code_for_contract(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if _TAX_CODE_10_PATTERN.fullmatch(raw):
        return raw
    if _TAX_CODE_9_PATTERN.fullmatch(raw):
        return f"0{raw}"
    raise ValueError(f"Invalid tax_code format: {raw}")


def _normalize_tax_code_for_response(value: Optional[str]) -> Optional[str]:
    try:
        return _normalize_tax_code_for_contract(value)
    except ValueError:
        return None


def _build_tax_code_lookup_candidates(value: Optional[str]) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []

    candidates = [raw]
    normalized = _normalize_tax_code_for_response(raw)
    if normalized and normalized not in candidates:
        candidates.append(normalized)

    if _TAX_CODE_10_PATTERN.fullmatch(raw) and raw.startswith("0"):
        legacy = raw[1:]
        if legacy not in candidates:
            candidates.append(legacy)

    return candidates


def _resolve_company_by_tax_code(db: Session, tax_code: str):
    for code in _build_tax_code_lookup_candidates(tax_code):
        company = db.query(models.Company).filter(models.Company.tax_code == code).first()
        if company:
            return company
    return None


def _classify_cluster(prob_30d: float, prob_60d: float, prob_90d: float) -> str:
    return classify_delinquency_cluster(prob_30d, prob_60d, prob_90d)


def _to_probability(value: Optional[float]) -> float:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, prob))


def _normalize_horizon_probs(
    prob_30d: Optional[float],
    prob_60d: Optional[float],
    prob_90d: Optional[float],
) -> tuple[float, float, float, bool]:
    """Keep 30/60/90 horizons monotonic: P30 <= P60 <= P90."""
    p30 = _to_probability(prob_30d)
    p60 = _to_probability(prob_60d)
    p90 = _to_probability(prob_90d)

    adjusted = False
    if p60 < p30:
        p60 = p30
        adjusted = True
    if p90 < p60:
        p90 = p60
        adjusted = True

    return round(p30, 4), round(p60, 4), round(p90, 4), adjusted


def _prediction_age_days(prediction_value: Optional[object]) -> Optional[int]:
    if prediction_value is None:
        return None

    parsed: Optional[date] = None
    if isinstance(prediction_value, datetime):
        parsed = prediction_value.date()
    elif isinstance(prediction_value, date):
        parsed = prediction_value
    elif isinstance(prediction_value, str):
        raw = prediction_value.strip()
        if raw:
            if "T" in raw:
                raw = raw.split("T", 1)[0]
            try:
                parsed = date.fromisoformat(raw)
            except ValueError:
                parsed = None

    if parsed is None:
        return None

    return max(0, (date.today() - parsed).days)


def _freshness_from_age(age_days: Optional[int]) -> str:
    if age_days is None:
        return "unknown"
    if age_days <= 7:
        return "fresh"
    if age_days <= 30:
        return "aging"
    return "stale"


def _normalize_freshness_filter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in {"fresh", "aging", "stale", "unknown"}:
        raise HTTPException(
            status_code=422,
            detail="freshness phải là một trong: fresh, aging, stale, unknown.",
        )
    return normalized


def _collect_model_versions(items: list[dict]) -> list[str]:
    versions = {
        str(item.get("model_version")).strip()
        for item in items
        if item.get("model_version")
    }
    return sorted(v for v in versions if v)


def _build_split_trigger_status_context(snapshot_source: str = "delinquency_detail") -> dict[str, Any]:
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


def _infer_score_source_from_model_version(model_version: Optional[str]) -> str:
    normalized = (model_version or "").strip().lower()
    if normalized in {"", "unknown", "none", "null"}:
        return "unknown"
    if normalized.startswith("baseline"):
        return "statistical_baseline"
    return "ml_model"


def _build_cache_health_summary(
    latest_predictions: list[dict],
    total_companies_with_payments: int,
    fresh_days: int,
    stale_days: int,
    sample_stale: int,
) -> dict:
    if fresh_days < 1 or stale_days < 1:
        raise ValueError("fresh_days và stale_days phải lớn hơn 0.")
    if fresh_days >= stale_days:
        raise ValueError("fresh_days phải nhỏ hơn stale_days.")

    total_predictions = len(latest_predictions)
    freshness_counter = Counter({"fresh": 0, "aging": 0, "stale": 0, "unknown": 0})
    source_counter = Counter()
    model_version_counter = Counter()
    stale_samples = []

    for row in latest_predictions:
        prediction_ref = row.get("prediction_ref")
        age_days = _prediction_age_days(prediction_ref)
        if age_days is None:
            freshness = "unknown"
        elif age_days <= fresh_days:
            freshness = "fresh"
        elif age_days <= stale_days:
            freshness = "aging"
        else:
            freshness = "stale"

        model_version = (row.get("model_version") or "").strip() or "unknown"
        score_source = _infer_score_source_from_model_version(model_version)

        freshness_counter[freshness] += 1
        source_counter[score_source] += 1
        model_version_counter[model_version] += 1

        if freshness == "stale":
            stale_samples.append(
                {
                    "tax_code": row.get("tax_code"),
                    "prediction_age_days": age_days,
                    "model_version": model_version,
                    "prob_90d": row.get("prob_90d"),
                }
            )

    stale_samples.sort(key=lambda item: item.get("prediction_age_days") or 0, reverse=True)

    if total_companies_with_payments > 0:
        coverage_ratio = round(total_predictions / total_companies_with_payments, 4)
    else:
        coverage_ratio = None

    freshness_ratios = {
        bucket: round((freshness_counter.get(bucket, 0) / total_predictions), 4) if total_predictions else 0.0
        for bucket in ("fresh", "aging", "stale", "unknown")
    }

    alerts = []
    status = "healthy"

    if total_companies_with_payments == 0:
        alerts.append(
            {
                "severity": "critical",
                "code": "no_tax_payment_data",
                "message": "Chưa có dữ liệu tax_payments để đánh giá coverage Delinquency cache.",
            }
        )
        status = "no_data"
    elif total_predictions == 0:
        alerts.append(
            {
                "severity": "critical",
                "code": "no_cache_predictions",
                "message": "Chưa có dự báo cache Delinquency. Cần chạy predict-batch/full refresh.",
            }
        )
        status = "critical"

    if coverage_ratio is not None:
        if coverage_ratio < 0.8:
            alerts.append(
                {
                    "severity": "critical",
                    "code": "low_coverage",
                    "message": f"Coverage thấp ({coverage_ratio:.1%}). Nên chạy full refresh để đạt >=95%.",
                }
            )
            status = "critical"
        elif coverage_ratio < 0.95:
            alerts.append(
                {
                    "severity": "warning",
                    "code": "partial_coverage",
                    "message": f"Coverage chưa tối ưu ({coverage_ratio:.1%}). Nên tăng tần suất refresh.",
                }
            )
            if status == "healthy":
                status = "warning"

    stale_ratio = freshness_ratios["stale"]
    unknown_ratio = freshness_ratios["unknown"]

    if stale_ratio > 0.3:
        alerts.append(
            {
                "severity": "critical",
                "code": "high_stale_ratio",
                "message": f"Tỷ lệ stale cao ({stale_ratio:.1%}). Cần refresh định kỳ hoặc tự động.",
            }
        )
        status = "critical"
    elif stale_ratio > 0.1:
        alerts.append(
            {
                "severity": "warning",
                "code": "stale_ratio_warning",
                "message": f"Tỷ lệ stale đáng chú ý ({stale_ratio:.1%}).",
            }
        )
        if status == "healthy":
            status = "warning"

    if unknown_ratio > 0.05:
        alerts.append(
            {
                "severity": "warning",
                "code": "unknown_freshness",
                "message": f"Có {unknown_ratio:.1%} prediction không xác định freshness.",
            }
        )
        if status == "healthy":
            status = "warning"

    return {
        "status": status,
        "generated_at": datetime.utcnow().isoformat(),
        "thresholds": {
            "fresh_days": fresh_days,
            "stale_days": stale_days,
        },
        "coverage": {
            "companies_with_payments": total_companies_with_payments,
            "companies_with_latest_prediction": total_predictions,
            "coverage_ratio": coverage_ratio,
        },
        "freshness": {
            "counts": {
                "fresh": freshness_counter.get("fresh", 0),
                "aging": freshness_counter.get("aging", 0),
                "stale": freshness_counter.get("stale", 0),
                "unknown": freshness_counter.get("unknown", 0),
            },
            "ratios": freshness_ratios,
        },
        "sources": {
            "ml_model": source_counter.get("ml_model", 0),
            "statistical_baseline": source_counter.get("statistical_baseline", 0),
            "unknown": source_counter.get("unknown", 0),
        },
        "model_versions": [
            {
                "model_version": version,
                "count": count,
            }
            for version, count in sorted(
                model_version_counter.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        ],
        "stale_samples": stale_samples[:sample_stale],
        "alerts": alerts,
    }


def _extract_health_threshold_alerts(summary: dict) -> list[dict]:
    alerts = summary.get("alerts") if isinstance(summary, dict) else None
    if not isinstance(alerts, list):
        return []
    return [
        alert
        for alert in alerts
        if isinstance(alert, dict) and alert.get("code") in _HEALTH_ALERT_CODES
    ]


def _emit_cache_health_events(summary: dict):
    threshold_alerts = _extract_health_threshold_alerts(summary)
    coverage = summary.get("coverage") if isinstance(summary, dict) else {}
    freshness = summary.get("freshness") if isinstance(summary, dict) else {}
    freshness_ratios = freshness.get("ratios") if isinstance(freshness, dict) else {}

    coverage_ratio = coverage.get("coverage_ratio") if isinstance(coverage, dict) else None
    stale_ratio = freshness_ratios.get("stale") if isinstance(freshness_ratios, dict) else None
    status = summary.get("status") if isinstance(summary, dict) else "unknown"

    if not threshold_alerts:
        log_event(
            logger,
            "info",
            "delinquency_cache_health_ok",
            status=status,
            coverage_ratio=coverage_ratio,
            stale_ratio=stale_ratio,
        )
        _health_alert_state["signature"] = None
        _health_alert_state["emitted_at"] = datetime.utcnow()
        return

    alert_codes = sorted({str(alert.get("code")) for alert in threshold_alerts})
    signature = "|".join(
        [
            str(status),
            ",".join(alert_codes),
            str(coverage_ratio),
            str(stale_ratio),
        ]
    )
    now = datetime.utcnow()

    last_signature = _health_alert_state.get("signature")
    last_emitted_at = _health_alert_state.get("emitted_at")
    should_emit = (
        signature != last_signature
        or not isinstance(last_emitted_at, datetime)
        or (now - last_emitted_at) >= _HEALTH_ALERT_COOLDOWN
    )
    if not should_emit:
        return

    has_critical = any(str(alert.get("severity", "")).lower() == "critical" for alert in threshold_alerts)
    severity = "critical" if has_critical else "warning"
    level = "error" if has_critical else "warning"

    log_event(
        logger,
        level,
        "delinquency_cache_health_threshold_breach",
        status=status,
        severity=severity,
        alert_codes=alert_codes,
        alert_count=len(threshold_alerts),
        coverage_ratio=coverage_ratio,
        stale_ratio=stale_ratio,
    )

    _health_alert_state["signature"] = signature
    _health_alert_state["emitted_at"] = now


def _get_delinquency_pipeline():
    """Lazy-load delinquency pipeline; return None if model artifact is unavailable."""
    global _delinquency_pipeline
    if _delinquency_pipeline is not None:
        return _delinquency_pipeline

    try:
        from ml_engine.delinquency_model import DelinquencyPipeline

        _delinquency_pipeline = DelinquencyPipeline()
        _delinquency_pipeline.load_models()
        return _delinquency_pipeline
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"[WARN] Failed to load delinquency pipeline: {exc}")
        return None


def _serialize_payments(payments: list[models.TaxPayment]) -> list[dict]:
    return [
        {
            "due_date": p.due_date,
            "actual_payment_date": p.actual_payment_date,
            "amount_due": float(p.amount_due or 0),
            "amount_paid": float(p.amount_paid or 0),
            "penalty_amount": float(p.penalty_amount or 0),
            "tax_period": p.tax_period,
            "status": p.status,
        }
        for p in payments
    ]


def _serialize_tax_returns(tax_returns: list[models.TaxReturn]) -> list[dict]:
    return [
        {
            "filing_date": tr.filing_date,
            "revenue": float(tr.revenue or 0),
            "expenses": float(tr.expenses or 0),
        }
        for tr in tax_returns
    ]


def _build_payment_history_summary(db: Session, tax_code: str) -> dict:
    """Build a compact payment history summary from tax_payments table."""
    recent_cutoff = date.today() - timedelta(days=730)  # 2 years

    payments = (
        db.query(models.TaxPayment)
        .filter(
            models.TaxPayment.tax_code == tax_code,
            models.TaxPayment.due_date >= recent_cutoff,
        )
        .order_by(models.TaxPayment.due_date.desc())
        .limit(50)
        .all()
    )

    if not payments:
        return {
            "total_periods": 0,
            "on_time_count": 0,
            "late_count": 0,
            "unpaid_count": 0,
            "avg_days_late": 0.0,
            "total_penalties": 0.0,
            "data_available": False,
        }

    on_time = 0
    late = 0
    unpaid = 0
    late_days_list = []
    total_penalties = 0.0

    for p in payments:
        total_penalties += float(p.penalty_amount or 0)
        if p.actual_payment_date is None:
            if p.due_date < date.today():
                unpaid += 1
            # else: not yet due
        elif p.actual_payment_date > p.due_date:
            late += 1
            late_days_list.append((p.actual_payment_date - p.due_date).days)
        else:
            on_time += 1

    return {
        "total_periods": len(payments),
        "on_time_count": on_time,
        "late_count": late,
        "unpaid_count": unpaid,
        "avg_days_late": round(sum(late_days_list) / max(1, len(late_days_list)), 1) if late_days_list else 0.0,
        "max_days_late": max(late_days_list) if late_days_list else 0,
        "total_penalties": round(total_penalties, 2),
        "data_available": True,
    }


def _build_early_warning_payload(
    db: Session,
    tax_code: str,
    payment_history_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build early warning signals for non-filer / under-filer workflows.

    Heuristics are intentionally conservative and use recent 180-day windows so
    they remain stable for list/detail rendering without requiring extra tables.
    """
    summary = payment_history_summary if isinstance(payment_history_summary, dict) else _build_payment_history_summary(db, tax_code)

    lookback_days = 180
    since = date.today() - timedelta(days=lookback_days)

    due_rows = (
        db.query(models.TaxPayment.due_date, models.TaxPayment.actual_payment_date)
        .filter(
            models.TaxPayment.tax_code == tax_code,
            models.TaxPayment.due_date >= since,
            models.TaxPayment.due_date <= date.today(),
        )
        .all()
    )
    due_periods = len(due_rows)
    unpaid_due_periods = sum(1 for due_date, paid_date in due_rows if due_date and due_date < date.today() and paid_date is None)

    filed_query = (
        db.query(func.count(models.TaxReturn.id))
        .filter(
            models.TaxReturn.tax_code == tax_code,
            models.TaxReturn.filing_date >= since,
            models.TaxReturn.filing_date <= date.today(),
        )
    )
    filed_returns = filed_query.scalar() if hasattr(filed_query, "scalar") else 0
    filed_returns_count = int(filed_returns or 0)

    filing_coverage = round(filed_returns_count / due_periods, 3) if due_periods > 0 else None
    unpaid_ratio = round(unpaid_due_periods / due_periods, 3) if due_periods > 0 else 0.0

    level_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    queue_order = {"monitor": 0, "watchlist": 1, "priority_review": 2}

    level = "low"
    queue = "monitor"
    tags: list[str] = []
    reasons: list[str] = []

    def _raise_priority(next_level: str, next_queue: str, tag: str, reason: str) -> None:
        nonlocal level, queue
        if level_order.get(next_level, 0) > level_order.get(level, 0):
            level = next_level
        if queue_order.get(next_queue, 0) > queue_order.get(queue, 0):
            queue = next_queue
        if tag not in tags:
            tags.append(tag)
        if reason and reason not in reasons:
            reasons.append(reason)

    if due_periods >= 2 and filed_returns_count == 0:
        _raise_priority(
            "critical",
            "priority_review",
            "non_filer",
            f"Không ghi nhận tờ khai trong {due_periods} kỳ nghĩa vụ gần nhất (6 tháng).",
        )
    elif due_periods >= 3 and filing_coverage is not None and filing_coverage < 0.5:
        _raise_priority(
            "high",
            "priority_review",
            "under_filer",
            f"Tỷ lệ nộp tờ khai thấp ({int(round(filing_coverage * 100))}%) so với số kỳ nghĩa vụ phát sinh.",
        )
    elif due_periods >= 3 and filing_coverage is not None and filing_coverage < 0.8:
        _raise_priority(
            "medium",
            "watchlist",
            "filing_gap",
            f"Tỷ lệ nộp tờ khai chưa đạt ngưỡng an toàn ({int(round(filing_coverage * 100))}%).",
        )

    if unpaid_ratio >= 0.4:
        _raise_priority(
            "high",
            "priority_review",
            "unpaid_due",
            f"Có {unpaid_due_periods}/{due_periods} kỳ quá hạn chưa nộp tiền thuế.",
        )
    elif unpaid_ratio >= 0.2:
        _raise_priority(
            "medium",
            "watchlist",
            "unpaid_due",
            f"Ghi nhận {unpaid_due_periods}/{due_periods} kỳ nghĩa vụ chưa nộp đúng hạn.",
        )

    late_count = int(summary.get("late_count") or 0)
    avg_days_late = float(summary.get("avg_days_late") or 0)
    if late_count >= 4 and avg_days_late >= 15:
        _raise_priority(
            "medium",
            "watchlist",
            "persistent_late",
            f"Lịch sử trễ hạn kéo dài ({late_count} kỳ, trung bình {avg_days_late:.1f} ngày).",
        )

    has_warning = level != "low"
    if has_warning:
        primary_reason = reasons[0] if reasons else "Phát hiện tín hiệu non-filer/under-filer cần theo dõi."
    else:
        primary_reason = "Chưa ghi nhận tín hiệu non-filer/under-filer đáng kể trong kỳ gần đây."

    return {
        "has_warning": has_warning,
        "queue": queue,
        "level": level,
        "tags": tags,
        "reason": primary_reason,
        "metrics": {
            "lookback_days": lookback_days,
            "due_periods": due_periods,
            "filed_returns": filed_returns_count,
            "filing_coverage": filing_coverage,
            "unpaid_due_periods": unpaid_due_periods,
            "unpaid_ratio": unpaid_ratio,
            "late_count": late_count,
            "avg_days_late": round(avg_days_late, 1),
        },
    }


def _build_intervention_uplift_payload(
    probability: Optional[float],
    prob_30d: Optional[float],
    prob_60d: Optional[float],
    prob_90d: Optional[float],
    payment_history_summary: Optional[dict[str, Any]] = None,
    early_warning: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build intervention recommendation and expected uplift for company-level detail."""
    p30, p60, p90, _ = _normalize_horizon_probs(prob_30d, prob_60d, prob_90d)
    overall_probability = max(_to_probability(probability), p30, p60, p90)

    summary = payment_history_summary if isinstance(payment_history_summary, dict) else {}
    warning = early_warning if isinstance(early_warning, dict) else {}

    queue = str(warning.get("queue") or "monitor").strip().lower()
    level = str(warning.get("level") or "low").strip().lower()
    warning_reason = str(warning.get("reason") or "").strip()

    total_periods = int(summary.get("total_periods") or 0)
    unpaid_count = int(summary.get("unpaid_count") or 0)
    late_count = int(summary.get("late_count") or 0)
    avg_days_late = float(summary.get("avg_days_late") or 0)
    total_penalties = float(summary.get("total_penalties") or 0)

    if queue == "priority_review" or level in {"critical", "high"} or overall_probability >= 0.8 or unpaid_count >= 4:
        action = "escalated_enforcement"
    elif overall_probability >= 0.65 or unpaid_count >= 2 or (late_count >= 4 and avg_days_late >= 15):
        action = "field_audit"
    elif overall_probability >= 0.4 or queue == "watchlist" or late_count >= 2:
        action = "structured_outreach"
    elif overall_probability >= 0.2 or late_count >= 1:
        action = "auto_reminder"
    else:
        action = "monitor"

    if total_periods >= 6:
        confidence = "high"
    elif total_periods >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    queue_bonus = 15 if queue == "priority_review" else (8 if queue == "watchlist" else 0)
    priority_score = int(
        round(
            min(
                100,
                max(
                    0,
                    overall_probability * 70 + unpaid_count * 7 + late_count * 4 + queue_bonus,
                ),
            )
        )
    )

    impact_ratio_map = {
        "monitor": 0.05,
        "auto_reminder": 0.12,
        "structured_outreach": 0.2,
        "field_audit": 0.3,
        "escalated_enforcement": 0.38,
    }
    impact_ratio = impact_ratio_map.get(action, 0.12)

    expected_risk_reduction_pp = round(overall_probability * impact_ratio * 100, 1)
    expected_penalty_saving = round(total_penalties * min(0.9, impact_ratio + 0.12), 2) if total_penalties > 0 else 0.0

    base_recoverable_vnd = max(0.0, unpaid_count * 2_500_000 + late_count * 800_000)
    expected_collection_uplift = round(base_recoverable_vnd * impact_ratio, 2)

    next_steps_map = {
        "monitor": [
            "Theo dõi tiếp 1-2 chu kỳ nộp thuế gần nhất.",
            "Duy trì nhắc lịch nộp tờ khai qua kênh tự động.",
        ],
        "auto_reminder": [
            "Gửi nhắc hạn nộp tờ khai/NNT trước hạn 7 ngày.",
            "Theo dõi phản hồi trong 72 giờ và cập nhật trạng thái.",
        ],
        "structured_outreach": [
            "Liên hệ bổ sung thông tin qua phone/email theo playbook delinquency.",
            "Đặt mốc cam kết nộp bổ sung trong vòng 5 ngày làm việc.",
        ],
        "field_audit": [
            "Mở soát xét hồ sơ chi tiết và đối chiếu lịch sử nộp/chậm nộp.",
            "Phân công cán bộ xử lý theo nhóm rủi ro cao.",
        ],
        "escalated_enforcement": [
            "Kích hoạt quy trình cưỡng chế theo quy định nội bộ.",
            "Phối hợp đội thu nợ để giảm tồn đọng trong 30 ngày.",
        ],
    }

    rationale_parts = []
    if warning_reason:
        rationale_parts.append(warning_reason)
    rationale_parts.append(f"Xác suất trễ hạn 90 ngày hiện tại {int(round(p90 * 100))}%.")
    if unpaid_count > 0:
        rationale_parts.append(f"Đang có {unpaid_count} kỳ quá hạn chưa nộp.")
    if late_count > 0:
        rationale_parts.append(f"Lịch sử ghi nhận {late_count} kỳ trễ hạn.")

    return {
        "recommended_action": action,
        "priority_score": priority_score,
        "expected_risk_reduction_pp": expected_risk_reduction_pp,
        "expected_penalty_saving": expected_penalty_saving,
        "expected_collection_uplift": expected_collection_uplift,
        "confidence": confidence,
        "rationale": " ".join(rationale_parts).strip() or "Chưa ghi nhận động lực can thiệp mạnh ở chu kỳ hiện tại.",
        "next_steps": next_steps_map.get(action, []),
    }


def _build_baseline_prediction(db: Session, tax_code: str, company_name: str) -> dict:
    """
    Build a baseline delinquency prediction from payment history statistics.
    This is used when no ML model prediction is cached yet (Phase 0 baseline).
    The real ML model (Program A) will replace this in Phase 1A.
    """
    normalized_tax_code = _normalize_tax_code_for_response(tax_code)
    if normalized_tax_code is None:
        return {}

    history = _build_payment_history_summary(db, tax_code)

    if not history["data_available"] or history["total_periods"] == 0:
        early_warning = _build_early_warning_payload(db, tax_code, history)
        intervention_uplift = _build_intervention_uplift_payload(
            probability=0.0,
            prob_30d=0.0,
            prob_60d=0.0,
            prob_90d=0.0,
            payment_history_summary=history,
            early_warning=early_warning,
        )
        return {
            "tax_code": normalized_tax_code,
            "company_name": company_name,
            "probability": 0.0,
            "prob_30d": 0.0,
            "prob_60d": 0.0,
            "prob_90d": 0.0,
            "cluster": "Chưa có dữ liệu",
            "top_reasons": [],
            "model_version": "baseline-statistical-v1",
            "model_confidence": None,
            "prediction_date": str(date.today()),
            "score_source": "no_data",
            "prediction_age_days": None,
            "freshness": "unknown",
            "monotonic_adjusted": False,
            "early_warning": early_warning,
            "intervention_uplift": intervention_uplift,
            "payment_history_summary": history,
        }

    # Simple statistical baseline: late ratio as probability proxy
    total = history["total_periods"]
    late_ratio = (history["late_count"] + history["unpaid_count"]) / max(1, total)
    unpaid_ratio = history["unpaid_count"] / max(1, total)
    avg_days = history["avg_days_late"]

    # Heuristic probability estimates (will be replaced by ML model)
    prob_30d = min(1.0, round(late_ratio * 0.6 + unpaid_ratio * 0.3 + min(avg_days / 90, 0.3), 4))
    prob_60d = min(1.0, round(prob_30d * 1.15 + unpaid_ratio * 0.1, 4))
    prob_90d = min(1.0, round(prob_60d * 1.10, 4))
    prob_30d, prob_60d, prob_90d, monotonic_adjusted = _normalize_horizon_probs(prob_30d, prob_60d, prob_90d)
    overall_prob = round(prob_90d, 4)

    reasons = []
    if history["late_count"] > 0:
        reasons.append({"reason": f"Đã trễ hạn {history['late_count']}/{total} kỳ gần đây", "weight": round(late_ratio, 2)})
    if history["unpaid_count"] > 0:
        reasons.append({"reason": f"Còn {history['unpaid_count']} kỳ chưa nộp", "weight": round(unpaid_ratio, 2)})
    if avg_days > 15:
        reasons.append({"reason": f"Trung bình trễ {avg_days:.0f} ngày", "weight": round(min(avg_days / 90, 1.0), 2)})
    if history["total_penalties"] > 0:
        reasons.append({"reason": f"Tổng phạt: {history['total_penalties']:,.0f} VNĐ", "weight": 0.1})

    cluster = _classify_cluster(prob_30d, prob_60d, prob_90d)
    early_warning = _build_early_warning_payload(db, tax_code, history)
    intervention_uplift = _build_intervention_uplift_payload(
        probability=overall_prob,
        prob_30d=prob_30d,
        prob_60d=prob_60d,
        prob_90d=prob_90d,
        payment_history_summary=history,
        early_warning=early_warning,
    )

    return {
        "tax_code": normalized_tax_code,
        "company_name": company_name,
        "probability": overall_prob,
        "prob_30d": prob_30d,
        "prob_60d": prob_60d,
        "prob_90d": prob_90d,
        "cluster": cluster,
        "top_reasons": reasons,
        "model_version": "baseline-statistical-v1",
        "model_confidence": None,
        "prediction_date": str(date.today()),
        "score_source": "statistical_baseline",
        "prediction_age_days": 0,
        "freshness": "fresh",
        "monotonic_adjusted": monotonic_adjusted,
        "early_warning": early_warning,
        "intervention_uplift": intervention_uplift,
        "payment_history_summary": history,
    }


@router.get("/delinquency", response_model=schemas.DelinquencyListResponse)
def get_delinquency_forecast(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_probability: float = Query(0.0, ge=0.0, le=1.0),
    cluster: Optional[str] = None,
    freshness: Optional[str] = Query(None, description="fresh | aging | stale | unknown"),
    db: Session = Depends(get_db),
):
    """
    Tab 3: Dự báo nguy cơ trễ hạn nộp thuế.
    Returns paginated delinquency predictions from the database.
    Falls back to statistical baseline if ML model has not yet been run.
    """
    freshness_filter = _normalize_freshness_filter(freshness)

    # Try cached ML predictions first
    query = db.query(models.DelinquencyPrediction)

    if min_probability > 0:
        query = query.filter(models.DelinquencyPrediction.prob_90d >= min_probability)
    if cluster:
        query = query.filter(models.DelinquencyPrediction.risk_cluster.ilike(f"%{cluster}%"))
    if freshness_filter:
        fresh_cutoff = date.today() - timedelta(days=7)
        aging_cutoff = date.today() - timedelta(days=30)

        if freshness_filter == "fresh":
            query = query.filter(models.DelinquencyPrediction.prediction_date >= fresh_cutoff)
        elif freshness_filter == "aging":
            query = query.filter(
                models.DelinquencyPrediction.prediction_date < fresh_cutoff,
                models.DelinquencyPrediction.prediction_date >= aging_cutoff,
            )
        elif freshness_filter == "stale":
            query = query.filter(models.DelinquencyPrediction.prediction_date < aging_cutoff)
        else:  # unknown
            query = query.filter(models.DelinquencyPrediction.prediction_date.is_(None))

    # Get latest prediction per company
    subq = (
        db.query(
            models.DelinquencyPrediction.tax_code,
            func.max(models.DelinquencyPrediction.created_at).label("latest"),
        )
        .group_by(models.DelinquencyPrediction.tax_code)
        .subquery()
    )

    query = query.join(
        subq,
        and_(
            models.DelinquencyPrediction.tax_code == subq.c.tax_code,
            models.DelinquencyPrediction.created_at == subq.c.latest,
        ),
    )

    total = query.count()
    offset = (page - 1) * page_size
    predictions_db = (
        query.order_by(desc(models.DelinquencyPrediction.prob_90d))
        .offset(offset)
        .limit(page_size)
        .all()
    )

    # If we have cached predictions, return them
    if predictions_db:
        tax_codes = [pred.tax_code for pred in predictions_db]
        lookup_tax_codes = []
        for code in tax_codes:
            for candidate in _build_tax_code_lookup_candidates(code):
                if candidate not in lookup_tax_codes:
                    lookup_tax_codes.append(candidate)

        company_rows = (
            db.query(models.Company.tax_code, models.Company.name)
            .filter(models.Company.tax_code.in_(lookup_tax_codes))
            .all()
        )
        company_name_map = {tc: name for tc, name in company_rows}

        items = []
        for pred in predictions_db:
            normalized_tax_code = _normalize_tax_code_for_response(pred.tax_code)
            if normalized_tax_code is None:
                continue

            company_name = company_name_map.get(pred.tax_code, "") or company_name_map.get(normalized_tax_code, "")
            payment_summary = _build_payment_history_summary(db, pred.tax_code)
            early_warning = _build_early_warning_payload(db, pred.tax_code, payment_summary)
            prob_30d, prob_60d, prob_90d, monotonic_adjusted = _normalize_horizon_probs(
                pred.prob_30d,
                pred.prob_60d,
                pred.prob_90d,
            )
            intervention_uplift = _build_intervention_uplift_payload(
                probability=prob_90d,
                prob_30d=prob_30d,
                prob_60d=prob_60d,
                prob_90d=prob_90d,
                payment_history_summary=payment_summary,
                early_warning=early_warning,
            )
            prediction_ref = pred.prediction_date or (pred.created_at.date() if pred.created_at else None)
            age_days = _prediction_age_days(prediction_ref)
            top_reasons = []
            if isinstance(pred.top_reasons, list):
                top_reasons = [
                    {"reason": r.get("reason", ""), "weight": float(r.get("weight", 0))}
                    for r in pred.top_reasons if isinstance(r, dict)
                ]

            items.append({
                "tax_code": normalized_tax_code,
                "company_name": company_name,
                "probability": prob_90d,
                "prob_30d": prob_30d,
                "prob_60d": prob_60d,
                "prob_90d": prob_90d,
                "cluster": pred.risk_cluster or "",
                "top_reasons": top_reasons,
                "model_version": pred.model_version,
                "model_confidence": float(pred.model_confidence) if pred.model_confidence else None,
                "prediction_date": str(pred.prediction_date) if pred.prediction_date else None,
                "score_source": "ml_model",
                "prediction_age_days": age_days,
                "freshness": _freshness_from_age(age_days),
                "monotonic_adjusted": monotonic_adjusted,
                "early_warning": early_warning,
                "intervention_uplift": intervention_uplift,
                "payment_history_summary": payment_summary,
            })

        stale_count = sum(1 for item in items if item.get("freshness") == "stale")
        fresh_count = sum(1 for item in items if item.get("freshness") == "fresh")
        model_versions = _collect_model_versions(items)
        model_version = model_versions[0] if len(model_versions) == 1 else ("mixed" if model_versions else None)

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "predictions": items,
            "model_info": {
                "source": "ml_model",
                "model_version": model_version,
                "model_versions": model_versions,
                "note": "Cached delinquency predictions from ML model.",
                "fresh_count": fresh_count,
                "stale_count": stale_count,
                "applied_freshness_filter": freshness_filter,
            },
        }

    # Fallback: build statistical baseline from payment data for top companies
    companies_with_payments = (
        db.query(models.TaxPayment.tax_code)
        .filter(models.TaxPayment.tax_code.isnot(None))
        .distinct()
        .limit(200)
        .all()
    )
    tax_codes = [r[0] for r in companies_with_payments if _normalize_tax_code_for_response(r[0]) is not None]

    company_rows = (
        db.query(models.Company.tax_code, models.Company.name)
        .filter(models.Company.tax_code.in_(tax_codes))
        .all()
        if tax_codes
        else []
    )
    company_name_map = {tc: name for tc, name in company_rows}

    if not tax_codes:
        # No payment data at all – return empty with guidance
        return {
            "total": 0,
            "page": page,
            "page_size": page_size,
            "predictions": [],
            "model_info": {
                "source": "no_data",
                "model_version": None,
                "note": "Chưa có dữ liệu thanh toán (tax_payments). Vui lòng import dữ liệu ngày nộp thực tế.",
                "applied_freshness_filter": freshness_filter,
            },
        }

    all_baselines = []
    for tc in tax_codes:
        company_name = company_name_map.get(tc, "")
        baseline = _build_baseline_prediction(db, tc, company_name)
        if not baseline:
            continue
        if baseline["probability"] >= min_probability:
            if not cluster or (cluster.lower() in baseline["cluster"].lower()):
                if not freshness_filter or baseline.get("freshness") == freshness_filter:
                    all_baselines.append(baseline)

    all_baselines.sort(key=lambda x: x["probability"], reverse=True)
    total_baselines = len(all_baselines)
    paged = all_baselines[offset:offset + page_size]

    return {
        "total": total_baselines,
        "page": page,
        "page_size": page_size,
        "predictions": paged,
        "model_info": {
            "source": "statistical_baseline",
            "model_version": "baseline-statistical-v1",
            "model_versions": ["baseline-statistical-v1"],
            "note": "Dự báo dựa trên thống kê lịch sử thanh toán. Model ML (Phase 1A) sẽ thay thế.",
            "applied_freshness_filter": freshness_filter,
        },
    }


@router.get("/delinquency/health/cache")
def get_delinquency_cache_health(
    fresh_days: int = Query(7, ge=1, le=90),
    stale_days: int = Query(30, ge=2, le=365),
    sample_stale: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Operational cache health endpoint for Delinquency predictions."""
    if fresh_days >= stale_days:
        raise HTTPException(
            status_code=422,
            detail="fresh_days phải nhỏ hơn stale_days.",
        )

    total_companies_with_payments = (
        db.query(models.TaxPayment.tax_code)
        .filter(models.TaxPayment.tax_code.isnot(None))
        .distinct()
        .count()
    )

    subq = (
        db.query(
            models.DelinquencyPrediction.tax_code,
            func.max(models.DelinquencyPrediction.created_at).label("latest"),
        )
        .group_by(models.DelinquencyPrediction.tax_code)
        .subquery()
    )

    latest_rows = (
        db.query(
            models.DelinquencyPrediction.tax_code,
            models.DelinquencyPrediction.prediction_date,
            models.DelinquencyPrediction.created_at,
            models.DelinquencyPrediction.model_version,
            models.DelinquencyPrediction.prob_90d,
        )
        .join(
            subq,
            and_(
                models.DelinquencyPrediction.tax_code == subq.c.tax_code,
                models.DelinquencyPrediction.created_at == subq.c.latest,
            ),
        )
        .all()
    )

    latest_predictions = []
    for row in latest_rows:
        prediction_ref = row.prediction_date or (row.created_at.date() if row.created_at else None)
        latest_predictions.append(
            {
                "tax_code": row.tax_code,
                "prediction_ref": prediction_ref,
                "model_version": row.model_version,
                "prob_90d": float(row.prob_90d) if row.prob_90d is not None else None,
            }
        )

    summary = _build_cache_health_summary(
        latest_predictions=latest_predictions,
        total_companies_with_payments=total_companies_with_payments,
        fresh_days=fresh_days,
        stale_days=stale_days,
        sample_stale=sample_stale,
    )
    _emit_cache_health_events(summary)
    return summary


@router.get("/delinquency/{tax_code}", response_model=schemas.DelinquencyPredictionItem)
def get_delinquency_detail(tax_code: str, db: Session = Depends(get_db)):
    """
    Get delinquency prediction detail for a specific company.
    """
    company = _resolve_company_by_tax_code(db, tax_code)
    if not company:
        raise HTTPException(status_code=404, detail="Không tìm thấy doanh nghiệp với MST này.")

    resolved_tax_code = company.tax_code

    # Check cached ML prediction first
    cached = (
        db.query(models.DelinquencyPrediction)
        .filter(models.DelinquencyPrediction.tax_code == resolved_tax_code)
        .order_by(desc(models.DelinquencyPrediction.created_at))
        .first()
    )

    if cached:
        split_trigger_status = _build_split_trigger_status_context(snapshot_source="delinquency_detail_cached")
        payment_summary = _build_payment_history_summary(db, resolved_tax_code)
        early_warning = _build_early_warning_payload(db, resolved_tax_code, payment_summary)
        prob_30d, prob_60d, prob_90d, monotonic_adjusted = _normalize_horizon_probs(
            cached.prob_30d,
            cached.prob_60d,
            cached.prob_90d,
        )
        intervention_uplift = _build_intervention_uplift_payload(
            probability=prob_90d,
            prob_30d=prob_30d,
            prob_60d=prob_60d,
            prob_90d=prob_90d,
            payment_history_summary=payment_summary,
            early_warning=early_warning,
        )
        prediction_ref = cached.prediction_date or (cached.created_at.date() if cached.created_at else None)
        age_days = _prediction_age_days(prediction_ref)
        top_reasons = []
        if isinstance(cached.top_reasons, list):
            top_reasons = [
                {"reason": r.get("reason", ""), "weight": float(r.get("weight", 0))}
                for r in cached.top_reasons if isinstance(r, dict)
            ]
        normalized_tax_code = _normalize_tax_code_for_response(cached.tax_code)
        if normalized_tax_code is None:
            raise HTTPException(status_code=422, detail="MST doanh nghiệp không hợp lệ theo chuẩn 10 chữ số.")
        return {
            "tax_code": normalized_tax_code,
            "company_name": company.name,
            "probability": prob_90d,
            "prob_30d": prob_30d,
            "prob_60d": prob_60d,
            "prob_90d": prob_90d,
            "cluster": cached.risk_cluster or "",
            "top_reasons": top_reasons,
            "model_version": cached.model_version,
            "model_confidence": float(cached.model_confidence) if cached.model_confidence else None,
            "prediction_date": str(cached.prediction_date) if cached.prediction_date else None,
            "score_source": "ml_model",
            "prediction_age_days": age_days,
            "freshness": _freshness_from_age(age_days),
            "monotonic_adjusted": monotonic_adjusted,
            "early_warning": early_warning,
            "intervention_uplift": intervention_uplift,
            "split_trigger_status": split_trigger_status,
            "payment_history_summary": payment_summary,
        }

    # Fallback to statistical baseline
    split_trigger_status = _build_split_trigger_status_context(snapshot_source="delinquency_detail_baseline")
    baseline_payload = _build_baseline_prediction(db, resolved_tax_code, company.name)
    if not baseline_payload:
        raise HTTPException(status_code=422, detail="MST doanh nghiệp không hợp lệ theo chuẩn 10 chữ số.")
    baseline_payload["split_trigger_status"] = split_trigger_status
    return baseline_payload


def _build_batch_candidate_tax_codes(
    db: Session,
    requested_tax_codes: Optional[list[str]],
    limit: int,
    refresh_existing: bool,
) -> list[str]:
    if requested_tax_codes:
        cleaned = []
        seen = set()
        for raw in requested_tax_codes:
            code = (raw or "").strip()
            if not code:
                continue
            if code in seen:
                continue
            seen.add(code)
            cleaned.append(code)

        if not cleaned:
            return []

        expanded = []
        expanded_seen = set()
        for code in cleaned:
            for candidate in _build_tax_code_lookup_candidates(code):
                if candidate in expanded_seen:
                    continue
                expanded_seen.add(candidate)
                expanded.append(candidate)

        rows = (
            db.query(models.TaxPayment.tax_code)
            .filter(models.TaxPayment.tax_code.in_(expanded))
            .distinct()
            .all()
        )
        candidates = [row[0] for row in rows if row and row[0]]
    else:
        rows = (
            db.query(models.TaxPayment.tax_code)
            .filter(models.TaxPayment.tax_code.isnot(None))
            .distinct()
            .order_by(models.TaxPayment.tax_code.asc())
            .all()
        )
        candidates = [row[0] for row in rows if row and row[0]]

    candidates = [code for code in candidates if _normalize_tax_code_for_response(code) is not None]

    if not refresh_existing and candidates:
        today = date.today()
        existing_today_rows = (
            db.query(models.DelinquencyPrediction.tax_code)
            .filter(
                models.DelinquencyPrediction.tax_code.in_(candidates),
                models.DelinquencyPrediction.prediction_date >= today,
            )
            .distinct()
            .all()
        )
        existing_today = {row[0] for row in existing_today_rows if row and row[0]}
        candidates = [code for code in candidates if code not in existing_today]

    return candidates[:limit]


def _predict_and_persist_tax_code(
    db: Session,
    tax_code: str,
    pipeline,
    refresh_existing: bool,
) -> dict:
    lookup_candidates = _build_tax_code_lookup_candidates(tax_code)
    company = None
    for code in lookup_candidates:
        company = db.query(models.Company).filter(models.Company.tax_code == code).first()
        if company:
            break

    resolved_tax_code = company.tax_code if company else tax_code
    normalized_tax_code = _normalize_tax_code_for_response(resolved_tax_code)
    if normalized_tax_code is None:
        return {
            "tax_code": "0000000000",
            "status": "failed",
            "score_source": "inference_error",
            "prob_90d": None,
            "model_version": None,
            "message": "MST không hợp lệ theo chuẩn 10 chữ số.",
        }

    company_name = company.name if company else ""

    payments = (
        db.query(models.TaxPayment)
        .filter(models.TaxPayment.tax_code.in_(lookup_candidates))
        .order_by(models.TaxPayment.due_date.asc())
        .all()
    )
    if not payments:
        return {
            "tax_code": normalized_tax_code,
            "status": "skipped",
            "score_source": "no_data",
            "prob_90d": None,
            "model_version": None,
            "message": "Không có dữ liệu tax_payments để dự báo.",
        }

    if pipeline is not None:
        try:
            import pandas as pd

            tax_returns = (
                db.query(models.TaxReturn)
                .filter(models.TaxReturn.tax_code.in_(lookup_candidates))
                .order_by(models.TaxReturn.filing_date.asc())
                .all()
            )

            payments_df = pd.DataFrame(_serialize_payments(payments))
            tax_returns_df = pd.DataFrame(_serialize_tax_returns(tax_returns))
            company_info = {
                "tax_code": resolved_tax_code,
                "name": company_name,
                "industry": company.industry if company else None,
                # Company model stores geographic slice as province.
                "region": getattr(company, "province", None) if company else None,
                "registration_date": company.registration_date if company else None,
            }

            pred = pipeline.predict_single(payments_df, tax_returns_df, company_info=company_info)
            prob_30d, prob_60d, prob_90d, monotonic_adjusted = _normalize_horizon_probs(
                pred.get("prob_30d"),
                pred.get("prob_60d"),
                pred.get("prob_90d"),
            )
            cluster = pred.get("cluster") or _classify_cluster(prob_30d, prob_60d, prob_90d)
            top_reasons = pred.get("top_reasons") if isinstance(pred.get("top_reasons"), list) else []
            model_version = pred.get("model_version") or "delinquency-temporal-v1"
            model_confidence = pred.get("model_confidence")
            score_source = "ml_model"
            info_message = "Dự báo bằng model ML temporal delinquency."
        except Exception as exc:
            print(f"[WARN] Delinquency batch predict failed for {resolved_tax_code}: {exc}")
            baseline = _build_baseline_prediction(db, resolved_tax_code, company_name)
            prob_30d, prob_60d, prob_90d, monotonic_adjusted = _normalize_horizon_probs(
                baseline.get("prob_30d"),
                baseline.get("prob_60d"),
                baseline.get("prob_90d"),
            )
            cluster = baseline.get("cluster") or _classify_cluster(prob_30d, prob_60d, prob_90d)
            top_reasons = baseline.get("top_reasons") or []
            model_version = baseline.get("model_version") or "baseline-statistical-v1"
            model_confidence = None
            score_source = baseline.get("score_source") or "inference_error"
            info_message = "ML không khả dụng, dùng baseline để không gián đoạn dịch vụ."
    else:
        baseline = _build_baseline_prediction(db, resolved_tax_code, company_name)
        prob_30d, prob_60d, prob_90d, monotonic_adjusted = _normalize_horizon_probs(
            baseline.get("prob_30d"),
            baseline.get("prob_60d"),
            baseline.get("prob_90d"),
        )
        cluster = baseline.get("cluster") or _classify_cluster(prob_30d, prob_60d, prob_90d)
        top_reasons = baseline.get("top_reasons") or []
        model_version = baseline.get("model_version") or "baseline-statistical-v1"
        model_confidence = None
        score_source = baseline.get("score_source") or "statistical_baseline"
        info_message = "Model ML chưa sẵn sàng, dùng baseline thống kê."

    if refresh_existing:
        db.query(models.DelinquencyPrediction).filter(
            models.DelinquencyPrediction.tax_code == resolved_tax_code,
            models.DelinquencyPrediction.prediction_date == date.today(),
        ).delete(synchronize_session=False)

    had_existing = (
        db.query(models.DelinquencyPrediction.id)
        .filter(models.DelinquencyPrediction.tax_code == resolved_tax_code)
        .first()
        is not None
    )

    row = models.DelinquencyPrediction(
        tax_code=resolved_tax_code,
        prediction_date=date.today(),
        prob_30d=prob_30d,
        prob_60d=prob_60d,
        prob_90d=prob_90d,
        risk_cluster=cluster,
        top_reasons=top_reasons,
        model_version=model_version,
        model_confidence=model_confidence,
    )
    db.add(row)
    db.commit()

    return {
        "tax_code": normalized_tax_code,
        "status": "updated" if had_existing else "created",
        "score_source": score_source,
        "prob_90d": prob_90d,
        "model_version": model_version,
        "message": f"{info_message}{' Đã hiệu chỉnh đơn điệu 30/60/90.' if monotonic_adjusted else ''}",
    }


@router.post("/delinquency/predict-batch", response_model=schemas.DelinquencyBatchPredictResponse)
def predict_delinquency_batch(
    payload: schemas.DelinquencyBatchPredictRequest,
    db: Session = Depends(get_db),
):
    """
    Batch predict delinquency and cache results to reduce baseline usage when new data arrives.
    """
    candidates = _build_batch_candidate_tax_codes(
        db,
        requested_tax_codes=payload.tax_codes,
        limit=payload.limit,
        refresh_existing=payload.refresh_existing,
    )

    if not candidates:
        return {
            "total_candidates": 0,
            "processed": 0,
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "items": [],
        }

    pipeline = _get_delinquency_pipeline()

    created = 0
    updated = 0
    skipped = 0
    failed = 0
    items = []

    for tax_code in candidates:
        try:
            result = _predict_and_persist_tax_code(
                db,
                tax_code=tax_code,
                pipeline=pipeline,
                refresh_existing=payload.refresh_existing,
            )
            items.append(result)
            if result["status"] == "created":
                created += 1
            elif result["status"] == "updated":
                updated += 1
            elif result["status"] == "skipped":
                skipped += 1
        except Exception as exc:
            db.rollback()
            failed += 1
            items.append({
                "tax_code": tax_code,
                "status": "failed",
                "score_source": "inference_error",
                "prob_90d": None,
                "model_version": None,
                "message": f"Batch predict lỗi: {exc}",
            })

    processed = len(candidates)
    return {
        "total_candidates": processed,
        "processed": processed,
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "items": items,
    }


@router.get("/delinquency/{tax_code}/payment-timeline")
def get_payment_timeline(
    tax_code: str,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """
    Return detailed payment timeline data for a specific company.
    Used by Timeline / Sparkline / Donut charts on the detail page.
    """
    company = _resolve_company_by_tax_code(db, tax_code)
    if not company:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy doanh nghiệp MST {tax_code}.")

    resolved_code = company.tax_code
    cutoff = date.today() - timedelta(days=730)

    payments = (
        db.query(models.TaxPayment)
        .filter(
            models.TaxPayment.tax_code == resolved_code,
            models.TaxPayment.due_date >= cutoff,
        )
        .order_by(models.TaxPayment.due_date.asc())
        .limit(limit)
        .all()
    )

    timeline = []
    amounts_over_time = []
    status_counts = {"on_time": 0, "late": 0, "unpaid": 0, "partial": 0}
    monthly_aggregation = {}

    for p in payments:
        due = p.due_date
        actual = p.actual_payment_date
        amount_due = float(p.amount_due or 0)
        amount_paid = float(p.amount_paid or 0)
        penalty = float(p.penalty_amount or 0)

        if actual is None and due < date.today():
            status = "unpaid"
            days_late = (date.today() - due).days
        elif actual is not None and actual > due:
            status = "late"
            days_late = (actual - due).days
        elif actual is not None and amount_paid < amount_due * 0.95:
            status = "partial"
            days_late = max(0, (actual - due).days) if actual > due else 0
        else:
            status = "on_time"
            days_late = 0

        status_counts[status] = status_counts.get(status, 0) + 1

        timeline.append({
            "due_date": str(due) if due else None,
            "actual_payment_date": str(actual) if actual else None,
            "amount_due": amount_due,
            "amount_paid": amount_paid,
            "penalty_amount": penalty,
            "tax_period": p.tax_period,
            "status": status,
            "days_late": days_late,
        })

        amounts_over_time.append({
            "date": str(due),
            "amount_due": amount_due,
            "amount_paid": amount_paid,
        })

        if due:
            month_key = due.strftime("%Y-%m")
            if month_key not in monthly_aggregation:
                monthly_aggregation[month_key] = {"due": 0, "paid": 0, "penalty": 0, "count": 0}
            monthly_aggregation[month_key]["due"] += amount_due
            monthly_aggregation[month_key]["paid"] += amount_paid
            monthly_aggregation[month_key]["penalty"] += penalty
            monthly_aggregation[month_key]["count"] += 1

    monthly_series = [
        {"month": k, "total_due": round(v["due"], 0), "total_paid": round(v["paid"], 0),
         "total_penalty": round(v["penalty"], 0), "payment_count": v["count"]}
        for k, v in sorted(monthly_aggregation.items())
    ]

    return {
        "tax_code": _normalize_tax_code_for_response(resolved_code) or resolved_code,
        "company_name": company.name,
        "total_records": len(timeline),
        "status_counts": status_counts,
        "timeline": timeline,
        "amounts_over_time": amounts_over_time,
        "monthly_series": monthly_series,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

