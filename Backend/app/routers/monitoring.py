from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
import os
import numpy as np
from sqlalchemy.orm import Session

from ..observability import get_structured_logger, log_event
from .. import schemas, auth, models
from ..database import get_db

router = APIRouter(prefix="/api/monitoring", tags=["MLOps"])

logger = get_structured_logger("taxinspector.monitoring")

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

AUDIT_QUALITY_REPORT_FILE = "audit_value_quality_report.json"
VAT_QUALITY_REPORT_FILE = "vat_refund_quality_report.json"
SPECIALIZED_PILOT_REPORT_FILE = "specialized_pilot_report.json"
SPECIALIZED_GO_NO_GO_REPORT_FILE = "specialized_go_no_go_report.json"

DRIFT_FEATURES = (
    "company_age_days",
    "invoice_amount_log",
    "is_reciprocal_ratio",
)

FRAUD_DRIFT_FEATURES = (
    "f1_divergence",
    "f2_ratio_limit",
    "f3_vat_structure",
    "f4_peer_comparison",
    "anomaly_score",
)

# Flagship: Delinquency model drift features (Program A)
DELINQUENCY_DRIFT_FEATURES = (
    "late_ratio_1yr",
    "avg_days_overdue",
    "unpaid_count",
    "penalty_trend",
    "payment_cv",
    "prob_30d",
    "prob_60d",
    "prob_90d",
)

OSINT_DRIFT_FEATURES = (
    "n_dom_subs",
    "n_rel_types",
    "max_own_pct",
    "inv_in_bn",
    "inv_out_bn",
    "max_dom_risk",
    "avg_dom_risk",
    "juris_risk",
)

SIMULATION_DRIFT_FEATURES = (
    "vat",
    "cit",
    "audit",
    "penalty",
    "interest",
    "growth",
    "base_rate",
    "avg_margin",
    "company_count",
)

VALID_POLICY_COMPARATORS = {">=", ">", "<=", "<", "=="}
ADMIN_POLICY_ROLES = {"admin"}
KPI_BLOCKING_STATUSES = {"fail", "insufficient_data", "no_metric", "cooldown_active"}
KPI_FAILLIKE_STATUSES = {"fail", "insufficient_data", "no_metric", "cooldown_active"}
TERMINAL_OUTCOME_STATUSES = ("recovered", "partial_recovered", "unrecoverable", "dismissed")
SYNTHETIC_MODEL_VERSION_TOKENS = ("seed-", "synthetic", "mock", "demo")
SYNTHETIC_LABEL_MARKERS = (
    "[AUTO-SEED-LARGE]",
    "Bootstrap label from assessment",
    "synthetic training label",
)
REAL_LABEL_ORIGINS = ("manual_inspector", "field_verified", "imported_casework")
SYNTHETIC_LABEL_ORIGINS = ("bootstrap_generated", "auto_seed")
DATA_REALITY_AUDIT_TABLE = "data_reality_audit_logs"


class FeedbackData(BaseModel):
    tax_code: Optional[str] = None
    invoice_number: Optional[str] = None
    path_id: Optional[str] = None
    is_fraud: bool
    expert_notes: Optional[str] = None


class MetricLog(BaseModel):
    metric_name: str
    value: float
    labels: dict = {}


class KPIPolicyInput(BaseModel):
    track_name: str
    metric_name: str
    comparator: Literal[">=", ">", "<=", "<", "=="] = ">="
    threshold: float
    min_sample: int = Field(default=50, ge=1)
    window_days: int = Field(default=28, ge=7, le=365)
    cooldown_days: int = Field(default=14, ge=0, le=365)
    enabled: bool = True
    rationale: Optional[str] = None


class KPIPolicyUpsertRequest(BaseModel):
    policies: list[KPIPolicyInput]
    replace_existing: bool = False


def _normalize_role(value: Any) -> str:
    return str(value or "").strip().lower()


def _require_policy_admin(current_user: models.User) -> None:
    role = _normalize_role(getattr(current_user, "role", ""))
    if role not in ADMIN_POLICY_ROLES:
        raise HTTPException(
            status_code=403,
            detail="Chỉ tài khoản admin mới được cập nhật KPI policy hoặc capture snapshots thủ công.",
        )


def _safe_log_monitoring_audit(
    db: Any,
    *,
    action: str,
    request: Optional[Request],
    current_user: Optional[models.User],
    detail: Optional[str] = None,
) -> None:
    """Best-effort immutable audit logging for monitoring governance actions."""
    if db is None or not hasattr(db, "add") or not hasattr(db, "commit"):
        return
    try:
        auth.log_audit(
            db,
            action=action,
            request=request,
            user_id=getattr(current_user, "id", None) if current_user is not None else None,
            badge_id=getattr(current_user, "badge_id", None) if current_user is not None else None,
            detail=detail,
        )
    except Exception as exc:
        logger.warning("Monitoring audit logging skipped: %s", str(exc))


def _parse_iso_timestamp(raw: str):
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        ts = datetime.fromisoformat(raw)
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    except ValueError:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _wasserstein_like(a: list[float], b: list[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return 0.0
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(arr_a, arr_b))
    except Exception:
        q = np.linspace(0.0, 1.0, 51)
        qa = np.quantile(arr_a, q)
        qb = np.quantile(arr_b, q)
        return float(np.mean(np.abs(qa - qb)))


def _load_baseline_stats() -> dict:
    baseline_path = MODEL_DIR / "drift_baseline.json"
    if not baseline_path.exists():
        return {}
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_fraud_baseline_stats() -> dict:
    baseline_path = MODEL_DIR / "fraud_drift_baseline.json"
    if not baseline_path.exists():
        return {}
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_osint_baseline_stats() -> dict:
    baseline_path = MODEL_DIR / "osint_drift_baseline.json"
    if not baseline_path.exists():
        return {}
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_simulation_baseline_stats() -> dict:
    baseline_path = MODEL_DIR / "simulation_drift_baseline.json"
    if not baseline_path.exists():
        return {}
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _artifact_report(path: Path) -> dict[str, Any]:
    payload = _safe_read_json(path)
    return {
        "exists": bool(path.exists()),
        "updated_at": _file_updated_at(path),
        "payload": payload,
    }


def _track_payload(pilot_payload: dict[str, Any], track_name: str) -> dict[str, Any]:
    tracks = pilot_payload.get("tracks")
    if not isinstance(tracks, dict):
        return {}
    value = tracks.get(track_name)
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _int_or_none(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) else None


def _file_updated_at(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z"
    except Exception:
        return None


def _criterion_actual(criteria: dict, key: str) -> Optional[float]:
    if not isinstance(criteria, dict):
        return None
    item = criteria.get(key)
    if not isinstance(item, dict):
        return None
    value = item.get("actual")
    return float(value) if isinstance(value, (int, float)) else None


def _metric_snapshot(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {
            "auc_roc": None,
            "pr_auc": None,
            "brier": None,
            "ece": None,
        }

    def _as_float(v):
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "auc_roc": _as_float(payload.get("auc_roc")),
        "pr_auc": _as_float(payload.get("pr_auc")),
        "brier": _as_float(payload.get("brier")),
        "ece": _as_float(payload.get("ece")),
    }


def _collect_feature_windows(
    metrics_path: Path,
    lookback_days: int,
    baseline_days: int,
    feature_names: tuple[str, ...],
    metric_names: Optional[set[str]] = None,
) -> tuple[dict, dict, int, int]:
    now = datetime.utcnow()
    recent_cutoff = now - timedelta(days=lookback_days)
    baseline_cutoff = now - timedelta(days=lookback_days + baseline_days)

    recent = {name: [] for name in feature_names}
    baseline = {name: [] for name in feature_names}
    recent_rows = 0
    baseline_rows = 0

    for entry in _read_jsonl(metrics_path):
        if metric_names and entry.get("metric") not in metric_names:
            continue

        labels = entry.get("labels")
        if not isinstance(labels, dict):
            continue

        ts = _parse_iso_timestamp(entry.get("timestamp", ""))
        if ts is None:
            continue

        if ts >= recent_cutoff:
            bucket = recent
            recent_rows += 1
        elif ts >= baseline_cutoff:
            bucket = baseline
            baseline_rows += 1
        else:
            continue

        for name in feature_names:
            value = labels.get(name)
            if isinstance(value, (int, float)):
                bucket[name].append(float(value))

    return recent, baseline, recent_rows, baseline_rows


def _build_drift_report(lookback_days: int, baseline_days: int, min_samples: int) -> dict:
    metrics_file = LOG_DIR / "ml_metrics.jsonl"
    recent, historical, recent_rows, baseline_rows = _collect_feature_windows(
        metrics_path=metrics_file,
        lookback_days=lookback_days,
        baseline_days=baseline_days,
        feature_names=DRIFT_FEATURES,
        metric_names={"inference_features", "graph_inference_features"},
    )
    baseline_stats = _load_baseline_stats()

    features_report = {}
    drifted_features = []
    sufficient_features = 0

    for name in DRIFT_FEATURES:
        recent_values = recent.get(name, [])

        model_baseline = baseline_stats.get("features", {}).get(name, {})
        if model_baseline:
            baseline_mean = float(model_baseline.get("mean", 0.0))
            baseline_std = float(model_baseline.get("std", 0.0))
            low = float(model_baseline.get("q0", baseline_mean))
            high = float(model_baseline.get("q100", baseline_mean))
            baseline_values = list(np.linspace(low, high, 101))
        else:
            baseline_values = historical.get(name, [])
            baseline_arr = np.asarray(baseline_values, dtype=float) if baseline_values else np.asarray([])
            baseline_mean = float(np.mean(baseline_arr)) if len(baseline_arr) else 0.0
            baseline_std = float(np.std(baseline_arr)) if len(baseline_arr) else 0.0

        if len(recent_values) < min_samples or len(baseline_values) < min_samples:
            features_report[name] = {
                "status": "insufficient_data",
                "recent_samples": len(recent_values),
                "baseline_samples": len(baseline_values),
            }
            continue

        sufficient_features += 1
        raw_distance = _wasserstein_like(recent_values, baseline_values)
        iqr = float(np.percentile(baseline_values, 75) - np.percentile(baseline_values, 25))
        scale = iqr if iqr > 1e-6 else max(baseline_std, 1.0)
        normalized_distance = raw_distance / scale

        if normalized_distance >= 1.0:
            status = "drift"
            drifted_features.append(name)
        elif normalized_distance >= 0.45:
            status = "warning"
        else:
            status = "stable"

        features_report[name] = {
            "wasserstein_distance": round(raw_distance, 6),
            "normalized_distance": round(normalized_distance, 6),
            "status": status,
            "recent_samples": len(recent_values),
            "baseline_samples": len(baseline_values),
            "recent_mean": round(float(np.mean(recent_values)), 6),
            "baseline_mean": round(baseline_mean, 6),
        }

    if sufficient_features == 0:
        recommendation = "Insufficient feature telemetry for drift decision. Continue collecting inference feature metrics."
        drift_detected = False
        severity = "insufficient_data"
    elif drifted_features:
        recommendation = "Potential drift detected. Validate model performance and schedule retraining if drift persists."
        drift_detected = True
        severity = "high" if len(drifted_features) >= 2 else "medium"
    else:
        recommendation = "No retraining required at this time."
        drift_detected = False
        severity = "low"

    return {
        "drift_detected": drift_detected,
        "drift_severity": severity,
        "drifted_features": drifted_features,
        "features": features_report,
        "recommendation": recommendation,
        "lookback_days": lookback_days,
        "baseline_days": baseline_days,
        "min_samples": min_samples,
        "sample_summary": {
            "recent_rows": recent_rows,
            "baseline_rows": baseline_rows,
            "baseline_source": "model_baseline_or_historical_logs",
        },
    }


def _build_fraud_drift_report(lookback_days: int, baseline_days: int, min_samples: int) -> dict:
    metrics_file = LOG_DIR / "ml_metrics.jsonl"
    recent, historical, recent_rows, baseline_rows = _collect_feature_windows(
        metrics_path=metrics_file,
        lookback_days=lookback_days,
        baseline_days=baseline_days,
        feature_names=FRAUD_DRIFT_FEATURES,
        metric_names={"fraud_inference_features"},
    )
    baseline_stats = _load_fraud_baseline_stats()

    features_report = {}
    drifted_features = []
    sufficient_features = 0

    for name in FRAUD_DRIFT_FEATURES:
        recent_values = recent.get(name, [])

        model_baseline = baseline_stats.get("features", {}).get(name, {}) if isinstance(baseline_stats, dict) else {}
        if model_baseline:
            baseline_mean = float(model_baseline.get("mean", 0.0))
            baseline_std = float(model_baseline.get("std", 0.0))
            low = float(model_baseline.get("q0", baseline_mean))
            high = float(model_baseline.get("q100", baseline_mean))
            baseline_values = list(np.linspace(low, high, 101))
            baseline_source = "model_baseline"
        else:
            baseline_values = historical.get(name, [])
            baseline_arr = np.asarray(baseline_values, dtype=float) if baseline_values else np.asarray([])
            baseline_mean = float(np.mean(baseline_arr)) if len(baseline_arr) else 0.0
            baseline_std = float(np.std(baseline_arr)) if len(baseline_arr) else 0.0
            baseline_source = "historical_window"

        if len(recent_values) < min_samples or len(baseline_values) < min_samples:
            features_report[name] = {
                "status": "insufficient_data",
                "recent_samples": len(recent_values),
                "baseline_samples": len(baseline_values),
            }
            continue

        sufficient_features += 1
        raw_distance = _wasserstein_like(recent_values, baseline_values)
        iqr = float(np.percentile(baseline_values, 75) - np.percentile(baseline_values, 25))
        scale = iqr if iqr > 1e-6 else max(baseline_std, 1.0)
        normalized_distance = raw_distance / scale

        if normalized_distance >= 1.0:
            status = "drift"
            drifted_features.append(name)
        elif normalized_distance >= 0.45:
            status = "warning"
        else:
            status = "stable"

        features_report[name] = {
            "wasserstein_distance": round(raw_distance, 6),
            "normalized_distance": round(normalized_distance, 6),
            "status": status,
            "recent_samples": len(recent_values),
            "baseline_samples": len(baseline_values),
            "recent_mean": round(float(np.mean(recent_values)), 6),
            "baseline_mean": round(baseline_mean, 6),
            "baseline_source": baseline_source,
        }

    if sufficient_features == 0:
        recommendation = "Insufficient fraud telemetry for drift decision. Continue collecting single-query inference signals."
        drift_detected = False
        severity = "insufficient_data"
    elif drifted_features:
        recommendation = "Fraud feature drift detected. Review calibration and schedule retraining if drift persists."
        drift_detected = True
        severity = "high" if len(drifted_features) >= 2 else "medium"
    else:
        recommendation = "Fraud feature distribution is stable. No immediate retraining required."
        drift_detected = False
        severity = "low"

    return {
        "drift_detected": drift_detected,
        "drift_severity": severity,
        "drifted_features": drifted_features,
        "features": features_report,
        "recommendation": recommendation,
        "lookback_days": lookback_days,
        "baseline_days": baseline_days,
        "min_samples": min_samples,
        "sample_summary": {
            "recent_rows": recent_rows,
            "baseline_rows": baseline_rows,
            "baseline_source": "fraud_drift_baseline_or_historical_logs",
        },
    }


def _build_fraud_quality_summary(include_criteria: bool = False) -> dict:
    quality_path = MODEL_DIR / "fraud_quality_report.json"
    calibrator_path = MODEL_DIR / "fraud_calibrator.joblib"
    xgb_path = MODEL_DIR / "xgboost_model.joblib"
    iso_path = MODEL_DIR / "isolation_forest.joblib"

    quality_payload = _safe_read_json(quality_path)
    drift_payload = _build_fraud_drift_report(lookback_days=7, baseline_days=30, min_samples=20)

    quality_available = bool(quality_payload)
    calibrator_available = calibrator_path.exists()

    gates = quality_payload.get("acceptance_gates", {}) if isinstance(quality_payload, dict) else {}
    criteria = gates.get("criteria", {}) if isinstance(gates, dict) else {}
    overall_pass = bool(gates.get("overall_pass")) if quality_available else None

    performance_keys = {"auc_roc_min", "pr_auc_min", "brier_max"}
    calibration_keys = {"ece_max", "brier_not_worse_than_raw"}
    perf_checks = [criteria[k].get("pass") for k in performance_keys if isinstance(criteria.get(k), dict)]
    calibration_checks = [criteria[k].get("pass") for k in calibration_keys if isinstance(criteria.get(k), dict)]

    soft_gate_keys = gates.get("soft_gate_criteria", []) if isinstance(gates, dict) else []
    if not isinstance(soft_gate_keys, list) or not soft_gate_keys:
        soft_gate_keys = [
            key for key, value in criteria.items() if isinstance(value, dict) and bool(value.get("soft_gate"))
        ]

    soft_gate_warnings = []
    soft_checks = []
    for key in soft_gate_keys:
        item = criteria.get(key)
        if not isinstance(item, dict):
            continue
        pass_value = item.get("pass")
        if pass_value is False:
            soft_gate_warnings.append(str(key))
        if pass_value is not None:
            soft_checks.append(bool(pass_value))

    soft_gate_pass = bool(all(soft_checks)) if soft_checks else None

    performance_pass = bool(all(perf_checks)) if perf_checks else overall_pass
    calibration_pass = bool(all(calibration_checks)) if calibration_checks else (
        bool(calibrator_available) if quality_available else None
    )
    all_pass = bool(performance_pass and (calibration_pass is not False))

    drift_severity = str(drift_payload.get("drift_severity", "insufficient_data"))
    if not quality_available and not calibrator_available:
        status = "unknown"
    elif performance_pass is False or calibration_pass is False:
        status = "degraded"
    elif soft_gate_warnings:
        status = "warning"
    elif drift_severity in {"high", "medium"}:
        status = "warning"
    elif performance_pass is True:
        status = "healthy"
    else:
        status = "warning"

    performance_payload = quality_payload.get("performance", {}) if isinstance(quality_payload, dict) else {}
    raw_metrics = _metric_snapshot(performance_payload.get("raw", {}))
    calibrated_metrics = _metric_snapshot(performance_payload.get("calibrated", {}))
    calibration_payload = quality_payload.get("calibration", {}) if isinstance(quality_payload, dict) else {}
    model_info = quality_payload.get("model_info", {}) if isinstance(quality_payload, dict) else {}
    dataset_info = quality_payload.get("dataset", {}) if isinstance(quality_payload, dict) else {}

    return {
        "status": status,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "model_version": model_info.get("model_version") if isinstance(model_info, dict) else None,
            "updated_at": _file_updated_at(xgb_path) or _file_updated_at(iso_path),
            "calibrator_available": calibrator_available,
            "quality_report_available": quality_available,
        },
        "gate_summary": {
            "performance_pass": performance_pass,
            "calibration_pass": calibration_pass,
            "all_pass": all_pass,
            "soft_gate_pass": soft_gate_pass,
            "soft_gate_warnings": soft_gate_warnings,
        },
        "performance": {
            "available": quality_available,
            "sample_size": int(dataset_info.get("test_size", 0)) if isinstance(dataset_info, dict) else 0,
            "metrics": calibrated_metrics,
            "criteria": criteria if include_criteria and isinstance(criteria, dict) else {},
        },
        "calibration": {
            "available": bool(calibration_payload.get("available", calibrator_available)),
            "method": calibration_payload.get(
                "method",
                model_info.get("calibration_method") if isinstance(model_info, dict) else None,
            ),
            "trained_at": quality_payload.get("generated_at") if isinstance(quality_payload, dict) else None,
            "brier_improvement": calibration_payload.get("brier_improvement") if isinstance(calibration_payload, dict) else None,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
            "criteria": criteria if include_criteria and isinstance(criteria, dict) else {},
        },
        "drift": {
            "detected": bool(drift_payload.get("drift_detected", False)),
            "severity": drift_severity,
            "drifted_features": drift_payload.get("drifted_features", []),
            "recommendation": drift_payload.get("recommendation", ""),
        },
    }


def _build_osint_quality_summary(include_criteria: bool = False) -> dict[str, Any]:
    quality_path = MODEL_DIR / "osint_quality_report.json"
    config_path = MODEL_DIR / "osint_config.json"
    model_path = MODEL_DIR / "osint_risk_model.joblib"

    quality_payload = _safe_read_json(quality_path)
    config_payload = _safe_read_json(config_path)
    drift_payload = _build_osint_drift_report(min_samples=10)

    quality_available = bool(quality_payload)
    model_available = model_path.exists()

    metrics = quality_payload.get("metrics", {}) if isinstance(quality_payload, dict) else {}
    gates = quality_payload.get("acceptance_gates", {}) if isinstance(quality_payload, dict) else {}
    criteria = gates.get("criteria", {}) if isinstance(gates, dict) else {}
    dataset = quality_payload.get("dataset", {}) if isinstance(quality_payload, dict) else {}

    auc = _float_or_none(metrics.get("auc"))
    pr_auc = _float_or_none(metrics.get("pr_auc"))

    if isinstance(gates, dict) and "overall_pass" in gates:
        performance_pass = bool(gates.get("overall_pass"))
    elif auc is not None and pr_auc is not None:
        performance_pass = bool(auc >= 0.60 and pr_auc >= 0.35)
    else:
        performance_pass = None

    drift_severity = str(drift_payload.get("drift_severity", "insufficient_data"))
    if not quality_available and not model_available:
        status = "unknown"
    elif performance_pass is False:
        status = "degraded"
    elif drift_severity in {"high", "medium"}:
        status = "warning"
    elif performance_pass is True:
        status = "healthy"
    else:
        status = "warning"

    return {
        "status": status,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "model_version": quality_payload.get("model_version") or config_payload.get("model_version"),
            "model_type": config_payload.get("model_type"),
            "updated_at": _file_updated_at(model_path) or _file_updated_at(config_path),
            "model_available": model_available,
            "quality_report_available": quality_available,
            "drift_baseline_available": bool(_load_osint_baseline_stats()),
        },
        "gate_summary": {
            "performance_pass": performance_pass,
            "all_pass": bool(performance_pass is True),
        },
        "performance": {
            "available": quality_available,
            "sample_size": int(dataset.get("total_samples", 0)) if isinstance(dataset, dict) else 0,
            "metrics": {
                "auc": auc,
                "pr_auc": pr_auc,
            },
            "criteria": criteria if include_criteria and isinstance(criteria, dict) else {},
        },
        "drift": {
            "detected": bool(drift_payload.get("drift_detected", False)),
            "severity": drift_severity,
            "drifted_features": drift_payload.get("drifted_features", []),
            "recommendation": drift_payload.get("recommendation", ""),
        },
    }


def _build_simulation_quality_summary(include_criteria: bool = False) -> dict[str, Any]:
    quality_path = MODEL_DIR / "simulation_quality_report.json"
    config_path = MODEL_DIR / "simulation_config.json"
    model_path = MODEL_DIR / "simulation_lgbm.joblib"

    quality_payload = _safe_read_json(quality_path)
    config_payload = _safe_read_json(config_path)
    drift_payload = _build_simulation_drift_report(min_samples=5)

    quality_available = bool(quality_payload)
    model_available = model_path.exists()

    metrics = quality_payload.get("metrics", {}) if isinstance(quality_payload, dict) else {}
    gates = quality_payload.get("acceptance_gates", {}) if isinstance(quality_payload, dict) else {}
    criteria = gates.get("criteria", {}) if isinstance(gates, dict) else {}
    dataset = quality_payload.get("dataset", {}) if isinstance(quality_payload, dict) else {}

    r2 = _float_or_none(metrics.get("r2"))
    rmse = _float_or_none(metrics.get("rmse"))
    mae = _float_or_none(metrics.get("mae"))

    if isinstance(gates, dict) and "overall_pass" in gates:
        performance_pass = bool(gates.get("overall_pass"))
    elif r2 is not None and rmse is not None:
        performance_pass = bool(r2 >= 0.90 and rmse <= 0.08)
    else:
        performance_pass = None

    drift_severity = str(drift_payload.get("drift_severity", "insufficient_data"))
    if not quality_available and not model_available:
        status = "unknown"
    elif performance_pass is False:
        status = "degraded"
    elif drift_severity in {"high", "medium"}:
        status = "warning"
    elif performance_pass is True:
        status = "healthy"
    else:
        status = "warning"

    return {
        "status": status,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "model_version": quality_payload.get("model_version") or config_payload.get("model_version"),
            "model_type": config_payload.get("model_type"),
            "updated_at": _file_updated_at(model_path) or _file_updated_at(config_path),
            "model_available": model_available,
            "quality_report_available": quality_available,
            "drift_baseline_available": bool(_load_simulation_baseline_stats()),
        },
        "gate_summary": {
            "performance_pass": performance_pass,
            "all_pass": bool(performance_pass is True),
        },
        "performance": {
            "available": quality_available,
            "sample_size": int(dataset.get("total_samples", 0)) if isinstance(dataset, dict) else 0,
            "metrics": {
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
            },
            "criteria": criteria if include_criteria and isinstance(criteria, dict) else {},
        },
        "drift": {
            "detected": bool(drift_payload.get("drift_detected", False)),
            "severity": drift_severity,
            "drifted_features": drift_payload.get("drifted_features", []),
            "recommendation": drift_payload.get("recommendation", ""),
        },
    }


def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            dbname=os.getenv("DB_NAME", "TaxInspector"),
        )
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None


def _resolve_db_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "TaxInspector")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def _empty_feature_snapshot(feature_names: tuple[str, ...]) -> dict[str, list[float]]:
    return {name: [] for name in feature_names}


def _fetch_osint_feature_snapshot() -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Build current OSINT feature snapshot from database tables."""
    snapshot = _empty_feature_snapshot(OSINT_DRIFT_FEATURES)
    metadata: dict[str, Any] = {
        "offshore_entities": 0,
        "ownership_links": 0,
        "invoice_rows": 0,
    }

    conn = None
    try:
        conn = psycopg2.connect(_resolve_db_url())
        with conn.cursor() as cur:
            cur.execute("SELECT tax_code, province FROM companies WHERE industry = 'Offshore Entity'")
            offshore_rows = cur.fetchall()
            metadata["offshore_entities"] = len(offshore_rows)

            if not offshore_rows:
                return snapshot, metadata

            cur.execute(
                """
                SELECT
                    ol.parent_tax_code AS offshore_tax_code,
                    ol.child_tax_code AS domestic_tax_code,
                    ol.relationship_type,
                    ol.ownership_percent,
                    COALESCE(c.risk_score, 0) AS risk_score
                FROM ownership_links ol
                JOIN companies c ON c.tax_code = ol.child_tax_code
                WHERE ol.parent_tax_code IN (
                    SELECT tax_code FROM companies WHERE industry = 'Offshore Entity'
                )
                """
            )
            link_rows = cur.fetchall()
            metadata["ownership_links"] = len(link_rows)

            cur.execute("SELECT seller_tax_code, buyer_tax_code, amount FROM invoices")
            invoice_rows = cur.fetchall()
            metadata["invoice_rows"] = len(invoice_rows)

        invoice_in: dict[str, float] = {}
        invoice_out: dict[str, float] = {}
        for seller_tax_code, buyer_tax_code, amount in invoice_rows:
            amount_value = float(amount or 0.0)
            seller = str(seller_tax_code)
            buyer = str(buyer_tax_code)
            invoice_out[seller] = invoice_out.get(seller, 0.0) + amount_value
            invoice_in[buyer] = invoice_in.get(buyer, 0.0) + amount_value

        links_by_parent: dict[str, list[tuple[Any, ...]]] = {}
        for row in link_rows:
            parent_tax_code = str(row[0])
            links_by_parent.setdefault(parent_tax_code, []).append(row)

        jurisdiction_risk = {
            "Cayman Islands": 5.0,
            "British Virgin Islands (BVI)": 5.0,
            "Panama": 5.0,
            "Seychelles": 4.0,
            "Bahamas": 4.0,
            "Cyprus": 3.0,
            "Hong Kong": 2.0,
            "Singapore": 2.0,
        }

        for offshore_tax_code, offshore_country in offshore_rows:
            tax_code = str(offshore_tax_code)
            country = str(offshore_country or "")
            links = links_by_parent.get(tax_code, [])

            n_dom_subs = float(len(links))
            n_rel_types = float(len({str(link[2] or "") for link in links}))
            max_own = float(max((float(link[3] or 0.0) for link in links), default=0.0))

            total_inv_in = 0.0
            total_inv_out = 0.0
            domestic_risks: list[float] = []

            for link in links:
                child_tax_code = str(link[1])
                total_inv_in += float(invoice_in.get(child_tax_code, 0.0))
                total_inv_out += float(invoice_out.get(child_tax_code, 0.0))
                domestic_risks.append(float(link[4] or 0.0))

            max_dom_risk = float(max(domestic_risks)) if domestic_risks else 0.0
            avg_dom_risk = float(sum(domestic_risks) / len(domestic_risks)) if domestic_risks else 0.0
            juris_risk = float(jurisdiction_risk.get(country, 1.0))

            snapshot["n_dom_subs"].append(n_dom_subs)
            snapshot["n_rel_types"].append(n_rel_types)
            snapshot["max_own_pct"].append(max_own)
            snapshot["inv_in_bn"].append(total_inv_in / 1e9)
            snapshot["inv_out_bn"].append(total_inv_out / 1e9)
            snapshot["max_dom_risk"].append(max_dom_risk)
            snapshot["avg_dom_risk"].append(avg_dom_risk)
            snapshot["juris_risk"].append(juris_risk)

    except Exception as exc:
        metadata["error"] = str(exc)
        log_event(logger, "warning", "monitoring_osint_snapshot_failed", error=str(exc))
    finally:
        if conn is not None:
            conn.close()

    return snapshot, metadata


def _fetch_simulation_feature_snapshot() -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Build current simulation feature snapshot from industry baseline aggregates."""
    snapshot = _empty_feature_snapshot(SIMULATION_DRIFT_FEATURES)
    metadata: dict[str, Any] = {"industries": 0}

    conn = None
    try:
        conn = psycopg2.connect(_resolve_db_url())
        with conn.cursor() as cur:
            where_sql = "c.industry IS NOT NULL AND c.industry != '' AND c.industry != 'Offshore Entity'"

            cur.execute(
                f"""
                SELECT
                    c.industry,
                    COUNT(DISTINCT c.tax_code) AS company_count,
                    COALESCE(AVG(tr.revenue), 0) AS avg_revenue
                FROM companies c
                LEFT JOIN tax_returns tr ON tr.tax_code = c.tax_code
                WHERE {where_sql}
                GROUP BY c.industry
                """
            )
            industry_rows = cur.fetchall()

            cur.execute(
                f"""
                SELECT
                    c.industry,
                    COUNT(DISTINCT dp.tax_code) AS delinquent_count,
                    COUNT(DISTINCT c.tax_code) AS total_count
                FROM companies c
                LEFT JOIN delinquency_predictions dp
                    ON dp.tax_code = c.tax_code AND dp.prob_90d >= 0.5
                WHERE {where_sql}
                GROUP BY c.industry
                """
            )
            delinquency_rows = cur.fetchall()

            cur.execute(
                """
                SELECT c.industry, COUNT(DISTINCT tp.tax_code)
                FROM tax_payments tp
                JOIN companies c ON c.tax_code = tp.tax_code
                WHERE tp.status IN ('overdue', 'partial')
                GROUP BY c.industry
                """
            )
            overdue_rows = cur.fetchall()

        delinquency_map = {
            str(industry): {
                "delinq_count": int(delinq_count or 0),
                "total": int(total_count or 0),
            }
            for industry, delinq_count, total_count in delinquency_rows
        }
        overdue_map = {
            str(industry): int(count or 0)
            for industry, count in overdue_rows
        }

        industry_margins = {
            "Xây dựng": 0.06,
            "Bất động sản": 0.12,
            "Thương mại XNK": 0.04,
            "Sản xuất công nghiệp": 0.08,
            "Nông nghiệp": 0.05,
            "Vận tải & Logistics": 0.07,
            "Công nghệ thông tin": 0.15,
            "Dịch vụ tài chính": 0.18,
            "Y tế & Dược phẩm": 0.14,
            "Giáo dục & Đào tạo": 0.10,
            "Thực phẩm & Đồ uống": 0.09,
            "May mặc & Giầy da": 0.06,
            "Khoáng sản & Năng lượng": 0.11,
            "Du lịch & Khách sạn": 0.08,
            "Viễn thông": 0.13,
        }

        for industry, company_count, _avg_revenue in industry_rows:
            industry_name = str(industry)
            count_value = int(company_count or 0)

            delinquency_info = delinquency_map.get(industry_name, {"delinq_count": 0, "total": count_value})
            base_rate = float(delinquency_info["delinq_count"]) / max(1, int(delinquency_info["total"]))
            if base_rate == 0:
                base_rate = float(overdue_map.get(industry_name, 0)) / max(1, count_value)

            avg_margin = float(industry_margins.get(industry_name, 0.08))
            if base_rate > 0:
                base_rate = max(0.02, min(0.95, base_rate))
            else:
                base_rate = max(0.05, avg_margin * 1.5)

            snapshot["vat"].append(10.0)
            snapshot["cit"].append(20.0)
            snapshot["audit"].append(5.0)
            snapshot["penalty"].append(1.0)
            snapshot["interest"].append(6.0)
            snapshot["growth"].append(6.5)
            snapshot["base_rate"].append(float(base_rate))
            snapshot["avg_margin"].append(float(avg_margin))
            snapshot["company_count"].append(float(count_value))

        metadata["industries"] = len(industry_rows)

    except Exception as exc:
        metadata["error"] = str(exc)
        log_event(logger, "warning", "monitoring_simulation_snapshot_failed", error=str(exc))
    finally:
        if conn is not None:
            conn.close()

    return snapshot, metadata


def _build_baseline_feature_drift_report(
    *,
    feature_names: tuple[str, ...],
    baseline_payload: dict[str, Any],
    current_snapshot: dict[str, list[float]],
    min_samples: int,
    warning_threshold: float = 0.45,
    drift_threshold: float = 1.0,
) -> dict[str, Any]:
    features_report: dict[str, Any] = {}
    drifted_features: list[str] = []
    sufficient_features = 0

    baseline_features = baseline_payload.get("features", {}) if isinstance(baseline_payload, dict) else {}

    for feature_name in feature_names:
        recent_values = [float(v) for v in current_snapshot.get(feature_name, []) if isinstance(v, (int, float))]
        model_baseline = baseline_features.get(feature_name, {}) if isinstance(baseline_features, dict) else {}

        baseline_values: list[float] = []
        baseline_mean = 0.0
        baseline_std = 0.0
        baseline_source = "missing"

        if isinstance(model_baseline, dict) and model_baseline:
            baseline_mean = float(model_baseline.get("mean", 0.0))
            baseline_std = float(model_baseline.get("std", 0.0))

            q0 = _float_or_none(model_baseline.get("q0"))
            q100 = _float_or_none(model_baseline.get("q100"))

            if q0 is None or q100 is None:
                spread = max(1e-6, 2.0 * baseline_std)
                q0 = baseline_mean - spread
                q100 = baseline_mean + spread

            if q100 < q0:
                q0, q100 = q100, q0

            baseline_values = list(np.linspace(q0, q100, 101))
            baseline_source = "model_baseline"

        if len(recent_values) < min_samples or len(baseline_values) < min_samples:
            features_report[feature_name] = {
                "status": "insufficient_data",
                "recent_samples": len(recent_values),
                "baseline_samples": len(baseline_values),
                "baseline_source": baseline_source,
            }
            continue

        sufficient_features += 1
        raw_distance = _wasserstein_like(recent_values, baseline_values)
        iqr = float(np.percentile(baseline_values, 75) - np.percentile(baseline_values, 25))
        scale = iqr if iqr > 1e-6 else max(baseline_std, 1.0)
        normalized_distance = raw_distance / scale

        if normalized_distance >= drift_threshold:
            status = "drift"
            drifted_features.append(feature_name)
        elif normalized_distance >= warning_threshold:
            status = "warning"
        else:
            status = "stable"

        features_report[feature_name] = {
            "wasserstein_distance": round(raw_distance, 6),
            "normalized_distance": round(normalized_distance, 6),
            "status": status,
            "recent_samples": len(recent_values),
            "baseline_samples": len(baseline_values),
            "recent_mean": round(float(np.mean(recent_values)), 6),
            "baseline_mean": round(float(baseline_mean), 6),
            "baseline_source": baseline_source,
        }

    if sufficient_features == 0:
        drift_detected = False
        drift_severity = "insufficient_data"
        recommendation = "Insufficient feature telemetry for drift decision."
    elif drifted_features:
        drift_detected = True
        drift_severity = "high" if len(drifted_features) >= 2 else "medium"
        recommendation = "Feature drift detected. Validate model quality and schedule retraining if drift persists."
    else:
        drift_detected = False
        drift_severity = "low"
        recommendation = "No significant drift detected."

    current_rows = max((len(values) for values in current_snapshot.values()), default=0)

    return {
        "drift_detected": drift_detected,
        "drift_severity": drift_severity,
        "drifted_features": drifted_features,
        "features": features_report,
        "recommendation": recommendation,
        "min_samples": int(min_samples),
        "sample_summary": {
            "current_rows": int(current_rows),
            "baseline_source": "model_drift_baseline",
        },
    }


def _build_osint_drift_report(min_samples: int = 10) -> dict[str, Any]:
    baseline_payload = _load_osint_baseline_stats()
    current_snapshot, metadata = _fetch_osint_feature_snapshot()

    report = _build_baseline_feature_drift_report(
        feature_names=OSINT_DRIFT_FEATURES,
        baseline_payload=baseline_payload,
        current_snapshot=current_snapshot,
        min_samples=min_samples,
    )
    report["model_version"] = baseline_payload.get("model_version")
    report["baseline_created_at"] = baseline_payload.get("created_at")
    report["sample_summary"] = {
        **report.get("sample_summary", {}),
        **metadata,
    }
    return report


def _build_simulation_drift_report(min_samples: int = 5) -> dict[str, Any]:
    baseline_payload = _load_simulation_baseline_stats()
    current_snapshot, metadata = _fetch_simulation_feature_snapshot()

    report = _build_baseline_feature_drift_report(
        feature_names=SIMULATION_DRIFT_FEATURES,
        baseline_payload=baseline_payload,
        current_snapshot=current_snapshot,
        min_samples=min_samples,
    )
    report["model_version"] = baseline_payload.get("model_version")
    report["baseline_created_at"] = baseline_payload.get("created_at")
    report["sample_summary"] = {
        **report.get("sample_summary", {}),
        **metadata,
    }
    return report


@router.post("/feedback")
async def submit_expert_feedback(feedback: FeedbackData):
    """
    Endpoint for tax inspectors to provide ground-truth feedback on AI predictions.
    This data is crucial for continuous learning and model retraining.
    """
    feedback_file = LOG_DIR / "expert_feedback.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "tax_code": feedback.tax_code,
        "invoice_number": feedback.invoice_number,
        "path_id": feedback.path_id,
        "is_fraud_ground_truth": feedback.is_fraud,
        "expert_notes": feedback.expert_notes,
    }

    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log_event(logger, "info", "monitoring_feedback_recorded", has_tax_code=bool(feedback.tax_code))

    return {"status": "success", "message": "Feedback recorded for next training cycle."}


@router.post("/log_metric")
async def log_ml_metric(metric: MetricLog):
    """
    Internal endpoint to log inference latency, fallback occurrences, etc.
    """
    metrics_file = LOG_DIR / "ml_metrics.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "metric": metric.metric_name,
        "value": metric.value,
        "labels": metric.labels,
    }

    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if metric.metric_name in {"inference_features", "graph_inference_features", "fraud_inference_features"}:
        log_event(logger, "info", "monitoring_feature_metric_logged", metric_name=metric.metric_name)

    return {"status": "success"}


@router.get("/health")
async def mlops_health_check():
    """
    Check the health of the ML serving infrastructure.
    """
    model_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models"

    status = {
        "status": "healthy",
        "models": {
            "graph_model": (model_dir / "gat_model.pt").exists(),
            "fraud_xgboost": (model_dir / "xgboost_model.joblib").exists(),
            "fraud_isolation_forest": (model_dir / "isolation_forest.joblib").exists(),
            "fraud_calibrator": (model_dir / "fraud_calibrator.joblib").exists(),
            "fraud_manifest": (model_dir / "fraud_model_manifest.json").exists(),
            "osint_model": (model_dir / "osint_risk_model.joblib").exists(),
            "osint_drift_baseline": (model_dir / "osint_drift_baseline.json").exists(),
            "simulation_model": (model_dir / "simulation_lgbm.joblib").exists(),
            "simulation_drift_baseline": (model_dir / "simulation_drift_baseline.json").exists(),
            "audit_value_model": (model_dir / "audit_value_model.joblib").exists(),
            "audit_value_calibrator": (model_dir / "audit_value_calibrator.joblib").exists(),
            "vat_refund_model": (model_dir / "vat_refund_model.joblib").exists(),
            "vat_refund_calibrator": (model_dir / "vat_refund_calibrator.joblib").exists(),
        },
        "db_connection": False,
    }

    conn = get_db_connection()
    if conn:
        status["db_connection"] = True
        conn.close()
    else:
        status["status"] = "degraded"

    critical_artifacts = (
        status["models"].get("graph_model"),
        status["models"].get("fraud_xgboost"),
        status["models"].get("fraud_isolation_forest"),
    )
    if not all(critical_artifacts):
        status["status"] = "degraded"

    return status


@router.get("/drift_report")
async def get_drift_report(
    lookback_days: int = 7,
    baseline_days: int = 30,
    min_samples: int = 20,
):
    """
    Compute feature drift from live metric logs.

    Data source priority:
        1) drift_baseline.json in model directory (if available)
        2) historical metrics window as baseline fallback
    """
    if lookback_days < 1 or baseline_days < 1 or min_samples < 1:
        raise HTTPException(status_code=400, detail="lookback_days, baseline_days and min_samples must be >= 1")

    report = _build_drift_report(
        lookback_days=lookback_days,
        baseline_days=baseline_days,
        min_samples=min_samples,
    )

    sufficient_features = sum(
        1
        for item in report.get("features", {}).values()
        if isinstance(item, dict) and item.get("status") != "insufficient_data"
    )

    log_event(
        logger,
        "info",
        "monitoring_drift_report_generated",
        drift_detected=report.get("drift_detected", False),
        drifted_features=len(report.get("drifted_features", [])),
        sufficient_features=sufficient_features,
    )
    return report


@router.get("/graph_quality", response_model=schemas.GraphQualityResponse)
async def get_graph_quality_summary(include_criteria: bool = False):
    """Graph-focused model quality summary for backend contract validation and frontend widgets."""
    serving_path = MODEL_DIR / "serving_e2e_report.json"
    stress_path = MODEL_DIR / "stress_evaluation_report.json"
    config_path = MODEL_DIR / "gat_config.json"
    model_path = MODEL_DIR / "gat_model.pt"

    serving_payload = _safe_read_json(serving_path)
    stress_payload = _safe_read_json(stress_path)
    config_payload = _safe_read_json(config_path)
    drift_payload = _build_drift_report(lookback_days=7, baseline_days=30, min_samples=20)

    serving_gates = serving_payload.get("acceptance_gates", {}) if isinstance(serving_payload, dict) else {}
    serving_criteria = serving_gates.get("criteria", {}) if isinstance(serving_gates, dict) else {}
    serving_test = serving_payload.get("serving_path_test", {}) if isinstance(serving_payload, dict) else {}
    serving_cmp = serving_payload.get("comparison", {}) if isinstance(serving_payload, dict) else {}

    stress_gates = stress_payload.get("stress_acceptance_gates", {}) if isinstance(stress_payload, dict) else {}
    stress_criteria = stress_gates.get("criteria", {}) if isinstance(stress_gates, dict) else {}
    stress_summary = stress_payload.get("stress_summary", {}) if isinstance(stress_payload, dict) else {}

    serving_available = bool(serving_payload)
    stress_available = bool(stress_payload)
    serving_pass = bool(serving_gates.get("overall_pass")) if serving_available else None
    stress_pass = bool(stress_gates.get("overall_pass")) if stress_available else None
    all_pass = bool(serving_pass and stress_pass)

    drift_severity = str(drift_payload.get("drift_severity", "insufficient_data"))
    if not serving_available and not stress_available:
        status = "unknown"
    elif serving_pass is False or stress_pass is False:
        status = "degraded"
    elif drift_severity in {"high", "medium"}:
        status = "warning"
    elif serving_pass is True and stress_pass is True:
        status = "healthy"
    else:
        status = "warning"

    response = {
        "status": status,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "model_version": config_payload.get("model_version") if isinstance(config_payload, dict) else None,
            "amount_feature_mode": config_payload.get("amount_feature_mode") if isinstance(config_payload, dict) else None,
            "updated_at": _file_updated_at(model_path) or _file_updated_at(config_path),
        },
        "gate_summary": {
            "serving_pass": serving_pass,
            "stress_pass": stress_pass,
            "all_pass": all_pass,
        },
        "serving": {
            "available": serving_available,
            "overall_pass": serving_pass,
            "node_f1": serving_test.get("node", {}).get("f1") if isinstance(serving_test, dict) else None,
            "edge_f1": serving_test.get("edge", {}).get("f1") if isinstance(serving_test, dict) else None,
            "node_pr_auc_delta": serving_cmp.get("node_pr_auc_delta") if isinstance(serving_cmp, dict) else None,
            "edge_pr_auc_delta": serving_cmp.get("edge_pr_auc_delta") if isinstance(serving_cmp, dict) else None,
            "criteria": serving_criteria if include_criteria and isinstance(serving_criteria, dict) else {},
            "updated_at": _file_updated_at(serving_path),
        },
        "stress": {
            "available": stress_available,
            "overall_pass": stress_pass,
            "worst_node_f1_delta": stress_summary.get("worst_node_f1_delta") if isinstance(stress_summary, dict) else None,
            "unseen_node_generalization_gap": _criterion_actual(stress_criteria, "unseen_node_generalization_gap"),
            "temporal_plus3m_edge_f1_drop": _criterion_actual(stress_criteria, "temporal_plus3m_edge_f1_drop"),
            "temporal_plus3m_edge_prauc_drop": _criterion_actual(stress_criteria, "temporal_plus3m_edge_prauc_drop"),
            "criteria": stress_criteria if include_criteria and isinstance(stress_criteria, dict) else {},
            "updated_at": _file_updated_at(stress_path),
        },
        "drift": {
            "detected": bool(drift_payload.get("drift_detected", False)),
            "severity": drift_severity,
            "drifted_features": drift_payload.get("drifted_features", []),
            "recommendation": drift_payload.get("recommendation", ""),
        },
    }

    log_event(
        logger,
        "info",
        "monitoring_graph_quality_generated",
        status=status,
        serving_available=serving_available,
        stress_available=stress_available,
        serving_pass=serving_pass,
        stress_pass=stress_pass,
    )
    return response


@router.get("/fraud_quality", response_model=schemas.FraudQualityResponse)
async def get_fraud_quality_summary(include_criteria: bool = False):
    """Fraud model quality summary including calibration, drift and acceptance gates."""
    response = _build_fraud_quality_summary(include_criteria=include_criteria)
    log_event(
        logger,
        "info",
        "monitoring_fraud_quality_generated",
        status=response.get("status", "unknown"),
        drift_severity=response.get("drift", {}).get("severity", "unknown"),
        quality_report_available=response.get("model_info", {}).get("quality_report_available", False),
    )
    return response


@router.get("/osint_quality")
async def get_osint_quality_summary(include_criteria: bool = False):
    """OSINT model quality summary including drift against training baseline."""
    response = _build_osint_quality_summary(include_criteria=include_criteria)
    log_event(
        logger,
        "info",
        "monitoring_osint_quality_generated",
        status=response.get("status", "unknown"),
        drift_severity=response.get("drift", {}).get("severity", "unknown"),
        quality_report_available=response.get("model_info", {}).get("quality_report_available", False),
    )
    return response


@router.get("/simulation_quality")
async def get_simulation_quality_summary(include_criteria: bool = False):
    """Simulation model quality summary including drift against baseline."""
    response = _build_simulation_quality_summary(include_criteria=include_criteria)
    log_event(
        logger,
        "info",
        "monitoring_simulation_quality_generated",
        status=response.get("status", "unknown"),
        drift_severity=response.get("drift", {}).get("severity", "unknown"),
        quality_report_available=response.get("model_info", {}).get("quality_report_available", False),
    )
    return response


@router.get("/osint_drift")
async def get_osint_drift_report(min_samples: int = 10):
    """Drift report for OSINT features vs. osint_drift_baseline.json."""
    if min_samples < 1:
        raise HTTPException(status_code=400, detail="min_samples must be >= 1")

    report = _build_osint_drift_report(min_samples=min_samples)
    log_event(
        logger,
        "info",
        "monitoring_osint_drift_generated",
        drift_detected=report.get("drift_detected", False),
        drifted_features=len(report.get("drifted_features", [])),
        severity=report.get("drift_severity", "unknown"),
    )
    return report


@router.get("/simulation_drift")
async def get_simulation_drift_report(min_samples: int = 5):
    """Drift report for simulation baseline features vs. simulation_drift_baseline.json."""
    if min_samples < 1:
        raise HTTPException(status_code=400, detail="min_samples must be >= 1")

    report = _build_simulation_drift_report(min_samples=min_samples)
    log_event(
        logger,
        "info",
        "monitoring_simulation_drift_generated",
        drift_detected=report.get("drift_detected", False),
        drifted_features=len(report.get("drifted_features", [])),
        severity=report.get("drift_severity", "unknown"),
    )
    return report


# ════════════════════════════════════════════════════════════════
#  Flagship: Delinquency Drift & Data Quality (Phase 0.2)
# ════════════════════════════════════════════════════════════════

@router.get("/delinquency_drift")
async def get_delinquency_drift():
    """
    Drift detection for the delinquency temporal model (Program A).
    Compares recent prediction distributions against baseline saved during training.
    """
    baseline_path = MODEL_DIR / "delinquency_drift_baseline.json"
    predictions_log = LOG_DIR / "delinquency_predictions.jsonl"

    baseline = {}
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline = json.load(f)
        except Exception:
            pass

    recent_preds = _read_jsonl(predictions_log)

    if not baseline and not recent_preds:
        return {
            "status": "no_data",
            "message": "Chưa có baseline hoặc prediction logs cho delinquency model.",
            "features": [],
            "overall_severity": "unknown",
            "generated_at": datetime.utcnow().isoformat(),
        }

    feature_drifts = []
    overall_severity = "healthy"

    for feat in DELINQUENCY_DRIFT_FEATURES:
        baseline_vals = baseline.get(feat, {})
        baseline_mean = baseline_vals.get("mean", 0.0)
        baseline_std = baseline_vals.get("std", 1.0)
        baseline_dist = baseline_vals.get("distribution", [])

        recent_vals = [r.get(feat, 0.0) for r in recent_preds[-500:] if feat in r]

        if not recent_vals:
            feature_drifts.append({
                "feature": feat,
                "status": "no_data",
                "drift_score": 0.0,
                "baseline_mean": round(baseline_mean, 4),
                "recent_mean": None,
                "recent_count": 0,
            })
            continue

        recent_mean = float(np.mean(recent_vals))
        recent_std = float(np.std(recent_vals)) if len(recent_vals) > 1 else 0.0

        # Compute drift using Wasserstein-like distance
        if baseline_dist:
            drift_score = _wasserstein_like(baseline_dist, recent_vals)
        else:
            drift_score = abs(recent_mean - baseline_mean) / max(baseline_std, 0.01)

        # Classify severity
        if drift_score > 2.0:
            severity = "critical"
            overall_severity = "critical"
        elif drift_score > 1.0:
            severity = "warning"
            if overall_severity == "healthy":
                overall_severity = "warning"
        else:
            severity = "healthy"

        feature_drifts.append({
            "feature": feat,
            "status": severity,
            "drift_score": round(drift_score, 4),
            "baseline_mean": round(baseline_mean, 4),
            "recent_mean": round(recent_mean, 4),
            "baseline_std": round(baseline_std, 4),
            "recent_std": round(recent_std, 4),
            "recent_count": len(recent_vals),
        })

    log_event(
        logger,
        "info",
        "monitoring_delinquency_drift_generated",
        overall_severity=overall_severity,
        features_checked=len(feature_drifts),
    )

    return {
        "status": overall_severity,
        "features": feature_drifts,
        "overall_severity": overall_severity,
        "model_version": baseline.get("model_version", "unknown"),
        "baseline_date": baseline.get("created_at", "unknown"),
        "recent_window_size": len(recent_preds),
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/data_quality")
def get_data_quality_gate():
    """
    Data quality gate: checks completeness and freshness of all flagship data sources.
    Returns pass/fail status for each data source with actionable guidance.
    """
    import psycopg2

    db_url = _resolve_db_url()

    checks = []
    overall = "pass"

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Check 1: Companies table
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT province) FROM companies")
        co_count, co_provinces = cur.fetchone()
        checks.append({
            "source": "companies",
            "status": "pass" if co_count > 0 else "fail",
            "row_count": co_count,
            "details": {
                "distinct_provinces": co_provinces,
                "has_province_data": co_provinces > 0,
            },
            "guidance": None if co_count > 0 else "Import danh sách doanh nghiệp vào bảng companies.",
        })

        # Check 2: Tax Returns
        cur.execute("SELECT COUNT(*), MIN(filing_date), MAX(filing_date) FROM tax_returns")
        tr_count, tr_min, tr_max = cur.fetchone()
        checks.append({
            "source": "tax_returns",
            "status": "pass" if tr_count > 10 else ("warning" if tr_count > 0 else "fail"),
            "row_count": tr_count,
            "details": {
                "earliest_filing": str(tr_min) if tr_min else None,
                "latest_filing": str(tr_max) if tr_max else None,
            },
            "guidance": None if tr_count > 10 else "Cần ít nhất 10 tờ khai thuế để model hoạt động.",
        })

        # Check 3: Invoices
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT seller_tax_code), COUNT(DISTINCT buyer_tax_code) FROM invoices")
        inv_count, inv_sellers, inv_buyers = cur.fetchone()
        checks.append({
            "source": "invoices",
            "status": "pass" if inv_count > 50 else ("warning" if inv_count > 0 else "fail"),
            "row_count": inv_count,
            "details": {"distinct_sellers": inv_sellers, "distinct_buyers": inv_buyers},
            "guidance": None if inv_count > 50 else "Cần hóa đơn để phân tích mạng lưới Graph Intelligence.",
        })

        # Check 4: Tax Payments (Program A core data)
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT tax_code) FROM tax_payments")
        tp_count, tp_companies = cur.fetchone()
        status_tp = "pass" if tp_count > 20 else ("warning" if tp_count > 0 else "fail")
        if status_tp != "pass":
            overall = "warning" if overall == "pass" else overall
        checks.append({
            "source": "tax_payments",
            "status": status_tp,
            "row_count": tp_count,
            "details": {"distinct_companies": tp_companies},
            "program": "A – Temporal Compliance",
            "guidance": None if tp_count > 20 else "Import dữ liệu ngày nộp thực tế (tax_payments) cho Program A.",
        })

        # Check 5: Invoice Line Items (Program B enrichment)
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT invoice_id) FROM invoice_line_items")
        li_count, li_invoices = cur.fetchone()
        checks.append({
            "source": "invoice_line_items",
            "status": "pass" if li_count > 0 else "optional",
            "row_count": li_count,
            "details": {"distinct_invoices": li_invoices},
            "program": "B – Graph Intelligence 2.0",
            "guidance": None if li_count > 0 else "Optional: chi tiết hóa đơn giúp Graph motif detection chính xác hơn.",
        })

        # Check 6: Ownership Links (Program B shell detection)
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT parent_tax_code) FROM ownership_links")
        ol_count, ol_parents = cur.fetchone()
        checks.append({
            "source": "ownership_links",
            "status": "pass" if ol_count > 0 else "optional",
            "row_count": ol_count,
            "details": {"distinct_parent_companies": ol_parents},
            "program": "B – Graph Intelligence 2.0",
            "guidance": None if ol_count > 0 else "Optional: quan hệ sở hữu giúp phát hiện công ty vỏ bọc.",
        })

        # Check 7: Inspector Labels (Retrain data)
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT label_type) FROM inspector_labels")
        il_count, il_types = cur.fetchone()
        checks.append({
            "source": "inspector_labels",
            "status": "pass" if il_count > 50 else ("building" if il_count > 0 else "empty"),
            "row_count": il_count,
            "details": {"distinct_label_types": il_types},
            "program": "All – Model Retraining",
            "guidance": f"Có {il_count} nhãn. Cần ≥50 nhãn để retrain hiệu quả." if il_count < 50 else None,
        })

        # Check 8: Delinquency Predictions cache
        cur.execute("SELECT COUNT(*), MAX(created_at) FROM delinquency_predictions")
        dp_count, dp_latest = cur.fetchone()
        checks.append({
            "source": "delinquency_predictions",
            "status": "pass" if dp_count > 0 else "empty",
            "row_count": dp_count,
            "details": {"latest_prediction": str(dp_latest) if dp_latest else None},
            "program": "A – Temporal Compliance",
            "guidance": None if dp_count > 0 else "Chạy batch prediction để populate cache.",
        })

        # Check 9: ai_risk_assessments lineage completeness (model_version)
        assessment_columns = _table_columns(cur, "ai_risk_assessments")
        if {"model_version", "created_at"}.issubset(assessment_columns):
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(*) FILTER (
                        WHERE model_version IS NOT NULL AND btrim(model_version) <> ''
                    ) AS with_model_version,
                    COUNT(*) FILTER (
                        WHERE created_at >= (NOW() - interval '30 days')
                    ) AS recent_rows,
                    COUNT(*) FILTER (
                        WHERE created_at >= (NOW() - interval '30 days')
                          AND model_version IS NOT NULL
                          AND btrim(model_version) <> ''
                    ) AS recent_with_model_version
                FROM ai_risk_assessments
                """
            )
            (
                assessments_total,
                assessments_with_version,
                assessments_recent,
                assessments_recent_with_version,
            ) = cur.fetchone()
            assessments_total = int(assessments_total or 0)
            assessments_with_version = int(assessments_with_version or 0)
            assessments_recent = int(assessments_recent or 0)
            assessments_recent_with_version = int(assessments_recent_with_version or 0)

            overall_coverage = _safe_div(assessments_with_version, assessments_total)
            recent_coverage = _safe_div(assessments_recent_with_version, assessments_recent)
            coverage_for_status = recent_coverage if assessments_recent >= 20 else overall_coverage

            if assessments_total == 0:
                status_lineage = "warning"
            elif coverage_for_status is None:
                status_lineage = "fail"
            elif coverage_for_status >= 0.95:
                status_lineage = "pass"
            elif coverage_for_status >= 0.80:
                status_lineage = "warning"
            else:
                status_lineage = "fail"

            checks.append(
                {
                    "source": "ai_risk_assessments_lineage",
                    "status": status_lineage,
                    "row_count": assessments_total,
                    "program": "B/C – Data Readiness",
                    "details": {
                        "overall_model_version_coverage": overall_coverage,
                        "recent_30d_model_version_coverage": recent_coverage,
                        "recent_30d_rows": assessments_recent,
                    },
                    "guidance": None
                    if status_lineage == "pass"
                    else "Bắt buộc ghi model_version cho mỗi bản ghi ai_risk_assessments để đảm bảo lineage.",
                }
            )

        # Check 10-13: inspector_labels completeness/lag/outlier + lineage
        inspector_columns = _table_columns(cur, "inspector_labels")

        if {"assessment_id", "model_version"}.issubset(inspector_columns):
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE assessment_id IS NOT NULL) AS linked_rows,
                    COUNT(*) FILTER (
                        WHERE assessment_id IS NOT NULL
                          AND model_version IS NOT NULL
                          AND btrim(model_version) <> ''
                    ) AS linked_with_model_version
                FROM inspector_labels
                """
            )
            linked_rows, linked_with_model_version = cur.fetchone()
            linked_rows = int(linked_rows or 0)
            linked_with_model_version = int(linked_with_model_version or 0)
            linked_coverage = _safe_div(linked_with_model_version, linked_rows)

            if linked_rows == 0:
                status_label_lineage = "building"
            elif linked_coverage is None:
                status_label_lineage = "fail"
            elif linked_coverage >= 0.85:
                status_label_lineage = "pass"
            elif linked_coverage >= 0.65:
                status_label_lineage = "warning"
            else:
                status_label_lineage = "fail"

            checks.append(
                {
                    "source": "inspector_labels_lineage",
                    "status": status_label_lineage,
                    "row_count": linked_rows,
                    "program": "B/C – Data Readiness",
                    "details": {
                        "linked_label_rows": linked_rows,
                        "linked_model_version_coverage": linked_coverage,
                    },
                    "guidance": None
                    if status_label_lineage in {"pass", "building"}
                    else "Backfill model_version cho inspector_labels đã gắn assessment_id.",
                }
            )

        if {"intervention_attempted", "outcome_status", "created_at"}.issubset(inspector_columns):
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE COALESCE(intervention_attempted, FALSE)
                          AND COALESCE(outcome_status, 'pending') IN ('pending', 'in_progress')
                    ) AS open_attempts,
                    COUNT(*) FILTER (
                        WHERE COALESCE(intervention_attempted, FALSE)
                          AND COALESCE(outcome_status, 'pending') IN ('pending', 'in_progress')
                          AND created_at < (NOW() - interval '30 days')
                    ) AS lagging_attempts
                FROM inspector_labels
                """
            )
            open_attempts, lagging_attempts = cur.fetchone()
            open_attempts = int(open_attempts or 0)
            lagging_attempts = int(lagging_attempts or 0)
            lagging_ratio = _safe_div(lagging_attempts, open_attempts)

            if open_attempts == 0 or lagging_attempts == 0:
                status_lag = "pass"
            elif lagging_ratio is not None and lagging_ratio <= 0.25:
                status_lag = "warning"
            else:
                status_lag = "fail"

            checks.append(
                {
                    "source": "inspector_outcome_lag",
                    "status": status_lag,
                    "row_count": open_attempts,
                    "program": "C – KPI Operational Loop",
                    "details": {
                        "open_attempts": open_attempts,
                        "lagging_attempts_over_30d": lagging_attempts,
                        "lagging_ratio": lagging_ratio,
                    },
                    "guidance": None
                    if status_lag == "pass"
                    else "Đóng outcome_status/outcome_recorded_at cho case pending quá 30 ngày.",
                }
            )

        if {
            "outcome_status",
            "outcome_recorded_at",
            "amount_recovered",
            "expected_recovery",
        }.issubset(inspector_columns):
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE outcome_status IN %s
                    ) AS terminal_rows,
                    COUNT(*) FILTER (
                        WHERE outcome_status IN %s
                          AND (
                                outcome_recorded_at IS NULL
                             OR expected_recovery IS NULL
                             OR (
                                    outcome_status IN ('recovered', 'partial_recovered')
                                AND amount_recovered IS NULL
                                )
                          )
                    ) AS incomplete_terminal_rows
                FROM inspector_labels
                """,
                (TERMINAL_OUTCOME_STATUSES, TERMINAL_OUTCOME_STATUSES),
            )
            terminal_rows, incomplete_terminal_rows = cur.fetchone()
            terminal_rows = int(terminal_rows or 0)
            incomplete_terminal_rows = int(incomplete_terminal_rows or 0)
            incomplete_ratio = _safe_div(incomplete_terminal_rows, terminal_rows)

            if terminal_rows == 0:
                status_terminal = "building"
            elif incomplete_terminal_rows == 0:
                status_terminal = "pass"
            elif incomplete_ratio is not None and incomplete_ratio <= 0.10:
                status_terminal = "warning"
            else:
                status_terminal = "fail"

            checks.append(
                {
                    "source": "inspector_terminal_completeness",
                    "status": status_terminal,
                    "row_count": terminal_rows,
                    "program": "C – KPI Operational Loop",
                    "details": {
                        "terminal_rows": terminal_rows,
                        "incomplete_terminal_rows": incomplete_terminal_rows,
                        "incomplete_ratio": incomplete_ratio,
                    },
                    "guidance": None
                    if status_terminal in {"pass", "building"}
                    else "Bổ sung expected_recovery / amount_recovered / outcome_recorded_at cho các case đã có outcome cuối.",
                }
            )

        numeric_columns = {
            "amount_recovered",
            "expected_recovery",
            "estimated_audit_cost",
            "actual_audit_cost",
        }
        if numeric_columns.issubset(inspector_columns):
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(*) FILTER (
                        WHERE COALESCE(amount_recovered, 0) < 0
                           OR COALESCE(expected_recovery, 0) < 0
                           OR COALESCE(estimated_audit_cost, 0) < 0
                           OR COALESCE(actual_audit_cost, 0) < 0
                    ) AS negative_rows,
                    COUNT(*) FILTER (
                        WHERE COALESCE(amount_recovered, 0) > 1e12
                           OR COALESCE(expected_recovery, 0) > 1e12
                    ) AS extreme_rows
                FROM inspector_labels
                """
            )
            total_rows, negative_rows, extreme_rows = cur.fetchone()
            total_rows = int(total_rows or 0)
            negative_rows = int(negative_rows or 0)
            extreme_rows = int(extreme_rows or 0)
            anomaly_rows = negative_rows + extreme_rows
            anomaly_ratio = _safe_div(anomaly_rows, total_rows)

            if total_rows == 0 or anomaly_rows == 0:
                status_numeric = "pass"
            elif anomaly_ratio is not None and anomaly_ratio <= 0.01:
                status_numeric = "warning"
            else:
                status_numeric = "fail"

            checks.append(
                {
                    "source": "inspector_numeric_outliers",
                    "status": status_numeric,
                    "row_count": total_rows,
                    "program": "B/C – Data Readiness",
                    "details": {
                        "negative_rows": negative_rows,
                        "extreme_rows_over_1e12": extreme_rows,
                        "anomaly_ratio": anomaly_ratio,
                    },
                    "guidance": None
                    if status_numeric == "pass"
                    else "Rà soát các bản ghi có giá trị âm/đột biến trong amount_recovered và expected_recovery.",
                }
            )

        cur.close()
        conn.close()

    except Exception as e:
        log_event(logger, "error", "data_quality_check_failed", error=str(e))
        return {
            "status": "error",
            "message": f"Không thể kết nối database: {str(e)}",
            "checks": [],
            "generated_at": datetime.utcnow().isoformat(),
        }

    fail_count = sum(1 for c in checks if c["status"] == "fail")
    warn_count = sum(1 for c in checks if c["status"] == "warning")
    if fail_count > 0:
        overall = "fail"
    elif warn_count > 0:
        overall = "warning"

    log_event(
        logger,
        "info",
        "data_quality_gate_checked",
        overall=overall,
        fail_count=fail_count,
        warn_count=warn_count,
    )

    return {
        "status": overall,
        "checks": checks,
        "summary": {
            "total_sources": len(checks),
            "pass": sum(1 for c in checks if c["status"] == "pass"),
            "warning": warn_count,
            "fail": fail_count,
            "optional": sum(1 for c in checks if c["status"] in ("optional", "empty", "building")),
        },
        "generated_at": datetime.utcnow().isoformat(),
    }


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0:
        return None
    return num / den


def _table_columns(cur, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table_name,),
    )
    return {str(row[0]) for row in cur.fetchall()}


def _normalize_policy_token(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    value = value.replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


def _normalize_kpi_policy_payload(policies: list[KPIPolicyInput]) -> list[dict[str, Any]]:
    if not policies:
        raise HTTPException(status_code=422, detail="policies không được rỗng.")

    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for policy in policies:
        track_name = _normalize_policy_token(policy.track_name)
        metric_name = _normalize_policy_token(policy.metric_name)
        if not track_name or not metric_name:
            raise HTTPException(status_code=422, detail="track_name và metric_name là bắt buộc.")

        comparator = str(policy.comparator or "").strip()
        if comparator not in VALID_POLICY_COMPARATORS:
            raise HTTPException(status_code=422, detail=f"comparator không hợp lệ: {comparator}")

        threshold = float(policy.threshold)
        if np.isnan(threshold) or np.isinf(threshold):
            raise HTTPException(status_code=422, detail="threshold phải là số hữu hạn.")

        key = (track_name, metric_name)
        if key in seen_keys:
            raise HTTPException(
                status_code=422,
                detail=f"Policy bị trùng cho track/metric: {track_name}/{metric_name}",
            )
        seen_keys.add(key)

        rationale = str(policy.rationale).strip() if policy.rationale is not None else None
        if rationale == "":
            rationale = None

        normalized.append(
            {
                "track_name": track_name,
                "metric_name": metric_name,
                "comparator": comparator,
                "threshold": threshold,
                "min_sample": max(1, int(policy.min_sample)),
                "window_days": max(7, min(365, int(policy.window_days))),
                "cooldown_days": max(0, min(365, int(policy.cooldown_days))),
                "enabled": bool(policy.enabled),
                "rationale": rationale,
            }
        )

    return normalized


def _upsert_kpi_policies(cur, policies: list[dict[str, Any]], replace_existing: bool = False) -> int:
    if replace_existing:
        cur.execute("SELECT track_name, metric_name FROM kpi_trigger_policies")
        existing = {(str(row[0]), str(row[1])) for row in cur.fetchall()}
        keep = {(p["track_name"], p["metric_name"]) for p in policies}
        for track_name, metric_name in sorted(existing - keep):
            cur.execute(
                "DELETE FROM kpi_trigger_policies WHERE track_name=%s AND metric_name=%s",
                (track_name, metric_name),
            )

    for policy in policies:
        cur.execute(
            """
            INSERT INTO kpi_trigger_policies (
                track_name, metric_name, comparator, threshold, min_sample,
                window_days, cooldown_days, enabled, rationale
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (track_name, metric_name)
            DO UPDATE SET
                comparator = EXCLUDED.comparator,
                threshold = EXCLUDED.threshold,
                min_sample = EXCLUDED.min_sample,
                window_days = EXCLUDED.window_days,
                cooldown_days = EXCLUDED.cooldown_days,
                enabled = EXCLUDED.enabled,
                rationale = EXCLUDED.rationale,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                policy["track_name"],
                policy["metric_name"],
                policy["comparator"],
                policy["threshold"],
                policy["min_sample"],
                policy["window_days"],
                policy["cooldown_days"],
                policy["enabled"],
                policy["rationale"],
            ),
        )

    return len(policies)


def _normalize_snapshot_source(source: Optional[str]) -> str:
    normalized = _normalize_policy_token(source or "split_trigger_status")
    return normalized[:80] if normalized else "split_trigger_status"


def _default_kpi_policies() -> list[dict[str, Any]]:
    return [
        {
            "track_name": "audit_value",
            "metric_name": "precision_top_50",
            "comparator": ">=",
            "threshold": 0.70,
            "min_sample": 50,
            "window_days": 28,
            "cooldown_days": 14,
            "enabled": True,
            "rationale": "Top-50 hồ sơ Audit Value cần precision ổn định trước khi cân nhắc split.",
        },
        {
            "track_name": "audit_value",
            "metric_name": "roi_positive_rate",
            "comparator": ">=",
            "threshold": 0.80,
            "min_sample": 50,
            "window_days": 28,
            "cooldown_days": 14,
            "enabled": True,
            "rationale": "Tối thiểu 80% case can thiệp phải cho net recovery dương.",
        },
        {
            "track_name": "vat_refund",
            "metric_name": "precision_top_100",
            "comparator": ">=",
            "threshold": 0.65,
            "min_sample": 80,
            "window_days": 28,
            "cooldown_days": 14,
            "enabled": True,
            "rationale": "Top-100 VAT queue cần precision đủ mạnh trước khi tách flow.",
        },
        {
            "track_name": "vat_refund",
            "metric_name": "false_negative_rate_high_risk",
            "comparator": "<=",
            "threshold": 0.12,
            "min_sample": 50,
            "window_days": 28,
            "cooldown_days": 14,
            "enabled": True,
            "rationale": "Giữ FN high-risk VAT ở mức thấp để tránh bỏ sót hồ sơ trọng điểm.",
        },
    ]


def _fetch_kpi_policies(cur) -> list[dict[str, Any]]:
    table_names = _table_columns(cur, "kpi_trigger_policies")
    if not table_names:
        return _default_kpi_policies()

    cur.execute(
        """
        SELECT track_name, metric_name, comparator, threshold, min_sample, window_days,
               cooldown_days, enabled, rationale
        FROM kpi_trigger_policies
        ORDER BY track_name, metric_name
        """
    )
    rows = cur.fetchall()
    if not rows:
        return _default_kpi_policies()

    return [
        {
            "track_name": r[0],
            "metric_name": r[1],
            "comparator": r[2],
            "threshold": float(r[3]),
            "min_sample": int(r[4]),
            "window_days": int(r[5]),
            "cooldown_days": int(r[6]),
            "enabled": bool(r[7]),
            "rationale": r[8],
        }
        for r in rows
    ]


def _evaluate_threshold(actual: Optional[float], comparator: str, threshold: float) -> Optional[bool]:
    if actual is None:
        return None
    if comparator == ">=":
        return actual >= threshold
    if comparator == ">":
        return actual > threshold
    if comparator == "<=":
        return actual <= threshold
    if comparator == "<":
        return actual < threshold
    if comparator == "==":
        return abs(actual - threshold) < 1e-9
    return None


def _fetch_recent_rule_failures(cur, policies: list[dict[str, Any]]) -> dict[tuple[str, str], datetime]:
    if not policies:
        return {}
    if not _table_columns(cur, "kpi_metric_snapshots"):
        return {}

    max_cooldown_days = 0
    for policy in policies:
        try:
            max_cooldown_days = max(max_cooldown_days, int(policy.get("cooldown_days") or 0))
        except (TypeError, ValueError):
            continue

    if max_cooldown_days <= 0:
        return {}

    cur.execute(
        """
        SELECT
            COALESCE(track_name, '') AS track_name,
            COALESCE(metric_name, '') AS metric_name,
            MAX(generated_at) AS last_failed_at
        FROM kpi_metric_snapshots
        WHERE generated_at >= (NOW() - (%s || ' days')::interval)
                    AND status IN ('fail', 'insufficient_data', 'no_metric')
        GROUP BY COALESCE(track_name, ''), COALESCE(metric_name, '')
        """,
        (max_cooldown_days,),
    )

    failures: dict[tuple[str, str], datetime] = {}
    for raw_track_name, raw_metric_name, last_failed_at in cur.fetchall():
        if not isinstance(last_failed_at, datetime):
            continue

        track_name = _normalize_policy_token(raw_track_name) or "unknown"
        metric_name = _normalize_policy_token(raw_metric_name)
        if not metric_name:
            continue

        normalized_ts = last_failed_at.replace(tzinfo=None) if last_failed_at.tzinfo else last_failed_at
        failures[(track_name, metric_name)] = normalized_ts

    return failures


def _compute_precision_at_k(cur, window_days: int, k: int) -> tuple[Optional[float], int]:
    if k <= 0:
        return None, 0

    cur.execute(
        """
        WITH ranked AS (
            SELECT
                COALESCE(expected_recovery, predicted_collection_uplift, 0) AS expected_value,
                COALESCE(amount_recovered, 0) AS recovered_value,
                COALESCE(label_type, '') AS label_type
            FROM inspector_labels
            WHERE created_at >= (NOW() - (%s || ' days')::interval)
            ORDER BY COALESCE(expected_recovery, predicted_collection_uplift, 0) DESC, created_at DESC
            LIMIT %s
        )
        SELECT
            COUNT(*) AS sample_count,
            COUNT(*) FILTER (
                WHERE recovered_value > 0
                   OR label_type IN ('fraud_confirmed', 'delinquency_confirmed')
            ) AS positive_count
        FROM ranked
        """,
        (int(window_days), int(k)),
    )
    sample_count, positive_count = cur.fetchone()
    sample_count = int(sample_count or 0)
    positive_count = int(positive_count or 0)
    return _safe_div(positive_count, sample_count), sample_count


def _compute_vat_precision_at_k(cur, window_days: int, k: int) -> tuple[Optional[float], int]:
    if k <= 0:
        return None, 0

    cur.execute(
        """
        WITH vat_cases AS (
            SELECT
                COALESCE(expected_recovery, predicted_collection_uplift, 0) AS expected_value,
                COALESCE(amount_recovered, 0) AS recovered_value,
                LOWER(COALESCE(label_type, '')) AS label_norm,
                COALESCE(outcome_status, '') AS outcome_status
            FROM inspector_labels
            WHERE created_at >= (NOW() - (%s || ' days')::interval)
              AND (
                          LOWER(COALESCE(label_type, '')) LIKE '%%vat%%'
                      OR LOWER(COALESCE(label_type, '')) LIKE '%%refund%%'
                      OR LOWER(COALESCE(label_type, '')) LIKE '%%invoice%%'
              )
            ORDER BY COALESCE(expected_recovery, predicted_collection_uplift, 0) DESC, created_at DESC
            LIMIT %s
        )
        SELECT
            COUNT(*) AS sample_count,
            COUNT(*) FILTER (
                WHERE recovered_value > 0
                   OR outcome_status IN ('recovered', 'partial_recovered')
                   OR label_norm LIKE '%%confirmed%%'
            ) AS positive_count
        FROM vat_cases
        """,
        (int(window_days), int(k)),
    )
    sample_count, positive_count = cur.fetchone()
    sample_count = int(sample_count or 0)
    positive_count = int(positive_count or 0)
    return _safe_div(positive_count, sample_count), sample_count


def _estimate_metric_sample_size(track_name: str, metric_name: str, summary: dict[str, Any]) -> int:
    total_labels = int(summary.get("total_labels") or 0)
    attempted_labels = int(summary.get("attempted_labels") or 0)
    confirmed_fraud_labels = int(summary.get("confirmed_fraud_labels") or 0)
    high_risk_labels = int(summary.get("high_risk_labels") or 0)

    if metric_name == "false_negative_rate_high_risk":
        return high_risk_labels if track_name == "vat_refund" else confirmed_fraud_labels
    if metric_name in {
        "roi_positive_rate",
        "expected_vs_actual_uplift_ratio",
        "expected_vs_actual_recovery_ratio",
        "terminal_rate",
        "recovered_rate",
    }:
        return attempted_labels
    return total_labels


def _build_synthetic_model_version_predicate(column_name: str = "model_version") -> str:
    prefix_predicates = [
        f"LOWER(COALESCE({column_name}, '')) LIKE '{token}%'"
        for token in SYNTHETIC_MODEL_VERSION_TOKENS
        if token.endswith("-")
    ]
    contains_predicates = [
        f"LOWER(COALESCE({column_name}, '')) LIKE '%{token}%'"
        for token in SYNTHETIC_MODEL_VERSION_TOKENS
        if not token.endswith("-")
    ]
    predicates = [*prefix_predicates, *contains_predicates]
    return " OR ".join(predicates) if predicates else "FALSE"


def _build_label_origin_predicate(column_name: str, origins: tuple[str, ...]) -> str:
    normalized = [str(origin or "").strip().lower() for origin in origins if str(origin or "").strip()]
    if not normalized:
        return "FALSE"
    quoted = ", ".join("'{}'".format(origin.replace("'", "''")) for origin in normalized)
    return f"LOWER(COALESCE({column_name}, '')) IN ({quoted})"


def _ensure_data_reality_audit_table(cur) -> None:
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DATA_REALITY_AUDIT_TABLE} (
            id SERIAL PRIMARY KEY,
            source_endpoint VARCHAR(80) NOT NULL,
            status VARCHAR(20) NOT NULL,
            ready_for_real_ops BOOLEAN NOT NULL,
            hard_ready BOOLEAN NOT NULL,
            reasons JSONB,
            hard_checks JSONB,
            soft_checks JSONB,
            metrics JSONB,
            generated_at TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _persist_data_reality_audit_log(cur, *, source_endpoint: str, data_reality: dict[str, Any]) -> None:
    if not isinstance(data_reality, dict):
        return

    _ensure_data_reality_audit_table(cur)

    generated_at = _parse_iso_timestamp(str(data_reality.get("generated_at") or "")) or datetime.utcnow()
    cur.execute(
        f"""
        INSERT INTO {DATA_REALITY_AUDIT_TABLE} (
            source_endpoint,
            status,
            ready_for_real_ops,
            hard_ready,
            reasons,
            hard_checks,
            soft_checks,
            metrics,
            generated_at
        )
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
        """,
        (
            str(source_endpoint or "unknown"),
            str(data_reality.get("status") or "unknown"),
            bool(data_reality.get("ready_for_real_ops", False)),
            bool(data_reality.get("hard_ready", False)),
            json.dumps(data_reality.get("reasons") or [], ensure_ascii=False),
            json.dumps(data_reality.get("hard_checks") or {}, ensure_ascii=False),
            json.dumps(data_reality.get("soft_checks") or {}, ensure_ascii=False),
            json.dumps(data_reality.get("metrics") or {}, ensure_ascii=False),
            generated_at,
        ),
    )


def _compute_specialized_data_reality(cur) -> dict[str, Any]:
    assessment_columns = _table_columns(cur, "ai_risk_assessments")
    label_columns = _table_columns(cur, "inspector_labels")

    has_assessment_model_version = "model_version" in assessment_columns
    has_label_evidence = "evidence_summary" in label_columns
    has_label_inspector = "inspector_id" in label_columns
    has_label_assessment_ref = "assessment_id" in label_columns
    has_label_origin = "label_origin" in label_columns

    total_assessments = 0
    synthetic_assessments = 0
    missing_assessment_model_version = 0
    if has_assessment_model_version:
        synthetic_predicate = _build_synthetic_model_version_predicate("model_version")
        cur.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (
                    WHERE model_version IS NULL OR btrim(model_version) = ''
                ) AS missing_model_version,
                COUNT(*) FILTER (
                    WHERE {synthetic_predicate}
                ) AS synthetic_rows
            FROM ai_risk_assessments
            """
        )
        total_assessments, missing_assessment_model_version, synthetic_assessments = cur.fetchone()

    total_assessments = int(total_assessments or 0)
    synthetic_assessments = int(synthetic_assessments or 0)
    missing_assessment_model_version = int(missing_assessment_model_version or 0)

    total_labels = 0
    synthetic_labels_by_marker = 0
    synthetic_labels_by_origin = 0
    synthetic_labels_combined = 0
    labels_with_inspector = 0
    manual_origin_labels = 0
    if has_label_evidence or has_label_inspector or has_label_origin:
        marker_clauses = []
        if has_label_evidence:
            for marker in SYNTHETIC_LABEL_MARKERS:
                escaped = marker.replace("'", "''")
                marker_clauses.append(f"COALESCE(evidence_summary, '') ILIKE '{escaped}%'")
                marker_clauses.append(f"COALESCE(evidence_summary, '') ILIKE '%{escaped}%'")

        synthetic_marker_sql = " OR ".join(marker_clauses) if marker_clauses else "FALSE"
        inspector_sql = "inspector_id IS NOT NULL" if has_label_inspector else "FALSE"
        manual_origin_sql = _build_label_origin_predicate("label_origin", REAL_LABEL_ORIGINS) if has_label_origin else "FALSE"
        synthetic_origin_sql = _build_label_origin_predicate("label_origin", SYNTHETIC_LABEL_ORIGINS) if has_label_origin else "FALSE"
        synthetic_combined_sql = f"({synthetic_marker_sql}) OR ({synthetic_origin_sql})"
        manual_feedback_sql = f"({manual_origin_sql}) OR ({inspector_sql})"

        cur.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (WHERE {synthetic_marker_sql}) AS synthetic_marker_rows,
                COUNT(*) FILTER (WHERE {synthetic_origin_sql}) AS synthetic_origin_rows,
                COUNT(*) FILTER (WHERE {synthetic_combined_sql}) AS synthetic_combined_rows,
                COUNT(*) FILTER (WHERE {inspector_sql}) AS inspector_rows,
                COUNT(*) FILTER (WHERE {manual_feedback_sql}) AS manual_feedback_rows
            FROM inspector_labels
            """
        )
        (
            total_labels,
            synthetic_labels_by_marker,
            synthetic_labels_by_origin,
            synthetic_labels_combined,
            labels_with_inspector,
            manual_origin_labels,
        ) = cur.fetchone()

    total_labels = int(total_labels or 0)
    synthetic_labels_by_marker = int(synthetic_labels_by_marker or 0)
    synthetic_labels_by_origin = int(synthetic_labels_by_origin or 0)
    synthetic_labels_combined = int(synthetic_labels_combined or 0)
    labels_with_inspector = int(labels_with_inspector or 0)
    manual_origin_labels = int(manual_origin_labels or 0)

    synthetic_labels_by_link = 0
    if has_label_assessment_ref and has_assessment_model_version:
        synthetic_predicate = _build_synthetic_model_version_predicate("a.model_version")
        exclusion_clauses: list[str] = []
        if has_label_origin:
            exclusion_clauses.append(_build_label_origin_predicate("l.label_origin", REAL_LABEL_ORIGINS))
        if has_label_inspector:
            exclusion_clauses.append("l.inspector_id IS NOT NULL")

        exclusion_sql = " OR ".join(f"({clause})" for clause in exclusion_clauses) if exclusion_clauses else "FALSE"
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM inspector_labels l
            JOIN ai_risk_assessments a ON a.id = l.assessment_id
            WHERE {synthetic_predicate}
              AND NOT ({exclusion_sql})
            """
        )
        synthetic_labels_by_link = int((cur.fetchone() or [0])[0] or 0)

    synthetic_labels = max(synthetic_labels_combined, synthetic_labels_by_link)

    assessment_real_rows = max(0, total_assessments - synthetic_assessments)
    label_real_rows = max(0, total_labels - synthetic_labels)

    assessment_real_ratio = _safe_div(assessment_real_rows, total_assessments)
    label_real_ratio = _safe_div(label_real_rows, total_labels)
    manual_label_ratio = _safe_div(manual_origin_labels, total_labels)

    audit_meta = _safe_read_json(MODEL_DIR / "audit_value_model_meta.json")
    vat_meta = _safe_read_json(MODEL_DIR / "vat_refund_model_meta.json")
    audit_quality = _safe_read_json(MODEL_DIR / AUDIT_QUALITY_REPORT_FILE)
    vat_quality = _safe_read_json(MODEL_DIR / VAT_QUALITY_REPORT_FILE)

    audit_quality_pass = bool((audit_quality.get("acceptance_gates") or {}).get("overall_pass", False))
    vat_quality_pass = bool((vat_quality.get("acceptance_gates") or {}).get("overall_pass", False))

    audit_model_file = MODEL_DIR / "audit_value_model.joblib"
    vat_model_file = MODEL_DIR / "vat_refund_model.joblib"
    artifacts_ready = bool(audit_model_file.exists() and vat_model_file.exists())

    audit_samples = int(audit_meta.get("sample_count") or 0) if isinstance(audit_meta, dict) else 0
    vat_samples = int(vat_meta.get("sample_count") or 0) if isinstance(vat_meta, dict) else 0

    hard_checks = {
        "enough_observation_rows": bool(total_assessments >= 500 and total_labels >= 500),
        "non_synthetic_assessment_ratio": bool(assessment_real_ratio >= 0.80),
        "non_synthetic_label_ratio": bool(label_real_ratio >= 0.80),
        "model_artifacts_ready": artifacts_ready,
        "quality_gates_pass": bool(audit_quality_pass and vat_quality_pass),
        "trained_sample_volume_ok": bool(audit_samples >= 200 and vat_samples >= 200),
    }

    assessed_lineage_ratio = _safe_div(
        total_assessments - missing_assessment_model_version,
        total_assessments,
    ) if has_assessment_model_version else 1.0
    soft_checks = {
        "manual_feedback_present": bool(manual_origin_labels >= 50 or manual_label_ratio >= 0.10),
        "assessment_model_version_lineage": bool(assessed_lineage_ratio >= 0.80),
    }

    failed_hard = [name for name, passed in hard_checks.items() if not passed]
    failed_soft = [name for name, passed in soft_checks.items() if not passed]

    reasons: list[str] = []
    if "enough_observation_rows" in failed_hard:
        reasons.append("Chưa đủ số mẫu đánh giá (cần tối thiểu 500 assessments và 500 labels).")
    if "non_synthetic_assessment_ratio" in failed_hard:
        reasons.append("Tỷ lệ assessment synthetic còn cao, chưa đạt ngưỡng dữ liệu thật.")
    if "non_synthetic_label_ratio" in failed_hard:
        reasons.append("Tỷ lệ nhãn synthetic/bootstrap còn cao, cần bổ sung nhãn thực tế.")
    if "model_artifacts_ready" in failed_hard:
        reasons.append("Thiếu artifact model Audit/VAT đã train.")
    if "quality_gates_pass" in failed_hard:
        reasons.append("Quality gate Audit/VAT chưa pass hoàn toàn.")
    if "trained_sample_volume_ok" in failed_hard:
        reasons.append("Mẫu huấn luyện trong metadata model chưa đạt ngưỡng tối thiểu.")
    if "manual_feedback_present" in failed_soft:
        reasons.append("Nhãn do thanh tra nhập thực tế còn ít, cần tăng phản hồi hiện trường.")
    if "assessment_model_version_lineage" in failed_soft:
        reasons.append("Độ phủ model_version trên assessment còn thấp, cần hoàn thiện lineage.")

    hard_ready = len(failed_hard) == 0
    ready_for_real_ops = hard_ready and len(failed_soft) == 0
    status = "ready" if ready_for_real_ops else ("warning" if hard_ready else "blocked")

    return {
        "status": status,
        "ready_for_real_ops": ready_for_real_ops,
        "hard_ready": hard_ready,
        "hard_checks": hard_checks,
        "soft_checks": soft_checks,
        "reasons": reasons,
        "reason_count": len(reasons),
        "metrics": {
            "total_assessments": total_assessments,
            "synthetic_assessments": synthetic_assessments,
            "real_assessment_ratio": round(float(assessment_real_ratio), 4),
            "missing_assessment_model_version": missing_assessment_model_version,
            "total_labels": total_labels,
            "synthetic_labels": synthetic_labels,
            "synthetic_labels_by_marker": synthetic_labels_by_marker,
            "synthetic_labels_by_origin": synthetic_labels_by_origin,
            "synthetic_labels_by_link": synthetic_labels_by_link,
            "real_label_ratio": round(float(label_real_ratio), 4),
            "labels_with_inspector": labels_with_inspector,
            "manual_origin_labels": manual_origin_labels,
            "manual_label_ratio": round(float(manual_label_ratio), 4),
            "audit_model_samples": audit_samples,
            "vat_model_samples": vat_samples,
        },
        "artifacts": {
            "audit_model": bool(audit_model_file.exists()),
            "vat_model": bool(vat_model_file.exists()),
            "audit_quality_report": bool(audit_quality),
            "vat_quality_report": bool(vat_quality),
            "audit_quality_pass": audit_quality_pass,
            "vat_quality_pass": vat_quality_pass,
            "audit_trained_at": audit_meta.get("trained_at") if isinstance(audit_meta, dict) else None,
            "vat_trained_at": vat_meta.get("trained_at") if isinstance(vat_meta, dict) else None,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def _compute_intervention_effectiveness(
    cur,
    window_days: int = 90,
    top_k: int = 50,
    include_data_reality: bool = False,
) -> dict[str, Any]:
    data_reality = _compute_specialized_data_reality(cur) if include_data_reality else None
    columns = _table_columns(cur, "inspector_labels")
    required = {
        "intervention_action",
        "intervention_attempted",
        "outcome_status",
        "predicted_collection_uplift",
        "expected_recovery",
        "expected_net_recovery",
        "estimated_audit_cost",
        "actual_audit_cost",
        "amount_recovered",
    }
    missing_columns = sorted(required - columns)

    if missing_columns:
        payload = {
            "window_days": int(window_days),
            "top_k": int(top_k),
            "schema_ready": False,
            "missing_columns": missing_columns,
            "message": "Schema KPI chưa sẵn sàng. Cần chạy migration mới cho inspector_labels.",
            "generated_at": datetime.utcnow().isoformat(),
        }
        if data_reality is not None:
            payload["data_reality"] = data_reality
        return payload

    cur.execute(
        """
        SELECT
            COUNT(*) AS total_labels,
            COUNT(*) FILTER (WHERE intervention_attempted) AS attempted_labels,
            COUNT(*) FILTER (
                WHERE outcome_status IN ('recovered', 'partial_recovered', 'unrecoverable', 'dismissed')
            ) AS terminal_labels,
            COUNT(*) FILTER (WHERE COALESCE(amount_recovered, 0) > 0) AS positive_recovery_labels,
            SUM(COALESCE(predicted_collection_uplift, 0)) AS total_predicted_uplift,
            SUM(COALESCE(expected_recovery, 0)) AS total_expected_recovery,
            SUM(COALESCE(expected_net_recovery, 0)) AS total_expected_net_recovery,
            SUM(COALESCE(amount_recovered, 0)) AS total_actual_recovered,
            SUM(COALESCE(actual_audit_cost, estimated_audit_cost, 0)) AS total_audit_cost,
            COUNT(*) FILTER (
                WHERE intervention_attempted
                  AND (COALESCE(amount_recovered, 0) - COALESCE(actual_audit_cost, estimated_audit_cost, 0)) > 0
            ) AS roi_positive_labels,
            COUNT(*) FILTER (WHERE label_type = 'fraud_confirmed') AS confirmed_fraud_labels,
            COUNT(*) FILTER (
                WHERE label_type = 'fraud_confirmed'
                  AND COALESCE(intervention_action, 'monitor') IN ('monitor', 'auto_reminder')
            ) AS low_intensity_confirmed
        FROM inspector_labels
        WHERE created_at >= (NOW() - (%s || ' days')::interval)
        """,
        (int(window_days),),
    )
    (
        total_labels,
        attempted_labels,
        terminal_labels,
        positive_recovery_labels,
        total_predicted_uplift,
        total_expected_recovery,
        total_expected_net_recovery,
        total_actual_recovered,
        total_audit_cost,
        roi_positive_labels,
        confirmed_fraud_labels,
        low_intensity_confirmed,
    ) = cur.fetchone()

    total_labels = int(total_labels or 0)
    attempted_labels = int(attempted_labels or 0)
    terminal_labels = int(terminal_labels or 0)
    positive_recovery_labels = int(positive_recovery_labels or 0)
    total_predicted_uplift = float(total_predicted_uplift or 0)
    total_expected_recovery = float(total_expected_recovery or 0)
    total_expected_net_recovery = float(total_expected_net_recovery or 0)
    total_actual_recovered = float(total_actual_recovered or 0)
    total_audit_cost = float(total_audit_cost or 0)
    roi_positive_labels = int(roi_positive_labels or 0)
    confirmed_fraud_labels = int(confirmed_fraud_labels or 0)
    low_intensity_confirmed = int(low_intensity_confirmed or 0)

    cur.execute(
        """
        SELECT
            COALESCE(intervention_action, 'unassigned') AS action,
            COUNT(*) AS sample_count,
            COUNT(*) FILTER (
                WHERE COALESCE(amount_recovered, 0) > 0
                   OR label_type IN ('fraud_confirmed', 'delinquency_confirmed')
            ) AS positive_count,
            SUM(COALESCE(predicted_collection_uplift, 0)) AS expected_uplift,
            SUM(COALESCE(amount_recovered, 0)) AS actual_recovered,
            SUM(COALESCE(actual_audit_cost, estimated_audit_cost, 0)) AS audit_cost
        FROM inspector_labels
        WHERE created_at >= (NOW() - (%s || ' days')::interval)
        GROUP BY COALESCE(intervention_action, 'unassigned')
        ORDER BY sample_count DESC
        """,
        (int(window_days),),
    )
    by_action = []
    for action, sample_count, positive_count, expected_uplift, actual_recovered, audit_cost in cur.fetchall():
        sample_count = int(sample_count or 0)
        positive_count = int(positive_count or 0)
        expected_uplift = float(expected_uplift or 0)
        actual_recovered = float(actual_recovered or 0)
        audit_cost = float(audit_cost or 0)
        by_action.append(
            {
                "intervention_action": action,
                "sample_count": sample_count,
                "positive_count": positive_count,
                "precision": _safe_div(positive_count, sample_count),
                "expected_uplift": expected_uplift,
                "actual_recovered": actual_recovered,
                "uplift_realization_ratio": _safe_div(actual_recovered, expected_uplift),
                "avg_net_recovery": _safe_div(actual_recovered - audit_cost, sample_count),
            }
        )

    precision_at_top_k, top_k_count = _compute_precision_at_k(cur, window_days=window_days, k=top_k)
    false_negative_rate_high_risk = _safe_div(low_intensity_confirmed, confirmed_fraud_labels)

    payload = {
        "window_days": int(window_days),
        "top_k": int(top_k),
        "schema_ready": True,
        "summary": {
            "total_labels": total_labels,
            "attempted_labels": attempted_labels,
            "terminal_labels": terminal_labels,
            "positive_recovery_labels": positive_recovery_labels,
            "confirmed_fraud_labels": confirmed_fraud_labels,
        },
        "metrics": {
            "precision_at_top_k": precision_at_top_k,
            "precision_top_k_sample": top_k_count,
            "roi_positive_rate": _safe_div(roi_positive_labels, attempted_labels),
            "expected_vs_actual_uplift_ratio": _safe_div(total_actual_recovered, total_predicted_uplift),
            "expected_vs_actual_recovery_ratio": _safe_div(total_actual_recovered, total_expected_recovery),
            "false_negative_rate_high_risk": false_negative_rate_high_risk,
            "average_recovery_per_label": _safe_div(total_actual_recovered, total_labels),
            "total_predicted_uplift": total_predicted_uplift,
            "total_expected_recovery": total_expected_recovery,
            "total_expected_net_recovery": total_expected_net_recovery,
            "total_actual_recovered": total_actual_recovered,
            "total_audit_cost": total_audit_cost,
            "net_recovery": total_actual_recovered - total_audit_cost,
        },
        "by_intervention_action": by_action,
        "generated_at": datetime.utcnow().isoformat(),
    }
    if data_reality is not None:
        payload["data_reality"] = data_reality
    return payload


def _compute_vat_refund_effectiveness(
    cur,
    window_days: int = 90,
    top_k: int = 100,
    include_data_reality: bool = False,
) -> dict[str, Any]:
    data_reality = _compute_specialized_data_reality(cur) if include_data_reality else None
    columns = _table_columns(cur, "inspector_labels")
    required = {
        "label_type",
        "intervention_action",
        "intervention_attempted",
        "outcome_status",
        "predicted_collection_uplift",
        "expected_recovery",
        "expected_net_recovery",
        "estimated_audit_cost",
        "actual_audit_cost",
        "amount_recovered",
    }
    missing_columns = sorted(required - columns)

    if missing_columns:
        payload = {
            "window_days": int(window_days),
            "top_k": int(top_k),
            "schema_ready": False,
            "missing_columns": missing_columns,
            "message": "Schema KPI VAT chưa sẵn sàng. Cần chạy migration mới cho inspector_labels.",
            "generated_at": datetime.utcnow().isoformat(),
        }
        if data_reality is not None:
            payload["data_reality"] = data_reality
        return payload

    cur.execute(
        """
        WITH vat_cases AS (
            SELECT *, LOWER(COALESCE(label_type, '')) AS label_norm
            FROM inspector_labels
            WHERE created_at >= (NOW() - (%s || ' days')::interval)
              AND (
                          LOWER(COALESCE(label_type, '')) LIKE '%%vat%%'
                      OR LOWER(COALESCE(label_type, '')) LIKE '%%refund%%'
                      OR LOWER(COALESCE(label_type, '')) LIKE '%%invoice%%'
              )
        )
        SELECT
            COUNT(*) AS total_labels,
            COUNT(*) FILTER (WHERE intervention_attempted) AS attempted_labels,
            COUNT(*) FILTER (
                WHERE outcome_status IN ('recovered', 'partial_recovered', 'unrecoverable', 'dismissed')
            ) AS terminal_labels,
            COUNT(*) FILTER (
                WHERE COALESCE(amount_recovered, 0) > 0
                   OR outcome_status IN ('recovered', 'partial_recovered')
            ) AS recovered_labels,
            COUNT(*) FILTER (WHERE label_norm LIKE '%%confirmed%%') AS confirmed_vat_labels,
            COUNT(*) FILTER (
                WHERE label_norm LIKE '%%high_risk%%'
                   OR label_norm LIKE '%%confirmed%%'
            ) AS high_risk_labels,
            COUNT(*) FILTER (
                WHERE (label_norm LIKE '%%high_risk%%' OR label_norm LIKE '%%confirmed%%')
                  AND COALESCE(intervention_action, 'monitor') IN ('monitor', 'auto_reminder')
            ) AS low_intensity_high_risk,
            SUM(COALESCE(predicted_collection_uplift, 0)) AS total_predicted_uplift,
            SUM(COALESCE(expected_recovery, 0)) AS total_expected_recovery,
            SUM(COALESCE(expected_net_recovery, 0)) AS total_expected_net_recovery,
            SUM(COALESCE(amount_recovered, 0)) AS total_actual_recovered,
            SUM(COALESCE(actual_audit_cost, estimated_audit_cost, 0)) AS total_audit_cost,
            COUNT(*) FILTER (
                WHERE intervention_attempted
                  AND (COALESCE(amount_recovered, 0) - COALESCE(actual_audit_cost, estimated_audit_cost, 0)) > 0
            ) AS roi_positive_labels
        FROM vat_cases
        """,
        (int(window_days),),
    )
    (
        total_labels,
        attempted_labels,
        terminal_labels,
        recovered_labels,
        confirmed_vat_labels,
        high_risk_labels,
        low_intensity_high_risk,
        total_predicted_uplift,
        total_expected_recovery,
        total_expected_net_recovery,
        total_actual_recovered,
        total_audit_cost,
        roi_positive_labels,
    ) = cur.fetchone()

    total_labels = int(total_labels or 0)
    attempted_labels = int(attempted_labels or 0)
    terminal_labels = int(terminal_labels or 0)
    recovered_labels = int(recovered_labels or 0)
    confirmed_vat_labels = int(confirmed_vat_labels or 0)
    high_risk_labels = int(high_risk_labels or 0)
    low_intensity_high_risk = int(low_intensity_high_risk or 0)
    total_predicted_uplift = float(total_predicted_uplift or 0)
    total_expected_recovery = float(total_expected_recovery or 0)
    total_expected_net_recovery = float(total_expected_net_recovery or 0)
    total_actual_recovered = float(total_actual_recovered or 0)
    total_audit_cost = float(total_audit_cost or 0)
    roi_positive_labels = int(roi_positive_labels or 0)

    precision_at_top_k, top_k_count = _compute_vat_precision_at_k(cur, window_days=window_days, k=top_k)
    false_negative_rate_high_risk = _safe_div(low_intensity_high_risk, high_risk_labels)

    cur.execute(
        """
        WITH vat_cases AS (
            SELECT *, LOWER(COALESCE(label_type, '')) AS label_norm
            FROM inspector_labels
            WHERE created_at >= (NOW() - (%s || ' days')::interval)
              AND (
                          LOWER(COALESCE(label_type, '')) LIKE '%%vat%%'
                      OR LOWER(COALESCE(label_type, '')) LIKE '%%refund%%'
                      OR LOWER(COALESCE(label_type, '')) LIKE '%%invoice%%'
              )
        )
        SELECT
            COALESCE(intervention_action, 'unassigned') AS action,
            COUNT(*) AS sample_count,
            COUNT(*) FILTER (
                WHERE COALESCE(amount_recovered, 0) > 0
                   OR outcome_status IN ('recovered', 'partial_recovered')
                     OR label_norm LIKE '%%confirmed%%'
            ) AS positive_count,
            SUM(COALESCE(expected_recovery, 0)) AS expected_recovery,
            SUM(COALESCE(amount_recovered, 0)) AS actual_recovered,
            SUM(COALESCE(actual_audit_cost, estimated_audit_cost, 0)) AS audit_cost
        FROM vat_cases
        GROUP BY COALESCE(intervention_action, 'unassigned')
        ORDER BY sample_count DESC
        """,
        (int(window_days),),
    )
    by_action = []
    for action, sample_count, positive_count, expected_recovery, actual_recovered, audit_cost in cur.fetchall():
        sample_count = int(sample_count or 0)
        positive_count = int(positive_count or 0)
        expected_recovery = float(expected_recovery or 0)
        actual_recovered = float(actual_recovered or 0)
        audit_cost = float(audit_cost or 0)
        by_action.append(
            {
                "intervention_action": action,
                "sample_count": sample_count,
                "positive_count": positive_count,
                "precision": _safe_div(positive_count, sample_count),
                "expected_recovery": expected_recovery,
                "actual_recovered": actual_recovered,
                "recovery_realization_ratio": _safe_div(actual_recovered, expected_recovery),
                "avg_net_recovery": _safe_div(actual_recovered - audit_cost, sample_count),
            }
        )

    payload = {
        "window_days": int(window_days),
        "top_k": int(top_k),
        "schema_ready": True,
        "label_scope": {
            "match_patterns": ["%vat%", "%refund%", "%invoice%"],
        },
        "summary": {
            "total_labels": total_labels,
            "attempted_labels": attempted_labels,
            "terminal_labels": terminal_labels,
            "recovered_labels": recovered_labels,
            "confirmed_vat_labels": confirmed_vat_labels,
            "high_risk_labels": high_risk_labels,
        },
        "metrics": {
            "precision_at_top_k": precision_at_top_k,
            "precision_top_k_sample": top_k_count,
            "false_negative_rate_high_risk": false_negative_rate_high_risk,
            "roi_positive_rate": _safe_div(roi_positive_labels, attempted_labels),
            "expected_vs_actual_uplift_ratio": _safe_div(total_actual_recovered, total_predicted_uplift),
            "expected_vs_actual_recovery_ratio": _safe_div(total_actual_recovered, total_expected_recovery),
            "total_predicted_uplift": total_predicted_uplift,
            "total_expected_recovery": total_expected_recovery,
            "total_expected_net_recovery": total_expected_net_recovery,
            "total_actual_recovered": total_actual_recovered,
            "total_audit_cost": total_audit_cost,
            "net_recovery": total_actual_recovered - total_audit_cost,
        },
        "by_intervention_action": by_action,
        "generated_at": datetime.utcnow().isoformat(),
    }
    if data_reality is not None:
        payload["data_reality"] = data_reality
    return payload


@router.get("/vat_effectiveness")
@router.get("/vat_refund_effectiveness")
def get_vat_refund_effectiveness(window_days: int = 90, top_k: int = 100):
    """
    Operational KPI snapshot for VAT refund effectiveness.
    Focuses on VAT-labelled inspector outcomes and false-negative control.
    """
    window_days = max(7, min(int(window_days or 90), 365))
    top_k = max(10, min(int(top_k or 100), 500))
    db_url = _resolve_db_url()

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        payload = _compute_vat_refund_effectiveness(
            cur,
            window_days=window_days,
            top_k=top_k,
            include_data_reality=True,
        )

        data_reality = payload.get("data_reality") if isinstance(payload, dict) else None
        if isinstance(data_reality, dict):
            try:
                _persist_data_reality_audit_log(
                    cur,
                    source_endpoint="vat_refund_effectiveness",
                    data_reality=data_reality,
                )
                conn.commit()
            except Exception:
                conn.rollback()

        cur.close()
        conn.close()
        return payload
    except Exception as e:
        return {
            "schema_ready": False,
            "window_days": window_days,
            "top_k": top_k,
            "message": f"Không thể tính KPI vat refund effectiveness: {str(e)}",
            "generated_at": datetime.utcnow().isoformat(),
        }


def _build_split_trigger_status_payload(cur, policies: list[dict[str, Any]]) -> dict[str, Any]:
    max_window = max(int(p.get("window_days", 28)) for p in policies) if policies else 28
    max_top_k = 50
    for policy in policies:
        metric_name = str(policy.get("metric_name") or "")
        if metric_name.startswith("precision_top_"):
            try:
                max_top_k = max(max_top_k, int(metric_name.replace("precision_top_", "")))
            except ValueError:
                continue

    intervention_effectiveness = _compute_intervention_effectiveness(cur, window_days=max_window, top_k=max_top_k)
    if not intervention_effectiveness.get("schema_ready"):
        return {
            "ready": False,
            "schema_ready": False,
            "reason": intervention_effectiveness.get("message", "Schema KPI chưa sẵn sàng."),
            "track_status": {},
            "totals": {
                "enabled_rules": 0,
                "passed_rules": 0,
            },
            "window_days": max_window,
            "generated_at": datetime.utcnow().isoformat(),
        }

    audit_effectiveness = _compute_audit_value_effectiveness(cur, window_days=max_window, top_k=max_top_k)
    vat_effectiveness = _compute_vat_refund_effectiveness(cur, window_days=max_window, top_k=max_top_k)

    intervention_metrics = dict(intervention_effectiveness.get("metrics") or {})
    audit_metrics = dict(audit_effectiveness.get("metrics") or {}) if audit_effectiveness.get("schema_ready") else {}
    vat_metrics = dict(vat_effectiveness.get("metrics") or {}) if vat_effectiveness.get("schema_ready") else {}

    metrics_by_track: dict[str, dict[str, Any]] = {
        "intervention": intervention_metrics,
        "audit_value": audit_metrics,
        "vat_refund": vat_metrics,
    }
    summary_by_track: dict[str, dict[str, Any]] = {
        "intervention": dict(intervention_effectiveness.get("summary") or {}),
        "audit_value": dict(audit_effectiveness.get("summary") or {}),
        "vat_refund": dict(vat_effectiveness.get("summary") or {}),
    }

    precision_cache: dict[int, tuple[Optional[float], int]] = {}
    vat_precision_cache: dict[int, tuple[Optional[float], int]] = {}
    recent_failures = _fetch_recent_rule_failures(cur, policies)
    now_utc = datetime.utcnow()
    track_rules: dict[str, list[dict[str, Any]]] = {}
    total_enabled = 0
    total_passed = 0

    for policy in policies:
        track_name = str(policy.get("track_name") or "unknown")
        metric_name = str(policy.get("metric_name") or "")
        comparator = str(policy.get("comparator") or ">=")
        threshold = float(policy.get("threshold") or 0.0)
        min_sample = int(policy.get("min_sample") or 0)
        window_days = int(policy.get("window_days") or max_window)
        cooldown_days = max(0, int(policy.get("cooldown_days") or 0))
        enabled = bool(policy.get("enabled", True))

        if metric_name.startswith("precision_top_"):
            try:
                k = int(metric_name.replace("precision_top_", ""))
            except ValueError:
                k = 50

            if track_name == "vat_refund":
                if k not in vat_precision_cache:
                    vat_precision_cache[k] = _compute_vat_precision_at_k(cur, window_days=window_days, k=k)
                actual, sample_size = vat_precision_cache[k]
            else:
                if k not in precision_cache:
                    precision_cache[k] = _compute_precision_at_k(cur, window_days=window_days, k=k)
                actual, sample_size = precision_cache[k]
        else:
            if track_name in metrics_by_track and metrics_by_track.get(track_name):
                track_metrics = metrics_by_track.get(track_name) or {}
                track_summary = summary_by_track.get(track_name) or {}
            else:
                track_metrics = metrics_by_track.get("intervention") or {}
                track_summary = summary_by_track.get("intervention") or {}

            if metric_name == "recovery_per_case":
                actual = track_metrics.get("average_recovery_per_label")
            else:
                actual = track_metrics.get(metric_name)

            sample_size = _estimate_metric_sample_size(track_name, metric_name, track_summary)

        status = "disabled"
        passed = None
        cooldown_active = False
        cooldown_until: Optional[datetime] = None
        cooldown_remaining_days = 0
        last_failed_at: Optional[datetime] = None

        if enabled:
            total_enabled += 1
            if sample_size < min_sample:
                status = "insufficient_data"
            else:
                passed = _evaluate_threshold(actual, comparator, threshold)
                if passed is True:
                    status = "pass"
                elif passed is False:
                    status = "fail"
                else:
                    status = "no_metric"

            if status == "pass" and cooldown_days > 0:
                policy_key = (_normalize_policy_token(track_name) or "unknown", _normalize_policy_token(metric_name) or "")
                recent_failure_at = recent_failures.get(policy_key)
                if isinstance(recent_failure_at, datetime):
                    maybe_cooldown_until = recent_failure_at + timedelta(days=cooldown_days)
                    if maybe_cooldown_until > now_utc:
                        cooldown_active = True
                        last_failed_at = recent_failure_at
                        cooldown_until = maybe_cooldown_until
                        cooldown_remaining_days = max(
                            1,
                            math.ceil((cooldown_until - now_utc).total_seconds() / 86400),
                        )
                        status = "cooldown_active"
                        passed = False

            if status == "pass":
                total_passed += 1

        track_rules.setdefault(track_name, []).append(
            {
                **policy,
                "actual": actual,
                "sample_size": sample_size,
                "status": status,
                "passed": passed,
                "cooldown_active": cooldown_active,
                "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
                "cooldown_remaining_days": cooldown_remaining_days,
                "last_failed_at": last_failed_at.isoformat() if last_failed_at else None,
            }
        )

    track_status: dict[str, dict[str, Any]] = {}
    for track_name, rules in track_rules.items():
        enabled_rules = [r for r in rules if r.get("enabled")]
        ready = bool(enabled_rules) and all(r.get("status") == "pass" for r in enabled_rules)
        blocking_rules = [r for r in enabled_rules if r.get("status") in KPI_BLOCKING_STATUSES]
        track_status[track_name] = {
            "ready_for_split": ready,
            "enabled_rule_count": len(enabled_rules),
            "blocking_rule_count": len(blocking_rules),
            "rules": rules,
        }

    readiness_score = round((_safe_div(total_passed, total_enabled) or 0) * 100, 1)
    critical_tracks = ["audit_value", "vat_refund"]
    present_critical_tracks = [track for track in critical_tracks if track in track_status]
    split_recommendation = bool(present_critical_tracks) and all(
        track_status[track].get("ready_for_split", False)
        for track in present_critical_tracks
    )

    return {
        "ready": split_recommendation,
        "schema_ready": True,
        "readiness_score": readiness_score,
        "critical_tracks": critical_tracks,
        "track_status": track_status,
        "totals": {
            "enabled_rules": total_enabled,
            "passed_rules": total_passed,
        },
        "window_days": max_window,
        "generated_at": datetime.utcnow().isoformat(),
    }


def _persist_kpi_snapshots(
    cur,
    split_status_payload: dict[str, Any],
    source: str = "split_trigger_status",
) -> int:
    if not _table_columns(cur, "kpi_metric_snapshots"):
        return 0

    track_status = split_status_payload.get("track_status")
    if not isinstance(track_status, dict):
        return 0

    generated_at = _parse_iso_timestamp(str(split_status_payload.get("generated_at") or ""))
    if generated_at is None:
        generated_at = datetime.utcnow()

    snapshot_source = _normalize_snapshot_source(source)
    inserted = 0

    for raw_track_name, track_payload in track_status.items():
        if not isinstance(track_payload, dict):
            continue
        track_name = _normalize_policy_token(raw_track_name) or "unknown"
        ready_for_split = bool(track_payload.get("ready_for_split", False))
        rules = track_payload.get("rules")
        if not isinstance(rules, list):
            continue

        for rule in rules:
            if not isinstance(rule, dict):
                continue

            metric_name = _normalize_policy_token(rule.get("metric_name"))
            if not metric_name:
                continue

            actual_raw = rule.get("actual")
            metric_value = float(actual_raw) if isinstance(actual_raw, (int, float)) else None
            sample_size = max(0, int(rule.get("sample_size") or 0))
            comparator = str(rule.get("comparator") or "").strip() or None
            threshold = float(rule.get("threshold")) if isinstance(rule.get("threshold"), (int, float)) else None
            status = str(rule.get("status") or "no_metric").strip().lower()
            if status not in {"pass", "fail", "insufficient_data", "no_metric", "disabled", "cooldown_active"}:
                status = "no_metric"

            window_days = int(rule.get("window_days") or split_status_payload.get("window_days") or 28)
            window_days = max(1, min(365, window_days))

            details_payload = {
                "enabled": bool(rule.get("enabled", True)),
                "passed": rule.get("passed"),
                "min_sample": int(rule.get("min_sample") or 0),
                "ready_for_split": ready_for_split,
                "readiness_score": split_status_payload.get("readiness_score"),
                "cooldown_active": bool(rule.get("cooldown_active", False)),
                "cooldown_days": int(rule.get("cooldown_days") or 0),
                "cooldown_remaining_days": int(rule.get("cooldown_remaining_days") or 0),
                "cooldown_until": rule.get("cooldown_until"),
                "last_failed_at": rule.get("last_failed_at"),
            }

            cur.execute(
                """
                INSERT INTO kpi_metric_snapshots (
                    track_name, metric_name, metric_value, sample_size,
                    comparator, threshold, status, window_days,
                    source, details, generated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    track_name,
                    metric_name,
                    metric_value,
                    sample_size,
                    comparator,
                    threshold,
                    status,
                    window_days,
                    snapshot_source,
                    json.dumps(details_payload, ensure_ascii=False),
                    generated_at,
                ),
            )
            inserted += 1

    return inserted


def _decode_json_blob(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return None


@router.get("/inspector_label_stats")
def get_inspector_label_stats():
    """
    Inspector label statistics for model retraining readiness.
    Shows distribution, recency, and basic outcome coverage signals.
    """
    db_url = _resolve_db_url()

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT label_type, confidence, COUNT(*) as cnt
            FROM inspector_labels
            GROUP BY label_type, confidence
            ORDER BY cnt DESC
            """
        )
        label_dist = [{"label_type": r[0], "confidence": r[1], "count": int(r[2])} for r in cur.fetchall()]

        cur.execute(
            """
            SELECT decision, COUNT(*) as cnt
            FROM inspector_labels
            WHERE decision IS NOT NULL
            GROUP BY decision
            ORDER BY cnt DESC
            """
        )
        decision_dist = [{"decision": r[0], "count": int(r[1])} for r in cur.fetchall()]

        cur.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT tax_code), MAX(created_at)
            FROM inspector_labels
            """
        )
        total, companies, latest = cur.fetchone()

        cur.execute(
            """
            SELECT
                SUM(COALESCE(amount_recovered, 0)),
                COUNT(*) FILTER (WHERE intervention_attempted),
                COUNT(*) FILTER (WHERE outcome_status IN ('recovered', 'partial_recovered', 'unrecoverable', 'dismissed'))
            FROM inspector_labels
            """
        )
        total_recovered, attempted_count, terminal_count = cur.fetchone()

        cur.close()
        conn.close()

        total = int(total or 0)
        companies = int(companies or 0)
        attempted_count = int(attempted_count or 0)
        terminal_count = int(terminal_count or 0)
        retrain_ready = total >= 50

        return {
            "total_labels": total,
            "distinct_companies": companies,
            "latest_label": str(latest) if latest else None,
            "total_amount_recovered": float(total_recovered or 0),
            "attempted_labels": attempted_count,
            "terminal_outcomes": terminal_count,
            "retrain_ready": retrain_ready,
            "retrain_message": "Đủ dữ liệu để retrain model." if retrain_ready else f"Cần thêm {50 - total} nhãn.",
            "label_distribution": label_dist,
            "decision_distribution": decision_dist,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "total_labels": 0,
            "retrain_ready": False,
            "retrain_message": f"Lỗi kiểm tra: {str(e)}",
            "label_distribution": [],
            "generated_at": datetime.utcnow().isoformat(),
        }


@router.get("/intervention_effectiveness")
def get_intervention_effectiveness(window_days: int = 90, top_k: int = 50):
    """
    Operational KPI snapshot for intervention and recovery effectiveness.
    Used as a readiness gate before split-trigger decisions.
    """
    window_days = max(7, min(int(window_days or 90), 365))
    top_k = max(10, min(int(top_k or 50), 500))
    db_url = _resolve_db_url()

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        payload = _compute_intervention_effectiveness(
            cur,
            window_days=window_days,
            top_k=top_k,
            include_data_reality=True,
        )

        data_reality = payload.get("data_reality") if isinstance(payload, dict) else None
        if isinstance(data_reality, dict):
            try:
                _persist_data_reality_audit_log(
                    cur,
                    source_endpoint="intervention_effectiveness",
                    data_reality=data_reality,
                )
                conn.commit()
            except Exception:
                conn.rollback()

        cur.close()
        conn.close()
        return payload
    except Exception as e:
        return {
            "schema_ready": False,
            "window_days": window_days,
            "top_k": top_k,
            "message": f"Không thể tính KPI effectiveness: {str(e)}",
            "generated_at": datetime.utcnow().isoformat(),
        }


def _compute_audit_value_effectiveness(
    cur,
    window_days: int = 90,
    top_k: int = 50,
    include_data_reality: bool = False,
) -> dict[str, Any]:
    data_reality = _compute_specialized_data_reality(cur) if include_data_reality else None
    columns = _table_columns(cur, "inspector_labels")
    required = {
        "expected_recovery",
        "expected_net_recovery",
        "estimated_audit_cost",
        "actual_audit_cost",
        "amount_recovered",
        "intervention_action",
        "intervention_attempted",
        "outcome_status",
    }
    missing_columns = sorted(required - columns)

    if missing_columns:
        payload = {
            "schema_ready": False,
            "window_days": int(window_days),
            "top_k": int(top_k),
            "missing_columns": missing_columns,
            "message": "Schema KPI audit value chưa sẵn sàng. Cần chạy migration mới cho inspector_labels.",
            "generated_at": datetime.utcnow().isoformat(),
        }
        if data_reality is not None:
            payload["data_reality"] = data_reality
        return payload

    cur.execute(
        """
        SELECT
            COUNT(*) AS total_labels,
            COUNT(*) FILTER (WHERE expected_recovery IS NOT NULL) AS expected_labeled,
            COUNT(*) FILTER (WHERE intervention_attempted) AS attempted_labels,
            COUNT(*) FILTER (
                WHERE outcome_status IN ('recovered', 'partial_recovered', 'unrecoverable', 'dismissed')
            ) AS terminal_labels,
            COUNT(*) FILTER (
                WHERE COALESCE(amount_recovered, 0) > 0
                   OR outcome_status IN ('recovered', 'partial_recovered')
            ) AS recovered_labels,
            SUM(COALESCE(expected_recovery, 0)) AS total_expected_recovery,
            SUM(COALESCE(expected_net_recovery, 0)) AS total_expected_net_recovery,
            SUM(COALESCE(amount_recovered, 0)) AS total_actual_recovered,
            SUM(COALESCE(actual_audit_cost, estimated_audit_cost, 0)) AS total_audit_cost,
            COUNT(*) FILTER (
                WHERE intervention_attempted
                  AND (COALESCE(amount_recovered, 0) - COALESCE(actual_audit_cost, estimated_audit_cost, 0)) > 0
            ) AS roi_positive_labels
        FROM inspector_labels
        WHERE created_at >= (NOW() - (%s || ' days')::interval)
        """,
        (int(window_days),),
    )
    (
        total_labels,
        expected_labeled,
        attempted_labels,
        terminal_labels,
        recovered_labels,
        total_expected_recovery,
        total_expected_net_recovery,
        total_actual_recovered,
        total_audit_cost,
        roi_positive_labels,
    ) = cur.fetchone()

    total_labels = int(total_labels or 0)
    expected_labeled = int(expected_labeled or 0)
    attempted_labels = int(attempted_labels or 0)
    terminal_labels = int(terminal_labels or 0)
    recovered_labels = int(recovered_labels or 0)
    total_expected_recovery = float(total_expected_recovery or 0)
    total_expected_net_recovery = float(total_expected_net_recovery or 0)
    total_actual_recovered = float(total_actual_recovered or 0)
    total_audit_cost = float(total_audit_cost or 0)
    roi_positive_labels = int(roi_positive_labels or 0)

    precision_at_top_k, top_k_sample = _compute_precision_at_k(cur, window_days=window_days, k=top_k)

    cur.execute(
        """
        SELECT
            COALESCE(intervention_action, 'unassigned') AS lane,
            COUNT(*) AS sample_count,
            COUNT(*) FILTER (
                WHERE COALESCE(amount_recovered, 0) > 0
                   OR outcome_status IN ('recovered', 'partial_recovered')
            ) AS recovered_count,
            SUM(COALESCE(expected_recovery, 0)) AS expected_recovery,
            SUM(COALESCE(amount_recovered, 0)) AS actual_recovered,
            SUM(COALESCE(actual_audit_cost, estimated_audit_cost, 0)) AS audit_cost
        FROM inspector_labels
        WHERE created_at >= (NOW() - (%s || ' days')::interval)
        GROUP BY COALESCE(intervention_action, 'unassigned')
        ORDER BY sample_count DESC
        """,
        (int(window_days),),
    )
    by_lane = []
    for lane, sample_count, recovered_count, expected_recovery, actual_recovered, audit_cost in cur.fetchall():
        sample_count = int(sample_count or 0)
        recovered_count = int(recovered_count or 0)
        expected_recovery = float(expected_recovery or 0)
        actual_recovered = float(actual_recovered or 0)
        audit_cost = float(audit_cost or 0)
        by_lane.append(
            {
                "lane": lane,
                "sample_count": sample_count,
                "recovered_count": recovered_count,
                "success_rate": _safe_div(recovered_count, sample_count),
                "expected_recovery": expected_recovery,
                "actual_recovered": actual_recovered,
                "recovery_realization_ratio": _safe_div(actual_recovered, expected_recovery),
                "avg_net_recovery": _safe_div(actual_recovered - audit_cost, sample_count),
            }
        )

    payload = {
        "schema_ready": True,
        "window_days": int(window_days),
        "top_k": int(top_k),
        "summary": {
            "total_labels": total_labels,
            "expected_labeled": expected_labeled,
            "attempted_labels": attempted_labels,
            "terminal_labels": terminal_labels,
            "recovered_labels": recovered_labels,
        },
        "metrics": {
            "precision_at_top_k": precision_at_top_k,
            "precision_top_k_sample": top_k_sample,
            "attempt_rate": _safe_div(attempted_labels, total_labels),
            "terminal_rate": _safe_div(terminal_labels, attempted_labels),
            "recovered_rate": _safe_div(recovered_labels, attempted_labels),
            "roi_positive_rate": _safe_div(roi_positive_labels, attempted_labels),
            "expected_vs_actual_recovery_ratio": _safe_div(total_actual_recovered, total_expected_recovery),
            "total_expected_recovery": total_expected_recovery,
            "total_expected_net_recovery": total_expected_net_recovery,
            "total_actual_recovered": total_actual_recovered,
            "total_audit_cost": total_audit_cost,
            "net_recovery": total_actual_recovered - total_audit_cost,
        },
        "by_lane": by_lane,
        "generated_at": datetime.utcnow().isoformat(),
    }
    if data_reality is not None:
        payload["data_reality"] = data_reality
    return payload


@router.get("/audit_value_effectiveness")
def get_audit_value_effectiveness(window_days: int = 90, top_k: int = 50):
    """
    Operational KPI snapshot for Audit Value lane effectiveness.
    Tracks realization of expected recovery vs actual recovery by intervention lane.
    """
    window_days = max(7, min(int(window_days or 90), 365))
    top_k = max(10, min(int(top_k or 50), 500))
    db_url = _resolve_db_url()

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        payload = _compute_audit_value_effectiveness(
            cur,
            window_days=window_days,
            top_k=top_k,
            include_data_reality=True,
        )

        data_reality = payload.get("data_reality") if isinstance(payload, dict) else None
        if isinstance(data_reality, dict):
            try:
                _persist_data_reality_audit_log(
                    cur,
                    source_endpoint="audit_value_effectiveness",
                    data_reality=data_reality,
                )
                conn.commit()
            except Exception:
                conn.rollback()

        cur.close()
        conn.close()
        return payload
    except Exception as e:
        return {
            "schema_ready": False,
            "window_days": window_days,
            "top_k": top_k,
            "message": f"Không thể tính KPI audit value effectiveness: {str(e)}",
            "generated_at": datetime.utcnow().isoformat(),
        }


@router.get("/kpi_policy")
def get_kpi_policy():
    """
    Return configured KPI split-trigger policies.
    Falls back to baked-in defaults when policy table is empty/unavailable.
    """
    db_url = _resolve_db_url()

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        has_table = bool(_table_columns(cur, "kpi_trigger_policies"))
        policies = _fetch_kpi_policies(cur)
        source = "database" if has_table else "default"
        return {
            "source": source,
            "policies": policies,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "source": "default",
            "policies": _default_kpi_policies(),
            "warning": f"Policy DB unavailable: {str(e)}",
            "generated_at": datetime.utcnow().isoformat(),
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@router.put("/kpi_policy")
def upsert_kpi_policy(
    request: Request,
    payload: KPIPolicyUpsertRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """
    Upsert KPI split-trigger policies.
    Set replace_existing=true to replace the entire policy set.
    """
    role = _normalize_role(getattr(current_user, "role", ""))
    if role not in ADMIN_POLICY_ROLES:
        _safe_log_monitoring_audit(
            db,
            action="KPI_POLICY_UPDATE_DENIED",
            request=request,
            current_user=current_user,
            detail="User role is not authorized to update KPI policy.",
        )
    _require_policy_admin(current_user)
    normalized_policies = _normalize_kpi_policy_payload(payload.policies)
    db_url = _resolve_db_url()
    audit_action = "KPI_POLICY_UPDATE_FAILED"
    audit_detail = f"replace_existing={bool(payload.replace_existing)} policies={len(normalized_policies)}"

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        if not _table_columns(cur, "kpi_trigger_policies"):
            raise HTTPException(
                status_code=503,
                detail="kpi_trigger_policies chưa tồn tại. Cần chạy migration trước.",
            )

        updated_count = _upsert_kpi_policies(
            cur,
            normalized_policies,
            replace_existing=bool(payload.replace_existing),
        )
        conn.commit()
        policies = _fetch_kpi_policies(cur)
        audit_action = "KPI_POLICY_UPDATE"
        audit_detail = f"updated_count={updated_count} replace_existing={bool(payload.replace_existing)}"

        return {
            "updated_count": updated_count,
            "replace_existing": bool(payload.replace_existing),
            "policies": policies,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise
    except Exception as e:
        if conn is not None:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Không thể cập nhật KPI policy: {str(e)}")
    finally:
        _safe_log_monitoring_audit(
            db,
            action=audit_action,
            request=request,
            current_user=current_user,
            detail=audit_detail,
        )
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def get_split_trigger_status_snapshot(
    persist_snapshot: bool = False,
    snapshot_source: str = "split_trigger_status",
) -> dict[str, Any]:
    """
    Shared helper for split-trigger readiness payload so non-monitoring routers
    can attach the same governance context to their own responses.
    """
    db_url = _resolve_db_url()

    conn = None
    cur = None

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        policies = _fetch_kpi_policies(cur)
        payload = _build_split_trigger_status_payload(cur, policies)

        if persist_snapshot:
            snapshots_captured = _persist_kpi_snapshots(
                cur,
                payload,
                source=snapshot_source,
            )
            conn.commit()
            payload["snapshots_captured"] = snapshots_captured

        return payload
    except Exception as e:
        return {
            "ready": False,
            "schema_ready": False,
            "readiness_score": 0,
            "reason": f"Không thể đánh giá split-trigger: {str(e)}",
            "track_status": {},
            "totals": {
                "enabled_rules": 0,
                "passed_rules": 0,
            },
            "generated_at": datetime.utcnow().isoformat(),
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@router.get("/split_trigger_status")
def get_split_trigger_status(
    request: Request,
    persist_snapshot: bool = False,
    snapshot_source: str = "split_trigger_status",
    db: Session = Depends(get_db),
):
    """
    Evaluate split-trigger readiness using KPI policies and current outcomes.
    """
    current_user: Optional[models.User] = None
    if persist_snapshot:
        try:
            current_user = auth.get_current_user(request=request, db=db)
        except HTTPException:
            _safe_log_monitoring_audit(
                db,
                action="KPI_SNAPSHOT_CAPTURE_DENIED",
                request=request,
                current_user=None,
                detail="Unauthenticated request tried to persist split-trigger snapshots.",
            )
            raise

        role = _normalize_role(getattr(current_user, "role", ""))
        if role not in ADMIN_POLICY_ROLES:
            _safe_log_monitoring_audit(
                db,
                action="KPI_SNAPSHOT_CAPTURE_DENIED",
                request=request,
                current_user=current_user,
                detail="User role is not authorized to persist split-trigger snapshots.",
            )
        _require_policy_admin(current_user)
    payload = get_split_trigger_status_snapshot(
        persist_snapshot=persist_snapshot,
        snapshot_source=snapshot_source,
    )
    if persist_snapshot:
        _safe_log_monitoring_audit(
            db,
            action="KPI_SNAPSHOT_CAPTURE",
            request=request,
            current_user=current_user,
            detail=f"snapshot_source={snapshot_source} via=split_trigger_status",
        )
    return payload


def _build_specialized_rollout_status_payload(include_split_snapshot: bool = True) -> dict[str, Any]:
    audit_quality = _artifact_report(MODEL_DIR / AUDIT_QUALITY_REPORT_FILE)
    vat_quality = _artifact_report(MODEL_DIR / VAT_QUALITY_REPORT_FILE)
    pilot_report = _artifact_report(MODEL_DIR / SPECIALIZED_PILOT_REPORT_FILE)
    go_no_go_report = _artifact_report(MODEL_DIR / SPECIALIZED_GO_NO_GO_REPORT_FILE)

    audit_quality_payload = audit_quality["payload"]
    vat_quality_payload = vat_quality["payload"]
    pilot_payload = pilot_report["payload"]
    go_no_go_payload = go_no_go_report["payload"]

    audit_quality_pass = bool((audit_quality_payload.get("acceptance_gates") or {}).get("overall_pass", False))
    vat_quality_pass = bool((vat_quality_payload.get("acceptance_gates") or {}).get("overall_pass", False))

    audit_track = _track_payload(pilot_payload, "audit_value")
    vat_track = _track_payload(pilot_payload, "vat_refund")

    audit_delta = _float_or_none(((audit_track.get("delta_model_minus_heuristic") or {}).get("f1_delta")))
    vat_delta = _float_or_none(((vat_track.get("delta_model_minus_heuristic") or {}).get("f1_delta")))
    audit_samples = _int_or_none(audit_track.get("samples_evaluated"))
    vat_samples = _int_or_none(vat_track.get("samples_evaluated"))

    go_no_go_summary = go_no_go_payload.get("summary")
    go_no_go_summary = go_no_go_summary if isinstance(go_no_go_summary, dict) else {}

    go_no_go_decision = go_no_go_payload.get("decision")
    go_no_go_decision = go_no_go_decision if isinstance(go_no_go_decision, dict) else {}

    hard_gates_pass = bool(go_no_go_summary.get("hard_gates_pass", False))
    split_gate_pass = bool(go_no_go_summary.get("split_gate_pass", False))
    stability_gate_pass = bool(go_no_go_summary.get("stability_gate_pass", False))
    soft_gates_pass = bool(split_gate_pass and stability_gate_pass)

    decision_status = str(go_no_go_decision.get("status") or "unavailable")
    go_live_phase_d = bool(go_no_go_decision.get("go_live_phase_d", False))

    split_snapshot = None
    if include_split_snapshot:
        split_snapshot = get_split_trigger_status_snapshot(
            persist_snapshot=False,
            snapshot_source="specialized_rollout_status",
        )

    availability_flags = {
        "audit_quality": bool(audit_quality.get("exists")),
        "vat_quality": bool(vat_quality.get("exists")),
        "pilot_report": bool(pilot_report.get("exists")),
        "go_no_go_report": bool(go_no_go_report.get("exists")),
    }
    available = all(availability_flags.values())

    if decision_status == "go_phase_d_candidate":
        rollout_status = "ready_for_phase_d"
    elif decision_status == "conditional_go_continue_integrated_first":
        rollout_status = "conditional_go"
    elif decision_status == "no_go_tune_models_or_data":
        rollout_status = "no_go"
    elif available:
        rollout_status = "review_required"
    else:
        rollout_status = "insufficient_artifacts"

    recommended_actions = go_no_go_decision.get("recommended_actions")
    if not isinstance(recommended_actions, list) or not recommended_actions:
        if rollout_status == "ready_for_phase_d":
            recommended_actions = [
                "Open Phase D scope with controlled rollout batches.",
                "Keep weekly pilot checks active during first release month.",
            ]
        elif rollout_status == "conditional_go":
            recommended_actions = [
                "Continue integrated-first mode and collect additional pilot cycles.",
                "Re-evaluate split/stability soft gates after next cycle.",
            ]
        elif rollout_status == "no_go":
            recommended_actions = [
                "Tune model/data quality on failing tracks and retrain.",
                "Re-run pilot before next go/no-go review.",
            ]
        else:
            recommended_actions = [
                "Generate missing quality/pilot/go-no-go artifacts via specialized pipeline.",
                "Re-open rollout review when artifacts are complete.",
            ]

    data_reality = None
    try:
        db_url = _resolve_db_url()
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        data_reality = _compute_specialized_data_reality(cur)
        if isinstance(data_reality, dict):
            try:
                _persist_data_reality_audit_log(
                    cur,
                    source_endpoint="specialized_rollout_status",
                    data_reality=data_reality,
                )
                conn.commit()
            except Exception:
                conn.rollback()
        cur.close()
        conn.close()
    except Exception as exc:
        data_reality = {
            "status": "error",
            "ready_for_real_ops": False,
            "hard_ready": False,
            "hard_checks": {},
            "soft_checks": {},
            "reasons": [f"Không thể kiểm tra dữ liệu train thật: {str(exc)}"],
            "reason_count": 1,
            "metrics": {},
            "artifacts": {},
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    if isinstance(data_reality, dict) and not bool(data_reality.get("ready_for_real_ops", False)):
        if rollout_status == "ready_for_phase_d":
            rollout_status = "review_required"

        reason_rows = data_reality.get("reasons") if isinstance(data_reality.get("reasons"), list) else []
        primary_reason = str(reason_rows[0]) if reason_rows else "Dữ liệu huấn luyện/chạy chưa đạt chuẩn vận hành thật."
        hard_action = f"Khóa kích hoạt Phase D: {primary_reason}"
        if hard_action not in recommended_actions:
            recommended_actions = [hard_action, *recommended_actions]

    return {
        "available": available,
        "availability": availability_flags,
        "rollout_status": rollout_status,
        "phase_d_candidate": bool(go_live_phase_d),
        "summary": {
            "hard_gates_pass": hard_gates_pass,
            "soft_gates_pass": soft_gates_pass,
            "split_gate_pass": split_gate_pass,
            "stability_gate_pass": stability_gate_pass,
            "audit_quality_pass": audit_quality_pass,
            "vat_quality_pass": vat_quality_pass,
        },
        "artifacts": {
            "audit_quality": {
                "updated_at": audit_quality.get("updated_at"),
                "overall_pass": audit_quality_pass,
                "calibrated_metrics": _metric_snapshot(
                    (audit_quality_payload.get("performance") or {}).get("calibrated") or {}
                ),
            },
            "vat_quality": {
                "updated_at": vat_quality.get("updated_at"),
                "overall_pass": vat_quality_pass,
                "calibrated_metrics": _metric_snapshot(
                    (vat_quality_payload.get("performance") or {}).get("calibrated") or {}
                ),
            },
            "pilot": {
                "updated_at": pilot_report.get("updated_at"),
                "audit_value": {
                    "samples_evaluated": audit_samples,
                    "f1_delta_model_minus_heuristic": audit_delta,
                },
                "vat_refund": {
                    "samples_evaluated": vat_samples,
                    "f1_delta_model_minus_heuristic": vat_delta,
                },
            },
            "go_no_go": {
                "updated_at": go_no_go_report.get("updated_at"),
                "decision_status": decision_status,
                "go_live_phase_d": go_live_phase_d,
                "message": str(go_no_go_decision.get("message") or ""),
            },
        },
        "recommended_actions": recommended_actions,
        "split_trigger_snapshot": split_snapshot,
        "data_reality": data_reality,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/specialized_rollout_status")
def get_specialized_rollout_status(include_split_snapshot: bool = True):
    """
    Consolidated operational status for specialized tracks rollout:
    quality gates + pilot deltas + go/no-go decision + optional split-trigger snapshot.
    """
    try:
        return _build_specialized_rollout_status_payload(
            include_split_snapshot=bool(include_split_snapshot),
        )
    except Exception as e:
        return {
            "available": False,
            "rollout_status": "error",
            "phase_d_candidate": False,
            "summary": {
                "hard_gates_pass": False,
                "soft_gates_pass": False,
                "split_gate_pass": False,
                "stability_gate_pass": False,
                "audit_quality_pass": False,
                "vat_quality_pass": False,
            },
            "artifacts": {},
            "recommended_actions": [
                "Inspect specialized artifacts and monitoring logs.",
                "Re-run specialized pipeline once underlying errors are fixed.",
            ],
            "error": f"Không thể tải specialized rollout status: {str(e)}",
            "generated_at": datetime.utcnow().isoformat(),
        }


@router.get("/data_reality_audit_logs")
def get_data_reality_audit_logs(
    limit: int = 200,
    source_endpoint: Optional[str] = None,
    status: Optional[str] = None,
    blocked_only: bool = False,
):
    """
    Dedicated audit endpoint for data_reality gate history.
    Useful to track when/why readiness is blocked across intervention/specialized endpoints.
    """
    safe_limit = max(1, min(int(limit or 200), 1000))
    db_url = _resolve_db_url()

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        if not _table_columns(cur, DATA_REALITY_AUDIT_TABLE):
            return {
                "items": [],
                "count": 0,
                "message": "Chưa có bảng data_reality_audit_logs.",
                "generated_at": datetime.utcnow().isoformat(),
            }

        where_clauses: list[str] = []
        query_params: list[Any] = []

        if source_endpoint:
            where_clauses.append("source_endpoint = %s")
            query_params.append(str(source_endpoint).strip())

        if status:
            where_clauses.append("LOWER(status) = %s")
            query_params.append(str(status).strip().lower())

        if blocked_only:
            where_clauses.append("ready_for_real_ops = FALSE")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        cur.execute(
            f"SELECT COUNT(*) FROM {DATA_REALITY_AUDIT_TABLE} {where_sql}",
            tuple(query_params),
        )
        total_rows = int((cur.fetchone() or [0])[0] or 0)

        cur.execute(
            f"""
            SELECT
                id,
                source_endpoint,
                status,
                ready_for_real_ops,
                hard_ready,
                reasons,
                hard_checks,
                soft_checks,
                metrics,
                generated_at,
                created_at
            FROM {DATA_REALITY_AUDIT_TABLE}
            {where_sql}
            ORDER BY id DESC
            LIMIT %s
            """,
            tuple([*query_params, safe_limit]),
        )

        items = []
        for row in cur.fetchall():
            (
                row_id,
                row_source,
                row_status,
                row_ready,
                row_hard_ready,
                row_reasons,
                row_hard_checks,
                row_soft_checks,
                row_metrics,
                row_generated_at,
                row_created_at,
            ) = row

            decoded_reasons = _decode_json_blob(row_reasons) or []

            items.append(
                {
                    "id": int(row_id),
                    "source_endpoint": row_source,
                    "status": row_status,
                    "ready_for_real_ops": bool(row_ready),
                    "hard_ready": bool(row_hard_ready),
                    "reasons": decoded_reasons,
                    "reason_count": len(decoded_reasons),
                    "hard_checks": _decode_json_blob(row_hard_checks) or {},
                    "soft_checks": _decode_json_blob(row_soft_checks) or {},
                    "metrics": _decode_json_blob(row_metrics) or {},
                    "generated_at": row_generated_at.isoformat() if isinstance(row_generated_at, datetime) else row_generated_at,
                    "created_at": row_created_at.isoformat() if isinstance(row_created_at, datetime) else row_created_at,
                }
            )

        return {
            "count": len(items),
            "total_rows": total_rows,
            "items": items,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        return {
            "count": 0,
            "total_rows": 0,
            "items": [],
            "message": f"Không thể đọc data_reality audit logs: {str(exc)}",
            "generated_at": datetime.utcnow().isoformat(),
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@router.post("/kpi_snapshots/capture")
def capture_kpi_snapshots(
    request: Request,
    snapshot_source: str = "manual_capture",
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """
    Capture current split-trigger KPI status into historical snapshots.
    """
    role = _normalize_role(getattr(current_user, "role", ""))
    if role not in ADMIN_POLICY_ROLES:
        _safe_log_monitoring_audit(
            db,
            action="KPI_SNAPSHOT_CAPTURE_DENIED",
            request=request,
            current_user=current_user,
            detail="User role is not authorized to capture KPI snapshots.",
        )
    _require_policy_admin(current_user)
    db_url = _resolve_db_url()
    audit_action = "KPI_SNAPSHOT_CAPTURE_FAILED"
    audit_detail = f"snapshot_source={snapshot_source}"

    conn = None
    cur = None

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        policies = _fetch_kpi_policies(cur)
        payload = _build_split_trigger_status_payload(cur, policies)

        snapshots_captured = _persist_kpi_snapshots(
            cur,
            payload,
            source=snapshot_source,
        )
        conn.commit()
        audit_action = "KPI_SNAPSHOT_CAPTURE"
        audit_detail = f"snapshot_source={snapshot_source} captured={snapshots_captured}"

        return {
            "captured": snapshots_captured,
            "ready": bool(payload.get("ready", False)),
            "schema_ready": bool(payload.get("schema_ready", False)),
            "readiness_score": float(payload.get("readiness_score") or 0),
            "window_days": int(payload.get("window_days") or 28),
            "reason": payload.get("reason") if not payload.get("schema_ready", True) else None,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể capture KPI snapshots: {str(e)}")
    finally:
        _safe_log_monitoring_audit(
            db,
            action=audit_action,
            request=request,
            current_user=current_user,
            detail=audit_detail,
        )
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@router.get("/kpi_snapshots")
def get_kpi_snapshots(
    track_name: Optional[str] = None,
    metric_name: Optional[str] = None,
    days: int = 30,
    limit: int = 200,
):
    """
    Return historical KPI snapshots for split-trigger governance reviews.
    """
    days = max(1, min(int(days or 30), 365))
    limit = max(1, min(int(limit or 200), 500))
    normalized_track = _normalize_policy_token(track_name) if track_name else None
    normalized_metric = _normalize_policy_token(metric_name) if metric_name else None

    db_url = _resolve_db_url()
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        if not _table_columns(cur, "kpi_metric_snapshots"):
            return {
                "available": False,
                "reason": "kpi_metric_snapshots chưa tồn tại. Cần chạy migration trước.",
                "snapshots": [],
                "generated_at": datetime.utcnow().isoformat(),
            }

        where_clauses = ["generated_at >= (NOW() - (%s || ' days')::interval)"]
        params: list[Any] = [days]
        if normalized_track:
            where_clauses.append("track_name = %s")
            params.append(normalized_track)
        if normalized_metric:
            where_clauses.append("metric_name = %s")
            params.append(normalized_metric)

        where_sql = " AND ".join(where_clauses)
        params.append(limit)

        cur.execute(
            f"""
            SELECT
                track_name,
                metric_name,
                metric_value,
                sample_size,
                comparator,
                threshold,
                status,
                window_days,
                source,
                details,
                generated_at
            FROM kpi_metric_snapshots
            WHERE {where_sql}
            ORDER BY generated_at DESC, id DESC
            LIMIT %s
            """,
            tuple(params),
        )
        rows = cur.fetchall()

        snapshots = []
        for row in rows:
            snapshots.append(
                {
                    "track_name": row[0],
                    "metric_name": row[1],
                    "metric_value": float(row[2]) if isinstance(row[2], (int, float)) else None,
                    "sample_size": int(row[3] or 0),
                    "comparator": row[4],
                    "threshold": float(row[5]) if isinstance(row[5], (int, float)) else None,
                    "status": row[6],
                    "window_days": int(row[7] or 0),
                    "source": row[8],
                    "details": _decode_json_blob(row[9]),
                    "generated_at": row[10].isoformat() if row[10] else None,
                }
            )

        status_breakdown = {
            "pass": 0,
            "fail": 0,
            "cooldown_active": 0,
            "insufficient_data": 0,
            "no_metric": 0,
            "disabled": 0,
            "other": 0,
        }
        for snapshot in snapshots:
            status = str(snapshot.get("status") or "").lower()
            if status in status_breakdown:
                status_breakdown[status] += 1
            else:
                status_breakdown["other"] += 1

        latest_by_metric: dict[tuple[str, str], dict[str, Any]] = {}
        for snapshot in snapshots:
            key = (snapshot["track_name"], snapshot["metric_name"])
            if key not in latest_by_metric:
                latest_by_metric[key] = {
                    "track_name": snapshot["track_name"],
                    "metric_name": snapshot["metric_name"],
                    "metric_value": snapshot["metric_value"],
                    "status": snapshot["status"],
                    "sample_size": snapshot["sample_size"],
                    "generated_at": snapshot["generated_at"],
                }

        pass_fail_total = status_breakdown["pass"] + status_breakdown["fail"] + status_breakdown["cooldown_active"]
        pass_rate = _safe_div(status_breakdown["pass"], pass_fail_total)

        return {
            "available": True,
            "filters": {
                "track_name": normalized_track,
                "metric_name": normalized_metric,
                "days": days,
                "limit": limit,
            },
            "total_snapshots": len(snapshots),
            "status_breakdown": status_breakdown,
            "pass_rate": pass_rate,
            "latest_generated_at": snapshots[0]["generated_at"] if snapshots else None,
            "latest_by_metric": list(latest_by_metric.values()),
            "snapshots": snapshots,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"Không thể truy vấn KPI snapshots: {str(e)}",
            "snapshots": [],
            "generated_at": datetime.utcnow().isoformat(),
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def _extract_active_kpi_breaches(split_payload: dict[str, Any]) -> list[dict[str, Any]]:
    track_status = split_payload.get("track_status")
    if not isinstance(track_status, dict):
        return []

    breaches: list[dict[str, Any]] = []
    for track_name, track_payload in track_status.items():
        if not isinstance(track_payload, dict):
            continue
        rules = track_payload.get("rules")
        if not isinstance(rules, list):
            continue

        for rule in rules:
            if not isinstance(rule, dict) or not bool(rule.get("enabled", True)):
                continue
            status = str(rule.get("status") or "").strip().lower()
            if status not in KPI_FAILLIKE_STATUSES:
                continue

            breaches.append(
                {
                    "track_name": track_name,
                    "metric_name": rule.get("metric_name"),
                    "status": status,
                    "actual": rule.get("actual"),
                    "threshold": rule.get("threshold"),
                    "comparator": rule.get("comparator"),
                    "sample_size": int(rule.get("sample_size") or 0),
                    "min_sample": int(rule.get("min_sample") or 0),
                    "window_days": int(rule.get("window_days") or 0),
                }
            )
    return breaches


@router.get("/split_trigger_alerts")
def get_split_trigger_alerts(
    days: int = 14,
    min_pass_rate: float = 0.70,
    min_recent_pass_rate: float = 0.65,
    min_drift_pp: float = 0.08,
    min_track_pass_rate: float = 0.65,
    stale_snapshot_hours: int = 12,
):
    """
    Alerting and trend view for split-trigger policy breaches.
    Helps operations teams detect pass-rate drift and blocking rules quickly.
    """
    days = max(1, min(int(days or 14), 90))
    min_pass_rate = max(0.0, min(float(min_pass_rate), 1.0))
    min_recent_pass_rate = max(0.0, min(float(min_recent_pass_rate), 1.0))
    min_drift_pp = max(0.0, min(float(min_drift_pp), 1.0))
    min_track_pass_rate = max(0.0, min(float(min_track_pass_rate), 1.0))
    stale_snapshot_hours = max(1, min(int(stale_snapshot_hours or 12), 168))

    split_status = get_split_trigger_status_snapshot(
        persist_snapshot=False,
        snapshot_source="split_trigger_alerts",
    )
    active_breaches = _extract_active_kpi_breaches(split_status)

    db_url = _resolve_db_url()
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        latest_snapshot_ts = None
        snapshot_age_hours = None

        def _build_pass_rate_summary(
            overall_pass_rate: Optional[float],
            recent_pass_rate: Optional[float],
            drift_pp: Optional[float],
        ) -> dict[str, Any]:
            return {
                "days": days,
                "overall_pass_rate": overall_pass_rate,
                "recent_3d_pass_rate": recent_pass_rate,
                "drift_pp": drift_pp,
                "min_pass_rate": min_pass_rate,
                "min_recent_pass_rate": min_recent_pass_rate,
                "min_drift_pp": min_drift_pp,
                "min_track_pass_rate": min_track_pass_rate,
            }

        def _build_readiness_summary(
            recent_readiness: Optional[float],
            readiness_drift_pp: Optional[float],
        ) -> dict[str, Any]:
            return {
                "current_readiness_score": float(split_status.get("readiness_score") or 0),
                "recent_3d_readiness_score": recent_readiness,
                "drift_pp": readiness_drift_pp,
                "min_drift_pp": min_drift_pp,
            }

        def _build_snapshot_freshness() -> dict[str, Any]:
            return {
                "latest_generated_at": latest_snapshot_ts.isoformat() if latest_snapshot_ts else None,
                "snapshot_age_hours": snapshot_age_hours,
                "stale_snapshot_hours": stale_snapshot_hours,
            }

        if not _table_columns(cur, "kpi_metric_snapshots"):
            alerts = [
                {
                    "code": "snapshot_table_missing",
                    "severity": "critical",
                    "message": "kpi_metric_snapshots chưa tồn tại. Không thể đánh giá pass-rate trend.",
                }
            ]
            if active_breaches:
                alerts.append(
                    {
                        "code": "active_rule_breaches",
                        "severity": "high",
                        "message": f"Hiện có {len(active_breaches)} KPI rule đang fail/insufficient_data/no_metric.",
                    }
                )

            return {
                "available": False,
                "ready": bool(split_status.get("ready", False)),
                "schema_ready": bool(split_status.get("schema_ready", False)),
                "readiness_score": float(split_status.get("readiness_score") or 0),
                "alert_level": "critical",
                "alerts": alerts,
                "active_breaches": active_breaches,
                "pass_rate_summary": _build_pass_rate_summary(None, None, None),
                "readiness_summary": _build_readiness_summary(None, None),
                "snapshot_freshness": _build_snapshot_freshness(),
                "track_pass_rates": [],
                "trend": [],
                "readiness_trend": [],
                "generated_at": datetime.utcnow().isoformat(),
            }

        cur.execute("SELECT MAX(generated_at) FROM kpi_metric_snapshots")
        latest_snapshot_ts = cur.fetchone()[0]
        if latest_snapshot_ts is not None:
            if getattr(latest_snapshot_ts, "tzinfo", None) is not None:
                latest_snapshot_ts = latest_snapshot_ts.replace(tzinfo=None)
            snapshot_age_hours = round(max(0.0, (datetime.utcnow() - latest_snapshot_ts).total_seconds() / 3600.0), 2)

        cur.execute(
            """
            SELECT
                date_trunc('day', generated_at)::date AS day,
                COUNT(*) FILTER (WHERE status = 'pass') AS pass_count,
                COUNT(*) AS total_count
            FROM kpi_metric_snapshots
            WHERE generated_at >= (NOW() - (%s || ' days')::interval)
            GROUP BY date_trunc('day', generated_at)::date
            ORDER BY day DESC
            """,
            (days,),
        )
        rows = cur.fetchall()

        trend = []
        total_pass = 0
        total_count = 0
        for day, pass_count, row_total in rows:
            pass_count = int(pass_count or 0)
            row_total = int(row_total or 0)
            total_pass += pass_count
            total_count += row_total
            trend.append(
                {
                    "day": day.isoformat() if hasattr(day, "isoformat") else str(day),
                    "pass_count": pass_count,
                    "total_count": row_total,
                    "pass_rate": _safe_div(pass_count, row_total),
                }
            )

        overall_pass_rate = _safe_div(total_pass, total_count)

        recent_rows = trend[:3]
        recent_pass = sum(int(item.get("pass_count") or 0) for item in recent_rows)
        recent_total = sum(int(item.get("total_count") or 0) for item in recent_rows)
        recent_pass_rate = _safe_div(recent_pass, recent_total)

        pass_rate_drift_pp = None
        if len(trend) >= 6:
            split_at = max(1, len(trend) // 2)
            recent_half = trend[:split_at]
            older_half = trend[split_at:]

            recent_half_pass = sum(int(item.get("pass_count") or 0) for item in recent_half)
            recent_half_total = sum(int(item.get("total_count") or 0) for item in recent_half)
            older_half_pass = sum(int(item.get("pass_count") or 0) for item in older_half)
            older_half_total = sum(int(item.get("total_count") or 0) for item in older_half)

            recent_half_rate = _safe_div(recent_half_pass, recent_half_total)
            older_half_rate = _safe_div(older_half_pass, older_half_total)
            if recent_half_rate is not None and older_half_rate is not None:
                pass_rate_drift_pp = float(recent_half_rate - older_half_rate)

        cur.execute(
            """
            SELECT
                date_trunc('day', generated_at)::date AS day,
                AVG(
                    CASE
                        WHEN details IS NOT NULL
                         AND jsonb_typeof(details) = 'object'
                         AND details ? 'readiness_score'
                         AND NULLIF(details->>'readiness_score', '') IS NOT NULL
                        THEN NULLIF(details->>'readiness_score', '')::float
                        ELSE NULL
                    END
                ) AS readiness_score_avg
            FROM kpi_metric_snapshots
            WHERE generated_at >= (NOW() - (%s || ' days')::interval)
            GROUP BY date_trunc('day', generated_at)::date
            ORDER BY day DESC
            """,
            (days,),
        )
        readiness_rows = cur.fetchall()
        readiness_trend = []
        for day, readiness_score_avg in readiness_rows:
            readiness_trend.append(
                {
                    "day": day.isoformat() if hasattr(day, "isoformat") else str(day),
                    "readiness_score": float(readiness_score_avg) if isinstance(readiness_score_avg, (int, float)) else None,
                }
            )

        readiness_recent_values = [
            float(item.get("readiness_score"))
            for item in readiness_trend[:3]
            if isinstance(item.get("readiness_score"), (int, float))
        ]
        recent_readiness_score = (
            float(sum(readiness_recent_values) / len(readiness_recent_values))
            if readiness_recent_values
            else None
        )

        readiness_values = [
            float(item.get("readiness_score"))
            for item in readiness_trend
            if isinstance(item.get("readiness_score"), (int, float))
        ]
        readiness_drift_pp = None
        if len(readiness_values) >= 6:
            split_at = max(1, len(readiness_values) // 2)
            recent_half_values = readiness_values[:split_at]
            older_half_values = readiness_values[split_at:]
            if older_half_values:
                recent_half_avg = float(sum(recent_half_values) / len(recent_half_values))
                older_half_avg = float(sum(older_half_values) / len(older_half_values))
                readiness_drift_pp = recent_half_avg - older_half_avg

        cur.execute(
            """
            SELECT
                track_name,
                COUNT(*) FILTER (WHERE status = 'pass') AS pass_count,
                COUNT(*) FILTER (WHERE status IN ('fail', 'cooldown_active')) AS fail_count,
                COUNT(*) FILTER (WHERE status IN ('pass', 'fail', 'cooldown_active')) AS pass_fail_count,
                COUNT(*) AS total_count
            FROM kpi_metric_snapshots
            WHERE generated_at >= (NOW() - (%s || ' days')::interval)
            GROUP BY track_name
            ORDER BY track_name
            """,
            (days,),
        )
        track_pass_rates = []
        for track_name, pass_count, fail_count, pass_fail_count, total_count in cur.fetchall():
            pass_count = int(pass_count or 0)
            fail_count = int(fail_count or 0)
            pass_fail_count = int(pass_fail_count or 0)
            total_count = int(total_count or 0)
            track_pass_rates.append(
                {
                    "track_name": track_name,
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "pass_fail_count": pass_fail_count,
                    "total_count": total_count,
                    "pass_rate": _safe_div(pass_count, pass_fail_count),
                }
            )

        alerts = []
        if not bool(split_status.get("schema_ready", False)):
            alerts.append(
                {
                    "code": "schema_not_ready",
                    "severity": "critical",
                    "message": str(split_status.get("reason") or "Schema KPI chưa sẵn sàng."),
                }
            )

        if active_breaches:
            alerts.append(
                {
                    "code": "active_rule_breaches",
                    "severity": "high",
                    "message": f"Hiện có {len(active_breaches)} KPI rule đang fail/insufficient_data/no_metric.",
                }
            )

        if snapshot_age_hours is not None and snapshot_age_hours > stale_snapshot_hours:
            stale_multiplier = _safe_div(snapshot_age_hours, stale_snapshot_hours) or 1.0
            stale_severity = "critical" if stale_multiplier >= 2.0 else "high"
            alerts.append(
                {
                    "code": "snapshot_stale",
                    "severity": stale_severity,
                    "message": (
                        f"Snapshot mới nhất đã {snapshot_age_hours:.1f}h, vượt ngưỡng {stale_snapshot_hours}h. "
                        "Cần kiểm tra scheduler hoặc trigger capture thủ công."
                    ),
                }
            )

        if overall_pass_rate is not None and overall_pass_rate < min_pass_rate:
            alerts.append(
                {
                    "code": "overall_pass_rate_low",
                    "severity": "medium",
                    "message": f"Overall pass-rate {overall_pass_rate:.2%} thấp hơn ngưỡng {min_pass_rate:.2%}.",
                }
            )

        if recent_pass_rate is not None and recent_pass_rate < min_recent_pass_rate:
            alerts.append(
                {
                    "code": "recent_pass_rate_low",
                    "severity": "high",
                    "message": f"Pass-rate 3 ngày gần nhất {recent_pass_rate:.2%} thấp hơn ngưỡng {min_recent_pass_rate:.2%}.",
                }
            )

        if pass_rate_drift_pp is not None and pass_rate_drift_pp <= -min_drift_pp:
            alerts.append(
                {
                    "code": "pass_rate_drift_down",
                    "severity": "medium",
                    "message": f"Pass-rate gần đây giảm {abs(pass_rate_drift_pp):.2%} so với giai đoạn trước.",
                }
            )

        if readiness_drift_pp is not None and readiness_drift_pp <= -(min_drift_pp * 100.0):
            alerts.append(
                {
                    "code": "readiness_drift_down",
                    "severity": "high",
                    "message": f"Readiness score gần đây giảm {abs(readiness_drift_pp):.1f} điểm so với giai đoạn trước.",
                }
            )

        critical_tracks = set(split_status.get("critical_tracks") or ["audit_value", "vat_refund"])
        for track_item in track_pass_rates:
            track_name = str(track_item.get("track_name") or "")
            pass_rate = track_item.get("pass_rate")
            pass_fail_count = int(track_item.get("pass_fail_count") or 0)
            if pass_fail_count <= 0 or pass_rate is None:
                continue
            if float(pass_rate) < min_track_pass_rate:
                alerts.append(
                    {
                        "code": "track_pass_rate_low",
                        "severity": "high" if track_name in critical_tracks else "medium",
                        "message": (
                            f"Track {track_name} có pass-rate {float(pass_rate):.2%} "
                            f"thấp hơn ngưỡng {min_track_pass_rate:.2%}."
                        ),
                        "track_name": track_name,
                    }
                )

        if any(alert.get("severity") == "critical" for alert in alerts):
            alert_level = "critical"
        elif any(alert.get("severity") == "high" for alert in alerts):
            alert_level = "high"
        elif any(alert.get("severity") == "medium" for alert in alerts):
            alert_level = "medium"
        else:
            alert_level = "low"

        return {
            "available": True,
            "ready": bool(split_status.get("ready", False)),
            "schema_ready": bool(split_status.get("schema_ready", False)),
            "readiness_score": float(split_status.get("readiness_score") or 0),
            "alert_level": alert_level,
            "alerts": alerts,
            "active_breaches": active_breaches,
            "pass_rate_summary": _build_pass_rate_summary(
                overall_pass_rate,
                recent_pass_rate,
                pass_rate_drift_pp,
            ),
            "readiness_summary": _build_readiness_summary(recent_readiness_score, readiness_drift_pp),
            "snapshot_freshness": _build_snapshot_freshness(),
            "track_pass_rates": track_pass_rates,
            "trend": trend,
            "readiness_trend": readiness_trend,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "available": False,
            "ready": bool(split_status.get("ready", False)),
            "schema_ready": bool(split_status.get("schema_ready", False)),
            "readiness_score": float(split_status.get("readiness_score") or 0),
            "alert_level": "critical",
            "alerts": [
                {
                    "code": "split_trigger_alerts_unavailable",
                    "severity": "critical",
                    "message": f"Không thể tải split-trigger alerts: {str(e)}",
                }
            ],
            "active_breaches": active_breaches,
            "pass_rate_summary": {
                "days": days,
                "overall_pass_rate": None,
                "recent_3d_pass_rate": None,
                "drift_pp": None,
                "min_pass_rate": min_pass_rate,
                "min_recent_pass_rate": min_recent_pass_rate,
                "min_drift_pp": min_drift_pp,
                "min_track_pass_rate": min_track_pass_rate,
            },
            "readiness_summary": {
                "current_readiness_score": float(split_status.get("readiness_score") or 0),
                "recent_3d_readiness_score": None,
                "drift_pp": None,
                "min_drift_pp": min_drift_pp,
            },
            "snapshot_freshness": {
                "latest_generated_at": None,
                "snapshot_age_hours": None,
                "stale_snapshot_hours": stale_snapshot_hours,
            },
            "track_pass_rates": [],
            "trend": [],
            "readiness_trend": [],
            "generated_at": datetime.utcnow().isoformat(),
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

