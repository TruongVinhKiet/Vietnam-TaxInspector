from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
import os
import numpy as np

from ..observability import get_structured_logger, log_event
from .. import schemas

router = APIRouter(prefix="/api/monitoring", tags=["MLOps"])

logger = get_structured_logger("taxinspector.monitoring")

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

DRIFT_FEATURES = (
    "company_age_days",
    "invoice_amount_log",
    "is_reciprocal_ratio",
)

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
        # Robust approximation when scipy is unavailable.
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


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


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


def _collect_feature_windows(metrics_path: Path, lookback_days: int, baseline_days: int) -> tuple[dict, dict, int, int]:
    now = datetime.utcnow()
    recent_cutoff = now - timedelta(days=lookback_days)
    baseline_cutoff = now - timedelta(days=lookback_days + baseline_days)

    recent = {name: [] for name in DRIFT_FEATURES}
    baseline = {name: [] for name in DRIFT_FEATURES}
    recent_rows = 0
    baseline_rows = 0

    for entry in _read_jsonl(metrics_path):
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

        for name in DRIFT_FEATURES:
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
    )
    baseline_stats = _load_baseline_stats()

    features_report = {}
    drifted_features = []
    sufficient_features = 0

    for name in DRIFT_FEATURES:
        recent_values = recent.get(name, [])

        # Prefer persisted training baseline if present.
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

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            dbname=os.getenv("DB_NAME", "TaxInspector")
        )
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

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

    if metric.metric_name in {"inference_features", "graph_inference_features"}:
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
            "gat_model": (model_dir / "gat_model.pt").exists(),
            "calibrator": (model_dir / "calibrator.pkl").exists(),
            "anomaly_detector": (model_dir / "anomaly_detector.pkl").exists(),
            "ensemble_meta": (model_dir / "ensemble_meta.pkl").exists(),
        },
        "db_connection": False
    }
    
    conn = get_db_connection()
    if conn:
        status["db_connection"] = True
        conn.close()
    else:
        status["status"] = "degraded"
        
    if not all(status["models"].values()):
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
        1 for item in report.get("features", {}).values()
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
