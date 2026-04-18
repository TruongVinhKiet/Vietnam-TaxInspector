"""
run_specialized_pilot_cohort.py

Pilot comparison between model path and heuristic path for specialized tracks:
- audit_value
- vat_refund

The script evaluates recent labeled rows and writes a comparison report to:
    data/models/specialized_pilot_report.json

Usage (from Backend directory):
    python app/scripts/run_specialized_pilot_cohort.py

Custom sampling:
    python app/scripts/run_specialized_pilot_cohort.py --lookback-days 540 --per-track-limit 3000 --model-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import func, or_

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from app.models import AIRiskAssessment, InspectorLabel
from app.routers import ai_analysis


REPORT_PATH = BACKEND_DIR / "data" / "models" / "specialized_pilot_report.json"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return float(default)
    return float(numerator / denominator)


@contextmanager
def _disable_specialized_prediction_for_heuristic():
    original = ai_analysis._predict_specialized_probability

    def _always_none(_track_name: str, _feature_map: dict[str, float]):
        return None

    ai_analysis._predict_specialized_probability = _always_none
    try:
        yield
    finally:
        ai_analysis._predict_specialized_probability = original


def _actual_label_audit(label: InspectorLabel) -> int:
    amount_recovered = _to_float(getattr(label, "amount_recovered", 0.0), 0.0)
    outcome = str(getattr(label, "outcome_status", "") or "").strip().lower()
    label_type = str(getattr(label, "label_type", "") or "").strip().lower()

    if amount_recovered > 0:
        return 1
    if outcome in {"recovered", "partial_recovered"}:
        return 1
    if "confirmed" in label_type:
        return 1
    return 0


def _actual_label_vat(label: InspectorLabel) -> int:
    amount_recovered = _to_float(getattr(label, "amount_recovered", 0.0), 0.0)
    outcome = str(getattr(label, "outcome_status", "") or "").strip().lower()
    label_type = str(getattr(label, "label_type", "") or "").strip().lower()

    if amount_recovered > 0:
        return 1
    if outcome in {"recovered", "partial_recovered"}:
        return 1
    if "confirmed" in label_type or "high_risk" in label_type:
        return 1
    return 0


def _binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {
            "samples": 0,
            "positive_rate": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "auc_roc": None,
            "pr_auc": None,
        }

    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
    fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))
    fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))

    precision = _safe_div(tp, tp + fp, 0.0)
    recall = _safe_div(tp, tp + fn, 0.0)
    f1 = _safe_div(2.0 * precision * recall, precision + recall, 0.0)
    accuracy = _safe_div(tp + tn, n, 0.0)

    auc_roc = None
    pr_auc = None
    if y_prob is not None and len(y_prob) == n and len(np.unique(y_true_np)) >= 2:
        probs = np.asarray(y_prob, dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        try:
            auc_roc = float(roc_auc_score(y_true_np, probs))
            pr_auc = float(average_precision_score(y_true_np, probs))
        except Exception:
            auc_roc = None
            pr_auc = None

    return {
        "samples": n,
        "positive_rate": round(float(np.mean(y_true_np)), 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "auc_roc": (round(auc_roc, 6) if isinstance(auc_roc, float) else None),
        "pr_auc": (round(pr_auc, 6) if isinstance(pr_auc, float) else None),
    }


def _load_track_rows(
    db,
    *,
    track_name: str,
    lookback_days: int,
    limit: int,
) -> list[tuple[InspectorLabel, AIRiskAssessment]]:
    since = datetime.utcnow() - timedelta(days=max(1, int(lookback_days)))

    query = (
        db.query(InspectorLabel, AIRiskAssessment)
        .join(AIRiskAssessment, AIRiskAssessment.id == InspectorLabel.assessment_id)
        .filter(
            InspectorLabel.assessment_id.isnot(None),
            InspectorLabel.created_at >= since,
            AIRiskAssessment.tax_code.isnot(None),
        )
    )

    if track_name == "vat_refund":
        query = query.filter(
            or_(
                func.lower(func.coalesce(InspectorLabel.label_type, "")).like("%vat%"),
                func.lower(func.coalesce(InspectorLabel.label_type, "")).like("%refund%"),
                func.lower(func.coalesce(InspectorLabel.label_type, "")).like("%invoice%"),
            )
        )
    else:
        query = query.filter(
            or_(
                InspectorLabel.outcome_status.isnot(None),
                func.coalesce(InspectorLabel.amount_recovered, 0) > 0,
                func.coalesce(InspectorLabel.expected_recovery, 0) > 0,
            )
        )

    return (
        query.order_by(InspectorLabel.created_at.desc(), InspectorLabel.id.desc())
        .limit(max(1, int(limit)))
        .all()
    )


def _evaluate_track(
    rows: list[tuple[InspectorLabel, AIRiskAssessment]],
    *,
    track_name: str,
    model_threshold: float,
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred_model: list[int] = []
    y_pred_heur: list[int] = []
    y_prob_model: list[float] = []

    skipped_missing_model = 0

    for label, assessment in rows:
        risk_score = _to_float(assessment.risk_score, 0.0)
        anomaly_score = _to_float(assessment.anomaly_score, 0.0)
        model_confidence = _to_float(assessment.model_confidence, 50.0)
        revenue = _to_float(assessment.revenue, 0.0)
        total_expenses = _to_float(assessment.total_expenses, 0.0)
        f1 = _to_float(assessment.f1_divergence, 0.0)
        f2 = _to_float(assessment.f2_ratio_limit, 0.0)
        f3 = _to_float(assessment.f3_vat_structure, 0.0)
        f4 = _to_float(assessment.f4_peer_comparison, 0.0)
        red_flags = assessment.red_flags if isinstance(assessment.red_flags, list) else []

        if track_name == "audit_value":
            feature_map = ai_analysis._build_audit_model_feature_map(
                risk_score=risk_score,
                anomaly_score=anomaly_score,
                model_confidence=model_confidence,
                red_flags=red_flags,
                revenue=revenue,
                total_expenses=total_expenses,
                f1_divergence=f1,
                f2_ratio_limit=f2,
                f3_vat_structure=f3,
                f4_peer_comparison=f4,
            )
            actual = _actual_label_audit(label)
            model_payload = ai_analysis._predict_specialized_probability("audit_value", feature_map)
            with _disable_specialized_prediction_for_heuristic():
                heuristic_payload = ai_analysis._build_audit_value_payload(
                    risk_score=risk_score,
                    anomaly_score=anomaly_score,
                    model_confidence=model_confidence,
                    red_flags=red_flags,
                    yearly_feature_scores=[],
                    revenue=revenue,
                    total_expenses=total_expenses,
                    f1_divergence=f1,
                    f2_ratio_limit=f2,
                    f3_vat_structure=f3,
                    f4_peer_comparison=f4,
                )
            heuristic_positive = str(heuristic_payload.get("recommended_lane") or "monitor") in {"targeted_audit", "priority_audit"}
        else:
            feature_map = ai_analysis._build_vat_refund_model_feature_map(
                risk_score=risk_score,
                anomaly_score=anomaly_score,
                f1_divergence=f1,
                f2_ratio_limit=f2,
                f3_vat_structure=f3,
                f4_peer_comparison=f4,
                revenue=revenue,
                total_expenses=total_expenses,
                red_flags=red_flags,
            )
            actual = _actual_label_vat(label)
            model_payload = ai_analysis._predict_specialized_probability("vat_refund", feature_map)
            with _disable_specialized_prediction_for_heuristic():
                heuristic_payload = ai_analysis._build_vat_refund_signals_payload(
                    risk_score=risk_score,
                    anomaly_score=anomaly_score,
                    red_flags=red_flags,
                    yearly_feature_scores=[],
                    f2_ratio_limit=f2,
                    f3_vat_structure=f3,
                    revenue=revenue,
                    total_expenses=total_expenses,
                    f1_divergence=f1,
                    f4_peer_comparison=f4,
                )
            heuristic_positive = bool(heuristic_payload.get("has_signal")) or str(heuristic_payload.get("queue") or "monitor") != "monitor"

        if not isinstance(model_payload, dict):
            skipped_missing_model += 1
            continue

        model_probability = float(np.clip(_to_float(model_payload.get("probability"), 0.0), 0.0, 1.0))
        model_positive = model_probability >= model_threshold

        y_true.append(int(actual))
        y_pred_model.append(1 if model_positive else 0)
        y_pred_heur.append(1 if heuristic_positive else 0)
        y_prob_model.append(model_probability)

    model_metrics = _binary_metrics(y_true, y_pred_model, y_prob_model)
    heuristic_metrics = _binary_metrics(y_true, y_pred_heur, None)

    delta = {
        "precision_delta": None,
        "recall_delta": None,
        "f1_delta": None,
        "accuracy_delta": None,
    }

    if isinstance(model_metrics.get("precision"), float) and isinstance(heuristic_metrics.get("precision"), float):
        delta["precision_delta"] = round(float(model_metrics["precision"] - heuristic_metrics["precision"]), 6)
    if isinstance(model_metrics.get("recall"), float) and isinstance(heuristic_metrics.get("recall"), float):
        delta["recall_delta"] = round(float(model_metrics["recall"] - heuristic_metrics["recall"]), 6)
    if isinstance(model_metrics.get("f1"), float) and isinstance(heuristic_metrics.get("f1"), float):
        delta["f1_delta"] = round(float(model_metrics["f1"] - heuristic_metrics["f1"]), 6)
    if isinstance(model_metrics.get("accuracy"), float) and isinstance(heuristic_metrics.get("accuracy"), float):
        delta["accuracy_delta"] = round(float(model_metrics["accuracy"] - heuristic_metrics["accuracy"]), 6)

    return {
        "samples_requested": len(rows),
        "samples_evaluated": len(y_true),
        "skipped_missing_model": skipped_missing_model,
        "model_threshold": model_threshold,
        "model": model_metrics,
        "heuristic": heuristic_metrics,
        "delta_model_minus_heuristic": delta,
    }


def run_pilot(
    *,
    lookback_days: int,
    per_track_limit: int,
    model_threshold: float,
    output_path: Path,
) -> int:
    db = SessionLocal()
    try:
        audit_rows = _load_track_rows(
            db,
            track_name="audit_value",
            lookback_days=lookback_days,
            limit=per_track_limit,
        )
        vat_rows = _load_track_rows(
            db,
            track_name="vat_refund",
            lookback_days=lookback_days,
            limit=per_track_limit,
        )

        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config": {
                "lookback_days": int(lookback_days),
                "per_track_limit": int(per_track_limit),
                "model_threshold": float(model_threshold),
            },
            "tracks": {
                "audit_value": _evaluate_track(
                    audit_rows,
                    track_name="audit_value",
                    model_threshold=model_threshold,
                ),
                "vat_refund": _evaluate_track(
                    vat_rows,
                    track_name="vat_refund",
                    model_threshold=model_threshold,
                ),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("[PILOT] Specialized pilot comparison done")
        print(f"[PILOT] report_path={output_path}")

        for track_name in ("audit_value", "vat_refund"):
            payload = report["tracks"][track_name]
            model_f1 = (payload.get("model") or {}).get("f1")
            heuristic_f1 = (payload.get("heuristic") or {}).get("f1")
            delta_f1 = (payload.get("delta_model_minus_heuristic") or {}).get("f1_delta")
            print(
                f"[PILOT] {track_name}: evaluated={payload.get('samples_evaluated')} "
                f"model_f1={model_f1} heuristic_f1={heuristic_f1} delta_f1={delta_f1}"
            )

        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pilot comparison for specialized model tracks")
    parser.add_argument("--lookback-days", type=int, default=540, help="Lookback window for labels")
    parser.add_argument("--per-track-limit", type=int, default=3000, help="Max rows per track")
    parser.add_argument("--model-threshold", type=float, default=0.50, help="Decision threshold for model probability")
    parser.add_argument("--output", type=str, default=str(REPORT_PATH), help="Output report path")

    args = parser.parse_args()

    if args.lookback_days < 1:
        print("[ERROR] --lookback-days must be >= 1")
        return 1
    if args.per_track_limit < 50:
        print("[ERROR] --per-track-limit should be >= 50")
        return 1
    if args.model_threshold < 0.0 or args.model_threshold > 1.0:
        print("[ERROR] --model-threshold must be in [0, 1]")
        return 1

    return run_pilot(
        lookback_days=int(args.lookback_days),
        per_track_limit=int(args.per_track_limit),
        model_threshold=float(args.model_threshold),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    raise SystemExit(main())
