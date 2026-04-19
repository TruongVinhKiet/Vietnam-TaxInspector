"""
train_vat_refund.py - Training script for Program D (VAT Refund Signals).
========================================================================

This script trains a binary classifier that estimates the probability of
material VAT refund risk requiring priority audit handling.

Artifacts:
    - data/models/vat_refund_model.joblib
    - data/models/vat_refund_calibrator.joblib (optional)
    - data/models/vat_refund_quality_report.json
    - data/models/vat_refund_drift_baseline.json
    - data/models/vat_refund_model_meta.json

Usage:
    python -m ml_engine.train_vat_refund
    python ml_engine/train_vat_refund.py --lookback-days 540
    python ml_engine/train_vat_refund.py --label-origin-policy exclude_synthetic
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Ensure backend root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


VAT_MODEL_VERSION = os.getenv("VAT_REFUND_MODEL_VERSION", "vat-refund-v1")
VAT_FEATURE_SET_VERSION = "vat_refund_features_v1"

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
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
REAL_LABEL_ORIGINS = ("manual_inspector", "field_verified", "imported_casework")
SYNTHETIC_LABEL_ORIGINS = ("bootstrap_generated", "auto_seed")
VALID_LABEL_ORIGIN_POLICIES = ("exclude_synthetic", "real_only", "all")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_json_like(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except Exception:
            return []
    return []


def _normalize_label_origin_policy(raw_policy: str | None) -> str:
    normalized = str(raw_policy or "").strip().lower()
    if normalized in VALID_LABEL_ORIGIN_POLICIES:
        return normalized
    return "exclude_synthetic"


def _build_label_origin_filter_sql(policy: str, canonical_origin_sql: str | None = None) -> str:
    resolved_policy = _normalize_label_origin_policy(policy)
    canonical_origin_sql = canonical_origin_sql or "COALESCE(NULLIF(btrim(LOWER(COALESCE(l.label_origin, ''))), ''), 'manual_inspector')"
    real_tokens = ", ".join(f"'{token}'" for token in REAL_LABEL_ORIGINS)
    synthetic_tokens = ", ".join(f"'{token}'" for token in SYNTHETIC_LABEL_ORIGINS)

    if resolved_policy == "all":
        return "TRUE"
    if resolved_policy == "real_only":
        return f"{canonical_origin_sql} IN ({real_tokens})"
    return f"{canonical_origin_sql} NOT IN ({synthetic_tokens})"


def _count_vat_flags(raw_flags: Any) -> int:
    flags = _coerce_json_like(raw_flags)
    if not isinstance(flags, list):
        return 0

    count = 0
    for item in flags:
        if not isinstance(item, dict):
            continue
        text_blob = " ".join(
            str(item.get(k, ""))
            for k in ("feature", "title", "reason", "description", "actual_value")
        ).lower()
        if any(keyword in text_blob for keyword in VAT_KEYWORDS):
            count += 1
    return count


def _build_feature_row(row: dict[str, Any]) -> dict[str, float]:
    revenue = max(0.0, _to_float(row.get("revenue"), 0.0))
    total_expenses = max(0.0, _to_float(row.get("total_expenses"), 0.0))

    cost_of_goods = total_expenses * 0.75
    vat_output = revenue * 0.10
    vat_input = cost_of_goods * 0.10
    vat_input_output_ratio = (vat_input / vat_output) if vat_output > 0 else (2.0 if vat_input > 0 else 0.0)
    estimated_refund_gap = max(0.0, vat_input - vat_output)

    expense_to_revenue_ratio = (total_expenses / revenue) if revenue > 0 else (2.0 if total_expenses > 0 else 0.0)

    return {
        "risk_score": max(0.0, min(100.0, _to_float(row.get("risk_score"), 0.0))),
        "anomaly_score": max(0.0, _to_float(row.get("anomaly_score"), 0.0)),
        "f2_ratio_limit": max(0.0, _to_float(row.get("f2_ratio_limit"), 0.0)),
        "f3_vat_structure": max(0.0, _to_float(row.get("f3_vat_structure"), 0.0)),
        "revenue": revenue,
        "total_expenses": total_expenses,
        "vat_input_output_ratio": max(0.0, vat_input_output_ratio),
        "estimated_refund_gap": estimated_refund_gap,
        "vat_flag_count": float(_count_vat_flags(row.get("red_flags"))),
        "expense_to_revenue_ratio": max(0.0, expense_to_revenue_ratio),
        "f1_divergence": max(0.0, _to_float(row.get("f1_divergence"), 0.0)),
        "f4_peer_comparison": max(0.0, _to_float(row.get("f4_peer_comparison"), 0.0)),
    }


def _build_target_label(row: dict[str, Any]) -> int:
    amount_recovered = _to_float(row.get("amount_recovered"), 0.0)
    outcome = str(row.get("outcome_status") or "").strip().lower()
    label_type = str(row.get("label_type") or "").strip().lower()

    if amount_recovered > 0:
        return 1
    if outcome in {"recovered", "partial_recovered"}:
        return 1
    if "confirmed" in label_type or "high_risk" in label_type:
        return 1
    return 0


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        low, high = boundaries[i], boundaries[i + 1]
        if i == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def _safe_split(X, y, test_size: float, random_state: int):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _metrics_snapshot(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    out: dict[str, Any] = {
        "samples": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)) if len(y_true) else 0.0, 6),
        "auc_roc": None,
        "pr_auc": None,
        "brier": None,
        "ece": None,
        "precision_at_0_5": None,
        "recall_at_0_5": None,
        "f1_at_0_5": None,
        "available": False,
    }

    if len(y_true) == 0:
        return out

    y_pred = (y_prob >= 0.5).astype(int)
    out["precision_at_0_5"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
    out["recall_at_0_5"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 6)
    out["f1_at_0_5"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 6)
    out["brier"] = round(float(brier_score_loss(y_true, y_prob)), 6)
    out["ece"] = round(float(_expected_calibration_error(y_true, y_prob, bins=10)), 6)

    if len(np.unique(y_true)) >= 2:
        out["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        out["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 6)
        out["available"] = True

    return out


def _build_drift_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    features_payload = {}
    for col in FEATURE_COLUMNS:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(values) == 0:
            continue
        features_payload[col] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
            "q0": round(float(np.min(values)), 6),
            "q25": round(float(np.percentile(values, 25)), 6),
            "q50": round(float(np.percentile(values, 50)), 6),
            "q75": round(float(np.percentile(values, 75)), 6),
            "q100": round(float(np.max(values)), 6),
        }
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": VAT_MODEL_VERSION,
        "feature_set_version": VAT_FEATURE_SET_VERSION,
        "sample_count": int(len(frame)),
        "features": features_payload,
    }


def _resolve_default_db_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    try:
        from app.database import SQLALCHEMY_DATABASE_URL

        if SQLALCHEMY_DATABASE_URL:
            return SQLALCHEMY_DATABASE_URL
    except Exception:
        pass

    return "postgresql://postgres:postgres@localhost/tax_inspector"


def _load_table_columns(conn, table_name: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ANY(current_schemas(false))
              AND table_name = %s
            """,
            (table_name,),
        )
        return {
            str(row[0]).strip().lower()
            for row in cur.fetchall()
            if row and row[0]
        }


def load_training_data(db_url: str, lookback_days: int, label_origin_policy: str) -> pd.DataFrame:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(db_url)
    try:
        inspector_label_columns = _load_table_columns(conn, "inspector_labels")
        has_label_origin = "label_origin" in inspector_label_columns
        has_model_version = "model_version" in inspector_label_columns

        canonical_origin_sql = (
            "COALESCE(NULLIF(btrim(LOWER(COALESCE(l.label_origin, ''))), ''), 'manual_inspector')"
            if has_label_origin
            else "'manual_inspector'"
        )
        origin_filter_sql = _build_label_origin_filter_sql(label_origin_policy, canonical_origin_sql)

        label_origin_select_sql = f"{canonical_origin_sql} AS label_origin"
        label_model_version_select_sql = (
            "l.model_version AS label_model_version"
            if has_model_version
            else "NULL::VARCHAR AS label_model_version"
        )

        query = f"""
        SELECT
            l.id AS label_id,
            l.tax_code,
            l.label_type,
            {label_origin_select_sql},
            {label_model_version_select_sql},
            l.outcome_status,
            l.amount_recovered,
            l.created_at AS label_created_at,
            a.risk_score,
            a.anomaly_score,
            a.revenue,
            a.total_expenses,
            a.f1_divergence,
            a.f2_ratio_limit,
            a.f3_vat_structure,
            a.f4_peer_comparison,
            a.red_flags
        FROM inspector_labels l
        LEFT JOIN LATERAL (
            SELECT
                risk_score,
                anomaly_score,
                revenue,
                total_expenses,
                f1_divergence,
                f2_ratio_limit,
                f3_vat_structure,
                f4_peer_comparison,
                red_flags,
                model_version,
                created_at
            FROM ai_risk_assessments a
            WHERE a.tax_code = l.tax_code
            ORDER BY a.created_at DESC
            LIMIT 1
        ) a ON TRUE
        WHERE l.created_at >= (NOW() - (%s || ' days')::interval)
          AND ({origin_filter_sql})
          AND (
                     LOWER(COALESCE(l.label_type, '')) LIKE '%%vat%%'
                 OR LOWER(COALESCE(l.label_type, '')) LIKE '%%refund%%'
                 OR LOWER(COALESCE(l.label_type, '')) LIKE '%%invoice%%'
          )
        ORDER BY l.created_at ASC
    """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (int(lookback_days),))
            rows = cur.fetchall()
    finally:
        conn.close()

    return pd.DataFrame(rows)


def main(db_url: str, lookback_days: int, min_samples: int, label_origin_policy: str) -> int:
    print("=" * 64)
    print("  VAT Refund Risk Model Training")
    print("=" * 64)

    print("\n[1/6] Loading VAT/refund labels...")
    resolved_policy = _normalize_label_origin_policy(label_origin_policy)
    print(f"       label_origin_policy={resolved_policy}")
    frame = load_training_data(
        db_url=db_url,
        lookback_days=lookback_days,
        label_origin_policy=resolved_policy,
    )
    print(f"       Loaded labels: {len(frame):,}")

    if len(frame) > 0 and "label_origin" in frame.columns:
        origin_counts = (
            frame["label_origin"]
            .fillna("manual_inspector")
            .astype(str)
            .str.strip()
            .replace({"": "manual_inspector"})
            .value_counts()
            .to_dict()
        )
        origin_summary = ", ".join(f"{k}:{int(v)}" for k, v in sorted(origin_counts.items()))
        print(f"       Label origins: {origin_summary}")

    if len(frame) < max(30, min_samples):
        print(f"[ABORT] Not enough labels for training (need >= {max(30, min_samples)}).")
        return 2

    print("\n[2/6] Building feature matrix...")
    feature_rows = []
    targets = []
    timestamps = []

    for row in frame.to_dict(orient="records"):
        feature_rows.append(_build_feature_row(row))
        targets.append(_build_target_label(row))
        timestamps.append(row.get("label_created_at"))

    features_df = pd.DataFrame(feature_rows)
    for col in FEATURE_COLUMNS:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[FEATURE_COLUMNS].fillna(0.0)

    X = features_df.to_numpy(dtype=float)
    y = np.asarray(targets, dtype=int)

    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    print(f"       Samples: {len(y):,} | Positive: {positive_count:,} | Negative: {negative_count:,}")

    if len(np.unique(y)) < 2:
        print("[ABORT] Labels contain only one class, cannot train classifier.")
        return 3

    print("\n[3/6] Splitting train/calibration/test...")
    X_train, X_holdout, y_train, y_holdout = _safe_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )
    X_calib, X_test, y_calib, y_test = _safe_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=43,
    )

    print(
        f"       Split sizes: train={len(y_train)} | calib={len(y_calib)} | test={len(y_test)}"
    )

    print("\n[4/6] Training base classifier...")
    model = RandomForestClassifier(
        n_estimators=320,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    raw_calib_prob = np.clip(model.predict_proba(X_calib)[:, 1], 0.0, 1.0)
    calibrator = None
    if len(y_calib) >= 30 and len(np.unique(y_calib)) >= 2:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_calib_prob, y_calib)
        print("       Calibrator: isotonic (enabled)")
    else:
        print("       Calibrator: skipped (insufficient calibration set)")

    print("\n[5/6] Evaluating model quality...")
    raw_test_prob = np.clip(model.predict_proba(X_test)[:, 1], 0.0, 1.0)
    calibrated_test_prob = (
        np.clip(np.asarray(calibrator.predict(raw_test_prob), dtype=float), 0.0, 1.0)
        if calibrator is not None
        else raw_test_prob
    )

    raw_metrics = _metrics_snapshot(y_test, raw_test_prob)
    calibrated_metrics = _metrics_snapshot(y_test, calibrated_test_prob)

    criteria = {
        "auc_roc_min": {
            "threshold": 0.60,
            "actual": calibrated_metrics.get("auc_roc"),
            "pass": (calibrated_metrics.get("auc_roc") or 0.0) >= 0.60,
        },
        "pr_auc_min": {
            "threshold": 0.50,
            "actual": calibrated_metrics.get("pr_auc"),
            "pass": (calibrated_metrics.get("pr_auc") or 0.0) >= 0.50,
        },
        "brier_max": {
            "threshold": 0.30,
            "actual": calibrated_metrics.get("brier"),
            "pass": (calibrated_metrics.get("brier") or 1.0) <= 0.30,
        },
        "ece_max": {
            "threshold": 0.16,
            "actual": calibrated_metrics.get("ece"),
            "pass": (calibrated_metrics.get("ece") or 1.0) <= 0.16,
            "soft_gate": True,
        },
    }
    hard_gate_keys = ["auc_roc_min", "pr_auc_min", "brier_max"]
    overall_pass = bool(all(criteria[key]["pass"] for key in hard_gate_keys))

    quality_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "track": "vat_refund",
            "model_version": VAT_MODEL_VERSION,
            "feature_set_version": VAT_FEATURE_SET_VERSION,
            "algorithm": "RandomForestClassifier",
            "calibration_method": "isotonic" if calibrator is not None else None,
        },
        "dataset": {
            "total_size": int(len(y)),
            "train_size": int(len(y_train)),
            "calibration_size": int(len(y_calib)),
            "test_size": int(len(y_test)),
            "positive_rate_total": round(float(np.mean(y)), 6),
            "time_coverage": {
                "earliest": str(pd.to_datetime(min(timestamps), errors="coerce")) if timestamps else None,
                "latest": str(pd.to_datetime(max(timestamps), errors="coerce")) if timestamps else None,
            },
        },
        "performance": {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
        "acceptance_gates": {
            "criteria": criteria,
            "soft_gate_criteria": ["ece_max"],
            "overall_pass": overall_pass,
        },
    }

    print(
        "       Metrics (calibrated): "
        f"AUC={calibrated_metrics.get('auc_roc')} | "
        f"PR-AUC={calibrated_metrics.get('pr_auc')} | "
        f"Brier={calibrated_metrics.get('brier')}"
    )

    print("\n[6/6] Saving artifacts...")
    model_path = MODEL_DIR / "vat_refund_model.joblib"
    calibrator_path = MODEL_DIR / "vat_refund_calibrator.joblib"
    quality_path = MODEL_DIR / "vat_refund_quality_report.json"
    baseline_path = MODEL_DIR / "vat_refund_drift_baseline.json"
    meta_path = MODEL_DIR / "vat_refund_model_meta.json"

    joblib.dump(model, model_path)
    if calibrator is not None:
        joblib.dump(calibrator, calibrator_path)
    elif calibrator_path.exists():
        calibrator_path.unlink()

    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    drift_baseline = _build_drift_baseline(features_df)
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(drift_baseline, f, ensure_ascii=False, indent=2)

    model_meta = {
        "track": "vat_refund",
        "model_version": VAT_MODEL_VERSION,
        "feature_set_version": VAT_FEATURE_SET_VERSION,
        "label_origin_policy": resolved_policy,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "sample_count": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 6),
        "calibration_enabled": calibrator is not None,
        "quality_report_file": quality_path.name,
        "drift_baseline_file": baseline_path.name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)

    print(f"       [OK] Model saved: {model_path}")
    print(f"       [OK] Metadata: {meta_path}")
    print(f"       [OK] Quality report: {quality_path}")
    print(f"       [OK] Drift baseline: {baseline_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    default_label_origin_policy = _normalize_label_origin_policy(
        os.getenv("VAT_REFUND_LABEL_ORIGIN_POLICY", "exclude_synthetic")
    )
    parser = argparse.ArgumentParser(description="Train VAT Refund model artifacts")
    parser.add_argument("--db-url", type=str, default=_resolve_default_db_url())
    parser.add_argument("--lookback-days", type=int, default=int(os.getenv("VAT_REFUND_LOOKBACK_DAYS", "365")))
    parser.add_argument("--min-samples", type=int, default=int(os.getenv("VAT_REFUND_MIN_SAMPLES", "80")))
    parser.add_argument(
        "--label-origin-policy",
        type=str,
        choices=list(VALID_LABEL_ORIGIN_POLICIES),
        default=default_label_origin_policy,
    )
    args = parser.parse_args()

    exit_code = main(
        db_url=args.db_url,
        lookback_days=max(30, int(args.lookback_days)),
        min_samples=max(20, int(args.min_samples)),
        label_origin_policy=args.label_origin_policy,
    )
    sys.exit(exit_code)
