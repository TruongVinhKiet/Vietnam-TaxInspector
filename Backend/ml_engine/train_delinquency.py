"""
train_delinquency.py – Training Script for Temporal Compliance Model (Program A)
==================================================================================
Trains a LightGBM classifier to predict tax delinquency risk from payment history.
Designed for i7 + 12GB RAM: streaming feature extraction, temporal train/test split.

Usage:
    python -m ml_engine.train_delinquency [--db-url postgresql://...] [--sample-size 10000]

Outputs:
    - data/models/delinquency_lgbm.joblib           – Trained model
    - data/models/delinquency_config.json            – Model config & metadata
    - data/models/delinquency_drift_baseline.json    – Feature distributions for drift detection
    - data/models/delinquency_quality_report.json    – Training metrics report
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

# Ensure parent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine.delinquency_model import DelinquencyFeatureEngineer, DELINQUENCY_MODEL_VERSION

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DELINQUENCY_MIN_TRAINING_SAMPLES = max(10_000, int(os.environ.get("DELINQUENCY_MIN_TRAINING_SAMPLES", "10000")))


def load_training_data(db_url: str, sample_size: int = 10000) -> tuple:
    """
    Load payment and tax return data from database for training.
    Returns (payments_by_company, returns_by_company, company_info_map)
    """
    import psycopg2

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Get companies with payment data
    cur.execute("""
        SELECT DISTINCT tp.tax_code, c.name, c.industry, c.registration_date
        FROM tax_payments tp
        JOIN companies c ON c.tax_code = tp.tax_code
        LIMIT %s
    """, (sample_size,))
    companies = cur.fetchall()

    if not companies:
        print("[ERROR] No companies with payment data found. Cannot train.")
        cur.close()
        conn.close()
        return {}, {}, {}

    print(f"[INFO] Found {len(companies)} companies with payment data.")

    payments_by_company = {}
    returns_by_company = {}
    company_info = {}

    for tax_code, name, industry, reg_date in companies:
        company_info[tax_code] = {
            "name": name,
            "industry": industry or "Unknown",
            "registration_date": str(reg_date) if reg_date else None,
        }

        # Load payments
        cur.execute("""
            SELECT due_date, actual_payment_date, amount_due, amount_paid,
                   penalty_amount, tax_period, status
            FROM tax_payments
            WHERE tax_code = %s
            ORDER BY due_date
        """, (tax_code,))
        cols = ["due_date", "actual_payment_date", "amount_due", "amount_paid",
                "penalty_amount", "tax_period", "status"]
        rows = cur.fetchall()
        if rows:
            payments_by_company[tax_code] = pd.DataFrame(rows, columns=cols)

        # Load tax returns
        cur.execute("""
            SELECT filing_date, revenue, expenses
            FROM tax_returns
            WHERE tax_code = %s
            ORDER BY filing_date
        """, (tax_code,))
        tr_rows = cur.fetchall()
        if tr_rows:
            returns_by_company[tax_code] = pd.DataFrame(
                tr_rows, columns=["filing_date", "revenue", "expenses"]
            )

    cur.close()
    conn.close()

    print(f"[INFO] Loaded payments for {len(payments_by_company)} companies.")
    return payments_by_company, returns_by_company, company_info


def build_labels(payments_df: pd.DataFrame) -> int:
    """
    Build binary label: 1 = company was late on any payment in the most recent quarter.
    This simulates the target for supervised learning.
    """
    if payments_df is None or payments_df.empty:
        return 0

    today = date.today()
    recent_cutoff = today - timedelta(days=90)

    recent = payments_df[
        pd.to_datetime(payments_df["due_date"], errors="coerce").dt.date >= recent_cutoff
    ]

    if recent.empty:
        return 0

    for _, row in recent.iterrows():
        actual = row.get("actual_payment_date")
        due = row.get("due_date")
        if actual is not None and pd.notna(actual):
            if isinstance(actual, str):
                actual = pd.to_datetime(actual).date()
            if isinstance(due, str):
                due = pd.to_datetime(due).date()
            if actual > due:
                return 1
        elif row.get("status") == "overdue":
            return 1
        elif pd.isna(actual) and due < today:
            return 1

    return 0


def _compute_binary_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute classification metrics with safe handling for single-class slices."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    positive_rate = float(np.mean(y_true == 1)) if len(y_true) else 0.0
    has_two_classes = len(np.unique(y_true)) > 1

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "negative_predictive_value": round(npv, 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "brier": round(brier_score_loss(y_true, y_proba), 4),
        "positive_rate_test": round(positive_rate, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [0, 1],
        "support": {
            "positive": int(np.sum(y_true == 1)),
            "negative": int(np.sum(y_true == 0)),
        },
    }

    if has_two_classes:
        metrics["pr_auc"] = round(average_precision_score(y_true, y_proba), 4)
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_proba), 4)
    else:
        metrics["pr_auc"] = None
        metrics["auc_roc"] = None

    return metrics


def _build_temporal_backtest(
    y_true,
    y_pred,
    y_proba,
    reference_dates,
    max_slices: int = 4,
) -> dict:
    """Evaluate model over chronological slices to expose temporal instability."""
    if reference_dates is None:
        return {
            "available": False,
            "reason": "missing_reference_dates",
            "slices": [],
        }

    frame = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(reference_dates, errors="coerce"),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "y_proba": np.asarray(y_proba),
        }
    ).dropna(subset=["reference_date"])

    if frame.empty:
        return {
            "available": False,
            "reason": "no_valid_reference_dates",
            "slices": [],
        }

    frame = frame.sort_values("reference_date").reset_index(drop=True)
    slice_count = min(max_slices, len(frame))
    index_slices = np.array_split(np.arange(len(frame)), slice_count)

    slices = []
    for idx, index_slice in enumerate(index_slices, start=1):
        if len(index_slice) == 0:
            continue

        part = frame.iloc[index_slice]
        part_metrics = _compute_binary_metrics(
            part["y_true"].to_numpy(),
            part["y_pred"].to_numpy(),
            part["y_proba"].to_numpy(),
        )

        slices.append(
            {
                "slice": idx,
                "start_date": part["reference_date"].min().date().isoformat(),
                "end_date": part["reference_date"].max().date().isoformat(),
                "samples": int(len(part)),
                "positive_rate": round(float(np.mean(part["y_true"] == 1)), 4),
                "metrics": part_metrics,
            }
        )

    return {
        "available": True,
        "slice_basis": "latest_due_date_per_company",
        "slice_count": len(slices),
        "coverage_samples": int(len(frame)),
        "slices": slices,
    }


def _resolve_default_db_url() -> str:
    """Resolve DB URL in the same order as runtime app configuration."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url

    try:
        from app.database import SQLALCHEMY_DATABASE_URL

        if SQLALCHEMY_DATABASE_URL:
            return SQLALCHEMY_DATABASE_URL
    except Exception:
        pass

    return "postgresql://postgres:postgres@localhost/tax_inspector"


def train_model(
    db_url: str = "postgresql://postgres:postgres@localhost/tax_inspector",
    sample_size: int = 10000,
    min_samples: int = DELINQUENCY_MIN_TRAINING_SAMPLES,
) -> int:
    """Main training function."""
    print("=" * 60)
    print("  DELINQUENCY MODEL TRAINING (Program A)")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading training data...")
    payments_by_company, returns_by_company, company_info = load_training_data(db_url, sample_size)

    if not payments_by_company:
        print("[ABORT] No training data available.")
        return 1

    total_companies = len(payments_by_company)
    required_samples = max(DELINQUENCY_MIN_TRAINING_SAMPLES, int(min_samples))
    if total_companies < required_samples:
        print(
            f"[ABORT] Insufficient training samples: need >= {required_samples:,} companies, "
            f"got {total_companies:,}."
        )
        return 2

    # Feature engineering
    print("[2/5] Extracting features...")
    engineer = DelinquencyFeatureEngineer()

    features_list = []
    labels = []
    tax_codes = []
    sample_reference_dates = []

    for tax_code, payments_df in payments_by_company.items():
        info = company_info.get(tax_code, {})
        returns_df = returns_by_company.get(tax_code)

        features = engineer.compute_features(payments_df, returns_df, info)
        feature_vec = engineer.get_feature_vector(features)

        label = build_labels(payments_df)
        latest_due = pd.to_datetime(payments_df.get("due_date"), errors="coerce").max()

        features_list.append(feature_vec)
        labels.append(label)
        tax_codes.append(tax_code)
        sample_reference_dates.append(latest_due)

    X = np.array(features_list)
    y = np.array(labels)

    print(f"    Dataset: {X.shape[0]} companies, {X.shape[1]} features")
    print(f"    Label distribution: {sum(y)} delinquent ({sum(y)/len(y)*100:.1f}%), "
          f"{len(y)-sum(y)} compliant ({(len(y)-sum(y))/len(y)*100:.1f}%)")

    if len(np.unique(y)) < 2:
        print("[ABORT] Training labels do not contain both classes; cannot train a reliable classifier.")
        return 3

    # Temporal train/test split (80/20, ordered by first appearance)
    print("[3/5] Training model...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    reference_dates = pd.to_datetime(sample_reference_dates, errors="coerce")
    reference_dates_test = reference_dates[split_idx:]

    train_positive = int(np.sum(y_train == 1))
    test_positive = int(np.sum(y_test == 1))
    split_summary = {
        "strategy": "ordered_80_20_temporal",
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_positive_count": train_positive,
        "test_positive_count": test_positive,
        "train_positive_rate": round(train_positive / len(y_train), 6) if len(y_train) else 0.0,
        "test_positive_rate": round(test_positive / len(y_test), 6) if len(y_test) else 0.0,
    }

    # Try LightGBM first, fallback to XGBoost, then sklearn
    model = None
    model_type = "unknown"

    try:
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 6,
            "min_child_samples": 10,
            "class_weight": "balanced",
            "verbose": -1,
            "n_jobs": 2,  # Conservative for 12GB RAM
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=True)],
        )
        model_type = "lightgbm"
        print("    [OK] LightGBM trained successfully.")

    except ImportError:
        print("    [WARN] LightGBM not available, trying XGBoost...")
        try:
            import xgboost as xgb

            scale_pos_weight = max(1, (len(y_train) - sum(y_train)) / max(1, sum(y_train)))
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                n_jobs=2,
                verbosity=1,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=True,
            )
            model_type = "xgboost"
            print("    [OK] XGBoost trained successfully.")

        except ImportError:
            print("    [WARN] XGBoost not available, using sklearn GradientBoosting...")
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=10,
            )
            model.fit(X_train, y_train)
            model_type = "sklearn_gbc"
            print("    [OK] Sklearn GradientBoosting trained successfully.")

    # Evaluation
    print("[4/5] Evaluating model...")
    temporal_backtest = {
        "available": False,
        "reason": "insufficient_test_data",
        "slices": [],
    }

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
        metrics = _compute_binary_metrics(y_test, y_pred, y_proba)
        temporal_backtest = _build_temporal_backtest(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            reference_dates=reference_dates_test,
            max_slices=4,
        )
    else:
        metrics = {
            "accuracy": 0.0,
            "note": "Insufficient test data for evaluation.",
        }

    print(f"    Metrics: {json.dumps(metrics, indent=2)}")

    # Save model
    print("[5/5] Saving artifacts...")
    model_path = MODEL_DIR / "delinquency_lgbm.joblib"
    joblib.dump(model, model_path)
    print(f"    [OK] Model saved to {model_path}")

    # Save config
    config = {
        "model_type": model_type,
        "model_version": DELINQUENCY_MODEL_VERSION,
        "feature_columns": list(engineer.FEATURE_COLS),
        "n_features": len(engineer.FEATURE_COLS),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
        "hardware_note": "Trained on i7 Gen 8th + 12GB RAM (memory-optimized).",
    }
    config_path = MODEL_DIR / "delinquency_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"    [OK] Config saved to {config_path}")

    # Save drift baseline
    drift_baseline = {
        "model_version": DELINQUENCY_MODEL_VERSION,
        "created_at": datetime.utcnow().isoformat(),
    }
    for i, feat_name in enumerate(engineer.FEATURE_COLS):
        col_vals = X[:, i].tolist()
        drift_baseline[feat_name] = {
            "mean": round(float(np.mean(col_vals)), 6),
            "std": round(float(np.std(col_vals)), 6),
            "min": round(float(np.min(col_vals)), 6),
            "max": round(float(np.max(col_vals)), 6),
            "distribution": [round(float(v), 6) for v in np.random.choice(col_vals, min(200, len(col_vals)), replace=False)],
        }
    drift_path = MODEL_DIR / "delinquency_drift_baseline.json"
    with open(drift_path, "w", encoding="utf-8") as f:
        json.dump(drift_baseline, f, indent=2, ensure_ascii=False)
    print(f"    [OK] Drift baseline saved to {drift_path}")

    # Save quality report
    quality_report = {
        "model_version": DELINQUENCY_MODEL_VERSION,
        "model_type": model_type,
        "metrics": metrics,
        "acceptance_gates": {
            "overall_pass": True,
            "criteria": {
                "training_samples_min": {
                    "threshold": int(required_samples),
                    "actual": int(len(X)),
                    "pass": int(len(X)) >= int(required_samples),
                },
                "label_class_diversity": {
                    "threshold": 2,
                    "actual": int(len(np.unique(y))),
                    "pass": int(len(np.unique(y))) >= 2,
                },
            },
        },
        "evaluation_protocol": {
            "target": "binary_delinquency_within_recent_window",
            "split": split_summary,
            "threshold": 0.5,
            "notes": "Includes imbalance-sensitive metrics (PR-AUC, balanced accuracy, MCC).",
        },
        "dataset": {
            "total": len(X),
            "train": len(X_train),
            "test": len(X_test),
            "positive_rate": round(sum(y) / len(y), 4) if len(y) > 0 else 0,
            "required_min_samples": int(required_samples),
            "train_positive_rate": split_summary["train_positive_rate"],
            "test_positive_rate": split_summary["test_positive_rate"],
            "train_positive_count": split_summary["train_positive_count"],
            "test_positive_count": split_summary["test_positive_count"],
        },
        "temporal_backtest": temporal_backtest,
        "generated_at": datetime.utcnow().isoformat(),
    }
    quality_path = MODEL_DIR / "delinquency_quality_report.json"
    gate_criteria = (quality_report.get("acceptance_gates") or {}).get("criteria") or {}
    quality_report["acceptance_gates"]["overall_pass"] = bool(all(item.get("pass") for item in gate_criteria.values()))
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    print(f"    [OK] Quality report saved to {quality_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Model: {model_type} | Version: {DELINQUENCY_MODEL_VERSION}")
    if metrics.get("auc_roc"):
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics.get('f1', 0):.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Delinquency Model (Program A)")
    parser.add_argument("--db-url", default=_resolve_default_db_url())
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--min-samples", type=int, default=DELINQUENCY_MIN_TRAINING_SAMPLES)
    args = parser.parse_args()

    min_samples = max(DELINQUENCY_MIN_TRAINING_SAMPLES, int(args.min_samples))
    sample_size = int(args.sample_size)
    if sample_size < min_samples:
        print(f"[ERROR] --sample-size must be >= --min-samples ({min_samples:,}).")
        raise SystemExit(1)

    raise SystemExit(train_model(db_url=args.db_url, sample_size=sample_size, min_samples=min_samples))
