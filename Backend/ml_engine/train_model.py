"""
train_model.py – Script huấn luyện mô hình AI trên Mock Data
===============================================================
Đọc tax_data_mock.csv -> Feature Engineering -> Train Models

Artifacts generated:
    - isolation_forest.joblib
    - xgboost_model.joblib
    - shap_background.joblib
    - fraud_calibrator.joblib
    - fraud_quality_report.json
    - fraud_drift_baseline.json

Usage:
    python -m ml_engine.train_model
    # or
    python ml_engine/train_model.py
"""

import os
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    brier_score_loss,
)
from sklearn.isotonic import IsotonicRegression

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine.feature_engineering import TaxFeatureEngineer


FRAUD_MODEL_VERSION = os.getenv("FRAUD_MODEL_VERSION", "fraud-hybrid-v2")
FRAUD_FEATURE_SET_VERSION = "fraud_inference_features_v1"
FRAUD_MIN_TRAINING_SAMPLES = max(10_000, int(os.getenv("FRAUD_MIN_REQUIRED_SAMPLES", "10000")))

TEMPORAL_AUC_DROP_MAX = 0.08
TEMPORAL_PR_AUC_DROP_MAX = 0.10
SLICE_MIN_AUC = 0.50
SLICE_MIN_SAMPLES = 80

PR_AUC_CORE_FLOOR = 0.10
PR_AUC_SLICE_FLOOR = 0.02
PR_AUC_THRESHOLD_CAP = 0.50
PR_AUC_CORE_LIFT = 2.2
PR_AUC_SLICE_LIFT = 0.5


def _expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        low = boundaries[idx]
        high = boundaries[idx + 1]
        if idx == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def _safe_train_test_split(X, y, test_size: float, random_state: int):
    """Use stratified split when possible; fallback safely for tiny/imbalanced sets."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _evaluate_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute robust probability metrics for binary fraud tasks."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    sample_count = int(len(y_true_arr))
    fraud_ratio = float(np.mean(y_true_arr)) if sample_count else 0.0

    snapshot = {
        "samples": sample_count,
        "fraud_ratio": round(fraud_ratio, 6),
        "auc_roc": None,
        "pr_auc": None,
        "brier": None,
        "ece": None,
        "available": False,
    }

    if sample_count == 0:
        return snapshot

    snapshot["brier"] = round(float(brier_score_loss(y_true_arr, y_prob_arr)), 6)
    snapshot["ece"] = round(float(_expected_calibration_error(y_true_arr, y_prob_arr, bins=10)), 6)

    if len(np.unique(y_true_arr)) >= 2:
        snapshot["auc_roc"] = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 6)
        snapshot["pr_auc"] = round(float(average_precision_score(y_true_arr, y_prob_arr)), 6)
        snapshot["available"] = True

    return snapshot


def _derive_pr_auc_thresholds(positive_rate: float) -> tuple[float, float]:
    """Derive class-imbalance-aware PR-AUC gates from the observed positive rate."""
    rate = float(np.clip(positive_rate, 1e-6, 0.5))

    core_threshold = min(PR_AUC_THRESHOLD_CAP, max(PR_AUC_CORE_FLOOR, rate * PR_AUC_CORE_LIFT))
    slice_threshold = min(core_threshold, max(PR_AUC_SLICE_FLOOR, rate * PR_AUC_SLICE_LIFT))

    return round(core_threshold, 6), round(slice_threshold, 6)


def _build_split_indices(frame: pd.DataFrame, y: np.ndarray) -> dict:
    """Prefer temporal split by year; fallback to stratified random split when needed."""
    year_series = pd.to_numeric(frame.get("year"), errors="coerce")
    years = sorted(int(v) for v in year_series.dropna().astype(int).unique().tolist())

    if len(years) >= 3:
        train_years = years[:-2]
        calib_year = years[-2]
        test_year = years[-1]

        train_idx = frame.index[frame["year"].isin(train_years)].to_numpy(dtype=int)
        calib_idx = frame.index[frame["year"] == calib_year].to_numpy(dtype=int)
        test_idx = frame.index[frame["year"] == test_year].to_numpy(dtype=int)

        temporal_valid = (
            len(train_idx) > 0
            and len(calib_idx) > 0
            and len(test_idx) > 0
            and len(np.unique(y[train_idx])) >= 2
            and len(np.unique(y[calib_idx])) >= 2
            and len(np.unique(y[test_idx])) >= 2
        )

        if temporal_valid:
            return {
                "mode": "temporal_year_split",
                "train_idx": train_idx,
                "calib_idx": calib_idx,
                "test_idx": test_idx,
                "train_years": train_years,
                "calibration_year": int(calib_year),
                "test_year": int(test_year),
                "fallback_reason": None,
            }

    all_indices = np.arange(len(frame), dtype=int)
    train_idx, holdout_idx = _safe_train_test_split(all_indices, y, test_size=0.3, random_state=42)
    holdout_idx = np.asarray(holdout_idx, dtype=int)
    calib_idx, test_idx = _safe_train_test_split(
        holdout_idx,
        y[holdout_idx],
        test_size=0.5,
        random_state=43,
    )

    return {
        "mode": "random_split_fallback",
        "train_idx": np.asarray(train_idx, dtype=int),
        "calib_idx": np.asarray(calib_idx, dtype=int),
        "test_idx": np.asarray(test_idx, dtype=int),
        "train_years": [],
        "calibration_year": None,
        "test_year": None,
        "fallback_reason": "insufficient_temporal_diversity_or_class_balance",
    }


def _build_slice_metrics(eval_frame: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[dict, dict]:
    """Compute slice metrics on evaluation split for fairness/stability visibility."""
    if eval_frame is None or eval_frame.empty or len(y_true) == 0:
        return {}, {
            "min_samples": SLICE_MIN_SAMPLES,
            "evaluated_group_count": 0,
            "min_auc_roc": None,
            "min_pr_auc": None,
        }

    payload = eval_frame.copy().reset_index(drop=True)
    payload["fraud_label"] = np.asarray(y_true, dtype=int)
    payload["fraud_probability"] = np.asarray(y_prob, dtype=float)

    revenue_series = pd.to_numeric(payload.get("revenue"), errors="coerce").fillna(0.0)
    payload["revenue_bucket"] = pd.cut(
        revenue_series,
        bins=[-np.inf, 1e9, 1e10, 1e11, np.inf],
        labels=["lt_1b", "1b_to_10b", "10b_to_100b", "gte_100b"],
        right=False,
    )

    dimensions = [
        ("industry", "industry"),
        ("province", "province"),
        ("revenue_bucket", "revenue_bucket"),
        ("year", "year"),
    ]

    results = {}
    auc_values = []
    pr_auc_values = []
    evaluated_group_count = 0

    for dimension_name, column in dimensions:
        if column not in payload.columns:
            continue

        dim_result = {}
        for raw_value, group in payload.groupby(column, dropna=False):
            if len(group) < SLICE_MIN_SAMPLES:
                continue

            group_metrics = _evaluate_probability_metrics(
                group["fraud_label"].to_numpy(dtype=int),
                group["fraud_probability"].to_numpy(dtype=float),
            )
            group_value = "unknown" if pd.isna(raw_value) else str(raw_value)
            dim_result[group_value] = group_metrics

            if group_metrics.get("auc_roc") is not None:
                auc_values.append(float(group_metrics["auc_roc"]))
            if group_metrics.get("pr_auc") is not None:
                pr_auc_values.append(float(group_metrics["pr_auc"]))
            evaluated_group_count += 1

        if dim_result:
            results[dimension_name] = dim_result

    summary = {
        "min_samples": SLICE_MIN_SAMPLES,
        "evaluated_group_count": evaluated_group_count,
        "min_auc_roc": round(float(min(auc_values)), 6) if auc_values else None,
        "min_pr_auc": round(float(min(pr_auc_values)), 6) if pr_auc_values else None,
    }
    return results, summary


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 60)
    print("  TaxInspector – AI Model Training Pipeline")
    print("=" * 60)

    # ---- 1. Load Data ----
    data_dir = Path(__file__).resolve().parent.parent / "data"
    csv_path = data_dir / "tax_data_mock.csv"

    if not csv_path.exists():
        print(f"[ERROR] Data file not found: {csv_path}")
        print("Run generate_mock_data.py first!")
        sys.exit(1)

    print(f"\n[1/6] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"       Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"       Fraud ratio: {df['fraud_label'].mean():.2%}")

    # ---- 2. Feature Engineering ----
    print("\n[2/6] Computing features (F1, F2, F3, F4)...")
    fe = TaxFeatureEngineer()
    df = fe.compute_features(df)
    print(f"       Feature columns: {fe.FEATURE_COLS}")

    # Keep full temporal rows for evaluation while preserving latest-year snapshot for drift baselines.
    model_df = df.sort_values(["year", "tax_code"]).reset_index(drop=True)
    latest_df = df.sort_values("year").groupby("tax_code").last().reset_index()
    if len(model_df) < FRAUD_MIN_TRAINING_SAMPLES:
        print(
            f"[ERROR] Fraud training requires at least {FRAUD_MIN_TRAINING_SAMPLES:,} rows; "
            f"got {len(model_df):,}."
        )
        sys.exit(2)
    print(f"       Companies (latest year): {len(latest_df):,}")

    X = fe.get_feature_matrix(model_df)
    y = model_df["fraud_label"].astype(int).values

    print(f"       X shape: {X.shape} | y balance: {y.sum()} fraud / {len(y)-y.sum()} normal")

    # ---- 3. Train Isolation Forest (Layer 1) ----
    print("\n[3/6] Training Isolation Forest (unsupervised)...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    split_info = _build_split_indices(model_df, y)
    train_idx = split_info["train_idx"]
    calib_idx = split_info["calib_idx"]
    test_idx = split_info["test_idx"]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_calib = X[calib_idx]
    y_calib = y[calib_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    if split_info["mode"] == "temporal_year_split":
        print(
            f"       Split: temporal train_years={split_info['train_years']} | "
            f"calib_year={split_info['calibration_year']} | test_year={split_info['test_year']}"
        )
    else:
        print("       [WARN] Temporal split unavailable, fallback to random split.")

    print(
        f"       Final split sizes: train={len(y_train)} | "
        f"calib={len(y_calib)} | test={len(y_test)}"
    )

    iso_forest.fit(X_train)

    # Evaluate on known labels
    raw_scores = iso_forest.decision_function(X_test)
    iso_predictions = iso_forest.predict(X_test)  # 1 = normal, -1 = anomaly
    iso_binary = (iso_predictions == -1).astype(int)
    iso_auc = roc_auc_score(y_test, -raw_scores)
    print(f"       Isolation Forest AUC: {iso_auc:.4f}")
    print(f"       Detected anomalies: {iso_binary.sum()} / {len(iso_binary)}")

    # ---- 4. Train XGBoost (Layer 2) ----
    print("\n[4/6] Training XGBoost classifier (supervised + calibration)...")

    # We DON'T add anomaly_score to X for XGBoost because
    # the pipeline uses the same feature set. Instead, XGBoost
    # learns directly from the financial features.
    # (anomaly_score is computed at inference time separately)

    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("       [WARN] XGBoost not installed, using sklearn GradientBoosting")
        xgb_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    xgb_model.fit(X_train, y_train)

    # Fit isotonic calibrator on dedicated calibration split
    y_prob_calib_raw = np.clip(xgb_model.predict_proba(X_calib)[:, 1], 0.0, 1.0)
    fraud_calibrator = None
    calibration_method = "identity"
    if len(np.unique(y_calib)) >= 2 and len(y_calib) >= 20:
        try:
            fraud_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            fraud_calibrator.fit(y_prob_calib_raw, y_calib)
            calibration_method = "isotonic"
            print("       [OK] Isotonic calibrator fitted.")
        except Exception as exc:
            print(f"       [WARN] Calibrator fit failed, fallback to raw probabilities: {exc}")
            fraud_calibrator = None
            calibration_method = "identity"
    else:
        print("       [WARN] Calibration split lacks class diversity, skip isotonic fitting.")

    # Evaluate on held-out test set
    y_prob_raw = np.clip(xgb_model.predict_proba(X_test)[:, 1], 0.0, 1.0)
    if fraud_calibrator is not None:
        y_prob_cal = np.clip(fraud_calibrator.predict(y_prob_raw), 0.0, 1.0)
    else:
        y_prob_cal = y_prob_raw.copy()

    y_pred = (y_prob_cal >= 0.5).astype(int)

    xgb_auc_raw = roc_auc_score(y_test, y_prob_raw)
    xgb_auc = roc_auc_score(y_test, y_prob_cal)
    xgb_pr_auc_raw = average_precision_score(y_test, y_prob_raw)
    xgb_pr_auc = average_precision_score(y_test, y_prob_cal)
    xgb_brier_raw = brier_score_loss(y_test, y_prob_raw)
    xgb_brier = brier_score_loss(y_test, y_prob_cal)
    xgb_ece_raw = _expected_calibration_error(y_test, y_prob_raw, bins=10)
    xgb_ece = _expected_calibration_error(y_test, y_prob_cal, bins=10)
    pr_auc_core_threshold, pr_auc_slice_threshold = _derive_pr_auc_thresholds(float(np.mean(y_test)))

    print(f"       XGBoost AUC (raw):        {xgb_auc_raw:.4f}")
    print(f"       XGBoost AUC (calibrated): {xgb_auc:.4f}")
    print(f"       PR-AUC   (raw/cal):       {xgb_pr_auc_raw:.4f} / {xgb_pr_auc:.4f}")
    print(f"       Brier    (raw/cal):       {xgb_brier_raw:.4f} / {xgb_brier:.4f}")
    print(f"       ECE      (raw/cal):       {xgb_ece_raw:.4f} / {xgb_ece:.4f}")
    print(f"       PR-AUC gate (core/slice): {pr_auc_core_threshold:.4f} / {pr_auc_slice_threshold:.4f}")
    print(f"\n       Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"       Confusion Matrix:")
    print(f"       {cm}")

    # ---- 4.1 Temporal Evaluation + Slice Metrics (soft gates) ----
    temporal_segments = {}
    split_indices_map = {
        "train": train_idx,
        "calibration": calib_idx,
        "test": test_idx,
    }
    for segment_name, segment_idx in split_indices_map.items():
        segment_X = X[segment_idx]
        segment_y = y[segment_idx]
        segment_prob_raw = np.clip(xgb_model.predict_proba(segment_X)[:, 1], 0.0, 1.0)
        if fraud_calibrator is not None:
            segment_prob_cal = np.clip(fraud_calibrator.predict(segment_prob_raw), 0.0, 1.0)
        else:
            segment_prob_cal = segment_prob_raw

        segment_metrics = _evaluate_probability_metrics(segment_y, segment_prob_cal)
        segment_years = sorted(
            int(v)
            for v in pd.to_numeric(model_df.iloc[segment_idx]["year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        temporal_segments[segment_name] = {
            "years": segment_years,
            "metrics": segment_metrics,
        }

    temporal_drop = {
        "auc_roc_drop_calibration_to_test": None,
        "pr_auc_drop_calibration_to_test": None,
    }
    calib_auc = temporal_segments["calibration"]["metrics"].get("auc_roc")
    test_auc = temporal_segments["test"]["metrics"].get("auc_roc")
    if calib_auc is not None and test_auc is not None:
        temporal_drop["auc_roc_drop_calibration_to_test"] = round(max(0.0, calib_auc - test_auc), 6)

    calib_pr_auc = temporal_segments["calibration"]["metrics"].get("pr_auc")
    test_pr_auc = temporal_segments["test"]["metrics"].get("pr_auc")
    if calib_pr_auc is not None and test_pr_auc is not None:
        temporal_drop["pr_auc_drop_calibration_to_test"] = round(max(0.0, calib_pr_auc - test_pr_auc), 6)

    temporal_evaluation = {
        "strategy": split_info["mode"],
        "fallback_reason": split_info.get("fallback_reason"),
        "segments": temporal_segments,
        "generalization_drop": temporal_drop,
        "train_years": split_info.get("train_years", []),
        "calibration_year": split_info.get("calibration_year"),
        "test_year": split_info.get("test_year"),
    }

    slice_eval_frame = model_df.iloc[test_idx].copy().reset_index(drop=True)
    slice_metrics, slice_summary = _build_slice_metrics(slice_eval_frame, y_test, y_prob_cal)

    # Feature importances
    print("\n       Feature Importances:")
    importances = xgb_model.feature_importances_
    for name, imp in sorted(zip(fe.FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"         {name:25s} {imp:.4f} {bar}")

    # ---- 5. Save Models + Calibration + Quality Artifacts ----
    print("\n[5/6] Saving models + quality artifacts...")
    model_dir = data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    iso_path = model_dir / "isolation_forest.joblib"
    xgb_path = model_dir / "xgboost_model.joblib"
    bg_path = model_dir / "shap_background.joblib"
    cal_path = model_dir / "fraud_calibrator.joblib"
    quality_path = model_dir / "fraud_quality_report.json"
    drift_baseline_path = model_dir / "fraud_drift_baseline.json"
    manifest_path = model_dir / "fraud_model_manifest.json"

    joblib.dump(iso_forest, iso_path)
    joblib.dump(xgb_model, xgb_path)

    # Save a small background sample for SHAP TreeExplainer (100 rows, stratified)
    bg_size = min(100, len(X))
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(X), size=bg_size, replace=False)
    bg_data = X[bg_indices]
    joblib.dump(bg_data, bg_path)

    joblib.dump(
        {
            "calibrator": fraud_calibrator,
            "method": calibration_method,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "model_version": FRAUD_MODEL_VERSION,
            "raw_metrics": {
                "auc_roc": float(xgb_auc_raw),
                "pr_auc": float(xgb_pr_auc_raw),
                "brier": float(xgb_brier_raw),
                "ece": float(xgb_ece_raw),
            },
            "calibrated_metrics": {
                "auc_roc": float(xgb_auc),
                "pr_auc": float(xgb_pr_auc),
                "brier": float(xgb_brier),
                "ece": float(xgb_ece),
            },
        },
        cal_path,
    )

    core_criteria = {
        "training_samples_min": {
            "threshold": FRAUD_MIN_TRAINING_SAMPLES,
            "actual": int(len(model_df)),
            "pass": bool(int(len(model_df)) >= FRAUD_MIN_TRAINING_SAMPLES),
        },
        "auc_roc_min": {
            "threshold": 0.70,
            "actual": round(float(xgb_auc), 6),
            "pass": bool(float(xgb_auc) >= 0.70),
        },
        "pr_auc_min": {
            "threshold": pr_auc_core_threshold,
            "actual": round(float(xgb_pr_auc), 6),
            "pass": bool(float(xgb_pr_auc) >= pr_auc_core_threshold),
        },
        "brier_max": {
            "threshold": 0.30,
            "actual": round(float(xgb_brier), 6),
            "pass": bool(float(xgb_brier) <= 0.30),
        },
        "ece_max": {
            "threshold": 0.15,
            "actual": round(float(xgb_ece), 6),
            "pass": bool(float(xgb_ece) <= 0.15),
        },
        "brier_not_worse_than_raw": {
            "threshold": 0.02,
            "actual": round(float(xgb_brier - xgb_brier_raw), 6),
            "pass": bool(float(xgb_brier - xgb_brier_raw) <= 0.02),
        },
    }

    soft_criteria = {}

    temporal_auc_drop = temporal_drop.get("auc_roc_drop_calibration_to_test")
    if temporal_auc_drop is None:
        soft_criteria["temporal_auc_drop_max_soft"] = {
            "threshold": TEMPORAL_AUC_DROP_MAX,
            "actual": None,
            "pass": True,
            "soft_gate": True,
            "note": "auc_drop_unavailable",
        }
    else:
        soft_criteria["temporal_auc_drop_max_soft"] = {
            "threshold": TEMPORAL_AUC_DROP_MAX,
            "actual": round(float(temporal_auc_drop), 6),
            "pass": bool(float(temporal_auc_drop) <= TEMPORAL_AUC_DROP_MAX),
            "soft_gate": True,
        }

    temporal_pr_drop = temporal_drop.get("pr_auc_drop_calibration_to_test")
    if temporal_pr_drop is None:
        soft_criteria["temporal_pr_auc_drop_max_soft"] = {
            "threshold": TEMPORAL_PR_AUC_DROP_MAX,
            "actual": None,
            "pass": True,
            "soft_gate": True,
            "note": "pr_auc_drop_unavailable",
        }
    else:
        soft_criteria["temporal_pr_auc_drop_max_soft"] = {
            "threshold": TEMPORAL_PR_AUC_DROP_MAX,
            "actual": round(float(temporal_pr_drop), 6),
            "pass": bool(float(temporal_pr_drop) <= TEMPORAL_PR_AUC_DROP_MAX),
            "soft_gate": True,
        }

    min_slice_auc = slice_summary.get("min_auc_roc")
    if min_slice_auc is None:
        soft_criteria["slice_min_auc_soft"] = {
            "threshold": SLICE_MIN_AUC,
            "actual": None,
            "pass": True,
            "soft_gate": True,
            "note": "slice_auc_unavailable",
        }
    else:
        soft_criteria["slice_min_auc_soft"] = {
            "threshold": SLICE_MIN_AUC,
            "actual": round(float(min_slice_auc), 6),
            "pass": bool(float(min_slice_auc) >= SLICE_MIN_AUC),
            "soft_gate": True,
        }

    min_slice_pr_auc = slice_summary.get("min_pr_auc")
    if min_slice_pr_auc is None:
        soft_criteria["slice_min_pr_auc_soft"] = {
            "threshold": pr_auc_slice_threshold,
            "actual": None,
            "pass": True,
            "soft_gate": True,
            "note": "slice_pr_auc_unavailable",
        }
    else:
        soft_criteria["slice_min_pr_auc_soft"] = {
            "threshold": pr_auc_slice_threshold,
            "actual": round(float(min_slice_pr_auc), 6),
            "pass": bool(float(min_slice_pr_auc) >= pr_auc_slice_threshold),
            "soft_gate": True,
        }

    criteria = {**core_criteria, **soft_criteria}
    core_overall_pass = all(bool(v.get("pass")) for v in core_criteria.values())

    quality_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "model_version": FRAUD_MODEL_VERSION,
            "feature_count": int(X.shape[1]),
            "calibration_method": calibration_method,
        },
        "dataset": {
            "total_size": int(len(model_df)),
            "required_min_samples": int(FRAUD_MIN_TRAINING_SAMPLES),
            "companies": int(len(latest_df)),
            "fraud_ratio": float(np.mean(y)),
            "train_size": int(len(y_train)),
            "calibration_size": int(len(y_calib)),
            "test_size": int(len(y_test)),
        },
        "performance": {
            "raw": {
                "auc_roc": round(float(xgb_auc_raw), 6),
                "pr_auc": round(float(xgb_pr_auc_raw), 6),
                "brier": round(float(xgb_brier_raw), 6),
                "ece": round(float(xgb_ece_raw), 6),
            },
            "calibrated": {
                "auc_roc": round(float(xgb_auc), 6),
                "pr_auc": round(float(xgb_pr_auc), 6),
                "brier": round(float(xgb_brier), 6),
                "ece": round(float(xgb_ece), 6),
            },
        },
        "calibration": {
            "available": fraud_calibrator is not None,
            "method": calibration_method,
            "brier_improvement": round(float(xgb_brier_raw - xgb_brier), 6),
        },
        "temporal_evaluation": temporal_evaluation,
        "slice_metrics": {
            "evaluation_scope": "test_split",
            "min_samples": SLICE_MIN_SAMPLES,
            "dimensions": slice_metrics,
            "summary": slice_summary,
        },
        "acceptance_gates": {
            "overall_pass": core_overall_pass,
            "core_criteria": sorted(core_criteria.keys()),
            "soft_gate_criteria": sorted(soft_criteria.keys()),
            "criteria": criteria,
        },
    }
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    baseline_features = latest_df[TaxFeatureEngineer.FEATURE_COLS].copy()
    latest_matrix = fe.get_feature_matrix(latest_df)
    baseline_raw_scores = iso_forest.decision_function(latest_matrix)
    baseline_features["anomaly_score"] = np.clip(0.5 - baseline_raw_scores, 0.0, 1.0)
    drift_baseline = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": FRAUD_MODEL_VERSION,
        "feature_set": FRAUD_FEATURE_SET_VERSION,
        "samples": int(len(baseline_features)),
        "features": {},
    }

    for feature_name in list(TaxFeatureEngineer.FEATURE_COLS) + ["anomaly_score"]:
        values = pd.to_numeric(baseline_features[feature_name], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
        if len(values) == 0:
            continue
        drift_baseline["features"][feature_name] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
            "q05": round(float(np.percentile(values, 5)), 6),
            "q95": round(float(np.percentile(values, 95)), 6),
            "q0": round(float(np.min(values)), 6),
            "q100": round(float(np.max(values)), 6),
        }

    with open(drift_baseline_path, "w", encoding="utf-8") as f:
        json.dump(drift_baseline, f, ensure_ascii=False, indent=2)

    artifact_files = {
        "isolation_forest": iso_path,
        "xgboost_model": xgb_path,
        "shap_background": bg_path,
        "fraud_calibrator": cal_path,
        "fraud_quality_report": quality_path,
        "fraud_drift_baseline": drift_baseline_path,
    }
    manifest_artifacts = {}
    for key, artifact_path in artifact_files.items():
        manifest_artifacts[key] = {
            "filename": artifact_path.name,
            "size_bytes": int(artifact_path.stat().st_size),
            "sha256": _file_sha256(artifact_path),
            "updated_at": datetime.utcfromtimestamp(artifact_path.stat().st_mtime).isoformat() + "Z",
        }

    manifest_payload = {
        "manifest_version": "fraud-model-manifest-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": FRAUD_MODEL_VERSION,
        "feature_contract": {
            "feature_set": FRAUD_FEATURE_SET_VERSION,
            "feature_columns": list(TaxFeatureEngineer.FEATURE_COLS),
            "feature_count": int(len(TaxFeatureEngineer.FEATURE_COLS)),
        },
        "calibration": {
            "available": fraud_calibrator is not None,
            "method": calibration_method,
        },
        "training_summary": {
            "company_count": int(len(latest_df)),
            "training_rows": int(len(model_df)),
            "fraud_ratio": float(np.mean(y)),
            "auc_roc": round(float(xgb_auc), 6),
            "pr_auc": round(float(xgb_pr_auc), 6),
            "brier": round(float(xgb_brier), 6),
            "ece": round(float(xgb_ece), 6),
            "split_strategy": split_info.get("mode"),
            "soft_gate_warning_count": int(
                sum(1 for item in soft_criteria.values() if item.get("pass") is False)
            ),
            "quality_gate_pass": bool(quality_report["acceptance_gates"]["overall_pass"]),
        },
        "artifacts": manifest_artifacts,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)

    print("\n[6/6] Finalizing artifacts...")

    print(f"       Saved: {iso_path}")
    print(f"       Saved: {xgb_path}")
    print(f"       Saved: {bg_path} ({bg_size} samples)")
    print(f"       Saved: {cal_path}")
    print(f"       Saved: {quality_path}")
    print(f"       Saved: {drift_baseline_path}")
    print(f"       Saved: {manifest_path}")
    print(f"       Iso Forest size: {os.path.getsize(iso_path)/1024:.1f} KB")
    print(f"       XGBoost size:    {os.path.getsize(xgb_path)/1024:.1f} KB")

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Isolation Forest AUC: {iso_auc:.4f}")
    print(f"  XGBoost AUC (cal):    {xgb_auc:.4f}")
    print(f"  PR-AUC (cal):         {xgb_pr_auc:.4f}")
    print(f"  Brier (cal):          {xgb_brier:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
