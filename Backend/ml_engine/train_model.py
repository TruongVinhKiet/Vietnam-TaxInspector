"""
train_model.py – Script huấn luyện mô hình AI trên Mock Data
===============================================================
Đọc tax_data_mock.csv -> Feature Engineering -> Train Models -> Save .joblib

Usage:
    python -m ml_engine.train_model
    # or
    python ml_engine/train_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine.feature_engineering import TaxFeatureEngineer


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

    print(f"\n[1/5] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"       Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"       Fraud ratio: {df['fraud_label'].mean():.2%}")

    # ---- 2. Feature Engineering ----
    print("\n[2/5] Computing features (F1, F2, F3, F4)...")
    fe = TaxFeatureEngineer()
    df = fe.compute_features(df)
    print(f"       Feature columns: {fe.FEATURE_COLS}")

    # Use latest year per company for training
    latest_df = df.sort_values("year").groupby("tax_code").last().reset_index()
    print(f"       Companies (latest year): {len(latest_df):,}")

    X = fe.get_feature_matrix(latest_df)
    y = latest_df["fraud_label"].values

    print(f"       X shape: {X.shape} | y balance: {y.sum()} fraud / {len(y)-y.sum()} normal")

    # ---- 3. Train Isolation Forest (Layer 1) ----
    print("\n[3/5] Training Isolation Forest (unsupervised)...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X)

    # Evaluate on known labels
    raw_scores = iso_forest.decision_function(X)
    iso_predictions = iso_forest.predict(X)  # 1 = normal, -1 = anomaly
    iso_binary = (iso_predictions == -1).astype(int)
    iso_auc = roc_auc_score(y, -raw_scores)
    print(f"       Isolation Forest AUC: {iso_auc:.4f}")
    print(f"       Detected anomalies: {iso_binary.sum()} / {len(iso_binary)}")

    # ---- 4. Train XGBoost (Layer 2) ----
    print("\n[4/5] Training XGBoost classifier (supervised)...")

    # Add anomaly_score as extra feature for XGBoost
    anomaly_scores = np.clip(0.5 - raw_scores, 0, 1).reshape(-1, 1)

    # We DON'T add anomaly_score to X for XGBoost because
    # the pipeline uses the same feature set. Instead, XGBoost
    # learns directly from the financial features.
    # (anomaly_score is computed at inference time separately)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    xgb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, y_prob)

    print(f"       XGBoost AUC: {xgb_auc:.4f}")
    print(f"\n       Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"       Confusion Matrix:")
    print(f"       {cm}")

    # Feature importances
    print("\n       Feature Importances:")
    importances = xgb_model.feature_importances_
    for name, imp in sorted(zip(fe.FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"         {name:25s} {imp:.4f} {bar}")

    # ---- 5. Save Models + SHAP Background Data ----
    print("\n[5/5] Saving models...")
    model_dir = data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    iso_path = model_dir / "isolation_forest.joblib"
    xgb_path = model_dir / "xgboost_model.joblib"
    bg_path  = model_dir / "shap_background.joblib"

    joblib.dump(iso_forest, iso_path)
    joblib.dump(xgb_model, xgb_path)

    # Save a small background sample for SHAP TreeExplainer (100 rows, stratified)
    bg_size = min(100, len(X))
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(X), size=bg_size, replace=False)
    bg_data = X[bg_indices]
    joblib.dump(bg_data, bg_path)

    print(f"       Saved: {iso_path}")
    print(f"       Saved: {xgb_path}")
    print(f"       Saved: {bg_path} ({bg_size} samples)")
    print(f"       Iso Forest size: {os.path.getsize(iso_path)/1024:.1f} KB")
    print(f"       XGBoost size:    {os.path.getsize(xgb_path)/1024:.1f} KB")

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Isolation Forest AUC: {iso_auc:.4f}")
    print(f"  XGBoost AUC:          {xgb_auc:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
