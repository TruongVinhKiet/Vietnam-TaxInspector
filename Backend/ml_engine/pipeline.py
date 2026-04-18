"""
pipeline.py – Hybrid AI Pipeline (Isolation Forest + XGBoost + SHAP)
======================================================================
Lớp 1: Isolation Forest (Unsupervised Anomaly Detection)
Lớp 2: XGBoost (Supervised Fraud Probability)
Lớp 3: SHAP (Explainable AI – Local per-sample explanation)

Hybrid Fusion Formula:
    risk_score = 0.80 * XGBoost_Probability * 100
               + 0.20 * IsolationForest_AnomalyScore * 100
    (capped at 0–100)

Pipeline có 2 chế độ:
    1. predict_single(features_dict) → dùng cho tra cứu real-time
    2. predict_batch(dataframe)      → dùng cho phân tích lô CSV
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .feature_engineering import TaxFeatureEngineer

try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Path to saved models
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
FRAUD_CALIBRATOR_FILENAME = "fraud_calibrator.joblib"
FRAUD_MANIFEST_FILENAME = "fraud_model_manifest.json"


def _fit_safe_pca_projection(matrix: np.ndarray) -> Optional[dict]:
    if not PCA_AVAILABLE or matrix is None:
        return None

    try:
        matrix_np = np.asarray(matrix, dtype=float)
    except Exception:
        return None

    if matrix_np.ndim != 2 or matrix_np.shape[0] < 2 or matrix_np.shape[1] < 2:
        return None

    matrix_np = np.nan_to_num(matrix_np, nan=0.0, posinf=0.0, neginf=0.0)
    non_zero_variance_cols = int(np.sum(np.var(matrix_np, axis=0) > 0.0))
    if non_zero_variance_cols < 2:
        return None

    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix_np)
        matrix_scaled = np.nan_to_num(matrix_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        pca_model = PCA(n_components=2)
        coords = pca_model.fit_transform(matrix_scaled)
    except Exception:
        return None

    explained = np.nan_to_num(
        getattr(pca_model, "explained_variance_ratio_", np.array([0.0, 0.0])),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return {
        "coords": coords,
        "explained_variance_ratio": explained,
        "scaler": scaler,
        "pca_model": pca_model,
    }


class TaxFraudPipeline:
    """
    Hybrid AI Pipeline for Tax Fraud Detection.
    Loads pre-trained models and runs inference.
    """

    # Hybrid fusion weights – tunable
    WEIGHT_XGB = 0.80
    WEIGHT_ISO = 0.20

    # Minimal raw contract for model serving.
    HARD_REQUIRED_COLUMNS = (
        "tax_code",
        "year",
        "revenue",
        "total_expenses",
        "net_profit",
        "vat_input",
        "vat_output",
    )

    DEFAULT_RAW_COLUMNS = {
        "company_name": "",
        "industry": "Unknown",
        "industry_avg_profit_margin": 0.08,
    }

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.feature_engineer = TaxFeatureEngineer()
        self.isolation_forest = None
        self.xgboost_model = None
        self.fraud_calibrator = None
        self.calibration_meta = {}
        self.model_manifest = {}
        self._shap_explainer = None  # Lazy-loaded SHAP TreeExplainer
        self._shap_background = None  # Background data for SHAP
        self._loaded = False

    def load_models(self):
        """Load pre-trained models from disk."""
        iso_path = self.model_dir / "isolation_forest.joblib"
        xgb_path = self.model_dir / "xgboost_model.joblib"

        if not iso_path.exists() or not xgb_path.exists():
            raise FileNotFoundError(
                f"Models not found in {self.model_dir}. "
                "Run train_model.py first to train and save models."
            )

        self.isolation_forest = joblib.load(iso_path)
        self.xgboost_model = joblib.load(xgb_path)

        # Load SHAP background data if available
        bg_path = self.model_dir / "shap_background.joblib"
        if bg_path.exists():
            self._shap_background = joblib.load(bg_path)
            print(f"[OK] SHAP background loaded ({len(self._shap_background)} samples)")

        self._load_model_manifest()
        self._load_fraud_calibrator()

        self._loaded = True
        print(f"[OK] Models loaded from {self.model_dir}")

    def ensure_loaded(self):
        if not self._loaded:
            self.load_models()

    def _load_model_manifest(self):
        manifest_path = self.model_dir / FRAUD_MANIFEST_FILENAME
        if not manifest_path.exists():
            self.model_manifest = {}
            return

        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
                self.model_manifest = payload if isinstance(payload, dict) else {}
            print(f"[OK] Fraud model manifest loaded from {manifest_path}")
        except Exception as exc:
            self.model_manifest = {}
            print(f"[WARN] Failed to load fraud model manifest: {exc}")

    def get_serving_metadata(self) -> dict:
        self.ensure_loaded()

        model_version = self.model_manifest.get("model_version") if isinstance(self.model_manifest, dict) else None
        if not model_version:
            model_version = self.calibration_meta.get("model_version")
        if not model_version:
            model_version = "fraud-hybrid-legacy"

        feature_contract = self.model_manifest.get("feature_contract", {}) if isinstance(self.model_manifest, dict) else {}
        return {
            "model_version": model_version,
            "manifest_version": self.model_manifest.get("manifest_version") if isinstance(self.model_manifest, dict) else None,
            "feature_set": feature_contract.get("feature_set") if isinstance(feature_contract, dict) else None,
            "calibration_method": self.calibration_meta.get("method", "identity"),
            "calibrator_available": bool(self.calibration_meta.get("available", False)),
        }

    def _load_fraud_calibrator(self):
        """Load optional fraud probability calibrator for better risk probability reliability."""
        cal_path = self.model_dir / FRAUD_CALIBRATOR_FILENAME
        if not cal_path.exists():
            self.fraud_calibrator = None
            self.calibration_meta = {
                "available": False,
                "method": "identity",
            }
            return

        try:
            payload = joblib.load(cal_path)
            if isinstance(payload, dict):
                self.fraud_calibrator = payload.get("calibrator")
                self.calibration_meta = {
                    "available": self.fraud_calibrator is not None,
                    "method": str(payload.get("method", "isotonic")),
                    "trained_at": payload.get("trained_at"),
                    "model_version": payload.get("model_version"),
                }
            else:
                self.fraud_calibrator = payload
                self.calibration_meta = {
                    "available": True,
                    "method": "isotonic",
                    "trained_at": None,
                }
            print(f"[OK] Fraud calibrator loaded from {cal_path}")
        except Exception as exc:
            self.fraud_calibrator = None
            self.calibration_meta = {
                "available": False,
                "method": "identity",
                "error": str(exc),
            }
            print(f"[WARN] Failed to load fraud calibrator: {exc}")

    def _calibrate_fraud_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibrator when available; otherwise use raw probabilities."""
        probs = np.asarray(raw_probs, dtype=float)
        if self.fraud_calibrator is None:
            return np.clip(probs, 0.0, 1.0)

        try:
            calibrated = self.fraud_calibrator.predict(probs)
            return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
        except Exception as exc:
            print(f"[WARN] Calibrator inference failed, fallback to raw probabilities: {exc}")
            return np.clip(probs, 0.0, 1.0)

    def _coerce_raw_input_dataframe(self, frame: pd.DataFrame, context: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            raise ValueError(f"{context}: input payload is empty")

        data = frame.copy()
        missing = [col for col in self.HARD_REQUIRED_COLUMNS if col not in data.columns]
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"{context}: missing required columns: {joined}")

        for col, default_value in self.DEFAULT_RAW_COLUMNS.items():
            if col not in data.columns:
                data[col] = default_value

        # Canonicalize tax_code to a trimmed string key used consistently across grouping/filtering.
        tax_code_series = data["tax_code"].astype(str).str.strip()
        tax_code_series = tax_code_series.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        tax_code_series = tax_code_series.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        data["tax_code"] = tax_code_series

        if "cost_of_goods" not in data.columns:
            data["cost_of_goods"] = pd.to_numeric(data["total_expenses"], errors="coerce") * 0.75
        if "operating_expenses" not in data.columns:
            data["operating_expenses"] = pd.to_numeric(data["total_expenses"], errors="coerce") * 0.25

        numeric_cols = (
            "year",
            "revenue",
            "cost_of_goods",
            "operating_expenses",
            "total_expenses",
            "net_profit",
            "vat_input",
            "vat_output",
            "industry_avg_profit_margin",
        )
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.replace([np.inf, -np.inf], np.nan)

        if data["tax_code"].isna().all() or (data["tax_code"].astype(str).str.strip() == "").all():
            raise ValueError(f"{context}: tax_code must contain at least one non-empty value")

        if data["year"].isna().any() or (data["year"] <= 0).any():
            raise ValueError(f"{context}: year must be a positive number for all records")

        data["year"] = data["year"].astype(int)

        non_nullable = ("revenue", "total_expenses", "net_profit", "vat_input", "vat_output")
        for col in non_nullable:
            if data[col].isna().any():
                raise ValueError(f"{context}: {col} contains invalid numeric values")

        return data

    def _validate_feature_frame(self, frame: pd.DataFrame, context: str) -> None:
        missing_features = [col for col in TaxFeatureEngineer.FEATURE_COLS if col not in frame.columns]
        if missing_features:
            joined = ", ".join(sorted(missing_features))
            raise ValueError(f"{context}: missing engineered features: {joined}")

        feature_values = frame[TaxFeatureEngineer.FEATURE_COLS].to_numpy(dtype=float)
        if not np.isfinite(feature_values).all():
            raise ValueError(f"{context}: engineered features contain non-finite values")

    def predict_single(self, company_data: dict) -> dict:
        """
        Real-time single query prediction.

        Args:
            company_data: Dict with 3 years of financial data for one company.
                          Should have keys matching CSV columns.

        Returns:
            Dict with risk_score, anomaly_score, red_flags, features, etc.
        """
        self.ensure_loaded()

        # Convert dict to DataFrame (single or multi-year rows)
        if isinstance(company_data, list):
            df = pd.DataFrame(company_data)
        else:
            df = pd.DataFrame([company_data])

        df = self._coerce_raw_input_dataframe(df, context="single_inference")

        # Feature engineering
        df = self.feature_engineer.compute_features(df)
        self._validate_feature_frame(df, context="single_inference")

        # Keep one snapshot per year to support temporal single-query charts.
        yearly_feature_frame = (
            df.sort_values(["year"])
            .groupby("year", as_index=False)
            .last()
            .sort_values("year")
            .reset_index(drop=True)
        )

        # Use the most recent year for primary scoring.
        latest_frame = yearly_feature_frame.sort_values("year", ascending=False).head(1)
        latest = latest_frame.iloc[0]
        X = self.feature_engineer.get_feature_matrix(latest_frame)

        # Build yearly feature/risk progression for single-query advanced charts.
        X_yearly = self.feature_engineer.get_feature_matrix(yearly_feature_frame)
        yearly_raw_anomaly = self.isolation_forest.decision_function(X_yearly)
        yearly_anomaly_scores = np.clip(0.5 - yearly_raw_anomaly, 0, 1)
        yearly_raw_probs = self.xgboost_model.predict_proba(X_yearly)[:, 1]
        yearly_fraud_probs = self._calibrate_fraud_probs(yearly_raw_probs)
        yearly_risk_scores = np.round(
            np.minimum(100.0, self.WEIGHT_XGB * yearly_fraud_probs * 100
                             + self.WEIGHT_ISO * yearly_anomaly_scores * 100),
            2,
        )

        yearly_feature_scores = []
        for idx, row in yearly_feature_frame.iterrows():
            yearly_feature_scores.append({
                "year": int(row.get("year", 0)),
                "risk_score": round(float(yearly_risk_scores[idx]), 2),
                "f1_divergence": round(float(row.get("f1_divergence", 0)), 4),
                "f2_ratio_limit": round(float(row.get("f2_ratio_limit", 0)), 4),
                "f3_vat_structure": round(float(row.get("f3_vat_structure", 0)), 4),
                "f4_peer_comparison": round(float(row.get("f4_peer_comparison", 0)), 4),
            })

        previous_year_features = yearly_feature_scores[-2] if len(yearly_feature_scores) >= 2 else None
        feature_deltas = {}
        if previous_year_features:
            for key in ("f1_divergence", "f2_ratio_limit", "f3_vat_structure", "f4_peer_comparison"):
                current_val = float(latest.get(key, 0))
                previous_val = float(previous_year_features.get(key, 0))
                feature_deltas[key] = round(current_val - previous_val, 4)

        # Layer 1: Isolation Forest anomaly score
        # decision_function: lower = more anomalous
        raw_anomaly = self.isolation_forest.decision_function(X)[0]
        # Normalize to 0-1 range (more positive = more anomalous)
        anomaly_score = float(max(0, min(1, 0.5 - raw_anomaly)))

        # Layer 2: XGBoost fraud probability
        raw_fraud_prob = float(self.xgboost_model.predict_proba(X)[0, 1])
        fraud_prob = float(self._calibrate_fraud_probs(np.array([raw_fraud_prob], dtype=float))[0])

        # ── Hybrid Fusion: weighted combination of both models ──
        risk_score = round(
            min(100.0, self.WEIGHT_XGB * fraud_prob * 100
                      + self.WEIGHT_ISO * anomaly_score * 100),
            2,
        )

        # Risk level classification
        risk_level = self._classify_risk_level(risk_score)

        # Layer 3: Generate explanations
        row_with_anomaly = latest.copy()
        row_with_anomaly["anomaly_score"] = anomaly_score
        red_flags = self.feature_engineer.generate_red_flags(row_with_anomaly)

        # SHAP explanation (real local SHAP values per sample)
        feature_names = TaxFeatureEngineer.FEATURE_COLS
        shap_values = self._compute_shap(X, feature_names)
        serving_meta = self.get_serving_metadata()

        return {
            "tax_code": str(latest.get("tax_code", "")),
            "company_name": str(latest.get("company_name", "")),
            "industry": str(latest.get("industry", "")),
            "year": int(latest.get("year", 0)),
            "revenue": float(latest.get("revenue", 0)),
            "total_expenses": float(latest.get("total_expenses", 0)),
            "f1_divergence": round(float(latest.get("f1_divergence", 0)), 4),
            "f2_ratio_limit": round(float(latest.get("f2_ratio_limit", 0)), 4),
            "f3_vat_structure": round(float(latest.get("f3_vat_structure", 0)), 4),
            "f4_peer_comparison": round(float(latest.get("f4_peer_comparison", 0)), 4),
            "anomaly_score": round(anomaly_score, 4),
            "model_confidence": round(max(fraud_prob, 1 - fraud_prob) * 100, 1),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "red_flags": red_flags,
            "shap_explanation": shap_values,
            "yearly_feature_scores": yearly_feature_scores,
            "previous_year_features": previous_year_features,
            "feature_deltas": feature_deltas,
            "model_version": serving_meta.get("model_version"),
            "calibration_method": serving_meta.get("calibration_method"),
        }

    def predict_batch(self, df: pd.DataFrame, batch_id: Optional[int] = None,
                      progress_callback=None) -> dict:
        """
        Batch prediction for a full CSV file.

        Args:
            df: Raw DataFrame from CSV upload.
            batch_id: Optional batch ID for progress tracking.
            progress_callback: Optional callable(processed, total) for progress updates.

        Returns:
            Dict with:
                - assessments: List of per-company risk dicts
                - statistics: Aggregated stats for dashboard charts
        """
        self.ensure_loaded()

        df = self._coerce_raw_input_dataframe(df, context="batch_inference")

        # Feature engineering on full dataset
        df = self.feature_engineer.compute_features(df)
        self._validate_feature_frame(df, context="batch_inference")

        # Calculate risk score for all rows to build trend data
        # IMPORTANT: Use the same Hybrid Fusion (80/20) as predict_single
        # to ensure consistency between batch trend and individual scores
        X_all = self.feature_engineer.get_feature_matrix(df)
        xgb_probs_all_raw = self.xgboost_model.predict_proba(X_all)[:, 1]
        xgb_probs_all = self._calibrate_fraud_probs(xgb_probs_all_raw)
        iso_raw_all = self.isolation_forest.decision_function(X_all)
        iso_scores_all = np.clip(0.5 - iso_raw_all, 0, 1)
        df["risk_score"] = np.round(
            np.minimum(100.0, self.WEIGHT_XGB * xgb_probs_all * 100
                             + self.WEIGHT_ISO * iso_scores_all * 100),
            2,
        )

        # Group by tax_code, use latest year
        latest_df = df.sort_values("year").groupby("tax_code").last().reset_index()
        X = self.feature_engineer.get_feature_matrix(latest_df)

        # Pre-index full history to avoid dtype mismatches and repeated O(n) filters in loop.
        history_by_tax_code = {
            key: group.sort_values("year")
            for key, group in df.groupby("tax_code", sort=False)
        }

        total = len(latest_df)
        assessments = []

        # Layer 1: Isolation Forest (vectorized)
        raw_anomaly_scores = self.isolation_forest.decision_function(X)
        anomaly_scores = np.clip(0.5 - raw_anomaly_scores, 0, 1)

        # Layer 2: XGBoost (vectorized)
        raw_fraud_probs = self.xgboost_model.predict_proba(X)[:, 1]
        fraud_probs = self._calibrate_fraud_probs(raw_fraud_probs)

        # ── Hybrid Fusion (vectorized) ──
        risk_scores = np.round(
            np.minimum(100.0, self.WEIGHT_XGB * fraud_probs * 100
                             + self.WEIGHT_ISO * anomaly_scores * 100),
            2,
        )

        # Process each company
        serving_meta = self.get_serving_metadata()
        for i in range(total):
            row = latest_df.iloc[i]
            risk_score = float(risk_scores[i])
            anomaly_score = float(anomaly_scores[i])

            risk_level = self._classify_risk_level(risk_score)

            row_data = row.copy()
            row_data["anomaly_score"] = anomaly_score
            red_flags = self.feature_engineer.generate_red_flags(row_data)

            # Extract multi-year history for this company from the full dataset.
            row_tax_code = row.get("tax_code", "")
            tc = str(row_tax_code).strip()
            company_history = history_by_tax_code.get(row_tax_code)
            if company_history is None or company_history.empty:
                company_history = history_by_tax_code.get(tc)
            if company_history is None:
                company_history = pd.DataFrame(columns=df.columns)
            yearly_history = [
                {
                    "year": int(h["year"]),
                    "revenue": round(float(h.get("revenue", 0)), 2),
                    "total_expenses": round(float(h.get("total_expenses", 0)), 2),
                }
                for _, h in company_history.iterrows()
            ]

            assessments.append({
                "batch_id": batch_id,
                "tax_code": tc,
                "company_name": str(row.get("company_name", "")),
                "industry": str(row.get("industry", "")),
                "year": int(row.get("year", 0)),
                "revenue": float(row.get("revenue", 0)),
                "total_expenses": float(row.get("total_expenses", 0)),
                "f1_divergence": round(float(row.get("f1_divergence", 0)), 4),
                "f2_ratio_limit": round(float(row.get("f2_ratio_limit", 0)), 4),
                "f3_vat_structure": round(float(row.get("f3_vat_structure", 0)), 4),
                "f4_peer_comparison": round(float(row.get("f4_peer_comparison", 0)), 4),
                "anomaly_score": round(anomaly_score, 4),
                "model_confidence": round(float(max(fraud_probs[i], 1 - fraud_probs[i]) * 100), 1),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "red_flags": red_flags,
                "yearly_history": yearly_history,
                "model_version": serving_meta.get("model_version"),
                "calibration_method": serving_meta.get("calibration_method"),
            })

            # Report progress
            if progress_callback and (i % 100 == 0 or i == total - 1):
                progress_callback(i + 1, total)

        # ---- Statistics for Dashboard ----
        risk_arr = np.array(risk_scores)
        stats = self._compute_batch_statistics(latest_df, risk_arr, anomaly_scores, full_df=df)

        return {
            "assessments": assessments,
            "statistics": stats,
            "total_companies": total,
        }

    def _compute_batch_statistics(self, df: pd.DataFrame, risk_scores: np.ndarray,
                                   anomaly_scores: np.ndarray, full_df: pd.DataFrame = None) -> dict:
        """Compute aggregated statistics for the ECharts dashboard."""

        # Risk distribution histogram (bins of 10)
        hist_counts, hist_edges = np.histogram(risk_scores, bins=10, range=(0, 100))
        risk_distribution = [
            {"range": f"{int(hist_edges[i])}-{int(hist_edges[i+1])}", "count": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ]

        # Risk by industry
        df_stats = df.copy()
        df_stats["risk_score"] = risk_scores
        industry_stats = (
            df_stats.groupby("industry")["risk_score"]
            .agg(["mean", "count", "max"])
            .reset_index()
            .rename(columns={"mean": "avg_risk", "count": "company_count", "max": "max_risk"})
            .sort_values("avg_risk", ascending=False)
            .to_dict("records")
        )

        # Risk by province
        if "province" in df_stats.columns:
            province_stats = (
                df_stats.groupby("province")["risk_score"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "avg_risk", "count": "company_count"})
                .sort_values("avg_risk", ascending=False)
                .to_dict("records")
            )
        else:
            province_stats = []

        # Scatter plot data – PCA 2D projection of features F1-F4
        scatter_data = []
        pca_features = ["f1_divergence", "f2_ratio_limit", "f3_vat_structure", "f4_peer_comparison"]
        pca_available_cols = [c for c in pca_features if c in df_stats.columns]

        pca_projection = None
        if PCA_AVAILABLE and len(pca_available_cols) >= 2:
            pca_matrix = df_stats[pca_available_cols].fillna(0).values
            pca_projection = _fit_safe_pca_projection(pca_matrix)

        if pca_projection is not None:
            coords_2d = pca_projection["coords"]

            for i in range(len(df_stats)):
                scatter_data.append({
                    "tax_code": str(df_stats.iloc[i].get("tax_code", "")),
                    "company_name": str(df_stats.iloc[i].get("company_name", "")),
                    "pc1": round(float(coords_2d[i, 0]), 4),
                    "pc2": round(float(coords_2d[i, 1]), 4),
                    "risk_score": float(risk_scores[i]),
                    "industry": str(df_stats.iloc[i].get("industry", "")),
                    "revenue": float(df_stats.iloc[i].get("revenue", 0)),
                })
        else:
            # Fallback: use revenue vs risk_score
            for i in range(len(df_stats)):
                scatter_data.append({
                    "tax_code": str(df_stats.iloc[i].get("tax_code", "")),
                    "company_name": str(df_stats.iloc[i].get("company_name", "")),
                    "pc1": round(float(np.log1p(max(0, df_stats.iloc[i].get("revenue", 0)))), 4),
                    "pc2": float(risk_scores[i]),
                    "risk_score": float(risk_scores[i]),
                    "industry": str(df_stats.iloc[i].get("industry", "")),
                    "revenue": float(df_stats.iloc[i].get("revenue", 0)),
                })

        # Correlation matrix (features)
        feature_cols = TaxFeatureEngineer.FEATURE_COLS
        available_cols = [c for c in feature_cols if c in df_stats.columns]
        corr_matrix = {}
        if len(available_cols) >= 2:
            corr_df = df_stats[available_cols].corr()
            corr_matrix = {
                "columns": available_cols,
                "values": corr_df.values.tolist(),
            }

        # Top 50 highest risk
        top_50_indices = np.argsort(risk_scores)[-50:][::-1]
        top_50 = []
        for idx in top_50_indices:
            row = df_stats.iloc[idx]
            top_50.append({
                "tax_code": str(row.get("tax_code", "")),
                "company_name": str(row.get("company_name", "")),
                "industry": str(row.get("industry", "")),
                "revenue": float(row.get("revenue", 0)),
                "risk_score": float(risk_scores[idx]),
                "anomaly_score": float(anomaly_scores[idx]),
            })

        # Summary stats
        summary = {
            "total_companies": len(risk_scores),
            "csv_total_rows": len(full_df) if full_df is not None else len(risk_scores),
            "avg_risk": round(float(np.mean(risk_scores)), 2),
            "median_risk": round(float(np.median(risk_scores)), 2),
            "critical_count": int(np.sum(risk_scores >= 80)),
            "high_count": int(np.sum((risk_scores >= 60) & (risk_scores < 80))),
            "medium_count": int(np.sum((risk_scores >= 40) & (risk_scores < 60))),
            "low_count": int(np.sum(risk_scores < 40)),
        }

        # ---- Revenue vs Risk (business-friendly scatter) ----
        revenue_risk_scatter = []
        for i in range(len(df_stats)):
            row = df_stats.iloc[i]
            revenue = float(row.get("revenue", 0) or 0)
            total_expenses = float(row.get("total_expenses", 0) or 0)
            expense_ratio = (total_expenses / max(revenue, 1.0)) if revenue > 0 else 0.0
            row_risk = float(row.get("risk_score", risk_scores[i]))
            revenue_risk_scatter.append({
                "tax_code": str(row.get("tax_code", "")),
                "company_name": str(row.get("company_name", "")),
                "industry": str(row.get("industry", "")),
                "revenue": revenue,
                "total_expenses": total_expenses,
                "expense_ratio": round(expense_ratio, 4),
                "risk_score": round(row_risk, 2),
            })

        # ---- Box Plot Data (risk score distribution per industry) ----
        box_plot_data = []
        for industry, group in df_stats.groupby("industry"):
            scores = group["risk_score"].values
            if len(scores) >= 5:
                q1, median, q3 = np.percentile(scores, [25, 50, 75])
                iqr = q3 - q1
                whisker_low = float(max(scores.min(), q1 - 1.5 * iqr))
                whisker_high = float(min(scores.max(), q3 + 1.5 * iqr))
                outliers = [float(v) for v in scores if v < whisker_low or v > whisker_high]
                box_plot_data.append({
                    "industry": str(industry),
                    "min": whisker_low,
                    "q1": round(float(q1), 2),
                    "median": round(float(median), 2),
                    "q3": round(float(q3), 2),
                    "max": whisker_high,
                    "outliers": outliers[:20],  # cap to 20 outliers per industry
                    "count": len(scores),
                })
        box_plot_data.sort(key=lambda x: x["median"], reverse=True)

        # ---- Key Drivers (aggregated feature importance) ----
        feature_labels = {
            "f1_divergence": "Lệch pha Tăng trưởng (F1)",
            "f2_ratio_limit": "Đội Chi phí/Doanh thu (F2)",
            "f3_vat_structure": "VAT Vòng lặp (F3)",
            "f4_peer_comparison": "Thấp hơn Ngành (F4)",
        }
        # Count how many high-risk companies trigger each flag
        high_risk_mask = risk_scores >= 60
        high_risk_df = df_stats[high_risk_mask]
        total_high = max(int(high_risk_mask.sum()), 1)

        key_drivers = []
        thresholds = {
            "f1_divergence": lambda v: v < -0.3,
            "f2_ratio_limit": lambda v: v > 0.95,
            "f3_vat_structure": lambda v: v > 0.90,
            "f4_peer_comparison": lambda v: v < -0.08,
        }
        for feat, label in feature_labels.items():
            if feat in high_risk_df.columns:
                triggered = high_risk_df[feat].apply(thresholds[feat]).sum()
                avg_val = float(high_risk_df[feat].mean()) if len(high_risk_df) > 0 else 0
                key_drivers.append({
                    "feature": feat,
                    "label": label,
                    "triggered_count": int(triggered),
                    "triggered_percent": round(triggered / total_high * 100, 1),
                    "avg_value": round(avg_val, 4),
                })
        key_drivers.sort(key=lambda x: x["triggered_count"], reverse=True)

        # ---- Year Trend Data ----
        year_trend_data = []
        if full_df is not None and "year" in full_df.columns:
            trend_group = full_df.groupby("year")
            for year, group in trend_group:
                avg_risk = round(float(group["risk_score"].mean()), 2)
                critical_count = int((group["risk_score"] >= 60).sum())
                total_rev = float(group["revenue"].sum()) / 1e9 if "revenue" in group.columns else 0
                year_trend_data.append({
                    "year": str(year),
                    "avg_risk": avg_risk,
                    "high_risk_count": critical_count,
                    "total_revenue_bn": round(total_rev, 2)
                })
            year_trend_data.sort(key=lambda x: x["year"])

        # ---- Cohort Transition Sankey (risk tier migration by year) ----
        cohort_transition_sankey = {
            "nodes": [],
            "links": [],
        }

        def _risk_tier_from_score(score: float) -> str:
            if score >= 80:
                return "critical"
            if score >= 60:
                return "high"
            if score >= 40:
                return "medium"
            return "low"

        if full_df is not None and {
            "tax_code", "year", "risk_score"
        }.issubset(set(full_df.columns)):
            cohort_df = full_df[["tax_code", "year", "risk_score"]].copy()
            cohort_df = cohort_df.dropna(subset=["tax_code", "year", "risk_score"])

            if not cohort_df.empty:
                cohort_df["tax_code"] = cohort_df["tax_code"].astype(str)
                cohort_df["year"] = cohort_df["year"].astype(int)
                cohort_df["risk_score"] = cohort_df["risk_score"].astype(float)
                cohort_df = cohort_df.sort_values(["tax_code", "year"])

                edge_counter = {}
                for _, company_rows in cohort_df.groupby("tax_code"):
                    by_year = (
                        company_rows.groupby("year", as_index=False)["risk_score"]
                        .mean()
                        .sort_values("year")
                    )
                    by_year["risk_tier"] = by_year["risk_score"].apply(_risk_tier_from_score)
                    records = by_year[["year", "risk_tier"]].to_dict("records")

                    for idx in range(len(records) - 1):
                        src = f"{int(records[idx]['year'])}:{records[idx]['risk_tier']}"
                        tgt = f"{int(records[idx + 1]['year'])}:{records[idx + 1]['risk_tier']}"
                        edge_counter[(src, tgt)] = edge_counter.get((src, tgt), 0) + 1

                if edge_counter:
                    node_names = set()
                    for src, tgt in edge_counter.keys():
                        node_names.add(src)
                        node_names.add(tgt)

                    tier_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

                    def _node_sort_key(name: str) -> tuple:
                        parts = name.split(":", 1)
                        if len(parts) != 2:
                            return (9999, 9999, name)
                        year_raw, tier_raw = parts
                        try:
                            year_val = int(year_raw)
                        except Exception:
                            year_val = 9999
                        return (year_val, tier_order.get(tier_raw, 99), name)

                    sorted_nodes = sorted(node_names, key=_node_sort_key)
                    cohort_transition_sankey["nodes"] = [
                        {
                            "name": node,
                            "year": int(node.split(":", 1)[0]),
                            "tier": node.split(":", 1)[1],
                        }
                        for node in sorted_nodes
                    ]
                    cohort_transition_sankey["links"] = [
                        {
                            "source": src,
                            "target": tgt,
                            "value": int(value),
                        }
                        for (src, tgt), value in sorted(edge_counter.items())
                    ]

        # ---- VAT anomaly heatmap (industry x year, based on F3 threshold) ----
        vat_threshold = 0.90
        vat_anomaly_heatmap = {
            "years": [],
            "industries": [],
            "values": [],
            "counts": [],
            "threshold": vat_threshold,
        }
        if full_df is not None and {
            "industry", "year", "f3_vat_structure"
        }.issubset(set(full_df.columns)):
            vat_df = full_df[["industry", "year", "f3_vat_structure"]].copy()
            vat_df = vat_df.dropna(subset=["industry", "year", "f3_vat_structure"])

            if not vat_df.empty:
                vat_df["industry"] = vat_df["industry"].astype(str)
                vat_df["year"] = vat_df["year"].astype(int)
                vat_df["f3_vat_structure"] = vat_df["f3_vat_structure"].astype(float)
                vat_df["is_anomaly"] = vat_df["f3_vat_structure"] > vat_threshold

                agg = (
                    vat_df.groupby(["industry", "year"], as_index=False)
                    .agg(
                        total_count=("f3_vat_structure", "size"),
                        anomaly_count=("is_anomaly", "sum"),
                    )
                )
                agg["anomaly_rate_pct"] = np.where(
                    agg["total_count"] > 0,
                    agg["anomaly_count"] / agg["total_count"] * 100,
                    0,
                )

                industries = (
                    agg.groupby("industry")["anomaly_count"]
                    .sum()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                years = sorted(agg["year"].unique().tolist())
                industry_to_idx = {name: idx for idx, name in enumerate(industries)}
                year_to_idx = {year: idx for idx, year in enumerate(years)}

                values = []
                counts = []
                for row in agg.itertuples(index=False):
                    xi = year_to_idx.get(int(row.year))
                    yi = industry_to_idx.get(str(row.industry))
                    if xi is None or yi is None:
                        continue
                    values.append([xi, yi, round(float(row.anomaly_rate_pct), 2)])
                    counts.append([xi, yi, int(row.anomaly_count), int(row.total_count)])

                vat_anomaly_heatmap = {
                    "years": [str(y) for y in years],
                    "industries": industries,
                    "values": values,
                    "counts": counts,
                    "threshold": vat_threshold,
                }

        # ---- Cumulative risk curve (Lorenz-style concentration) ----
        cumulative_risk_curve = {
            "points": [],
            "total_companies": int(len(risk_scores)),
            "total_risk": 0.0,
            "top_10pct_risk_share": 0.0,
            "top_20pct_risk_share": 0.0,
        }
        if len(risk_scores) > 0:
            sorted_scores = np.sort(risk_scores)[::-1]
            cumulative_scores = np.cumsum(sorted_scores)
            total_risk = float(cumulative_scores[-1]) if cumulative_scores.size > 0 else 0.0
            safe_total_risk = total_risk if total_risk > 0 else 1.0
            company_count = len(sorted_scores)

            step = max(1, company_count // 120)
            points = []
            for idx in range(0, company_count, step):
                points.append({
                    "company_count": int(idx + 1),
                    "percent_companies": round((idx + 1) / company_count * 100, 2),
                    "percent_risk": round(float(cumulative_scores[idx]) / safe_total_risk * 100, 2),
                })

            if not points or points[-1]["company_count"] != company_count:
                points.append({
                    "company_count": int(company_count),
                    "percent_companies": 100.0,
                    "percent_risk": 100.0,
                })

            top_10_count = max(1, int(np.ceil(company_count * 0.10)))
            top_20_count = max(1, int(np.ceil(company_count * 0.20)))

            cumulative_risk_curve = {
                "points": points,
                "total_companies": int(company_count),
                "total_risk": round(total_risk, 2),
                "top_10pct_risk_share": round(float(cumulative_scores[top_10_count - 1]) / safe_total_risk * 100, 2),
                "top_20pct_risk_share": round(float(cumulative_scores[top_20_count - 1]) / safe_total_risk * 100, 2),
            }

        # ---- Global XGBoost Feature Importance (AI-native) ----
        global_feature_importance = []
        try:
            feature_names = TaxFeatureEngineer.FEATURE_COLS
            importances = self.xgboost_model.feature_importances_
            total_imp = float(np.sum(importances))
            feature_label_map = {
                "f1_divergence": "Lệch pha Tăng trưởng (F1)",
                "f2_ratio_limit": "Đội Chi phí/Doanh thu (F2)",
                "f3_vat_structure": "Cấu trúc VAT (F3)",
                "f4_peer_comparison": "So sánh Ngành (F4)",
                "revenue_log": "Quy mô Doanh thu",
                "expense_log": "Quy mô Chi phí",
                "profit_margin": "Biên Lợi Nhuận",
                "revenue_growth_rate": "Tăng trưởng DT",
                "expense_growth_rate": "Tăng trưởng CP",
                "vat_net_ratio": "VAT Ròng",
            }
            for name, imp in zip(feature_names, importances):
                global_feature_importance.append({
                    "feature": name,
                    "label": feature_label_map.get(name, name),
                    "importance": round(float(imp), 4),
                    "importance_pct": round(float(imp) / total_imp * 100, 1) if total_imp > 0 else 0,
                })
            global_feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        except Exception:
            pass

        # ---- Contour Grid (Isolation Forest decision boundary in PCA space) ----
        contour_data = None
        if pca_projection is not None:
            try:
                scaler = pca_projection["scaler"]
                pca_model = pca_projection["pca_model"]
                coords = pca_projection["coords"]

                # Build a grid over PCA space
                x_min, x_max = coords[:, 0].min() - 1, coords[:, 0].max() + 1
                y_min, y_max = coords[:, 1].min() - 1, coords[:, 1].max() + 1
                grid_res = 40  # 40x40 grid for performance
                xx = np.linspace(x_min, x_max, grid_res)
                yy = np.linspace(y_min, y_max, grid_res)
                xx_grid, yy_grid = np.meshgrid(xx, yy)
                grid_2d = np.c_[xx_grid.ravel(), yy_grid.ravel()]

                # Inverse PCA to get back to feature space, then score with Isolation Forest
                grid_features = pca_model.inverse_transform(grid_2d)
                grid_original = scaler.inverse_transform(grid_features)
                # Pad to full feature matrix if needed
                full_feat_count = len(TaxFeatureEngineer.FEATURE_COLS)
                if grid_original.shape[1] < full_feat_count:
                    padding = np.zeros((grid_original.shape[0], full_feat_count - grid_original.shape[1]))
                    grid_original = np.hstack([grid_original, padding])
                
                iso_scores = self.isolation_forest.decision_function(grid_original)
                # Normalize scores: lower = more anomalous → flip to 0-1
                z_values = np.clip(0.5 - iso_scores, 0, 1)
                z_grid = z_values.reshape(grid_res, grid_res)

                contour_data = {
                    "x": xx.tolist(),
                    "y": yy.tolist(),
                    "z": z_grid.tolist(),
                }
            except Exception as e:
                print(f"[WARN] Contour grid failed: {e}")
                contour_data = None

        return {
            "summary": summary,
            "risk_distribution": risk_distribution,
            "industry_stats": industry_stats,
            "province_stats": province_stats,
            "scatter_data": scatter_data,
            "revenue_risk_scatter": revenue_risk_scatter,
            "correlation_matrix": corr_matrix,
            "top_50_risky": top_50,
            "box_plot_data": box_plot_data,
            "key_drivers": key_drivers,
            "year_trend": year_trend_data,
            "cohort_transition_sankey": cohort_transition_sankey,
            "vat_anomaly_heatmap": vat_anomaly_heatmap,
            "cumulative_risk_curve": cumulative_risk_curve,
            "global_feature_importance": global_feature_importance,
            "contour_data": contour_data,
        }

    def predict_whatif(self, base_data: dict, adjustments: dict) -> dict:
        """
        What-If Scenario Simulation.
        Takes a base company data dict and applies percentage adjustments,
        then re-scores through the pipeline.

        Args:
            base_data: Original financial data dict (single year)
            adjustments: Dict of {field_name: percentage_change} e.g. {"revenue": -20, "total_expenses": 30}

        Returns:
            Dict with original and simulated risk scores + delta.
        """
        self.ensure_loaded()
        import copy

        simulated = copy.deepcopy(base_data)
        adjustment_details = []

        for field, pct_change in adjustments.items():
            if field in simulated and simulated[field] is not None:
                original_val = float(simulated[field])
                new_val = original_val * (1 + pct_change / 100)
                simulated[field] = new_val
                adjustment_details.append({
                    "field": field,
                    "original": round(original_val, 2),
                    "simulated": round(new_val, 2),
                    "change_pct": pct_change,
                })

        # Recalculate derived fields
        if "revenue" in adjustments or "total_expenses" in adjustments:
            rev = float(simulated.get("revenue", 0))
            exp = float(simulated.get("total_expenses", 0))
            simulated["net_profit"] = rev - exp
            simulated["cost_of_goods"] = exp * 0.75
            simulated["operating_expenses"] = exp * 0.25
            simulated["vat_output"] = rev * 0.10
            simulated["vat_input"] = simulated["cost_of_goods"] * 0.10

        # Run pipeline on simulated data
        result = self.predict_single(simulated)

        return {
            "simulated_risk_score": result["risk_score"],
            "simulated_risk_level": result["risk_level"],
            "simulated_anomaly_score": result["anomaly_score"],
            "simulated_confidence": result["model_confidence"],
            "simulated_red_flags": result["red_flags"],
            "adjustments": adjustment_details,
            "simulated_features": {
                "f1_divergence": result["f1_divergence"],
                "f2_ratio_limit": result["f2_ratio_limit"],
                "f3_vat_structure": result["f3_vat_structure"],
                "f4_peer_comparison": result["f4_peer_comparison"],
            },
        }

    # ── Helper: risk level classification ──
    @staticmethod
    def _classify_risk_level(score: float) -> str:
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        return "low"

    # ── SHAP: Real local explanation per sample ──
    def _get_shap_explainer(self):
        """Lazy-initialize SHAP TreeExplainer (cached after first call)."""
        if self._shap_explainer is None and SHAP_AVAILABLE:
            try:
                booster = self.xgboost_model.get_booster()
                # Monkey-patch: ensure base_score is a plain float
                # (fixes serialization conflict between XGBoost >=2.0 and SHAP)
                # Some XGBoost versions return base_score as "[0.5]" instead of "0.5"
                try:
                    raw_base = booster.attr('base_score')
                    if raw_base is not None:
                        import re
                        cleaned = re.sub(r'[\[\]\s]', '', str(raw_base))
                        booster.set_attr(base_score=str(float(cleaned)))
                except Exception:
                    pass  # Safe to ignore if attr not present

                if self._shap_background is not None:
                    self._shap_explainer = shap.TreeExplainer(
                        booster,
                        data=self._shap_background,
                        feature_perturbation="interventional",
                    )
                    print("[OK] SHAP TreeExplainer initialized (interventional + background)")
                else:
                    self._shap_explainer = shap.TreeExplainer(booster)
                    print("[OK] SHAP TreeExplainer initialized")
            except Exception as e:
                print(f"[WARN] SHAP TreeExplainer init failed: {e}")
        return self._shap_explainer

    def _compute_shap(self, X: np.ndarray, feature_names: list) -> list[dict]:
        """
        Compute real per-sample SHAP values using shap.TreeExplainer.
        Falls back to global feature importances if SHAP is unavailable.
        """
        try:
            explainer = self._get_shap_explainer()
            if explainer is not None:
                # Real local SHAP values for this specific sample
                sv = explainer.shap_values(X)
                # For binary classification, shap_values may return a list
                if isinstance(sv, list):
                    sv = sv[1]  # class-1 (fraud) SHAP values
                sample_shap = sv[0]  # first (only) row

                explanations = []
                for idx, name in enumerate(feature_names):
                    explanations.append({
                        "feature": name,
                        "shap_value": round(float(sample_shap[idx]), 6),
                        "importance": round(abs(float(sample_shap[idx])), 6),
                        "direction": "risk" if sample_shap[idx] > 0 else "safe",
                        "value": round(float(X[0][idx]), 4) if len(X) > 0 else 0,
                    })
                explanations.sort(key=lambda x: x["importance"], reverse=True)
                return explanations
        except Exception as e:
            print(f"[WARN] SHAP computation failed, falling back to global: {e}")

        # Fallback: global feature importances
        return self._compute_shap_fallback(X, feature_names)

    def _compute_shap_fallback(self, X: np.ndarray, feature_names: list) -> list[dict]:
        """Fallback SHAP-like explanation using global tree feature importances."""
        try:
            importances = self.xgboost_model.feature_importances_
            explanations = []
            for idx, (name, imp) in enumerate(zip(feature_names, importances)):
                explanations.append({
                    "feature": name,
                    "shap_value": round(float(imp), 6),
                    "importance": round(float(imp), 6),
                    "direction": "risk",
                    "value": round(float(X[0][idx]), 4) if len(X) > 0 else 0,
                })
            explanations.sort(key=lambda x: x["importance"], reverse=True)
            return explanations
        except Exception:
            return []
