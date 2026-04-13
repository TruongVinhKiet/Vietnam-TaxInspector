"""
pipeline.py – Hybrid AI Pipeline (Isolation Forest + XGBoost + SHAP)
======================================================================
Lớp 1: Isolation Forest (Unsupervised Anomaly Detection)
Lớp 2: XGBoost (Supervised Fraud Probability)
Lớp 3: SHAP (Explainable AI – Red Flag Generation)

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

# Path to saved models
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


class TaxFraudPipeline:
    """
    Hybrid AI Pipeline for Tax Fraud Detection.
    Loads pre-trained models and runs inference.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.feature_engineer = TaxFeatureEngineer()
        self.isolation_forest = None
        self.xgboost_model = None
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
        self._loaded = True
        print(f"[OK] Models loaded from {self.model_dir}")

    def ensure_loaded(self):
        if not self._loaded:
            self.load_models()

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

        # Feature engineering
        df = self.feature_engineer.compute_features(df)

        # Use the most recent year for scoring
        latest = df.sort_values("year", ascending=False).iloc[0]
        X = self.feature_engineer.get_feature_matrix(df.tail(1))

        # Layer 1: Isolation Forest anomaly score
        # decision_function: lower = more anomalous
        raw_anomaly = self.isolation_forest.decision_function(X)[0]
        # Normalize to 0-1 range (more positive = more anomalous)
        anomaly_score = float(max(0, min(1, 0.5 - raw_anomaly)))

        # Layer 2: XGBoost fraud probability
        fraud_prob = float(self.xgboost_model.predict_proba(X)[0, 1])

        # Combined risk score (0–100)
        risk_score = round(fraud_prob * 100, 2)

        # Risk level classification
        if risk_score >= 80:
            risk_level = "critical"
        elif risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Layer 3: Generate explanations
        row_with_anomaly = latest.copy()
        row_with_anomaly["anomaly_score"] = anomaly_score
        red_flags = self.feature_engineer.generate_red_flags(row_with_anomaly)

        # SHAP explanation (simplified – feature importances)
        feature_names = TaxFeatureEngineer.FEATURE_COLS
        shap_values = self._compute_shap_simple(X, feature_names)

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

        # Feature engineering on full dataset
        df = self.feature_engineer.compute_features(df)

        # Calculate risk score for all rows to build trend data
        X_all = self.feature_engineer.get_feature_matrix(df)
        df["risk_score"] = np.round(self.xgboost_model.predict_proba(X_all)[:, 1] * 100, 2)

        # Group by tax_code, use latest year
        latest_df = df.sort_values("year").groupby("tax_code").last().reset_index()
        X = self.feature_engineer.get_feature_matrix(latest_df)

        total = len(latest_df)
        assessments = []

        # Layer 1: Isolation Forest (vectorized)
        raw_anomaly_scores = self.isolation_forest.decision_function(X)
        anomaly_scores = np.clip(0.5 - raw_anomaly_scores, 0, 1)

        # Layer 2: XGBoost (vectorized)
        fraud_probs = self.xgboost_model.predict_proba(X)[:, 1]
        risk_scores = np.round(fraud_probs * 100, 2)

        # Process each company
        for i in range(total):
            row = latest_df.iloc[i]
            risk_score = float(risk_scores[i])
            anomaly_score = float(anomaly_scores[i])

            if risk_score >= 80:
                risk_level = "critical"
            elif risk_score >= 60:
                risk_level = "high"
            elif risk_score >= 40:
                risk_level = "medium"
            else:
                risk_level = "low"

            row_data = row.copy()
            row_data["anomaly_score"] = anomaly_score
            red_flags = self.feature_engineer.generate_red_flags(row_data)

            assessments.append({
                "batch_id": batch_id,
                "tax_code": str(row.get("tax_code", "")),
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
                "risk_score": risk_score,
                "risk_level": risk_level,
                "red_flags": red_flags,
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

        if PCA_AVAILABLE and len(pca_available_cols) >= 2:
            from sklearn.preprocessing import StandardScaler
            pca_matrix = df_stats[pca_available_cols].fillna(0).values
            pca_matrix_scaled = StandardScaler().fit_transform(pca_matrix)
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(pca_matrix_scaled)
            explained = pca.explained_variance_ratio_

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
            "avg_risk": round(float(np.mean(risk_scores)), 2),
            "median_risk": round(float(np.median(risk_scores)), 2),
            "critical_count": int(np.sum(risk_scores >= 80)),
            "high_count": int(np.sum((risk_scores >= 60) & (risk_scores < 80))),
            "medium_count": int(np.sum((risk_scores >= 40) & (risk_scores < 60))),
            "low_count": int(np.sum(risk_scores < 40)),
        }

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

        return {
            "summary": summary,
            "risk_distribution": risk_distribution,
            "industry_stats": industry_stats,
            "province_stats": province_stats,
            "scatter_data": scatter_data,
            "correlation_matrix": corr_matrix,
            "top_50_risky": top_50,
            "box_plot_data": box_plot_data,
            "key_drivers": key_drivers,
            "year_trend": year_trend_data,
        }

    def _compute_shap_simple(self, X: np.ndarray, feature_names: list) -> list[dict]:
        """
        Simplified SHAP-like explanation using tree feature importances.
        For full SHAP, use: shap.TreeExplainer(self.xgboost_model).shap_values(X)
        """
        try:
            importances = self.xgboost_model.feature_importances_
            # Pair feature names with importance values
            explanations = []
            for name, imp in zip(feature_names, importances):
                explanations.append({
                    "feature": name,
                    "importance": round(float(imp), 4),
                    "value": round(float(X[0][feature_names.index(name)]), 4) if len(X) > 0 else 0,
                })
            # Sort by importance descending
            explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)
            return explanations
        except Exception:
            return []
