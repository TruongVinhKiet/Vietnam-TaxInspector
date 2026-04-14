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


class TaxFraudPipeline:
    """
    Hybrid AI Pipeline for Tax Fraud Detection.
    Loads pre-trained models and runs inference.
    """

    # Hybrid fusion weights – tunable
    WEIGHT_XGB = 0.80
    WEIGHT_ISO = 0.20

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.feature_engineer = TaxFeatureEngineer()
        self.isolation_forest = None
        self.xgboost_model = None
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
        # IMPORTANT: Use the same Hybrid Fusion (80/20) as predict_single
        # to ensure consistency between batch trend and individual scores
        X_all = self.feature_engineer.get_feature_matrix(df)
        xgb_probs_all = self.xgboost_model.predict_proba(X_all)[:, 1]
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

        total = len(latest_df)
        assessments = []

        # Layer 1: Isolation Forest (vectorized)
        raw_anomaly_scores = self.isolation_forest.decision_function(X)
        anomaly_scores = np.clip(0.5 - raw_anomaly_scores, 0, 1)

        # Layer 2: XGBoost (vectorized)
        fraud_probs = self.xgboost_model.predict_proba(X)[:, 1]

        # ── Hybrid Fusion (vectorized) ──
        risk_scores = np.round(
            np.minimum(100.0, self.WEIGHT_XGB * fraud_probs * 100
                             + self.WEIGHT_ISO * anomaly_scores * 100),
            2,
        )

        # Process each company
        for i in range(total):
            row = latest_df.iloc[i]
            risk_score = float(risk_scores[i])
            anomaly_score = float(anomaly_scores[i])

            risk_level = self._classify_risk_level(risk_score)

            row_data = row.copy()
            row_data["anomaly_score"] = anomaly_score
            red_flags = self.feature_engineer.generate_red_flags(row_data)

            # Extract multi-year history for this company from the full dataset
            tc = str(row.get("tax_code", ""))
            company_history = df[df["tax_code"] == tc].sort_values("year")
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
        if PCA_AVAILABLE and len(pca_available_cols) >= 2:
            try:
                from sklearn.preprocessing import StandardScaler
                pca_matrix = df_stats[pca_available_cols].fillna(0).values
                scaler = StandardScaler()
                pca_matrix_scaled = scaler.fit_transform(pca_matrix)
                pca_model = PCA(n_components=2)
                coords = pca_model.fit_transform(pca_matrix_scaled)

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
            "correlation_matrix": corr_matrix,
            "top_50_risky": top_50,
            "box_plot_data": box_plot_data,
            "key_drivers": key_drivers,
            "year_trend": year_trend_data,
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
