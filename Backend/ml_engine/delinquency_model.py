"""
delinquency_model.py – Temporal Compliance Intelligence (Program A)
====================================================================
LightGBM/XGBoost-based model for predicting tax delinquency risk.
Predicts probability of overdue payment within 30/60/90 day windows.

Features extracted from payment history:
    - Historical late payment ratio (1yr, 2yr, all-time)
    - Average days overdue
    - Payment amount consistency (coefficient of variation)
    - Penalty accumulation trend
    - Seasonal patterns (quarter/month effects)
    - Company age and industry risk profile
    - Revenue trend features from tax returns

Architecture:
    - Optimized for i7 + 12GB RAM: batched feature computation, streaming I/O
    - Supports both batch prediction and single-company inference
    - Produces calibrated probabilities + top reason explanations

Artifacts generated:
    - delinquency_lgbm.joblib
    - delinquency_config.json
    - delinquency_quality_report.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

from app.risk_utils import classify_delinquency_cluster

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
DELINQUENCY_MODEL_VERSION = os.getenv("DELINQUENCY_MODEL_VERSION", "delinquency-temporal-v1")


class DelinquencyFeatureEngineer:
    """
    Extract temporal compliance features from payment and tax return history.
    Designed for memory efficiency on 12GB RAM systems.
    """

    FEATURE_COLS = (
        "late_ratio_1yr",
        "late_ratio_2yr",
        "late_ratio_all",
        "avg_days_overdue",
        "max_days_overdue",
        "payment_cv",            # coefficient of variation of payment amounts
        "penalty_trend",         # penalty growth rate
        "unpaid_count",
        "partial_payment_ratio",
        "recent_late_acceleration",  # are late payments getting more frequent?
        "seasonal_q1_late",      # Q1 late ratio
        "seasonal_q2_late",      # Q2 late ratio
        "seasonal_q3_late",      # Q3 late ratio
        "seasonal_q4_late",      # Q4 late ratio
        "company_age_years",
        "revenue_trend",         # revenue growth/decline slope
        "expense_ratio_trend",   # expense/revenue ratio trend
        "industry_risk_index",   # encoded industry risk level
    )

    # Industry risk priors (domain knowledge from Vietnamese tax inspection)
    INDUSTRY_RISK_MAP = {
        "Xây dựng": 0.7,
        "Thương mại": 0.5,
        "Bất động sản": 0.65,
        "Vận tải & Logistics": 0.55,
        "Sản xuất": 0.4,
        "Công nghệ thông tin": 0.3,
        "Dịch vụ tài chính": 0.45,
        "Thực phẩm & Đồ uống": 0.35,
        "Dệt may": 0.45,
        "Nông nghiệp": 0.5,
    }

    def compute_features(
        self,
        payments_df: pd.DataFrame,
        tax_returns_df: Optional[pd.DataFrame] = None,
        company_info: Optional[dict] = None,
    ) -> dict:
        """
        Compute features for a single company from payment history.

        Args:
            payments_df: DataFrame with columns [due_date, actual_payment_date, amount_due, amount_paid, penalty_amount, tax_period, status]
            tax_returns_df: Optional DataFrame with [filing_date, revenue, expenses]
            company_info: Optional dict with [registration_date, industry]

        Returns:
            dict of feature_name -> float
        """
        features = {col: 0.0 for col in self.FEATURE_COLS}
        today = date.today()

        if payments_df is None or payments_df.empty:
            return features

        # Parse dates
        payments = payments_df.copy()
        payments["due_date"] = pd.to_datetime(payments["due_date"], errors="coerce").dt.date
        payments["actual_payment_date"] = pd.to_datetime(payments["actual_payment_date"], errors="coerce").dt.date
        payments = payments.dropna(subset=["due_date"])
        payments = payments.sort_values("due_date")

        if payments.empty:
            return features

        # Compute days overdue per payment
        days_overdue = []
        for _, row in payments.iterrows():
            if pd.notna(row["actual_payment_date"]) and row["actual_payment_date"] > row["due_date"]:
                days_overdue.append((row["actual_payment_date"] - row["due_date"]).days)
            elif pd.isna(row["actual_payment_date"]) and row["due_date"] < today:
                days_overdue.append((today - row["due_date"]).days)
            else:
                days_overdue.append(0)

        payments["days_overdue"] = days_overdue
        payments["is_late"] = (payments["days_overdue"] > 0).astype(int)

        # Late ratios by time window
        one_year_ago = today - timedelta(days=365)
        two_years_ago = today - timedelta(days=730)

        mask_1yr = payments["due_date"] >= one_year_ago
        mask_2yr = payments["due_date"] >= two_years_ago

        n_1yr = max(1, mask_1yr.sum())
        n_2yr = max(1, mask_2yr.sum())
        n_all = max(1, len(payments))

        features["late_ratio_1yr"] = round(payments.loc[mask_1yr, "is_late"].sum() / n_1yr, 4)
        features["late_ratio_2yr"] = round(payments.loc[mask_2yr, "is_late"].sum() / n_2yr, 4)
        features["late_ratio_all"] = round(payments["is_late"].sum() / n_all, 4)

        # Days overdue statistics
        late_days = [d for d in days_overdue if d > 0]
        features["avg_days_overdue"] = round(np.mean(late_days), 2) if late_days else 0.0
        features["max_days_overdue"] = max(late_days) if late_days else 0

        # Payment amount consistency
        amounts = pd.to_numeric(payments["amount_paid"], errors="coerce").dropna()
        if len(amounts) > 1 and amounts.mean() > 0:
            features["payment_cv"] = round(float(amounts.std() / amounts.mean()), 4)

        # Penalty trend
        penalties = pd.to_numeric(payments["penalty_amount"], errors="coerce").fillna(0)
        if len(penalties) >= 4:
            half = len(penalties) // 2
            early_penalty = penalties.iloc[:half].mean()
            late_penalty = penalties.iloc[half:].mean()
            features["penalty_trend"] = round(float(late_penalty - early_penalty), 2)

        # Unpaid count
        features["unpaid_count"] = int((payments["status"] == "overdue").sum() + (payments["actual_payment_date"].isna() & (payments["due_date"] < today)).sum())

        # Partial payment ratio
        amount_due = pd.to_numeric(payments["amount_due"], errors="coerce").fillna(0)
        amount_paid = pd.to_numeric(payments["amount_paid"], errors="coerce").fillna(0)
        partial_mask = (amount_paid > 0) & (amount_paid < amount_due * 0.95)
        features["partial_payment_ratio"] = round(partial_mask.sum() / n_all, 4)

        # Recent late acceleration (is it getting worse?)
        recent_payments = payments[mask_1yr]
        if len(recent_payments) >= 4:
            half = len(recent_payments) // 2
            early_late = recent_payments.iloc[:half]["is_late"].mean()
            recent_late = recent_payments.iloc[half:]["is_late"].mean()
            features["recent_late_acceleration"] = round(float(recent_late - early_late), 4)

        # Seasonal patterns
        payments["quarter"] = pd.to_datetime(payments["due_date"].astype(str)).dt.quarter
        for q in range(1, 5):
            q_mask = payments["quarter"] == q
            q_count = max(1, q_mask.sum())
            features[f"seasonal_q{q}_late"] = round(payments.loc[q_mask, "is_late"].sum() / q_count, 4)

        # Company metadata
        if company_info:
            reg_date = company_info.get("registration_date")
            if reg_date:
                if isinstance(reg_date, str):
                    reg_date = date.fromisoformat(reg_date)
                features["company_age_years"] = (today - reg_date).days / 365.25

            industry = company_info.get("industry", "")
            features["industry_risk_index"] = self.INDUSTRY_RISK_MAP.get(industry, 0.4)

        # Revenue trend from tax returns
        if tax_returns_df is not None and not tax_returns_df.empty:
            tr = tax_returns_df.copy()
            tr["revenue"] = pd.to_numeric(tr.get("revenue"), errors="coerce").fillna(0)
            tr["expenses"] = pd.to_numeric(tr.get("expenses"), errors="coerce").fillna(0)
            tr["filing_date"] = pd.to_datetime(tr["filing_date"], errors="coerce")
            tr = tr.dropna(subset=["filing_date"]).sort_values("filing_date")

            if len(tr) >= 2:
                revenues = tr["revenue"].values
                x = np.arange(len(revenues), dtype=float)
                if np.std(x) > 0 and np.std(revenues) > 0:
                    slope = np.polyfit(x, revenues, 1)[0]
                    features["revenue_trend"] = round(float(slope / max(1, np.mean(revenues))), 4)

                # Expense ratio trend
                expense_ratios = (tr["expenses"] / tr["revenue"].replace(0, np.nan)).dropna().values
                if len(expense_ratios) >= 2:
                    x_er = np.arange(len(expense_ratios), dtype=float)
                    if np.std(x_er) > 0:
                        er_slope = np.polyfit(x_er, expense_ratios, 1)[0]
                        features["expense_ratio_trend"] = round(float(er_slope), 4)

        return features

    def get_feature_vector(self, features: dict) -> np.ndarray:
        """Convert feature dict to ordered numpy array for model input."""
        return np.array([features.get(col, 0.0) for col in self.FEATURE_COLS], dtype=float)


class DelinquencyPipeline:
    """
    Temporal Compliance Intelligence pipeline.
    Loads pre-trained delinquency model and runs inference.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model = None
        self.config = {}
        self.feature_engineer = DelinquencyFeatureEngineer()
        self._loaded = False

    def load_models(self):
        """Load pre-trained delinquency model from disk."""
        model_path = self.model_dir / "delinquency_lgbm.joblib"
        config_path = self.model_dir / "delinquency_config.json"

        if not model_path.exists():
            print("[WARN] Delinquency model not found. Inference will use statistical baseline.")
            self._loaded = False
            return

        self.model = joblib.load(model_path)

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self._loaded = True
        print(f"[OK] Delinquency model loaded from {model_path}")

    @staticmethod
    def _normalize_horizon_probs(
        prob_30d: float,
        prob_60d: float,
        prob_90d: float,
    ) -> tuple[float, float, float, bool]:
        """Normalize horizon probabilities to satisfy P30 <= P60 <= P90."""
        p30 = max(0.0, min(1.0, float(prob_30d)))
        p60 = max(0.0, min(1.0, float(prob_60d)))
        p90 = max(0.0, min(1.0, float(prob_90d)))

        adjusted = False
        if p60 < p30:
            p60 = p30
            adjusted = True
        if p90 < p60:
            p90 = p60
            adjusted = True

        return round(p30, 4), round(p60, 4), round(p90, 4), adjusted

    def predict_single(
        self,
        payments_df: pd.DataFrame,
        tax_returns_df: Optional[pd.DataFrame] = None,
        company_info: Optional[dict] = None,
    ) -> dict:
        """
        Predict delinquency risk for a single company.

        Returns:
            dict with prob_30d, prob_60d, prob_90d, top_reasons, cluster, model_version
        """
        features = self.feature_engineer.compute_features(
            payments_df, tax_returns_df, company_info
        )
        feature_frame = pd.DataFrame(
            [[features.get(col, 0.0) for col in self.feature_engineer.FEATURE_COLS]],
            columns=self.feature_engineer.FEATURE_COLS,
        )

        if not self._loaded or self.model is None:
            # Statistical baseline fallback
            late_ratio = features.get("late_ratio_1yr", 0.0)
            unpaid = features.get("unpaid_count", 0)
            avg_days = features.get("avg_days_overdue", 0.0)

            prob_30d = min(1.0, late_ratio * 0.5 + min(unpaid * 0.1, 0.3) + min(avg_days / 120, 0.2))
            prob_60d = min(1.0, prob_30d * 1.15)
            prob_90d = min(1.0, prob_60d * 1.10)

            prob_30d, prob_60d, prob_90d, monotonic_adjusted = self._normalize_horizon_probs(
                prob_30d,
                prob_60d,
                prob_90d,
            )

            return self._build_result(
                prob_30d,
                prob_60d,
                prob_90d,
                features,
                "baseline-statistical-v1",
                monotonic_adjusted=monotonic_adjusted,
            )

        # ML model prediction
        try:
            # Model outputs 3 probabilities: P(30d), P(60d), P(90d)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(feature_frame)
                if probs.shape[1] >= 2:
                    # Binary classifier: probability of positive class
                    base_prob = float(probs[0, 1])
                    prob_30d = round(base_prob * 0.7, 4)
                    prob_60d = round(base_prob * 0.85, 4)
                    prob_90d = round(base_prob, 4)
                else:
                    prob_30d = prob_60d = prob_90d = float(probs[0, 0])
            else:
                pred = float(self.model.predict(feature_frame)[0])
                prob_30d = round(min(1.0, max(0.0, pred * 0.7)), 4)
                prob_60d = round(min(1.0, max(0.0, pred * 0.85)), 4)
                prob_90d = round(min(1.0, max(0.0, pred)), 4)
        except Exception as exc:
            print(f"[WARN] Delinquency model inference failed: {exc}")
            return self._build_result(
                0.0,
                0.0,
                0.0,
                features,
                "inference-error",
                monotonic_adjusted=False,
            )

        prob_30d, prob_60d, prob_90d, monotonic_adjusted = self._normalize_horizon_probs(
            prob_30d,
            prob_60d,
            prob_90d,
        )

        return self._build_result(
            prob_30d,
            prob_60d,
            prob_90d,
            features,
            DELINQUENCY_MODEL_VERSION,
            monotonic_adjusted=monotonic_adjusted,
        )

    def _build_result(
        self,
        prob_30d: float,
        prob_60d: float,
        prob_90d: float,
        features: dict,
        model_version: str,
        monotonic_adjusted: bool = False,
    ) -> dict:
        """Build standardized prediction result with top reasons."""
        # Generate top reasons from features
        reasons = []
        if features.get("late_ratio_1yr", 0) > 0.3:
            reasons.append({"reason": f"Tỷ lệ trễ hạn 1 năm: {features['late_ratio_1yr']:.0%}", "weight": features["late_ratio_1yr"]})
        if features.get("unpaid_count", 0) > 0:
            reasons.append({"reason": f"Số kỳ chưa nộp: {int(features['unpaid_count'])}", "weight": min(1.0, features["unpaid_count"] * 0.2)})
        if features.get("avg_days_overdue", 0) > 15:
            reasons.append({"reason": f"TB trễ: {features['avg_days_overdue']:.0f} ngày", "weight": min(1.0, features["avg_days_overdue"] / 90)})
        if features.get("recent_late_acceleration", 0) > 0.1:
            reasons.append({"reason": "Xu hướng trễ tăng gần đây", "weight": features["recent_late_acceleration"]})
        if features.get("penalty_trend", 0) > 0:
            reasons.append({"reason": "Phạt đang tăng", "weight": min(1.0, features["penalty_trend"] / 1e6)})
        if features.get("revenue_trend", 0) < -0.1:
            reasons.append({"reason": "Doanh thu đang giảm", "weight": abs(features["revenue_trend"])})

        reasons.sort(key=lambda x: x["weight"], reverse=True)
        reasons = reasons[:5]  # Top 5

        cluster = classify_delinquency_cluster(prob_30d, prob_60d, prob_90d)

        confidence = max(prob_90d, 1 - prob_90d) * 100

        return {
            "prob_30d": round(prob_30d, 4),
            "prob_60d": round(prob_60d, 4),
            "prob_90d": round(prob_90d, 4),
            "top_reasons": reasons,
            "cluster": cluster,
            "model_version": model_version,
            "model_confidence": round(confidence, 1),
            "monotonic_adjusted": monotonic_adjusted,
        }

    def predict_batch(self, companies_data: list[dict]) -> list[dict]:
        """
        Batch prediction for multiple companies.
        Memory-efficient: processes one company at a time.
        """
        results = []
        for company in companies_data:
            payments_df = pd.DataFrame(company.get("payments", []))
            tax_returns_df = pd.DataFrame(company.get("tax_returns", []))
            company_info = company.get("info", {})

            result = self.predict_single(payments_df, tax_returns_df, company_info)
            result["tax_code"] = company.get("tax_code", "")
            results.append(result)

        return results
