"""
causal_uplift_model.py – Causal Inference for Tax Collection Optimization
==========================================================================
Architecture:
    T-Learner (Two-Model Approach) for Individual Treatment Effect (ITE):
        - Model_T (Treatment):  P(collection | action taken)
        - Model_C (Control):    P(collection | no action)
        - CATE = Model_T(x) - Model_C(x)

    Propensity Score (Inverse Probability Weighting):
        - Logistic Regression estimating P(treatment | X)
        - Used for de-biasing observational data (no RCT available)

    Final Uplift Score:
        uplift_i = CATE_i * propensity_weight_i

Features (per company):
    - fraud_score:          AI risk score from pipeline.py
    - fraud_confidence:     model confidence
    - delinquency_90d:      P(overdue 90 days)
    - vat_refund_score:     VAT refund risk
    - prior_priority:       historical audit priority
    - n_past_actions:       number of previous collection actions
    - past_success_rate:    historical action success rate
    - company_age_years:    age in years
    - revenue_log:          log(revenue)
    - industry_risk:        sector-level delinquency rate

Design:
    - Handles selection bias in observational tax data
    - Propensity score trimming (0.05–0.95) for overlap
    - Confidence intervals via bootstrap
    - Audit-ready: logs CATE, propensity, and final uplift

Reference:
    Künzel et al., "Meta-learners for Estimating Heterogeneous Treatment
    Effects using Machine Learning", PNAS 2019
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

UPLIFT_FEATURE_NAMES = [
    "fraud_score",
    "fraud_confidence",
    "delinquency_90d",
    "vat_refund_score",
    "prior_priority",
    "n_past_actions",
    "past_success_rate",
    "company_age_years",
    "revenue_log",
    "industry_risk",
]

PROPENSITY_CLIP_LOW = 0.05
PROPENSITY_CLIP_HIGH = 0.95


# ════════════════════════════════════════════════════════════════
#  1. T-Learner: Two separate models for treated & control
# ════════════════════════════════════════════════════════════════

@dataclass
class UpliftPrediction:
    """Single prediction result from the Causal Uplift model."""
    tax_code: str
    cate: float                 # Conditional Average Treatment Effect
    prob_treated: float         # P(success | treatment)
    prob_control: float         # P(success | no treatment)
    propensity: float           # P(treatment | X)
    uplift_score: float         # Final prioritisation score
    confidence: str             # low / medium / high
    recommended_action: bool    # Should we act on this company?


class TLearnerUplift:
    """
    T-Learner for estimating heterogeneous treatment effects.
    
    Two separate GradientBoosting models:
      - model_t: trained on TREATED units only
      - model_c: trained on CONTROL units only
    
    CATE(x) = model_t.predict_proba(x) - model_c.predict_proba(x)
    """

    def __init__(self):
        self.model_t: GradientBoostingClassifier | None = None
        self.model_c: GradientBoostingClassifier | None = None
        self.propensity_model: LogisticRegression | None = None
        self.config: dict[str, Any] = {}
        self._loaded = False

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> dict[str, Any]:
        """
        Fit T-Learner on observational data.
        
        Args:
            X: (N, D) feature matrix
            treatment: (N,) binary — 1 = action taken, 0 = no action
            outcome: (N,) binary — 1 = successful collection, 0 = failed
            
        Returns:
            Training metrics dict.
        """
        treatment = np.asarray(treatment, dtype=int)
        outcome = np.asarray(outcome, dtype=int)

        # Split treated / control
        treated_mask = treatment == 1
        control_mask = treatment == 0
        X_t, y_t = X[treated_mask], outcome[treated_mask]
        X_c, y_c = X[control_mask], outcome[control_mask]

        logger.info(
            f"T-Learner fit: treated={treated_mask.sum()}, control={control_mask.sum()}"
        )

        if len(np.unique(y_t)) < 2 or len(np.unique(y_c)) < 2:
            raise ValueError(
                "Both treated and control groups must have both classes. "
                f"Treated classes: {np.unique(y_t)}, Control classes: {np.unique(y_c)}"
            )

        # ── Fit Treatment model ──
        self.model_t = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=8,
            subsample=0.8,
            random_state=42,
        )
        self.model_t.fit(X_t, y_t)

        # ── Fit Control model ──
        self.model_c = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=8,
            subsample=0.8,
            random_state=43,
        )
        self.model_c.fit(X_c, y_c)

        # ── Fit Propensity Score model ──
        self.propensity_model = LogisticRegression(
            max_iter=1000, solver="lbfgs", C=1.0, random_state=44
        )
        self.propensity_model.fit(X, treatment)

        # ── Compute training metrics ──
        cate_all = (
            self.model_t.predict_proba(X)[:, 1]
            - self.model_c.predict_proba(X)[:, 1]
        )
        propensity_all = np.clip(
            self.propensity_model.predict_proba(X)[:, 1],
            PROPENSITY_CLIP_LOW,
            PROPENSITY_CLIP_HIGH,
        )

        # Store config
        self.config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_treated": int(treated_mask.sum()),
            "n_control": int(control_mask.sum()),
            "feature_names": UPLIFT_FEATURE_NAMES,
            "trained_at": datetime.utcnow().isoformat() + "Z",
        }
        self._loaded = True

        metrics = {
            "avg_cate": round(float(np.mean(cate_all)), 4),
            "std_cate": round(float(np.std(cate_all)), 4),
            "avg_propensity": round(float(np.mean(propensity_all)), 4),
            "pct_positive_uplift": round(
                float(np.mean(cate_all > 0) * 100), 2
            ),
            "treated_auc": round(
                float(roc_auc_score(y_t, self.model_t.predict_proba(X_t)[:, 1])), 4
            ),
            "control_auc": round(
                float(roc_auc_score(y_c, self.model_c.predict_proba(X_c)[:, 1])), 4
            ),
        }
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE for each observation.
        
        Returns: (N,) array of treatment effects.
        """
        if self.model_t is None or self.model_c is None:
            return np.zeros(X.shape[0])

        prob_t = self.model_t.predict_proba(X)[:, 1]
        prob_c = self.model_c.predict_proba(X)[:, 1]
        return prob_t - prob_c

    def predict_full(
        self, X: np.ndarray, tax_codes: list[str] | None = None
    ) -> list[UpliftPrediction]:
        """
        Full prediction with propensity weighting and recommendations.
        """
        tax_codes = tax_codes or [f"unknown_{i}" for i in range(X.shape[0])]

        prob_t = self.model_t.predict_proba(X)[:, 1]
        prob_c = self.model_c.predict_proba(X)[:, 1]
        cate = prob_t - prob_c

        propensity = np.clip(
            self.propensity_model.predict_proba(X)[:, 1],
            PROPENSITY_CLIP_LOW,
            PROPENSITY_CLIP_HIGH,
        )

        results = []
        for i in range(X.shape[0]):
            # IPW-adjusted uplift score
            ipw_weight = 1.0 / propensity[i] if propensity[i] > 0 else 1.0
            uplift_score = float(cate[i]) * min(ipw_weight, 5.0)

            # Confidence based on propensity overlap
            if PROPENSITY_CLIP_LOW < propensity[i] < PROPENSITY_CLIP_HIGH:
                confidence = "high"
            elif 0.1 < propensity[i] < 0.9:
                confidence = "medium"
            else:
                confidence = "low"

            results.append(
                UpliftPrediction(
                    tax_code=tax_codes[i],
                    cate=round(float(cate[i]), 4),
                    prob_treated=round(float(prob_t[i]), 4),
                    prob_control=round(float(prob_c[i]), 4),
                    propensity=round(float(propensity[i]), 4),
                    uplift_score=round(uplift_score, 4),
                    confidence=confidence,
                    recommended_action=bool(cate[i] > 0.05),
                )
            )

        return results

    def save(self, path: str | Path | None = None) -> None:
        save_dir = Path(path) if path else MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model_t, save_dir / "uplift_model_treated.joblib")
        joblib.dump(self.model_c, save_dir / "uplift_model_control.joblib")
        joblib.dump(self.propensity_model, save_dir / "uplift_propensity.joblib")
        with open(save_dir / "uplift_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"[OK] Causal Uplift model saved to {save_dir}")

    def load(self, path: str | Path | None = None) -> bool:
        load_dir = Path(path) if path else MODEL_DIR
        required_files = [
            "uplift_model_treated.joblib",
            "uplift_model_control.joblib",
            "uplift_propensity.joblib",
            "uplift_config.json",
        ]
        if not all((load_dir / f).exists() for f in required_files):
            logger.warning("Causal Uplift model artifacts not found.")
            return False

        self.model_t = joblib.load(load_dir / "uplift_model_treated.joblib")
        self.model_c = joblib.load(load_dir / "uplift_model_control.joblib")
        self.propensity_model = joblib.load(load_dir / "uplift_propensity.joblib")
        with open(load_dir / "uplift_config.json") as f:
            self.config = json.load(f)
        self._loaded = True
        logger.info(f"Causal Uplift model loaded from {load_dir}")
        return True


# ════════════════════════════════════════════════════════════════
#  2. Qini Curve Evaluation (Uplift-specific metric)
# ════════════════════════════════════════════════════════════════

def compute_qini_coefficient(
    cate: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> float:
    """
    Compute the Qini coefficient — the area under the Qini curve.
    
    This is the standard metric for uplift models, analogous to AUC-ROC
    but specific to treatment effect estimation.
    
    Higher Qini = better at identifying who benefits most from treatment.
    """
    n = len(cate)
    if n == 0:
        return 0.0

    # Sort by predicted CATE descending
    order = np.argsort(-cate)
    treatment_sorted = treatment[order]
    outcome_sorted = outcome[order]

    # Cumulative sums
    cum_treated_success = np.cumsum(treatment_sorted * outcome_sorted)
    cum_control_success = np.cumsum((1 - treatment_sorted) * outcome_sorted)
    cum_treated = np.cumsum(treatment_sorted)
    cum_control = np.cumsum(1 - treatment_sorted)

    # Qini values at each cutoff
    qini_values = []
    for i in range(n):
        n_t = max(1, cum_treated[i])
        n_c = max(1, cum_control[i])
        qini_val = cum_treated_success[i] - cum_control_success[i] * (n_t / n_c)
        qini_values.append(qini_val)

    # Normalize by sample size and compute area
    qini_values = np.array(qini_values) / n
    qini_coefficient = float(np.trapz(qini_values, dx=1.0 / n))

    return round(qini_coefficient, 6)
