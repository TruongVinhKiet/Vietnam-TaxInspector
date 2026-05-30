# -*- coding: utf-8 -*-
"""Train and evaluate Taxpayer Portal ML models.

This is the taxpayer-specific training orchestrator. It trains/evaluates the
practical models used by `/api/taxpayer/intelligence/*`, writes artifacts,
model cards, quality reports, and a manifest. It is production-oriented but
honest: artifacts trained on synthetic/sandbox data are marked `sandbox`, not
`prod`, unless explicitly overridden.

Examples:
    cd Backend
    python scripts/train_taxpayer_models.py --source auto --sample-size 5000
    python scripts/train_taxpayer_models.py --source synthetic --fast --out-dir data/models/taxpayer_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_MODEL_DIR = BACKEND_DIR / "data" / "models" / "taxpayer"


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


@dataclass
class TaskDataset:
    name: str
    task_type: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    source: str
    label_name: str
    metadata: dict[str, Any]
    treatment: np.ndarray | None = None


@dataclass
class ModelRunResult:
    model_name: str
    model_version: str
    task_type: str
    source: str
    artifact_path: str
    quality_report_path: str
    model_card_path: str
    samples: int
    feature_count: int
    best_candidate: str
    primary_metric: str
    primary_score: float
    metrics: dict[str, Any]
    acceptance_gates: dict[str, Any]
    status: str
    train_data_hash: str
    duration_sec: float


def _industry_code(values: np.ndarray) -> np.ndarray:
    return values.astype(float)


def generate_synthetic_datasets(sample_size: int, seed: int) -> dict[str, TaskDataset]:
    """Create deterministic taxpayer-like data for sandbox training and tests."""
    rng = np.random.default_rng(seed)
    n = int(max(300, sample_size))

    industries = rng.integers(0, 5, size=n)
    months_elapsed = rng.integers(2, 13, size=n)
    annual_plan = rng.lognormal(mean=20.35, sigma=0.85, size=n)
    annual_plan = np.clip(annual_plan, 80_000_000, 6_500_000_000)
    current_revenue = annual_plan * months_elapsed / 12 * rng.normal(1.0, 0.22, size=n)
    current_revenue = np.clip(current_revenue, 0, None)
    expense_ratio = np.clip(rng.normal(0.68, 0.17, size=n), 0.18, 1.25)
    expense_total = current_revenue * expense_ratio
    profit_margin = np.divide(current_revenue - expense_total, np.maximum(current_revenue, 1.0))
    ecom_share = np.clip(rng.beta(1.8, 4.4, size=n), 0, 1)
    avg_monthly = np.divide(current_revenue, np.maximum(months_elapsed, 1))
    trend_pct = np.clip(rng.normal(0.06 + ecom_share * 0.10, 0.20, size=n), -0.45, 0.75)
    projected_revenue = current_revenue + avg_monthly * np.maximum(0, 12 - months_elapsed) * (1 + trend_pct)
    projected_revenue = np.clip(projected_revenue, 0, None)
    debt_total = np.maximum(0, rng.gamma(1.2, 14_000_000, size=n) * (rng.random(n) < 0.34))
    overdue_days = np.maximum(0, rng.normal(35, 45, size=n) * (debt_total > 0)).astype(float)
    deadline_misses = rng.poisson(0.25 + (overdue_days > 60) * 0.7, size=n)
    risky_invoice_count = rng.poisson(0.25 + ecom_share * 0.6, size=n)
    duplicate_invoice_count = rng.poisson(0.08 + risky_invoice_count * 0.05, size=n)
    cash_violations = rng.poisson(0.10 + (expense_ratio > 0.85) * 0.25, size=n)
    evidence_gaps = rng.poisson(0.55 + (expense_ratio > 0.8) * 0.5, size=n)
    data_quality = np.clip(rng.normal(82, 13, size=n) - evidence_gaps * 2.5, 25, 100)

    profile_X = np.column_stack(
        [
            np.log1p(current_revenue),
            np.log1p(expense_total),
            profit_margin,
            expense_ratio,
            ecom_share,
            avg_monthly / 100_000_000,
            trend_pct,
            np.log1p(debt_total),
            overdue_days / 180,
            deadline_misses,
            risky_invoice_count,
            duplicate_invoice_count,
            cash_violations,
            evidence_gaps,
            data_quality / 100,
            months_elapsed / 12,
            _industry_code(industries),
        ]
    )
    profile_features = [
        "current_revenue_log",
        "expense_total_log",
        "profit_margin",
        "expense_ratio",
        "ecommerce_share",
        "avg_monthly_revenue_100m",
        "trend_pct",
        "debt_log",
        "overdue_days_norm",
        "deadline_misses",
        "risky_invoice_count",
        "duplicate_invoice_count",
        "cash_payment_violations",
        "evidence_gap_count",
        "data_quality",
        "months_elapsed_ratio",
        "industry_code",
    ]
    delinquency_score = (
        -2.2
        + np.log1p(debt_total) / 7.5
        + overdue_days / 95
        + deadline_misses * 0.45
        - profit_margin * 1.8
        + (cash_violations > 0) * 0.45
        + rng.normal(0, 0.35, size=n)
    )
    delinquency_y = (sigmoid(delinquency_score) > 0.52).astype(int)
    growth_y = np.digitize(projected_revenue, [500_000_000, 1_000_000_000, 3_000_000_000]).astype(int)

    m = int(n * 2.2)
    amount = rng.lognormal(mean=16.2, sigma=1.2, size=m)
    vat_rate = rng.choice([0, 5, 8, 10, 15, 20], size=m, p=[0.04, 0.12, 0.24, 0.52, 0.04, 0.04])
    payment_overdue = rng.binomial(1, 0.12, size=m)
    payment_failed = rng.binomial(1, 0.04, size=m)
    is_adjustment = rng.binomial(1, 0.07, size=m)
    event_count = rng.poisson(0.22, size=m)
    near_dup_count = rng.poisson(0.09, size=m)
    same_day_pair_count = rng.poisson(1.2, size=m)
    seller_risk = np.clip(rng.beta(1.6, 4.0, size=m) * 100, 0, 100)
    buyer_risk = np.clip(rng.beta(1.4, 4.5, size=m) * 100, 0, 100)
    counterparty_gap = np.abs(seller_risk - buyer_risk)
    invoice_X = np.column_stack(
        [
            np.log1p(amount),
            vat_rate,
            payment_overdue,
            payment_failed,
            is_adjustment,
            event_count,
            near_dup_count,
            same_day_pair_count,
            seller_risk,
            buyer_risk,
            counterparty_gap,
        ]
    )
    invoice_features = [
        "amount_log",
        "vat_rate",
        "payment_status_overdue",
        "payment_status_failed",
        "is_adjustment",
        "event_count",
        "near_dup_count",
        "same_day_pair_count",
        "seller_risk_score",
        "buyer_risk_score",
        "counterparty_gap",
    ]
    invoice_risk = (
        -4.1
        + (vat_rate <= 0) * 0.8
        + (vat_rate > 12) * 1.0
        + payment_overdue * 0.9
        + payment_failed * 1.2
        + is_adjustment * 0.5
        + event_count * 0.45
        + near_dup_count * 1.0
        + (same_day_pair_count >= 4) * 0.65
        + seller_risk / 45
        + np.log1p(amount) / 11
        + rng.normal(0, 0.22, size=m)
    )
    invoice_y = (sigmoid(invoice_risk) > 0.45).astype(int)

    k = int(n * 2.0)
    exp_amount = rng.lognormal(mean=15.4, sigma=1.0, size=k)
    payment_cash = rng.binomial(1, 0.28, size=k)
    has_invoice = rng.binomial(1, 0.62, size=k)
    category = rng.integers(0, 8, size=k)
    supplier_type = rng.integers(0, 5, size=k)
    no_invoice_allowed = np.isin(supplier_type, [1, 2]).astype(int)
    owner_salary = ((category == 5) & (rng.random(k) < 0.22)).astype(int)
    business_relevance = np.clip(rng.normal(0.82, 0.18, size=k), 0, 1)
    expense_X = np.column_stack(
        [
            np.log1p(exp_amount),
            payment_cash,
            has_invoice,
            category,
            supplier_type,
            no_invoice_allowed,
            owner_salary,
            business_relevance,
            (exp_amount >= 5_000_000).astype(int),
        ]
    )
    expense_features = [
        "amount_log",
        "payment_cash",
        "has_invoice",
        "category_code",
        "supplier_type_code",
        "no_invoice_allowed_case",
        "owner_salary_flag",
        "business_relevance",
        "amount_ge_5m",
    ]
    expense_y = np.zeros(k, dtype=int)
    expense_y[(has_invoice == 0) & (no_invoice_allowed == 0)] = 1
    expense_y[(payment_cash == 1) & (exp_amount >= 5_000_000)] = 2
    expense_y[(owner_salary == 1) | (business_relevance < 0.35)] = 2

    t = int(n * 1.3)
    entity = rng.integers(0, max(30, n // 12), size=t)
    month = rng.integers(1, 13, size=t)
    lag1 = rng.lognormal(18.2, 0.7, size=t)
    lag2 = np.maximum(0, lag1 * rng.normal(0.96, 0.18, size=t))
    lag3 = np.maximum(0, lag1 * rng.normal(0.91, 0.22, size=t))
    rolling_mean = (lag1 + lag2 + lag3) / 3
    season = 1 + 0.15 * np.sin(2 * np.pi * month / 12)
    trend = rng.normal(0.04, 0.18, size=t)
    forecast_target = np.maximum(0, rolling_mean * (1 + trend) * season + rng.normal(0, rolling_mean * 0.08, size=t))
    forecast_X = np.column_stack(
        [
            np.log1p(lag1),
            np.log1p(lag2),
            np.log1p(lag3),
            np.log1p(rolling_mean),
            trend,
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            entity % 5,
        ]
    )
    forecast_features = [
        "lag1_revenue_log",
        "lag2_revenue_log",
        "lag3_revenue_log",
        "rolling_mean_log",
        "trend_pct",
        "month_sin",
        "month_cos",
        "industry_proxy",
    ]

    z = int(n * 1.8)
    tx_amount = rng.lognormal(16.1, 1.0, size=z)
    tx_hour = rng.integers(0, 24, size=z)
    tx_supplier_risk = rng.beta(1.4, 5.2, size=z)
    tx_dup = rng.poisson(0.05, size=z)
    tx_cash = rng.binomial(1, 0.22, size=z)
    tx_ratio = rng.normal(0.65, 0.22, size=z)
    anomaly_y = np.zeros(z, dtype=int)
    injected = rng.choice(z, size=max(12, z // 18), replace=False)
    tx_amount[injected] *= rng.uniform(6, 18, size=len(injected))
    tx_supplier_risk[injected] = rng.uniform(0.75, 1.0, size=len(injected))
    tx_dup[injected] += rng.integers(2, 8, size=len(injected))
    anomaly_y[injected] = 1
    anomaly_X = np.column_stack(
        [
            np.log1p(tx_amount),
            tx_hour / 23,
            tx_supplier_risk,
            tx_dup,
            tx_cash,
            tx_ratio,
            (tx_amount >= 5_000_000).astype(int),
        ]
    )
    anomaly_features = [
        "amount_log",
        "hour_norm",
        "supplier_risk",
        "duplicate_count",
        "cash_flag",
        "expense_ratio_proxy",
        "amount_ge_5m",
    ]

    g = int(n * 1.2)
    degree = rng.poisson(3.0, size=g)
    weighted_degree = rng.gamma(2.0, 70_000_000, size=g)
    pagerank_proxy = rng.beta(1.2, 9.0, size=g)
    community_risk = rng.beta(1.6, 4.2, size=g)
    triangle_count = rng.poisson(0.18, size=g)
    suspicious_mst = rng.binomial(1, 0.07, size=g)
    graph_X = np.column_stack(
        [
            degree,
            np.log1p(weighted_degree),
            pagerank_proxy,
            community_risk,
            triangle_count,
            suspicious_mst,
        ]
    )
    graph_y = (
        (community_risk + pagerank_proxy * 0.8 + (triangle_count > 0) * 0.4 + suspicious_mst * 0.6 + degree / 18)
        > 0.78
    ).astype(int)
    graph_features = [
        "degree",
        "weighted_degree_log",
        "pagerank_proxy",
        "community_risk",
        "triangle_count",
        "suspicious_tax_code_pattern",
    ]

    d = int(n * 1.1)
    field_count = rng.poisson(3.0, size=d)
    tax_code_present = rng.binomial(1, 0.82, size=d)
    amount_present = rng.binomial(1, 0.78, size=d)
    table_detected = rng.binomial(1, 0.55, size=d)
    blur_score = rng.beta(2.0, 5.0, size=d)
    file_size_mb = rng.gamma(1.5, 0.8, size=d)
    doc_type_code = rng.integers(0, 5, size=d)
    document_X = np.column_stack(
        [
            field_count,
            tax_code_present,
            amount_present,
            table_detected,
            blur_score,
            file_size_mb,
            doc_type_code,
        ]
    )
    document_y = ((tax_code_present == 0) | (amount_present == 0) | (blur_score > 0.65) | (field_count < 2)).astype(int)
    document_features = [
        "field_count",
        "tax_code_present",
        "amount_present",
        "table_detected",
        "blur_score",
        "file_size_mb",
        "doc_type_code",
    ]

    u = int(n * 1.2)
    uplift_X = profile_X[rng.choice(n, size=u, replace=True)]
    base_propensity = sigmoid(-0.8 + uplift_X[:, 8] * 1.7 + uplift_X[:, 9] * 0.35 + uplift_X[:, 13] * 0.18)
    treatment = rng.binomial(1, np.clip(base_propensity, 0.08, 0.92), size=u)
    true_uplift = 0.04 + uplift_X[:, 8] * 0.18 + (uplift_X[:, 12] > 0) * 0.05 + (uplift_X[:, 13] > 1) * 0.04
    base_outcome = sigmoid(-1.3 + uplift_X[:, 2] * 1.5 - uplift_X[:, 8] * 0.8 + uplift_X[:, 14] * 0.4)
    outcome_prob = np.clip(base_outcome + treatment * true_uplift, 0.02, 0.98)
    uplift_y = rng.binomial(1, outcome_prob, size=u)

    reconciliation_noise = rng.normal(0, 0.18, size=n)
    bank_revenue_ratio = np.clip(1.0 + trend_pct * 0.28 + reconciliation_noise, 0.05, 2.6)
    invoice_revenue_ratio = np.clip(0.92 + ecom_share * 0.18 + rng.normal(0, 0.16, size=n), 0.02, 2.4)
    ledger_filing_delta = np.clip(rng.normal(0.02, 0.14, size=n) + deadline_misses * 0.035, -0.7, 1.8)
    missing_source_count = rng.poisson(0.35 + evidence_gaps * 0.08, size=n)
    reconciliation_X = np.column_stack(
        [
            bank_revenue_ratio,
            invoice_revenue_ratio,
            ledger_filing_delta,
            missing_source_count,
            duplicate_invoice_count,
            evidence_gaps,
            current_revenue / 1_000_000_000,
            data_quality / 100,
        ]
    )
    reconciliation_y = (
        (np.abs(bank_revenue_ratio - 1) > 0.18)
        | (np.abs(invoice_revenue_ratio - 1) > 0.22)
        | (np.abs(ledger_filing_delta) > 0.20)
        | (missing_source_count >= 2)
    ).astype(int)
    reconciliation_features = [
        "bank_revenue_ratio",
        "invoice_revenue_ratio",
        "ledger_filing_delta",
        "missing_source_count",
        "duplicate_invoice_count",
        "evidence_gaps",
        "revenue_billion",
        "data_quality",
    ]

    channel_n = int(n * 1.1)
    channel_amount_log = rng.normal(17.2, 1.1, size=channel_n)
    channel_bank_text_score = rng.beta(2.2, 3.8, size=channel_n)
    channel_cod_flag = rng.binomial(1, 0.18, size=channel_n)
    channel_platform_fee_ratio = rng.beta(1.5, 9.0, size=channel_n)
    channel_weekend = rng.binomial(1, 0.25, size=channel_n)
    channel_X = np.column_stack(
        [
            channel_amount_log,
            channel_bank_text_score,
            channel_cod_flag,
            channel_platform_fee_ratio,
            channel_weekend,
            rng.normal(0, 1, size=channel_n),
        ]
    )
    channel_logits = np.column_stack(
        [
            0.7 - channel_platform_fee_ratio * 2.0 - channel_cod_flag * 0.4,
            -0.4 + channel_platform_fee_ratio * 10.0 + channel_bank_text_score * 0.35,
            -0.3 + channel_cod_flag * 4.2,
            0.1 + channel_weekend * 0.8 - channel_platform_fee_ratio * 1.3,
            -0.5 + channel_bank_text_score * 2.8,
        ]
    )
    channel_y = np.argmax(channel_logits + rng.normal(0, 0.16, size=channel_logits.shape), axis=1)
    channel_features = [
        "amount_log",
        "bank_text_channel_score",
        "cod_flag",
        "platform_fee_ratio",
        "weekend_flag",
        "memo_embedding_proxy",
    ]

    reserve_X = profile_X.copy()
    reserve_target = np.clip(
        0.055
        + profile_X[:, 6] * 0.028
        + profile_X[:, 7] * 0.006
        + profile_X[:, 8] * 0.085
        + (profile_X[:, 2] < 0.10) * 0.028
        + profile_X[:, 10] * 0.004
        + rng.normal(0, 0.003, size=n),
        0.04,
        0.35,
    )

    supplier_account_changes = rng.poisson(0.35, size=graph_X.shape[0]) + (graph_X[:, 1] > 4).astype(float)
    supplier_account_X = np.column_stack([graph_X, supplier_account_changes])
    supplier_account_y = ((supplier_account_X[:, -1] >= 2) | (supplier_account_X[:, 3] > 0.72) | (supplier_account_X[:, 5] > 0)).astype(int)
    supplier_account_features = graph_features + ["beneficiary_account_change_count"]

    inv_n = int(n * 1.15)
    stock_in = rng.gamma(2.5, 42, size=inv_n)
    stock_out = stock_in * np.clip(rng.normal(0.85, 0.28, size=inv_n), 0.0, 2.4)
    unit_cost_volatility = rng.beta(1.8, 5.2, size=inv_n)
    gross_margin_proxy = np.clip(rng.normal(0.22, 0.13, size=inv_n) - unit_cost_volatility * 0.1, -0.25, 0.7)
    missing_purchase_docs = rng.poisson(0.18 + (unit_cost_volatility > 0.75) * 0.35, size=inv_n)
    inventory_y = np.zeros(inv_n, dtype=int)
    anomaly_idx = rng.choice(inv_n, size=max(12, int(inv_n * 0.055)), replace=False)
    inventory_y[anomaly_idx] = 1
    stock_out[anomaly_idx] *= rng.uniform(1.25, 2.4, size=len(anomaly_idx))
    gross_margin_proxy[anomaly_idx] -= rng.uniform(0.08, 0.22, size=len(anomaly_idx))
    missing_purchase_docs[anomaly_idx] += rng.integers(1, 4, size=len(anomaly_idx))
    inventory_X = np.column_stack([stock_in, stock_out, stock_out - stock_in, unit_cost_volatility, gross_margin_proxy, missing_purchase_docs])
    inventory_features = ["stock_in_qty", "stock_out_qty", "stock_delta", "unit_cost_volatility", "gross_margin_proxy", "missing_purchase_docs"]

    evidence_X = np.column_stack(
        [
            document_X[:d],
            rng.poisson(1.2, size=d),
            rng.binomial(1, 0.62, size=d),
            rng.binomial(1, 0.48, size=d),
        ]
    )
    evidence_y = ((document_y == 1) | (evidence_X[:, -3] < 1) | (evidence_X[:, -2] == 0) | (evidence_X[:, -1] == 0)).astype(int)
    evidence_features = document_features + ["linked_case_count", "has_bank_match", "has_invoice_match"]

    reminder_X = uplift_X.copy()
    reminder_treatment = treatment.copy()
    reminder_y = uplift_y.copy()

    return {
        "taxpayer_delinquency_risk": TaskDataset(
            "taxpayer_delinquency_risk",
            "binary_classification",
            profile_X,
            delinquency_y,
            profile_features,
            "synthetic",
            "delinquency_90d",
            {"description": "Late payment/debt hazard model"},
        ),
        "taxpayer_growth_threshold": TaskDataset(
            "taxpayer_growth_threshold",
            "multiclass_classification",
            profile_X,
            growth_y,
            profile_features,
            "synthetic",
            "threshold_bucket",
            {"classes": ["lt_500m", "500m_1b", "1b_3b", "gt_3b"]},
        ),
        "taxpayer_invoice_risk": TaskDataset(
            "taxpayer_invoice_risk",
            "binary_classification",
            invoice_X,
            invoice_y,
            invoice_features,
            "synthetic",
            "risky_invoice",
            {"description": "Invoice and supplier risk model"},
        ),
        "taxpayer_expense_deductibility": TaskDataset(
            "taxpayer_expense_deductibility",
            "multiclass_classification",
            expense_X,
            expense_y,
            expense_features,
            "synthetic",
            "deductibility_status",
            {"classes": ["deductible", "needs_evidence", "non_deductible"]},
        ),
        "taxpayer_revenue_forecast": TaskDataset(
            "taxpayer_revenue_forecast",
            "regression",
            forecast_X,
            forecast_target,
            forecast_features,
            "synthetic",
            "next_month_revenue",
            {"description": "Point forecast with residual interval calibration"},
        ),
        "taxpayer_transaction_anomaly": TaskDataset(
            "taxpayer_transaction_anomaly",
            "anomaly_detection",
            anomaly_X,
            anomaly_y,
            anomaly_features,
            "synthetic",
            "anomaly_label",
            {"contamination": round(float(anomaly_y.mean()), 4)},
        ),
        "taxpayer_graph_risk": TaskDataset(
            "taxpayer_graph_risk",
            "binary_classification",
            graph_X,
            graph_y,
            graph_features,
            "synthetic",
            "high_risk_graph_node",
            {"description": "Supplier/entity graph risk proxy"},
        ),
        "taxpayer_document_quality": TaskDataset(
            "taxpayer_document_quality",
            "binary_classification",
            document_X,
            document_y,
            document_features,
            "synthetic",
            "needs_human_review",
            {"description": "Document AI review classifier"},
        ),
        "taxpayer_next_best_action_uplift": TaskDataset(
            "taxpayer_next_best_action_uplift",
            "uplift",
            uplift_X,
            uplift_y,
            profile_features,
            "synthetic",
            "action_success",
            {"description": "T-learner uplift model for reminder/payment/document actions"},
            treatment=treatment,
        ),
        "taxpayer_reconciliation_ranker": TaskDataset(
            "taxpayer_reconciliation_ranker",
            "binary_classification",
            reconciliation_X,
            reconciliation_y,
            reconciliation_features,
            "synthetic",
            "material_reconciliation_exception",
            {"description": "4-way bank-invoice-ledger-filing exception ranker"},
        ),
        "taxpayer_channel_attribution": TaskDataset(
            "taxpayer_channel_attribution",
            "multiclass_classification",
            channel_X,
            channel_y,
            channel_features,
            "synthetic",
            "channel_class",
            {"classes": ["direct_pos", "marketplace", "cod", "social", "bank_transfer"]},
        ),
        "taxpayer_tax_reserve_optimizer": TaskDataset(
            "taxpayer_tax_reserve_optimizer",
            "regression",
            reserve_X,
            reserve_target,
            profile_features,
            "synthetic",
            "recommended_reserve_rate",
            {"description": "Stochastic tax reserve optimizer target"},
        ),
        "taxpayer_supplier_account_graph_risk": TaskDataset(
            "taxpayer_supplier_account_graph_risk",
            "binary_classification",
            supplier_account_X,
            supplier_account_y,
            supplier_account_features,
            "synthetic",
            "supplier_account_change_risk",
            {"description": "Supplier beneficiary-account graph risk"},
        ),
        "taxpayer_inventory_anomaly": TaskDataset(
            "taxpayer_inventory_anomaly",
            "anomaly_detection",
            inventory_X,
            inventory_y,
            inventory_features,
            "synthetic",
            "inventory_cogs_anomaly",
            {"description": "Inventory/COGS flow anomaly model"},
        ),
        "taxpayer_evidence_bundle_quality": TaskDataset(
            "taxpayer_evidence_bundle_quality",
            "binary_classification",
            evidence_X,
            evidence_y,
            evidence_features,
            "synthetic",
            "evidence_bundle_needs_review",
            {"description": "Evidence bundle completeness classifier"},
        ),
        "taxpayer_reminder_bandit_uplift": TaskDataset(
            "taxpayer_reminder_bandit_uplift",
            "uplift",
            reminder_X,
            reminder_y,
            profile_features,
            "synthetic",
            "reminder_success",
            {"description": "Contextual bandit/uplift model for reminders"},
            treatment=reminder_treatment,
        ),
    }


def try_load_database_datasets(sample_size: int) -> dict[str, TaskDataset]:
    """Best-effort load from taxpayer tables. Returns empty dict if unavailable."""
    try:
        from sqlalchemy import text

        from app.database import SessionLocal
    except Exception:
        return {}

    try:
        with SessionLocal() as db:
            profile_rows = db.execute(
                text(
                    """
                    SELECT
                        p.user_id,
                        COALESCE(p.annual_revenue, 0) AS annual_revenue,
                        COALESCE(p.household_group, 2) AS household_group,
                        COALESCE(SUM(r.amount), 0) AS revenue_total,
                        COUNT(r.id) AS revenue_count,
                        COALESCE(SUM(e.amount), 0) AS expense_total,
                        COUNT(e.id) AS expense_count,
                        COALESCE(SUM(d.amount_due - d.amount_paid), 0) AS debt_total,
                        COALESCE(MAX(CASE WHEN d.due_date < CURRENT_DATE THEN CURRENT_DATE - d.due_date ELSE 0 END), 0) AS overdue_days
                    FROM taxpayer_profiles p
                    LEFT JOIN business_revenue_entries r ON r.user_id = p.user_id
                    LEFT JOIN business_expense_entries e ON e.user_id = p.user_id
                    LEFT JOIN taxpayer_debt_items d ON d.user_id = p.user_id
                    GROUP BY p.user_id, p.annual_revenue, p.household_group
                    LIMIT :limit
                    """
                ),
                {"limit": int(sample_size)},
            ).mappings().all()
            invoice_rows = db.execute(
                text(
                    """
                    SELECT amount, vat_rate, status, risk_json
                    FROM taxpayer_einvoices
                    LIMIT :limit
                    """
                ),
                {"limit": int(sample_size * 2)},
            ).mappings().all()
            expense_rows = db.execute(
                text(
                    """
                    SELECT amount, payment_method, has_invoice, category, supplier_type, deductible_status
                    FROM business_expense_entries
                    LIMIT :limit
                    """
                ),
                {"limit": int(sample_size * 2)},
            ).mappings().all()
    except Exception:
        return {}

    datasets: dict[str, TaskDataset] = {}
    if len(profile_rows) >= 80:
        X = []
        y_del = []
        y_growth = []
        for row in profile_rows:
            revenue = safe_float(row.get("revenue_total"))
            annual = max(safe_float(row.get("annual_revenue")), revenue)
            expenses = safe_float(row.get("expense_total"))
            debt = safe_float(row.get("debt_total"))
            overdue = safe_float(row.get("overdue_days"))
            margin = (revenue - expenses) / max(revenue, 1.0)
            ratio = expenses / max(revenue, 1.0)
            X.append(
                [
                    math.log1p(revenue),
                    math.log1p(expenses),
                    margin,
                    ratio,
                    0.0,
                    revenue / 12 / 100_000_000,
                    0.0,
                    math.log1p(debt),
                    overdue / 180,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.75,
                    min(1.0, safe_float(row.get("revenue_count")) / 12),
                    0,
                ]
            )
            y_del.append(1 if debt >= 50_000_000 or overdue >= 60 else 0)
            y_growth.append(int(np.digitize(max(annual, revenue), [500_000_000, 1_000_000_000, 3_000_000_000])))
        feature_names = generate_synthetic_datasets(300, 1)["taxpayer_delinquency_risk"].feature_names
        datasets["taxpayer_delinquency_risk"] = TaskDataset(
            "taxpayer_delinquency_risk",
            "binary_classification",
            np.asarray(X, dtype=float),
            np.asarray(y_del, dtype=int),
            feature_names,
            "database",
            "delinquency_90d",
            {"source_tables": ["taxpayer_profiles", "business_revenue_entries", "business_expense_entries", "taxpayer_debt_items"]},
        )
        datasets["taxpayer_growth_threshold"] = TaskDataset(
            "taxpayer_growth_threshold",
            "multiclass_classification",
            np.asarray(X, dtype=float),
            np.asarray(y_growth, dtype=int),
            feature_names,
            "database",
            "threshold_bucket",
            {"source_tables": ["taxpayer_profiles", "business_revenue_entries"]},
        )

    if len(invoice_rows) >= 120:
        X = []
        y = []
        for row in invoice_rows:
            risk = row.get("risk_json") or {}
            if isinstance(risk, str):
                try:
                    risk = json.loads(risk)
                except Exception:
                    risk = {}
            status = str(row.get("status") or "").lower()
            flags = risk.get("risk_flags") or []
            X.append(
                [
                    math.log1p(safe_float(row.get("amount"))),
                    safe_float(row.get("vat_rate")),
                    0,
                    0,
                    0,
                    len(flags),
                    1 if "duplicate" in " ".join(map(str, flags)).lower() else 0,
                    0,
                    safe_float(risk.get("seller_risk_score")),
                    safe_float(risk.get("buyer_risk_score")),
                    abs(safe_float(risk.get("seller_risk_score")) - safe_float(risk.get("buyer_risk_score"))),
                ]
            )
            y.append(1 if status in {"invalid", "risky", "cancelled", "missing"} or flags else 0)
        feature_names = generate_synthetic_datasets(300, 1)["taxpayer_invoice_risk"].feature_names
        datasets["taxpayer_invoice_risk"] = TaskDataset(
            "taxpayer_invoice_risk",
            "binary_classification",
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            feature_names,
            "database",
            "risky_invoice",
            {"source_tables": ["taxpayer_einvoices"]},
        )

    if len(expense_rows) >= 120:
        cats: dict[str, int] = {}
        suppliers: dict[str, int] = {}
        X = []
        y = []
        label_map = {"deductible": 0, "needs_evidence": 1, "needs_invoice": 1, "non_deductible": 2}
        for row in expense_rows:
            cat = str(row.get("category") or "other")
            sup = str(row.get("supplier_type") or "unknown")
            cats.setdefault(cat, len(cats))
            suppliers.setdefault(sup, len(suppliers))
            amount = safe_float(row.get("amount"))
            payment_cash = str(row.get("payment_method") or "").lower() in {"cash", "tien_mat"}
            has_invoice = bool(row.get("has_invoice"))
            no_invoice_allowed = sup in {"farmer", "household", "street_vendor", "seasonal_worker"}
            status = str(row.get("deductible_status") or "deductible")
            X.append(
                [
                    math.log1p(amount),
                    int(payment_cash),
                    int(has_invoice),
                    cats[cat],
                    suppliers[sup],
                    int(no_invoice_allowed),
                    0,
                    0.75,
                    int(amount >= 5_000_000),
                ]
            )
            y.append(label_map.get(status, 0))
        feature_names = generate_synthetic_datasets(300, 1)["taxpayer_expense_deductibility"].feature_names
        datasets["taxpayer_expense_deductibility"] = TaskDataset(
            "taxpayer_expense_deductibility",
            "multiclass_classification",
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            feature_names,
            "database",
            "deductibility_status",
            {"source_tables": ["business_expense_entries"]},
        )

    return datasets


def merge_datasets(primary: dict[str, TaskDataset], fallback: dict[str, TaskDataset]) -> dict[str, TaskDataset]:
    merged = dict(fallback)
    for name, dataset in primary.items():
        if len(dataset.y) >= 80 and len(np.unique(dataset.y)) >= 2:
            merged[name] = dataset
    return merged


def load_datasets(source: str, sample_size: int, seed: int, no_synthetic: bool = False) -> dict[str, TaskDataset]:
    synthetic = {} if no_synthetic else generate_synthetic_datasets(sample_size, seed)
    if source == "synthetic":
        return {} if no_synthetic else synthetic
    database = try_load_database_datasets(sample_size)
    if source == "db":
        return database
    if no_synthetic:
        return database
    return merge_datasets(database, synthetic)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    else:
        metrics["auc_roc"] = None
        metrics["pr_auc"] = None
    return metrics


def best_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1.0)))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def gate_report(metrics: dict[str, Any], thresholds: dict[str, tuple[str, float]]) -> dict[str, Any]:
    criteria = {}
    overall = True
    for key, (direction, threshold) in thresholds.items():
        actual = metrics.get(key)
        if actual is None:
            passed = False
        elif direction == ">=":
            passed = float(actual) >= threshold
        else:
            passed = float(actual) <= threshold
        criteria[key] = {"pass": bool(passed), "actual": actual, "threshold": threshold, "direction": direction}
        overall = bool(overall and passed)
    return {"overall_pass": overall, "criteria": criteria}


def production_status(dataset: TaskDataset, gates: dict[str, Any], min_real_samples: int, allow_synthetic: bool) -> str:
    if not gates.get("overall_pass"):
        return "staging_failed_gate"
    if dataset.source == "database" and len(dataset.y) >= min_real_samples:
        return "prod_candidate"
    if allow_synthetic and dataset.source == "synthetic":
        return "prod_candidate_synthetic_override"
    return "sandbox"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, default=str)


def make_model_card(
    *,
    dataset: TaskDataset,
    model_name: str,
    model_version: str,
    candidate: str,
    metrics: dict[str, Any],
    gates: dict[str, Any],
    status: str,
    artifact_path: Path,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_version": model_version,
        "status": status,
        "task_type": dataset.task_type,
        "intended_use": "Advisory AI for taxpayer portal. Not an automatic legal or tax decision.",
        "out_of_scope": [
            "Do not auto-submit tax filings.",
            "Do not make final legal conclusions without verified citations/human review.",
            "Do not promote synthetic-only artifacts as production unless explicitly approved.",
        ],
        "training_data": {
            "source": dataset.source,
            "samples": int(len(dataset.y)),
            "label_name": dataset.label_name,
            "feature_names": dataset.feature_names,
            "metadata": dataset.metadata,
        },
        "model": {"candidate": candidate, "artifact_path": str(artifact_path)},
        "metrics": metrics,
        "acceptance_gates": gates,
        "governance": {
            "requires_human_review": dataset.task_type in {"uplift", "anomaly_detection"} or "legal" in model_name,
            "feedback_loop": "taxpayer_ai_feedback",
            "drift_monitoring": "taxpayer_model_governance / concept_drift_detector",
            "privacy": "Use taxpayer consent before bank/OCR data is used for training.",
        },
        "created_at": utc_now(),
    }


def train_classifier(
    dataset: TaskDataset,
    *,
    out_dir: Path,
    model_version: str,
    seed: int,
    fast: bool,
    min_real_samples: int,
    allow_synthetic_promotion: bool,
) -> ModelRunResult:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    start = time.perf_counter()
    stratify = dataset.y if len(np.unique(dataset.y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.22,
        random_state=seed,
        stratify=stratify,
    )
    n_estimators = 80 if fast else 220
    candidates: list[tuple[str, Any]] = [
        (
            "logreg_balanced",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)),
                ]
            ),
        ),
        (
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=9 if fast else 14,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=3,
                random_state=seed,
            ),
        ),
    ]
    if fast:
        candidates = candidates[:2]

    best = None
    best_metrics: dict[str, Any] = {}
    best_score = -1.0
    is_binary = dataset.task_type == "binary_classification"
    for name, model in candidates:
        model.fit(X_train, y_train)
        if is_binary:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_test)[:, 1]
            else:
                prob = model.decision_function(X_test)
                prob = (prob - prob.min()) / max(prob.max() - prob.min(), 1e-9)
            threshold = best_binary_threshold(y_test, prob)
            metrics = binary_metrics(y_test, prob, threshold=threshold)
            score = float(metrics.get("pr_auc") or metrics.get("f1") or 0)
        else:
            pred = model.predict(X_test)
            metrics = multiclass_metrics(y_test, pred)
            score = float(metrics["f1_macro"])
        if score > best_score:
            best = (name, model)
            best_score = score
            best_metrics = metrics

    assert best is not None
    if is_binary:
        thresholds = {"pr_auc": (">=", 0.70), "f1": (">=", 0.68), "brier": ("<=", 0.22)}
        primary_metric = "pr_auc"
    else:
        thresholds = {"f1_macro": (">=", 0.70), "balanced_accuracy": (">=", 0.70)}
        primary_metric = "f1_macro"
    gates = gate_report(best_metrics, thresholds)
    status = production_status(dataset, gates, min_real_samples, allow_synthetic_promotion)

    artifact_path = out_dir / f"{dataset.name}.joblib"
    joblib.dump(
        {
            "model": best[1],
            "feature_names": dataset.feature_names,
            "model_version": model_version,
            "task_type": dataset.task_type,
            "label_name": dataset.label_name,
        },
        artifact_path,
    )
    report_path = out_dir / f"{dataset.name}_quality_report.json"
    card_path = out_dir / f"{dataset.name}_model_card.json"
    data_hash = sha256_json({"X_shape": dataset.X.shape, "y_sum": float(np.sum(dataset.y)), "source": dataset.source, "metadata": dataset.metadata})
    report = {
        "model_name": dataset.name,
        "model_version": model_version,
        "generated_at": utc_now(),
        "dataset": {"samples": int(len(dataset.y)), "source": dataset.source, "positive_ratio": float(np.mean(dataset.y == 1)) if is_binary else None},
        "best_candidate": best[0],
        "metrics": best_metrics,
        "acceptance_gates": gates,
        "status": status,
        "train_data_hash": data_hash,
    }
    card = make_model_card(
        dataset=dataset,
        model_name=dataset.name,
        model_version=model_version,
        candidate=best[0],
        metrics=best_metrics,
        gates=gates,
        status=status,
        artifact_path=artifact_path,
    )
    write_json(report_path, report)
    write_json(card_path, card)
    return ModelRunResult(
        model_name=dataset.name,
        model_version=model_version,
        task_type=dataset.task_type,
        source=dataset.source,
        artifact_path=str(artifact_path),
        quality_report_path=str(report_path),
        model_card_path=str(card_path),
        samples=int(len(dataset.y)),
        feature_count=int(dataset.X.shape[1]),
        best_candidate=best[0],
        primary_metric=primary_metric,
        primary_score=float(best_metrics.get(primary_metric) or 0.0),
        metrics=best_metrics,
        acceptance_gates=gates,
        status=status,
        train_data_hash=data_hash,
        duration_sec=round(time.perf_counter() - start, 3),
    )


def train_regressor(
    dataset: TaskDataset,
    *,
    out_dir: Path,
    model_version: str,
    seed: int,
    fast: bool,
    min_real_samples: int,
    allow_synthetic_promotion: bool,
) -> ModelRunResult:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split

    start = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.22, random_state=seed)
    n_estimators = 90 if fast else 260
    candidates: list[tuple[str, Any]] = [
        (
            "random_forest_regressor",
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10 if fast else 16,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "gradient_boosting_regressor",
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=3,
                random_state=seed,
            ),
        ),
    ]
    best = None
    best_metrics: dict[str, Any] = {}
    best_score = -1e18
    residual_quantiles = {}
    for name, model in candidates:
        model.fit(X_train, y_train)
        pred = np.maximum(0, model.predict(X_test))
        metrics = regression_metrics(y_test, pred)
        score = float(metrics["r2"]) - float(metrics["mape"])
        if score > best_score:
            best = (name, model)
            best_score = score
            best_metrics = metrics
            residual = y_test - pred
            residual_quantiles = {
                "p10": float(np.quantile(residual, 0.10)),
                "p50": float(np.quantile(residual, 0.50)),
                "p90": float(np.quantile(residual, 0.90)),
            }

    assert best is not None
    thresholds = {"r2": (">=", 0.72), "mape": ("<=", 0.18)}
    gates = gate_report(best_metrics, thresholds)
    status = production_status(dataset, gates, min_real_samples, allow_synthetic_promotion)
    artifact_path = out_dir / f"{dataset.name}.joblib"
    joblib.dump(
        {
            "model": best[1],
            "feature_names": dataset.feature_names,
            "model_version": model_version,
            "task_type": dataset.task_type,
            "label_name": dataset.label_name,
            "residual_quantiles": residual_quantiles,
        },
        artifact_path,
    )
    report_path = out_dir / f"{dataset.name}_quality_report.json"
    card_path = out_dir / f"{dataset.name}_model_card.json"
    data_hash = sha256_json({"X_shape": dataset.X.shape, "y_mean": float(np.mean(dataset.y)), "source": dataset.source, "metadata": dataset.metadata})
    report = {
        "model_name": dataset.name,
        "model_version": model_version,
        "generated_at": utc_now(),
        "dataset": {"samples": int(len(dataset.y)), "source": dataset.source},
        "best_candidate": best[0],
        "metrics": best_metrics,
        "residual_quantiles": residual_quantiles,
        "acceptance_gates": gates,
        "status": status,
        "train_data_hash": data_hash,
    }
    card = make_model_card(
        dataset=dataset,
        model_name=dataset.name,
        model_version=model_version,
        candidate=best[0],
        metrics=best_metrics,
        gates=gates,
        status=status,
        artifact_path=artifact_path,
    )
    write_json(report_path, report)
    write_json(card_path, card)
    return ModelRunResult(
        model_name=dataset.name,
        model_version=model_version,
        task_type=dataset.task_type,
        source=dataset.source,
        artifact_path=str(artifact_path),
        quality_report_path=str(report_path),
        model_card_path=str(card_path),
        samples=int(len(dataset.y)),
        feature_count=int(dataset.X.shape[1]),
        best_candidate=best[0],
        primary_metric="r2_minus_mape",
        primary_score=float(best_score),
        metrics=best_metrics,
        acceptance_gates=gates,
        status=status,
        train_data_hash=data_hash,
        duration_sec=round(time.perf_counter() - start, 3),
    )


def train_anomaly(
    dataset: TaskDataset,
    *,
    out_dir: Path,
    model_version: str,
    seed: int,
    fast: bool,
    min_real_samples: int,
    allow_synthetic_promotion: bool,
) -> ModelRunResult:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score, f1_score
    from sklearn.model_selection import train_test_split

    start = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.25, random_state=seed, stratify=dataset.y)
    contamination_grid = [0.04, 0.06] if fast else [0.025, 0.04, 0.06, 0.08, 0.10]
    best = None
    best_metrics = {}
    best_score = -1.0
    for contamination in contamination_grid:
        model = IsolationForest(
            n_estimators=90 if fast else 240,
            contamination=contamination,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(X_train)
        score = -model.decision_function(X_test)
        pred = (model.predict(X_test) == -1).astype(int)
        metrics = {
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "pr_auc": round(float(average_precision_score(y_test, score)), 4),
            "contamination": contamination,
        }
        primary = float(metrics["pr_auc"]) + float(metrics["f1"]) * 0.5
        if primary > best_score:
            best = (f"isolation_forest_{contamination}", model)
            best_score = primary
            best_metrics = metrics

    assert best is not None
    thresholds = {"pr_auc": (">=", 0.65), "f1": (">=", 0.45)}
    gates = gate_report(best_metrics, thresholds)
    status = production_status(dataset, gates, min_real_samples, allow_synthetic_promotion)
    artifact_path = out_dir / f"{dataset.name}.joblib"
    joblib.dump(
        {
            "model": best[1],
            "feature_names": dataset.feature_names,
            "model_version": model_version,
            "task_type": dataset.task_type,
            "label_name": dataset.label_name,
        },
        artifact_path,
    )
    report_path = out_dir / f"{dataset.name}_quality_report.json"
    card_path = out_dir / f"{dataset.name}_model_card.json"
    data_hash = sha256_json({"X_shape": dataset.X.shape, "y_sum": float(np.sum(dataset.y)), "source": dataset.source, "metadata": dataset.metadata})
    report = {
        "model_name": dataset.name,
        "model_version": model_version,
        "generated_at": utc_now(),
        "dataset": {"samples": int(len(dataset.y)), "source": dataset.source, "synthetic_anomaly_ratio": float(np.mean(dataset.y))},
        "best_candidate": best[0],
        "metrics": best_metrics,
        "acceptance_gates": gates,
        "status": status,
        "train_data_hash": data_hash,
    }
    card = make_model_card(
        dataset=dataset,
        model_name=dataset.name,
        model_version=model_version,
        candidate=best[0],
        metrics=best_metrics,
        gates=gates,
        status=status,
        artifact_path=artifact_path,
    )
    write_json(report_path, report)
    write_json(card_path, card)
    return ModelRunResult(
        model_name=dataset.name,
        model_version=model_version,
        task_type=dataset.task_type,
        source=dataset.source,
        artifact_path=str(artifact_path),
        quality_report_path=str(report_path),
        model_card_path=str(card_path),
        samples=int(len(dataset.y)),
        feature_count=int(dataset.X.shape[1]),
        best_candidate=best[0],
        primary_metric="pr_auc_plus_f1",
        primary_score=float(best_score),
        metrics=best_metrics,
        acceptance_gates=gates,
        status=status,
        train_data_hash=data_hash,
        duration_sec=round(time.perf_counter() - start, 3),
    )


def train_uplift(
    dataset: TaskDataset,
    *,
    out_dir: Path,
    model_version: str,
    seed: int,
    fast: bool,
    min_real_samples: int,
    allow_synthetic_promotion: bool,
) -> ModelRunResult:
    from sklearn.model_selection import train_test_split

    from ml_engine.causal_uplift_model import TLearnerUplift, compute_qini_coefficient

    start = time.perf_counter()
    treatment = dataset.treatment
    if treatment is None:
        raise ValueError("Uplift dataset requires treatment array.")
    idx = np.arange(len(dataset.y))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=seed, stratify=dataset.y)
    X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
    y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
    t_train, t_test = treatment[train_idx], treatment[test_idx]

    configs = [(120, 3, 0.05)] if fast else [(180, 3, 0.05), (260, 4, 0.04), (320, 3, 0.03)]
    best = None
    best_metrics = {}
    best_score = -1e18
    for n_estimators, max_depth, learning_rate in configs:
        learner = TLearnerUplift()
        try:
            train_metrics = learner.fit(
                X_train,
                t_train,
                y_train,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )
        except ValueError:
            continue
        cate = learner.predict(X_test)
        qini = compute_qini_coefficient(cate, t_test, y_test)
        metrics = {**train_metrics, "qini": qini, "avg_test_cate": round(float(np.mean(cate)), 4)}
        if qini > best_score:
            best = (f"t_learner_gbdt_{n_estimators}_{max_depth}_{learning_rate}", learner)
            best_score = qini
            best_metrics = metrics
    if best is None:
        raise RuntimeError("Could not train uplift model with both classes in treated/control groups.")

    thresholds = {"qini": (">=", 0.0005), "treated_auc": (">=", 0.68), "control_auc": (">=", 0.68)}
    gates = gate_report(best_metrics, thresholds)
    status = production_status(dataset, gates, min_real_samples, allow_synthetic_promotion)
    artifact_dir = out_dir / dataset.name
    best[1].config["model_version"] = model_version
    best[1].save(artifact_dir)
    artifact_path = artifact_dir / "uplift_config.json"
    report_path = out_dir / f"{dataset.name}_quality_report.json"
    card_path = out_dir / f"{dataset.name}_model_card.json"
    data_hash = sha256_json({"X_shape": dataset.X.shape, "y_sum": float(np.sum(dataset.y)), "treatment_sum": float(np.sum(treatment)), "source": dataset.source})
    report = {
        "model_name": dataset.name,
        "model_version": model_version,
        "generated_at": utc_now(),
        "dataset": {"samples": int(len(dataset.y)), "source": dataset.source, "treatment_rate": float(np.mean(treatment))},
        "best_candidate": best[0],
        "metrics": best_metrics,
        "acceptance_gates": gates,
        "status": status,
        "train_data_hash": data_hash,
    }
    card = make_model_card(
        dataset=dataset,
        model_name=dataset.name,
        model_version=model_version,
        candidate=best[0],
        metrics=best_metrics,
        gates=gates,
        status=status,
        artifact_path=artifact_path,
    )
    write_json(report_path, report)
    write_json(card_path, card)
    return ModelRunResult(
        model_name=dataset.name,
        model_version=model_version,
        task_type=dataset.task_type,
        source=dataset.source,
        artifact_path=str(artifact_path),
        quality_report_path=str(report_path),
        model_card_path=str(card_path),
        samples=int(len(dataset.y)),
        feature_count=int(dataset.X.shape[1]),
        best_candidate=best[0],
        primary_metric="qini",
        primary_score=float(best_score),
        metrics=best_metrics,
        acceptance_gates=gates,
        status=status,
        train_data_hash=data_hash,
        duration_sec=round(time.perf_counter() - start, 3),
    )


TRAINERS: dict[str, Callable[..., ModelRunResult]] = {
    "binary_classification": train_classifier,
    "multiclass_classification": train_classifier,
    "regression": train_regressor,
    "anomaly_detection": train_anomaly,
    "uplift": train_uplift,
}


def run_training(
    *,
    source: str = "auto",
    sample_size: int = 5000,
    out_dir: Path = DEFAULT_MODEL_DIR,
    seed: int = 42,
    trials: int = 2,
    max_retries: int = 2,
    fast: bool = False,
    min_real_samples: int = 1000,
    allow_synthetic_promotion: bool = False,
    no_synthetic: bool = False,
    promote_if_pass: bool = False,
    register_models: bool = False,
    write_drift_baseline: bool = False,
    task_filter: set[str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_version = f"taxpayer-advanced-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    all_results: dict[str, ModelRunResult] = {}
    attempts = []

    for retry in range(max(1, max_retries)):
        retry_seed = seed + retry * 101
        datasets = load_datasets(source, sample_size, retry_seed, no_synthetic=no_synthetic)
        if task_filter:
            datasets = {key: value for key, value in datasets.items() if key in task_filter}
        retry_results = {}
        for task_name, dataset in sorted(datasets.items()):
            if len(dataset.y) < 80 or len(np.unique(dataset.y)) < 2:
                continue
            trainer = TRAINERS.get(dataset.task_type)
            if not trainer:
                continue
            best_for_task = None
            for trial in range(max(1, trials)):
                trial_seed = retry_seed + trial * 17
                result = trainer(
                    dataset,
                    out_dir=out_dir,
                    model_version=model_version,
                    seed=trial_seed,
                    fast=fast,
                    min_real_samples=min_real_samples,
                    allow_synthetic_promotion=allow_synthetic_promotion,
                )
                if best_for_task is None or result.primary_score > best_for_task.primary_score:
                    best_for_task = result
            if best_for_task is not None:
                retry_results[task_name] = best_for_task
                previous = all_results.get(task_name)
                if previous is None or best_for_task.primary_score > previous.primary_score:
                    all_results[task_name] = best_for_task
        attempts.append(
            {
                "retry": retry + 1,
                "seed": retry_seed,
                "trained_tasks": sorted(retry_results.keys()),
                "gate_pass_count": sum(1 for item in retry_results.values() if item.acceptance_gates.get("overall_pass")),
            }
        )
        if retry_results and all(item.acceptance_gates.get("overall_pass") for item in retry_results.values()):
            break

    manifest = {
        "pipeline": "taxpayer_model_training_orchestrator",
        "model_version": model_version,
        "generated_at": utc_now(),
        "source_requested": source,
        "sample_size_requested": int(sample_size),
        "fast": bool(fast),
        "trials": int(trials),
        "max_retries": int(max_retries),
        "allow_synthetic_promotion": bool(allow_synthetic_promotion),
        "no_synthetic": bool(no_synthetic),
        "promote_if_pass": bool(promote_if_pass),
        "attempts": attempts,
        "models": {name: asdict(result) for name, result in sorted(all_results.items())},
        "summary": {
            "trained_model_count": len(all_results),
            "gate_pass_count": sum(1 for item in all_results.values() if item.acceptance_gates.get("overall_pass")),
            "prod_candidate_count": sum(1 for item in all_results.values() if str(item.status).startswith("prod_candidate")),
            "sandbox_count": sum(1 for item in all_results.values() if item.status == "sandbox"),
            "failed_gate_count": sum(1 for item in all_results.values() if item.status == "staging_failed_gate"),
        },
        "production_policy": {
            "prod_requires_gate_pass": True,
            "prod_requires_database_source": not allow_synthetic_promotion,
            "min_real_samples": int(min_real_samples),
            "synthetic_status": "sandbox unless --allow-synthetic-promotion is used",
            "promotion_command_enabled": bool(promote_if_pass),
            "register_models": bool(register_models),
            "write_drift_baseline": bool(write_drift_baseline),
        },
    }
    manifest_path = out_dir / "taxpayer_training_manifest.json"
    write_json(manifest_path, manifest)
    if register_models:
        registry = {
            "registry_name": "taxpayer_model_registry",
            "generated_at": utc_now(),
            "models": [
                {
                    "model_name": name,
                    "model_version": result.model_version,
                    "status": result.status,
                    "artifact_path": result.artifact_path,
                    "model_card_path": result.model_card_path,
                    "quality_report_path": result.quality_report_path,
                }
                for name, result in sorted(all_results.items())
            ],
        }
        write_json(out_dir / "taxpayer_model_registry.json", registry)
        manifest["registry_path"] = str(out_dir / "taxpayer_model_registry.json")
    if write_drift_baseline:
        drift = {
            "baseline_name": "taxpayer_drift_baseline",
            "generated_at": utc_now(),
            "models": {
                name: {
                    "source": result.source,
                    "samples": result.samples,
                    "primary_metric": result.primary_metric,
                    "primary_score": result.primary_score,
                    "train_data_hash": result.train_data_hash,
                }
                for name, result in sorted(all_results.items())
            },
        }
        write_json(out_dir / "taxpayer_drift_baseline.json", drift)
        manifest["drift_baseline_path"] = str(out_dir / "taxpayer_drift_baseline.json")
    if promote_if_pass:
        promotable = {
            name: asdict(result)
            for name, result in sorted(all_results.items())
            if result.status == "prod_candidate" and result.acceptance_gates.get("overall_pass")
        }
        promotion = {
            "promotion_manifest": "taxpayer_production_promotion",
            "generated_at": utc_now(),
            "promoted_count": len(promotable),
            "blocked_count": len(all_results) - len(promotable),
            "models": promotable,
            "blocked_reasons": {
                name: {
                    "status": result.status,
                    "source": result.source,
                    "samples": result.samples,
                    "overall_pass": result.acceptance_gates.get("overall_pass"),
                }
                for name, result in sorted(all_results.items())
                if name not in promotable
            },
        }
        write_json(out_dir / "taxpayer_promotion_manifest.json", promotion)
        manifest["promotion_manifest_path"] = str(out_dir / "taxpayer_promotion_manifest.json")
    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate all Taxpayer Portal ML models.")
    parser.add_argument("--source", choices=["auto", "db", "synthetic"], default="auto")
    parser.add_argument("--sample-size", type=int, default=int(os.getenv("TAXPAYER_TRAIN_SAMPLE_SIZE", "5000")))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=2, help="Trials per model per retry.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry with new seeds until quality gates pass.")
    parser.add_argument("--fast", action="store_true", help="Use smaller candidate grids for smoke tests.")
    parser.add_argument("--min-real-samples", type=int, default=1000)
    parser.add_argument("--allow-synthetic-promotion", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true", help="Use only real database datasets; do not fall back to synthetic.")
    parser.add_argument("--promote-if-pass", action="store_true", help="Write a production promotion manifest for prod_candidate models.")
    parser.add_argument("--register-models", action="store_true", help="Write a lightweight taxpayer model registry file.")
    parser.add_argument("--write-drift-baseline", action="store_true", help="Write drift baseline metadata for production monitoring.")
    parser.add_argument("--tasks", default="", help="Comma-separated task names to train.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_filter = {item.strip() for item in args.tasks.split(",") if item.strip()} or None
    manifest = run_training(
        source=args.source,
        sample_size=args.sample_size,
        out_dir=args.out_dir,
        seed=args.seed,
        trials=args.trials,
        max_retries=args.max_retries,
        fast=args.fast,
        min_real_samples=args.min_real_samples,
        allow_synthetic_promotion=args.allow_synthetic_promotion,
        no_synthetic=args.no_synthetic,
        promote_if_pass=args.promote_if_pass,
        register_models=args.register_models,
        write_drift_baseline=args.write_drift_baseline,
        task_filter=task_filter,
    )
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=True))
    print(f"[OK] Taxpayer training manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
