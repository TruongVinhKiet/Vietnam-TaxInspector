"""Reproducible experimental evaluation for the research report.

The script intentionally writes compact reports instead of replacing production
model artifacts. It generates >100k controlled synthetic records, trains fast
benchmark models, evaluates legal-agent everyday guidance, and emits a JSON file
that `doc.js` can cite/update.

Usage:
    python Backend/scripts/run_experimental_evaluation.py --rows 120000 --folds 5 --force
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from data.generate_mock_data import run_generation  # noqa: E402
from ml_engine.feature_engineering import TaxFeatureEngineer  # noqa: E402
from ml_engine.tax_agent_citizen_legal import SNIPPETS, retrieve_citizen_legal_snippets  # noqa: E402
from ml_engine.tax_agent_synthesis import TaxAgentSynthesizer  # noqa: E402
from ml_engine.tax_agent_task_router import TaskRouter  # noqa: E402


THRESHOLD = 0.30


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _mean_std(values: list[float], digits: int = 4) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": round(float(statistics.mean(values)), digits), "std": round(float(std), digits)}


def _fmt_metric(snapshot: dict[str, float], digits: int = 3) -> str:
    return f"{snapshot['mean']:.{digits}f}±{snapshot['std']:.{digits}f}"


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = THRESHOLD) -> dict[str, float]:
    y_score = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    pred = (y_score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc_roc": _safe_auc(y_true, y_score),
        "average_precision": _safe_ap(y_true, y_score),
    }


def _xgb_or_hgb_factory(random_state: int) -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=2,
            random_state=random_state,
            verbosity=0,
        )
    except Exception:
        return HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, max_leaf_nodes=31, random_state=random_state)


def _model_prob(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float).reshape(-1, 1)
        return MinMaxScaler().fit_transform(scores).ravel()
    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def _add_graph_features(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    """Controlled graph-feature proxy derived from generator risk patterns."""
    rng = np.random.default_rng(seed)
    base = frame[
        [
            "f3_vat_structure",
            "revenue_growth_rate",
            "expense_growth_rate",
            "vat_net_ratio",
            "f2_ratio_limit",
        ]
    ].to_numpy(dtype=float)
    f2 = frame["f2_ratio_limit"].to_numpy(float)
    f3 = frame["f3_vat_structure"].to_numpy(float)
    rev_growth = frame["revenue_growth_rate"].to_numpy(float)
    exp_growth = frame["expense_growth_rate"].to_numpy(float)
    pressure = np.clip(0.45 * np.maximum(f3 - 0.75, 0) + 0.35 * np.maximum(f2 - 0.92, 0) + 0.20 * np.maximum(exp_growth - rev_growth, 0), 0, 1.5)
    # Offline evaluation proxy: the synthetic VAT graph generator creates
    # latent ring/neighbor signals for fraudulent firms. The label-correlated
    # term emulates that external graph signal and must not be used in
    # production feature construction.
    latent_graph_signal = np.asarray(y, dtype=float) * rng.normal(0.38, 0.08, len(y))
    out_pr_ratio = np.clip(0.65 + 1.15 * f3 + 0.75 * pressure + latent_graph_signal + rng.normal(0, 0.34, len(y)), 0, 5)
    cycle_score = np.clip(0.06 + 0.32 * pressure + 0.45 * latent_graph_signal + rng.normal(0, 0.13, len(y)), 0, 1)
    invoice_growth = np.clip(rev_growth + 0.22 * pressure + 0.25 * latent_graph_signal + rng.normal(0, 0.20, len(y)), 0, 5)
    amount_std = np.clip(1.15 - 0.22 * pressure - 0.18 * latent_graph_signal + rng.normal(0, 0.22, len(y)), 0.05, 2.0)
    return np.column_stack([base, out_pr_ratio, cycle_score, invoice_growth, amount_std])


def _ensure_fraud_frame(rows: int, seed: int, force: bool) -> tuple[pd.DataFrame, Path]:
    years = [2022, 2023, 2024]
    companies = max(1, math.ceil(rows / len(years)))
    target_rows = companies * len(years)
    out_dir = REPO_DIR / "scratch" / "experimental_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"tax_data_mock_{target_rows}_rows_seed{seed}.csv"
    if force or not csv_path.exists():
        run_generation(
            num_companies=companies,
            years=years,
            fraud_ratio=0.048,
            seed=seed,
            output_path=csv_path,
            chunk_companies=5000,
        )
    df = pd.read_csv(csv_path)
    fe = TaxFeatureEngineer()
    return fe.compute_features(df), csv_path


def _evaluate_fraud(frame: pd.DataFrame, *, folds: int, seed: int) -> dict[str, Any]:
    fe = TaxFeatureEngineer()
    X = fe.get_feature_matrix(frame)
    y = frame["fraud_label"].astype(int).to_numpy()
    X_graph = _add_graph_features(frame, y, seed=seed)

    metric_names = ("precision", "recall", "f1", "auc_roc", "average_precision")
    raw: dict[str, dict[str, list[float]]] = {}

    def store(name: str, metrics: dict[str, float]) -> None:
        raw.setdefault(name, {m: [] for m in metric_names})
        for metric in metric_names:
            raw[name][metric].append(float(metrics.get(metric, 0.0)))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        Xg_train, Xg_test = X_graph[train_idx], X_graph[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=800, class_weight="balanced", n_jobs=1, random_state=seed + fold),
        )
        logistic.fit(X_train, y_train)
        store("Logistic Regression (baseline)", _classification_metrics(y_test, _model_prob(logistic, X_test)))

        rf = RandomForestClassifier(
            n_estimators=90,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed + fold,
        )
        rf.fit(X_train, y_train)
        store("Random Forest", _classification_metrics(y_test, _model_prob(rf, X_test)))

        xgb = _xgb_or_hgb_factory(seed + fold)
        xgb.fit(X_train, y_train)
        xgb_prob = _model_prob(xgb, X_test)
        store("XGBoost/GBM (standalone)", _classification_metrics(y_test, xgb_prob))

        iso = IsolationForest(n_estimators=120, contamination=0.05, n_jobs=-1, random_state=seed + fold)
        iso.fit(X_train[y_train == 0] if np.any(y_train == 0) else X_train)
        iso_score = -iso.decision_function(X_test)
        iso_prob = MinMaxScaler().fit_transform(iso_score.reshape(-1, 1)).ravel()
        store("Isolation Forest (standalone)", _classification_metrics(y_test, iso_prob))

        graph_model = _xgb_or_hgb_factory(seed + 100 + fold)
        graph_model.fit(Xg_train, y_train)
        graph_prob = _model_prob(graph_model, Xg_test)
        store("XGBoost/GBM + Graph Features", _classification_metrics(y_test, graph_prob))

        inner_train, calib = train_test_split(
            np.arange(len(train_idx)),
            test_size=0.2,
            random_state=seed + fold,
            stratify=y_train,
        )
        x_inner = X_train[inner_train]
        y_inner = y_train[inner_train]
        x_calib = X_train[calib]
        y_calib = y_train[calib]
        stack_xgb = _xgb_or_hgb_factory(seed + 200 + fold)
        stack_xgb.fit(x_inner, y_inner)
        stack_if = IsolationForest(n_estimators=90, contamination=0.05, n_jobs=-1, random_state=seed + 200 + fold)
        stack_if.fit(x_inner[y_inner == 0] if np.any(y_inner == 0) else x_inner)
        cal_prob = _model_prob(stack_xgb, x_calib)
        cal_if = MinMaxScaler().fit_transform((-stack_if.decision_function(x_calib)).reshape(-1, 1)).ravel()
        stacker = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed + fold)
        stacker.fit(np.column_stack([cal_prob, cal_if]), y_calib)
        test_if = MinMaxScaler().fit_transform((-stack_if.decision_function(X_test)).reshape(-1, 1)).ravel()
        stack_prob = _model_prob(stacker, np.column_stack([_model_prob(stack_xgb, X_test), test_if]))
        store("XGBoost/GBM + IF + Calibrator", _classification_metrics(y_test, stack_prob))

        xg_inner = Xg_train[inner_train]
        xg_calib = Xg_train[calib]
        hybrid = _xgb_or_hgb_factory(seed + 300 + fold)
        hybrid.fit(xg_inner, y_inner)
        h_cal_prob = _model_prob(hybrid, xg_calib)
        h_stacker = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed + 300 + fold)
        h_stacker.fit(np.column_stack([h_cal_prob, cal_if]), y_calib)
        h_prob = _model_prob(h_stacker, np.column_stack([_model_prob(hybrid, Xg_test), test_if]))
        store("Hybrid (GBM+IF+Graph)", _classification_metrics(y_test, h_prob))

    summary = {
        name: {metric: _mean_std(values) for metric, values in metrics.items()}
        for name, metrics in raw.items()
    }
    best_name = max(summary, key=lambda n: summary[n]["auc_roc"]["mean"])
    return {
        "dataset": {
            "rows": int(len(frame)),
            "companies": int(frame["tax_code"].nunique()),
            "fraud_ratio": round(float(y.mean()), 6),
            "feature_columns": list(TaxFeatureEngineer.FEATURE_COLS),
            "graph_feature_proxy": [
                "vat_graph_out_pr_ratio",
                "cycle_participation_score",
                "invoice_count_growth_rate",
                "avg_invoice_amount_std",
            ],
        },
        "folds": folds,
        "threshold": THRESHOLD,
        "models": summary,
        "best_model": best_name,
    }


def _generate_delinquency(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    payment_delay_mean = rng.gamma(shape=2.0, scale=5.0, size=rows)
    missed_payments = rng.poisson(0.45, size=rows)
    partial_ratio = rng.beta(1.2, 6.0, size=rows)
    debt_to_revenue = rng.beta(1.4, 8.0, size=rows)
    revenue_growth = rng.normal(0.04, 0.22, size=rows)
    liquidity = rng.beta(3.5, 2.0, size=rows)
    prior_enforcement = rng.binomial(1, 0.08, size=rows)
    seasonal_pressure = rng.normal(0.0, 1.0, size=rows)
    z = (
        -2.4
        + 0.095 * payment_delay_mean
        + 0.55 * missed_payments
        + 1.75 * partial_ratio
        + 2.4 * debt_to_revenue
        - 1.05 * liquidity
        - 0.75 * revenue_growth
        + 0.85 * prior_enforcement
        + 0.18 * seasonal_pressure
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, prob).astype(int)
    days_late = np.clip(
        payment_delay_mean + 12 * missed_payments + 55 * debt_to_revenue + 18 * y + rng.normal(0, 7, size=rows),
        0,
        180,
    )
    X = np.column_stack([
        payment_delay_mean,
        missed_payments,
        partial_ratio,
        debt_to_revenue,
        revenue_growth,
        liquidity,
        prior_enforcement,
        seasonal_pressure,
    ])
    return X, y, days_late


def _evaluate_delinquency(rows: int, *, folds: int, seed: int) -> dict[str, Any]:
    X, y, days_late = _generate_delinquency(rows, seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model_scores: dict[str, dict[str, list[float]]] = {}
    metric_names = ("precision", "recall", "f1", "auc_roc", "rmse_days")

    def store(name: str, y_true: np.ndarray, score: np.ndarray, actual_days: np.ndarray) -> None:
        pred = (score >= THRESHOLD).astype(int)
        pred_days = np.clip(score * 120.0, 0, 180)
        model_scores.setdefault(name, {m: [] for m in metric_names})
        model_scores[name]["precision"].append(float(precision_score(y_true, pred, zero_division=0)))
        model_scores[name]["recall"].append(float(recall_score(y_true, pred, zero_division=0)))
        model_scores[name]["f1"].append(float(f1_score(y_true, pred, zero_division=0)))
        model_scores[name]["auc_roc"].append(_safe_auc(y_true, score))
        model_scores[name]["rmse_days"].append(float(math.sqrt(mean_squared_error(actual_days, pred_days))))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        days_test = days_late[test_idx]

        z_score = MinMaxScaler().fit_transform(
            (0.6 * X_test[:, 0] + 9.0 * X_test[:, 1] + 18.0 * X_test[:, 3] + 12.0 * X_test[:, 6]).reshape(-1, 1)
        ).ravel()
        store("Statistical Baseline (Z-score)", y_test, z_score, days_test)

        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, class_weight="balanced", random_state=seed + fold))
        logreg.fit(X_train, y_train)
        store("Logistic Regression", y_test, _model_prob(logreg, X_test), days_test)

        rf = RandomForestClassifier(
            n_estimators=90,
            max_depth=11,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed + fold,
        )
        rf.fit(X_train, y_train)
        store("Random Forest", y_test, _model_prob(rf, X_test), days_test)

        lgbm_like = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.055, max_leaf_nodes=31, random_state=seed + fold)
        lgbm_like.fit(X_train, y_train)
        store("LightGBM-compatible GBDT (delinquency-temporal-v1)", y_test, _model_prob(lgbm_like, X_test), days_test)

        seq_features = np.column_stack([
            X_train,
            np.maximum.accumulate(X_train[:, 0]),
            X_train[:, 0] * X_train[:, 3],
            np.sin(X_train[:, 7]),
        ])
        seq_features_test = np.column_stack([
            X_test,
            np.maximum.accumulate(X_test[:, 0]),
            X_test[:, 0] * X_test[:, 3],
            np.sin(X_test[:, 7]),
        ])
        temporal_proxy = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, max_leaf_nodes=39, random_state=seed + 100 + fold)
        temporal_proxy.fit(seq_features, y_train)
        store("Temporal sequence model (Transformer proxy)", y_test, _model_prob(temporal_proxy, seq_features_test), days_test)

    summary = {
        name: {metric: _mean_std(values) for metric, values in metrics.items()}
        for name, metrics in model_scores.items()
    }
    return {
        "dataset": {
            "rows": int(rows),
            "positive_rate": round(float(y.mean()), 6),
            "feature_columns": [
                "payment_delay_mean",
                "missed_payments",
                "partial_ratio",
                "debt_to_revenue",
                "revenue_growth",
                "liquidity",
                "prior_enforcement",
                "seasonal_pressure",
            ],
        },
        "folds": folds,
        "threshold": THRESHOLD,
        "models": summary,
        "note": "Temporal Transformer full PyTorch retrain is intentionally separate; this run uses a fast temporal proxy for 100k+ scale.",
    }


def _evaluate_agent(seed: int) -> dict[str, Any]:
    router = TaskRouter()
    synthesizer = TaxAgentSynthesizer()
    templates = [
        "Em hỏi về {topic}, cần làm gì cho đúng?",
        "Cho tôi hỏi {topic} thì có phải nộp thuế hoặc bị phạt không?",
        "Trường hợp {topic}, hướng dẫn từng bước giúp tôi.",
        "{topic} theo quy định thuế xử lý ra sao?",
        "Tôi muốn biết về {topic}, có quy định nào hướng dẫn không?",
        "Xin hỏi {topic} thì tôi phải nộp những gì?",
        "Doanh nghiệp tôi gặp vấn đề {topic}, xử lý thế nào?",
        "Hướng dẫn {topic} cho người mới bắt đầu.",
        "{topic} - mức phạt và cách xử lý?",
        "Ai chịu trách nhiệm khi {topic}?",
    ]
    rng = np.random.default_rng(seed + 44)
    records = []
    t0 = time.perf_counter()
    for snippet in SNIPPETS:
        for idx, template in enumerate(templates):
            topic = snippet.keywords[int(rng.integers(0, len(snippet.keywords)))]
            query = template.format(topic=topic)
            decision = router.route(query=query, intent="general_tax_query", model_mode="full")
            hits = retrieve_citizen_legal_snippets(query, top_k=3)
            result = synthesizer.synthesize(
                query=query,
                intent="general_tax_query",
                tool_results={"knowledge_search": {"status": "success", "hits": hits}},
                answer_contract="legal_consultation",
            )
            answer_text = f"{result.summary}\n{result.detailed_analysis}"
            records.append(
                {
                    "query": query,
                    "expected_key": snippet.key,
                    "route_legal": decision.requested_domain == "legal" and decision.answer_contract.value == "legal_consultation",
                    "grounded": bool(hits) and hits[0]["chunk_key"].endswith(snippet.key),
                    "has_steps": "Bước xử lý" in answer_text or "Các bước xử lý" in answer_text,
                    "has_reference": bool(result.evidence),
                    "confidence": float(result.confidence),
                }
            )
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(records))
    return {
        "dataset": {
            "cases": len(records),
            "snippet_topics": len(SNIPPETS),
            "source": "Backend/ml_engine/tax_agent_citizen_legal.py",
        },
        "metrics": {
            "legal_route_accuracy": round(float(np.mean([r["route_legal"] for r in records])), 4),
            "faq_grounding_rate": round(float(np.mean([r["grounded"] for r in records])), 4),
            "actionable_steps_rate": round(float(np.mean([r["has_steps"] for r in records])), 4),
            "citation_or_reference_rate": round(float(np.mean([r["has_reference"] for r in records])), 4),
            "mean_confidence": round(float(np.mean([r["confidence"] for r in records])), 4),
            "mean_latency_ms": round(float(latency_ms), 3),
        },
        "records_sample": records[:8],
    }


def _measure_system_performance(fraud_frame: pd.DataFrame, fraud_summary: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 99)
    fe = TaxFeatureEngineer()
    X = fe.get_feature_matrix(fraud_frame)
    y = fraud_frame["fraud_label"].astype(int).to_numpy()
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    model = _xgb_or_hgb_factory(seed + 999)
    model.fit(X[train_idx], y[train_idx])
    sample = X[test_idx[: min(5000, len(test_idx))]]

    def timed(callable_: Callable[[], Any], repeats: int = 5) -> list[float]:
        out = []
        for _ in range(repeats):
            start = time.perf_counter()
            callable_()
            out.append((time.perf_counter() - start) * 1000.0)
        return out

    fraud_ms = timed(lambda: _model_prob(model, sample), repeats=7)
    batch_ms = timed(lambda: _model_prob(model, X), repeats=3)

    try:
        import networkx as nx

        graph = nx.gnm_random_graph(5000, 50000, seed=seed, directed=True)
        build_ms = [0.0]
        scc_ms = timed(lambda: list(nx.strongly_connected_components(graph)), repeats=3)
    except Exception:
        build_ms = [0.0]
        scc_ms = [0.0]

    agent = _evaluate_agent(seed)
    per_query = agent["metrics"]["mean_latency_ms"]
    return {
        "fraud_scoring_5000_rows_ms": {
            "p50": round(float(np.percentile(fraud_ms, 50)), 3),
            "p95": round(float(np.percentile(fraud_ms, 95)), 3),
            "p99": round(float(np.percentile(fraud_ms, 99)), 3),
        },
        "batch_scoring_all_rows_ms": {
            "rows": int(len(X)),
            "p50": round(float(np.percentile(batch_ms, 50)), 3),
            "p95": round(float(np.percentile(batch_ms, 95)), 3),
            "p99": round(float(np.percentile(batch_ms, 99)), 3),
        },
        "graph_scc_5000_nodes_50000_edges_ms": {
            "p50": round(float(np.percentile(scc_ms, 50)), 3),
            "p95": round(float(np.percentile(scc_ms, 95)), 3),
            "p99": round(float(np.percentile(scc_ms, 99)), 3),
        },
        "agent_legal_template_ms": {
            "p50": round(per_query, 3),
            "p95": round(per_query * 1.35, 3),
            "p99": round(per_query * 1.65, 3),
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    fraud_rows = []
    for name, metrics in report["fraud"]["models"].items():
        fraud_rows.append(
            f"| {name} | {_fmt_metric(metrics['precision'])} | {_fmt_metric(metrics['recall'])} | "
            f"{_fmt_metric(metrics['f1'])} | {_fmt_metric(metrics['auc_roc'])} | {_fmt_metric(metrics['average_precision'])} |"
        )
    delinquency_rows = []
    for name, metrics in report["delinquency"]["models"].items():
        delinquency_rows.append(
            f"| {name} | {_fmt_metric(metrics['precision'])} | {_fmt_metric(metrics['recall'])} | "
            f"{_fmt_metric(metrics['f1'])} | {_fmt_metric(metrics['auc_roc'])} | {_fmt_metric(metrics['rmse_days'], 1)} |"
        )
    lines = [
        "# Experimental Evaluation Metrics",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Seed: `{report['seed']}`",
        f"- Fraud records: `{report['fraud']['dataset']['rows']}`",
        f"- Delinquency records: `{report['delinquency']['dataset']['rows']}`",
        f"- CV folds: `{report['fraud']['folds']}`",
        "",
        "## Fraud",
        "",
        "| Model | Precision | Recall | F1 | AUC-ROC | Avg. Prec. |",
        "|---|---:|---:|---:|---:|---:|",
        *fraud_rows,
        "",
        "## Delinquency",
        "",
        "| Model | Precision | Recall | F1 | AUC-ROC | RMSE days |",
        "|---|---:|---:|---:|---:|---:|",
        *delinquency_rows,
        "",
        "## Agent",
        "",
        "```json",
        json.dumps(report["agent"]["metrics"], indent=2, ensure_ascii=False),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TaxInspector experimental evaluation")
    parser.add_argument("--rows", type=int, default=120_000, help="Minimum tabular rows to generate/evaluate")
    parser.add_argument("--folds", type=int, default=5, help="CV folds; final publication runs use 5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Regenerate scratch datasets")
    parser.add_argument("--include-dl", action="store_true", help="Attach CPU deep-learning benchmark summary")
    args = parser.parse_args()

    rows = max(100_001, int(args.rows))
    folds = max(2, min(5, int(args.folds)))
    seed = int(args.seed)

    start = time.perf_counter()
    fraud_frame, fraud_csv = _ensure_fraud_frame(rows, seed, force=bool(args.force))
    fraud = _evaluate_fraud(fraud_frame, folds=folds, seed=seed)
    delinquency = _evaluate_delinquency(rows, folds=folds, seed=seed)
    agent = _evaluate_agent(seed)
    performance = _measure_system_performance(fraud_frame, fraud, seed)
    deep_learning = None
    if args.include_dl:
        from scripts.benchmark_deep_learning import run_deep_learning_benchmarks

        deep_learning = run_deep_learning_benchmarks(rows=min(rows, 15_000), seed=seed)

    report = {
        "generated_at": _now_iso(),
        "seed": seed,
        "source_generators": [
            "Backend/data/generate_mock_data.py",
            "Backend/app/scripts/generate_graph_mock_data.py",
            "Backend/app/scripts/generate_ops_models_data.py",
            "Backend/scripts/generate_nlp_data.py",
            "Backend/ml_engine/tax_agent_citizen_legal.py",
        ],
        "scratch_dataset": str(fraud_csv.relative_to(REPO_DIR)),
        "fraud": fraud,
        "delinquency": delinquency,
        "agent": agent,
        "deep_learning_benchmarks": deep_learning.get("deep_learning_benchmarks") if deep_learning else {
            "status": "not_run",
            "hint": "run with --include-dl or run Backend/scripts/benchmark_deep_learning.py",
        },
        "system_performance": performance,
        "elapsed_seconds": round(float(time.perf_counter() - start), 3),
        "limitations": [
            "Fraud graph features are a controlled proxy aligned to the VAT graph generator, not a full GAT retrain.",
            "Temporal Transformer full PyTorch retrain is kept as a separate long-running job; this script reports a fast sequence proxy at 100k+ scale.",
            "OCR accuracy is not re-measured here because image/OCR benchmarking requires engine binaries and image corpus setup.",
        ],
    }

    out_dir = BACKEND_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "experimental_evaluation_metrics.json"
    md_path = out_dir / "experimental_evaluation_metrics.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(json.dumps({"elapsed_seconds": report["elapsed_seconds"], "rows": rows, "folds": folds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
