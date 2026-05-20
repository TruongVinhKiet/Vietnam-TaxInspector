"""Systematic ablation study for the TaxInspector fraud model stack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.feature_engineering import TaxFeatureEngineer  # noqa: E402
from scripts.run_experimental_evaluation import (  # noqa: E402
    THRESHOLD,
    _add_graph_features,
    _ensure_fraud_frame,
    _model_prob,
    _xgb_or_hgb_factory,
)
from scripts.statistical_tests import delong_auc_test, mcnemar_test, metric_ci_bundle, paired_t_test  # noqa: E402


REPORT_DIR = BACKEND_DIR / "reports"


def _metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else 0.0,
    }


# ── Precision-Recall Trade-off Analysis (Policy-relevant) ────────────────────
TRADEOFF_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Estimated cost per false-positive audit (VND) and benefit per detected fraud (VND)
COST_PER_FP_AUDIT_VND = 50_000_000       # 50M VND per wrongful audit
BENEFIT_PER_TP_DETECT_VND = 800_000_000  # 800M VND avg recovered per fraud case


def _threshold_tradeoff(y_true: np.ndarray, score: np.ndarray) -> list[dict[str, float]]:
    """Compute Precision/Recall/F1 at multiple thresholds for cost-benefit analysis."""
    total_positives = int(y_true.sum())
    total_negatives = int(len(y_true) - total_positives)
    rows = []
    for t in TRADEOFF_THRESHOLDS:
        pred = (score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        # Cost-benefit in billions VND (per 10,000 companies screened)
        scale = 10_000 / max(len(y_true), 1)
        cost_fp = fp * scale * COST_PER_FP_AUDIT_VND / 1e9
        benefit_tp = tp * scale * BENEFIT_PER_TP_DETECT_VND / 1e9
        net_benefit = benefit_tp - cost_fp
        rows.append({
            "threshold": round(t, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "flagged_cases": int(pred.sum()),
            "true_positives": tp,
            "false_positives": fp,
            "missed_frauds": fn,
            "cost_fp_billion_vnd": round(cost_fp, 3),
            "benefit_tp_billion_vnd": round(benefit_tp, 3),
            "net_benefit_billion_vnd": round(net_benefit, 3),
        })
    return rows


def _summarize_fold_metrics(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric in ("precision", "recall", "f1", "auc_roc", "average_precision"):
        vals = np.asarray([v[metric] for v in values], dtype=float)
        out[metric] = {
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std(ddof=0)), 6),
            "fold_values": [round(float(v), 6) for v in vals],
        }
    return out


def _vae_anomaly_score(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast VAE proxy using PCA reconstruction error fitted on normal rows."""
    scaler = StandardScaler()
    normal_train = X_train[y_train == 0] if np.any(y_train == 0) else X_train
    normal_scaled = scaler.fit_transform(normal_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    n_components = max(2, min(6, X_train_scaled.shape[1] - 1))
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(normal_scaled)

    def reconstruction_error(values: np.ndarray) -> np.ndarray:
        recon = pca.inverse_transform(pca.transform(values))
        return np.mean((values - recon) ** 2, axis=1)

    train_err = reconstruction_error(X_train_scaled)
    test_err = reconstruction_error(X_test_scaled)
    scaler_score = MinMaxScaler().fit(train_err.reshape(-1, 1))
    return scaler_score.transform(train_err.reshape(-1, 1)).ravel(), scaler_score.transform(test_err.reshape(-1, 1)).ravel()


def _stack_two_scores(
    train_score_a: np.ndarray,
    train_score_b: np.ndarray,
    y_train: np.ndarray,
    test_score_a: np.ndarray,
    test_score_b: np.ndarray,
    seed: int,
) -> np.ndarray:
    stacker = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    stacker.fit(np.column_stack([train_score_a, train_score_b]), y_train)
    return _model_prob(stacker, np.column_stack([test_score_a, test_score_b]))


def run_ablation(*, rows: int = 120000, folds: int = 5, seed: int = 42, force: bool = False, bootstrap: int = 250) -> dict[str, Any]:
    t0 = time.perf_counter()
    frame, csv_path = _ensure_fraud_frame(rows, seed, force=force)
    fe = TaxFeatureEngineer()
    X = fe.get_feature_matrix(frame)
    y = frame["fraud_label"].astype(int).to_numpy()
    X_graph = _add_graph_features(frame, y, seed)

    config_order = [
        "B0_Logistic",
        "B1_XGBoost",
        "C1_XGB_Graph",
        "C2_XGB_IF",
        "C3_XGB_IF_Graph",
        "C4_XGB_VAE",
        "C5_Full_Hybrid",
    ]
    fold_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in config_order}
    oof_scores: dict[str, np.ndarray] = {name: np.zeros(len(y), dtype=float) for name in config_order}

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        Xg_train, Xg_test = X_graph[train_idx], X_graph[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=900, class_weight="balanced", n_jobs=1, random_state=seed + fold))
        logistic.fit(X_train, y_train)
        b0 = _model_prob(logistic, X_test)

        b1_model = _xgb_or_hgb_factory(seed + fold)
        b1_model.fit(X_train, y_train)
        b1 = _model_prob(b1_model, X_test)

        graph_model = _xgb_or_hgb_factory(seed + 100 + fold)
        graph_model.fit(Xg_train, y_train)
        c1 = _model_prob(graph_model, Xg_test)

        iso = IsolationForest(n_estimators=120, contamination=0.05, n_jobs=-1, random_state=seed + 200 + fold)
        iso.fit(X_train[y_train == 0] if np.any(y_train == 0) else X_train)
        train_if = MinMaxScaler().fit_transform((-iso.decision_function(X_train)).reshape(-1, 1)).ravel()
        test_if = MinMaxScaler().fit_transform((-iso.decision_function(X_test)).reshape(-1, 1)).ravel()

        inner, calib = train_test_split(np.arange(len(train_idx)), test_size=0.2, random_state=seed + fold, stratify=y_train)
        base_stack = _xgb_or_hgb_factory(seed + 300 + fold)
        base_stack.fit(X_train[inner], y_train[inner])
        cal_b1 = _model_prob(base_stack, X_train[calib])
        c2 = _stack_two_scores(cal_b1, train_if[calib], y_train[calib], _model_prob(base_stack, X_test), test_if, seed + fold)

        graph_stack = _xgb_or_hgb_factory(seed + 400 + fold)
        graph_stack.fit(Xg_train[inner], y_train[inner])
        cal_graph = _model_prob(graph_stack, Xg_train[calib])
        c3_stack = _stack_two_scores(cal_graph, train_if[calib], y_train[calib], _model_prob(graph_stack, Xg_test), test_if, seed + 20 + fold)
        c3 = np.clip(0.75 * c1 + 0.25 * c3_stack, 0.0, 1.0)

        train_vae, test_vae = _vae_anomaly_score(X_train, X_test, y_train, seed + fold)
        vae_model = _xgb_or_hgb_factory(seed + 500 + fold)
        vae_model.fit(np.column_stack([X_train, train_vae]), y_train)
        c4 = _model_prob(vae_model, np.column_stack([X_test, test_vae]))

        full_model = _xgb_or_hgb_factory(seed + 600 + fold)
        full_train = np.column_stack([Xg_train, train_vae])
        full_test = np.column_stack([Xg_test, test_vae])
        full_model.fit(full_train[inner], y_train[inner])
        full_prob = _model_prob(full_model, full_test)
        c5 = np.clip(0.50 * full_prob + 0.30 * c1 + 0.15 * c4 + 0.05 * test_if, 0.0, 1.0)

        fold_scores = {
            "B0_Logistic": b0,
            "B1_XGBoost": b1,
            "C1_XGB_Graph": c1,
            "C2_XGB_IF": c2,
            "C3_XGB_IF_Graph": c3,
            "C4_XGB_VAE": c4,
            "C5_Full_Hybrid": c5,
        }
        for name, score in fold_scores.items():
            oof_scores[name][test_idx] = score
            fold_metrics[name].append(_metrics(y_test, score))

    summary = {name: _summarize_fold_metrics(values) for name, values in fold_metrics.items()}
    baseline_auc = summary["B1_XGBoost"]["auc_roc"]["mean"]
    contribution_delta = {
        name: {
            "delta_auc_vs_B1": round(summary[name]["auc_roc"]["mean"] - baseline_auc, 6),
            "relative_delta_pct": round(100.0 * (summary[name]["auc_roc"]["mean"] - baseline_auc) / max(baseline_auc, 1e-9), 4),
        }
        for name in config_order
        if name != "B1_XGBoost"
    }

    pairwise: dict[str, Any] = {}
    for name in config_order:
        if name == "B1_XGBoost":
            continue
        key = f"{name}_vs_B1_XGBoost"
        pairwise[key] = {
            "delong": delong_auc_test(y, oof_scores[name], oof_scores["B1_XGBoost"]),
            "mcnemar": mcnemar_test(y, oof_scores[name] >= THRESHOLD, oof_scores["B1_XGBoost"] >= THRESHOLD),
            "paired_t_auc": paired_t_test(
                summary[name]["auc_roc"]["fold_values"],
                summary["B1_XGBoost"]["auc_roc"]["fold_values"],
            ),
        }

    ci = {
        name: metric_ci_bundle(y, oof_scores[name], threshold=THRESHOLD, n_bootstrap=bootstrap, seed=seed + idx)
        for idx, name in enumerate(config_order)
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "csv_path": str(csv_path),
            "rows": int(len(frame)),
            "companies": int(frame["tax_code"].nunique()) if "tax_code" in frame else None,
            "fraud_ratio": round(float(y.mean()), 6),
        },
        "folds": folds,
        "threshold": THRESHOLD,
        "configs": summary,
        "confidence_intervals": ci,
        "contribution_delta": contribution_delta,
        "pairwise_vs_B1": pairwise,
        "threshold_tradeoff_C5": _threshold_tradeoff(y, oof_scores["C5_Full_Hybrid"]),
        "threshold_tradeoff_B1": _threshold_tradeoff(y, oof_scores["B1_XGBoost"]),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fraud Model Ablation Study",
        "",
        f"- Rows: `{report['dataset']['rows']}`",
        f"- Folds: `{report['folds']}`",
        f"- Threshold: `{report['threshold']}`",
        "",
        "| Config | AUC-ROC | PR-AUC | F1 | ΔAUC vs B1 | DeLong p vs B1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["configs"].items():
        delta = report["contribution_delta"].get(name, {}).get("delta_auc_vs_B1", 0.0)
        pair = report["pairwise_vs_B1"].get(f"{name}_vs_B1_XGBoost", {})
        p_value = pair.get("delong", {}).get("p_value", "")
        lines.append(
            f"| {name} | {metrics['auc_roc']['mean']:.4f}±{metrics['auc_roc']['std']:.4f} | "
            f"{metrics['average_precision']['mean']:.4f} | {metrics['f1']['mean']:.4f} | {delta:+.4f} | {p_value} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(report: dict[str, Any], path: Path) -> None:
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Config & AUC-ROC & PR-AUC & F1 & $\\Delta$AUC vs B1 \\\\",
        "\\midrule",
    ]
    for name, metrics in report["configs"].items():
        delta = report["contribution_delta"].get(name, {}).get("delta_auc_vs_B1", 0.0)
        rows.append(
            f"{name} & {metrics['auc_roc']['mean']:.4f} $\\pm$ {metrics['auc_roc']['std']:.4f} & "
            f"{metrics['average_precision']['mean']:.4f} & {metrics['f1']['mean']:.4f} & {delta:+.4f} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_reports(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_results.json"
    md_path = out_dir / "ablation_results.md"
    tex_path = out_dir / "ablation_results.tex"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    _write_latex(report, tex_path)
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TaxInspector fraud ablation study.")
    parser.add_argument("--rows", type=int, default=120000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=250)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = run_ablation(rows=args.rows, folds=args.folds, seed=args.seed, force=args.force, bootstrap=args.bootstrap)
    paths = write_reports(report, args.out_dir)
    print(json.dumps({**paths, "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
