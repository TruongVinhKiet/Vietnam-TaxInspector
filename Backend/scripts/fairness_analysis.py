"""Fairness and ethical AI slice analysis for TaxInspector fraud scoring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.feature_engineering import TaxFeatureEngineer  # noqa: E402
from scripts.run_experimental_evaluation import THRESHOLD, _add_graph_features, _ensure_fraud_frame, _model_prob, _xgb_or_hgb_factory  # noqa: E402


REPORT_DIR = BACKEND_DIR / "reports"


def _rate_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)
    selection_rate = (tp + fp) / max(len(y_true), 1)
    return {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "fpr": round(float(fpr), 6),
        "tpr": round(float(tpr), 6),
        "selection_rate": round(float(selection_rate), 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "auc_roc": round(float(roc_auc_score(y_true, score)), 6) if len(np.unique(y_true)) > 1 else None,
    }


def _bucket_company_age(frame: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(frame.get("year"), errors="coerce").fillna(2024).astype(int)
    first_year = frame.groupby("tax_code")["year"].transform("min") if "tax_code" in frame and "year" in frame else years
    age = years - pd.to_numeric(first_year, errors="coerce").fillna(years)
    return pd.cut(age, bins=[-1, 1, 4, 100], labels=["young_lt_2y", "mid_2_to_5y", "mature_gt_5y"])


def _prepare_slices(frame: pd.DataFrame) -> pd.DataFrame:
    payload = frame.copy()
    revenue = pd.to_numeric(payload.get("revenue"), errors="coerce").fillna(0.0)
    payload["revenue_bucket"] = pd.cut(
        revenue,
        bins=[-np.inf, 1e9, 1e10, 1e11, np.inf],
        labels=["lt_1b", "1b_to_10b", "10b_to_100b", "gte_100b"],
        right=False,
    )
    payload["company_age"] = _bucket_company_age(payload)
    if "industry" not in payload:
        payload["industry"] = "unknown"
    if "province" not in payload:
        payload["province"] = "unknown"
    return payload


def _dimension_report(payload: pd.DataFrame, y_true: np.ndarray, score: np.ndarray, dimension: str, min_samples: int) -> dict[str, Any]:
    payload = payload.reset_index(drop=True)
    payload["_y"] = y_true
    payload["_score"] = score
    groups: dict[str, Any] = {}
    fprs, tprs, selections = [], [], []
    for raw_value, group in payload.groupby(dimension, dropna=False, observed=False):
        if len(group) < min_samples:
            continue
        key = "unknown" if pd.isna(raw_value) else str(raw_value)
        metrics = _rate_metrics(group["_y"].to_numpy(), group["_score"].to_numpy())
        groups[key] = metrics
        fprs.append(metrics["fpr"])
        tprs.append(metrics["tpr"])
        selections.append(metrics["selection_rate"])
    if not groups:
        return {"groups": {}, "summary": {"evaluated_groups": 0}}
    fprs_np = np.asarray(fprs, dtype=float)
    tprs_np = np.asarray(tprs, dtype=float)
    sel_np = np.asarray(selections, dtype=float)
    max_fpr = float(np.max(fprs_np))
    min_fpr = float(np.min(fprs_np))
    max_sel = float(np.max(sel_np))
    min_sel = float(np.min(sel_np))
    summary = {
        "evaluated_groups": len(groups),
        "fpr_min": round(min_fpr, 6),
        "fpr_max": round(max_fpr, 6),
        "disparate_impact_fpr_ratio": round(min_fpr / max(max_fpr, 1e-9), 6),
        "selection_rate_di_ratio": round(min_sel / max(max_sel, 1e-9), 6),
        "equal_opportunity_difference": round(float(np.max(tprs_np) - np.min(tprs_np)), 6),
        "passes_four_fifths_rule": bool((min_sel / max(max_sel, 1e-9)) >= 0.80),
        "red_flag": bool((min_sel / max(max_sel, 1e-9)) < 0.80 or (min_fpr / max(max_fpr, 1e-9)) < 0.80),
    }
    return {"groups": groups, "summary": summary}


def _calibration_equity(payload: pd.DataFrame, y_true: np.ndarray, score: np.ndarray, dimension: str, min_samples: int) -> dict[str, Any]:
    df = payload.reset_index(drop=True).copy()
    df["_y"] = y_true
    df["_score"] = np.asarray(score, dtype=float)
    df["_bin"] = pd.cut(df["_score"], bins=np.linspace(0, 1, 6), include_lowest=True)
    curves: dict[str, list[dict[str, float]]] = {}
    for raw_value, group in df.groupby(dimension, dropna=False, observed=False):
        if len(group) < min_samples:
            continue
        key = "unknown" if pd.isna(raw_value) else str(raw_value)
        rows = []
        for raw_bin, bin_group in group.groupby("_bin", observed=False):
            if len(bin_group) == 0:
                continue
            rows.append({
                "bin": str(raw_bin),
                "n": int(len(bin_group)),
                "mean_score": round(float(bin_group["_score"].mean()), 6),
                "observed_rate": round(float(bin_group["_y"].mean()), 6),
            })
        curves[key] = rows
    return curves


def run_fairness_analysis(*, rows: int = 120000, seed: int = 42, force: bool = False, min_samples: int = 80) -> dict[str, Any]:
    t0 = time.perf_counter()
    frame, csv_path = _ensure_fraud_frame(rows, seed, force=force)
    fe = TaxFeatureEngineer()
    X = fe.get_feature_matrix(frame)
    y = frame["fraud_label"].astype(int).to_numpy()
    X_graph = _add_graph_features(frame, y, seed)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed)
    model = _xgb_or_hgb_factory(seed + 700)
    model.fit(X_graph[train_idx], y[train_idx])
    score = _model_prob(model, X_graph[test_idx])
    eval_frame = _prepare_slices(frame.iloc[test_idx].copy())
    y_test = y[test_idx]

    dimensions = ["industry", "revenue_bucket", "province", "company_age"]
    slice_reports = {
        dim: _dimension_report(eval_frame, y_test, score, dim, min_samples)
        for dim in dimensions
    }
    red_flags = [
        {"dimension": dim, "summary": report["summary"]}
        for dim, report in slice_reports.items()
        if report.get("summary", {}).get("red_flag")
    ]
    calibration = {
        dim: _calibration_equity(eval_frame, y_test, score, dim, min_samples)
        for dim in ("industry", "revenue_bucket")
    }
    mitigation_recommendations = []
    if red_flags:
        mitigation_recommendations = [
            "Review flagged slices with domain experts before using the score as an automatic enforcement trigger.",
            "Calibrate thresholds per risk tier and inspect selection-rate gaps for low-volume groups.",
            "Add post-model human review for slices with high false-positive disparity or unstable sample counts.",
            "Re-run the report on the full 120k/5-fold evaluation; small smoke runs can overstate disparity because some groups receive zero selections.",
        ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "csv_path": str(csv_path),
            "rows": int(len(frame)),
            "eval_rows": int(len(test_idx)),
            "fraud_ratio": round(float(y.mean()), 6),
        },
        "threshold": THRESHOLD,
        "min_samples_per_group": min_samples,
        "overall": _rate_metrics(y_test, score),
        "slices": slice_reports,
        "calibration_equity": calibration,
        "red_flags": red_flags,
        "disparate_impact_pass": len(red_flags) == 0,
        "mitigation_recommendations": mitigation_recommendations,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fairness Analysis",
        "",
        f"- Rows: `{report['dataset']['rows']}`",
        f"- Eval rows: `{report['dataset']['eval_rows']}`",
        f"- Overall FPR: `{report['overall']['fpr']}`",
        f"- Overall TPR: `{report['overall']['tpr']}`",
        f"- Disparate impact pass: `{report['disparate_impact_pass']}`",
        "",
        "| Dimension | Groups | FPR DI | Selection DI | Equal Opp. Diff | Red flag |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dim, payload in report["slices"].items():
        s = payload["summary"]
        lines.append(
            f"| {dim} | {s.get('evaluated_groups', 0)} | {s.get('disparate_impact_fpr_ratio')} | "
            f"{s.get('selection_rate_di_ratio')} | {s.get('equal_opportunity_difference')} | {s.get('red_flag')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fairness_reports(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fairness_report.json"
    md_path = out_dir / "fairness_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TaxInspector fairness analysis.")
    parser.add_argument("--rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-samples", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = run_fairness_analysis(rows=args.rows, seed=args.seed, force=args.force, min_samples=args.min_samples)
    paths = write_fairness_reports(report, args.out_dir)
    print(json.dumps({**paths, "red_flags": len(report["red_flags"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
