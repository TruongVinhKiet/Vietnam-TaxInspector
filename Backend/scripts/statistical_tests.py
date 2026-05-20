"""Statistical testing utilities for TaxInspector experimental evaluation.

The module is dependency-light: SciPy is used when available, otherwise normal
and chi-square(df=1) approximations are used.  It is designed for paired model
comparisons on the same fraud test folds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


EPS = 1e-12


def _normal_two_sided_p(z_value: float) -> float:
    return float(math.erfc(abs(float(z_value)) / math.sqrt(2.0)))


def _chi2_df1_sf(statistic: float) -> float:
    return float(math.erfc(math.sqrt(max(float(statistic), 0.0) / 2.0)))


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def auc_variance_delong(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Single-model DeLong AUC variance using influence values."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return {"auc": 0.0, "variance": 0.0, "std_error": 0.0}

    comparisons = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :]).astype(float)
    v10 = comparisons.mean(axis=1)
    v01 = comparisons.mean(axis=0)
    auc = float(v10.mean())
    variance = float(np.var(v10, ddof=1) / max(m, 1) + np.var(v01, ddof=1) / max(n, 1))
    return {"auc": auc, "variance": max(variance, 0.0), "std_error": math.sqrt(max(variance, 0.0))}


def delong_auc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    """Paired DeLong test for two correlated ROC-AUC values."""
    y_true = np.asarray(y_true, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    pos_a, neg_a = score_a[y_true == 1], score_a[y_true == 0]
    pos_b, neg_b = score_b[y_true == 1], score_b[y_true == 0]
    m, n = len(pos_a), len(neg_a)
    if m == 0 or n == 0:
        return {"auc_a": 0.0, "auc_b": 0.0, "delta_auc": 0.0, "z": 0.0, "p_value": 1.0, "ci95_low": 0.0, "ci95_high": 0.0}

    cmp_a = (pos_a[:, None] > neg_a[None, :]).astype(float) + 0.5 * (pos_a[:, None] == neg_a[None, :]).astype(float)
    cmp_b = (pos_b[:, None] > neg_b[None, :]).astype(float) + 0.5 * (pos_b[:, None] == neg_b[None, :]).astype(float)
    v10_a, v01_a = cmp_a.mean(axis=1), cmp_a.mean(axis=0)
    v10_b, v01_b = cmp_b.mean(axis=1), cmp_b.mean(axis=0)
    auc_a, auc_b = float(v10_a.mean()), float(v10_b.mean())
    delta = auc_a - auc_b
    cov10 = float(np.cov(v10_a, v10_b, ddof=1)[0, 1]) if m > 1 else 0.0
    cov01 = float(np.cov(v01_a, v01_b, ddof=1)[0, 1]) if n > 1 else 0.0
    var_a = float(np.var(v10_a, ddof=1) / m + np.var(v01_a, ddof=1) / n) if m > 1 and n > 1 else 0.0
    var_b = float(np.var(v10_b, ddof=1) / m + np.var(v01_b, ddof=1) / n) if m > 1 and n > 1 else 0.0
    var_delta = max(var_a + var_b - 2.0 * (cov10 / m + cov01 / n), EPS)
    se_delta = math.sqrt(var_delta)
    z_value = delta / se_delta
    return {
        "auc_a": round(auc_a, 6),
        "auc_b": round(auc_b, 6),
        "delta_auc": round(delta, 6),
        "z": round(float(z_value), 6),
        "p_value": round(_normal_two_sided_p(z_value), 8),
        "ci95_low": round(delta - 1.96 * se_delta, 6),
        "ci95_high": round(delta + 1.96 * se_delta, 6),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float | int]:
    """McNemar test with continuity correction."""
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    both_correct = int(np.sum(correct_a & correct_b))
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    denom = a_correct_b_wrong + a_wrong_b_correct
    chi2 = 0.0 if denom == 0 else (abs(a_correct_b_wrong - a_wrong_b_correct) - 1.0) ** 2 / denom
    odds_ratio = (a_correct_b_wrong + 0.5) / (a_wrong_b_correct + 0.5)
    return {
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "chi2": round(float(chi2), 6),
        "p_value": round(_chi2_df1_sf(chi2), 8),
        "odds_ratio": round(float(odds_ratio), 6),
    }


def paired_t_test(values_a: list[float] | np.ndarray, values_b: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b
    if len(diff) < 2:
        return {"mean_delta": float(diff.mean()) if len(diff) else 0.0, "t": 0.0, "p_value": 1.0}
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
    t_value = 0.0 if se <= EPS else float(diff.mean() / se)
    try:
        from scipy.stats import t as student_t

        p_value = float(2.0 * student_t.sf(abs(t_value), df=len(diff) - 1))
    except Exception:
        p_value = _normal_two_sided_p(t_value)
    return {"mean_delta": round(float(diff.mean()), 6), "t": round(t_value, 6), "p_value": round(p_value, 8)}


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    method: str = "bca",
) -> dict[str, float]:
    """Bootstrap confidence interval. BCa is used when jackknife is feasible."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return {"estimate": 0.0, "low": 0.0, "high": 0.0, "method": method}
    estimate = float(metric_fn(y_true, y_score))
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        samples.append(float(metric_fn(y_true[idx], y_score[idx])))
    if not samples:
        return {"estimate": round(estimate, 6), "low": round(estimate, 6), "high": round(estimate, 6), "method": method}
    samples_np = np.sort(np.asarray(samples, dtype=float))
    alpha = 1.0 - confidence

    if method.lower() == "bca" and n <= 5000:
        normal = NormalDist()
        prop_less = np.clip(np.mean(samples_np < estimate), 1e-6, 1 - 1e-6)
        z0 = normal.inv_cdf(float(prop_less))
        jack = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if len(np.unique(y_true[mask])) < 2:
                continue
            jack.append(float(metric_fn(y_true[mask], y_score[mask])))
        if len(jack) >= 3:
            jack_np = np.asarray(jack)
            jack_mean = float(jack_np.mean())
            numerator = np.sum((jack_mean - jack_np) ** 3)
            denominator = 6.0 * (np.sum((jack_mean - jack_np) ** 2) ** 1.5 + EPS)
            accel = float(numerator / denominator)
            z_low = normal.inv_cdf(alpha / 2.0)
            z_high = normal.inv_cdf(1.0 - alpha / 2.0)
            adj_low = normal.cdf(z0 + (z0 + z_low) / (1.0 - accel * (z0 + z_low) + EPS))
            adj_high = normal.cdf(z0 + (z0 + z_high) / (1.0 - accel * (z0 + z_high) + EPS))
            low, high = np.quantile(samples_np, [np.clip(adj_low, 0, 1), np.clip(adj_high, 0, 1)])
        else:
            low, high = np.quantile(samples_np, [alpha / 2.0, 1.0 - alpha / 2.0])
    else:
        low, high = np.quantile(samples_np, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {"estimate": round(estimate, 6), "low": round(float(low), 6), "high": round(float(high), 6), "method": method}


def metric_ci_bundle(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.30, n_bootstrap: int = 300, seed: int = 42) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    def _auc(a, b):
        return 0.0 if len(np.unique(a)) < 2 else float(roc_auc_score(a, b))

    def _ap(a, b):
        return 0.0 if len(np.unique(a)) < 2 else float(average_precision_score(a, b))

    def _precision(a, b):
        return float(precision_score(a, b >= threshold, zero_division=0))

    def _recall(a, b):
        return float(recall_score(a, b >= threshold, zero_division=0))

    def _f1(a, b):
        return float(f1_score(a, b >= threshold, zero_division=0))

    return {
        "auc_roc": bootstrap_ci(y_true, y_score, _auc, n_bootstrap=n_bootstrap, seed=seed),
        "average_precision": bootstrap_ci(y_true, y_score, _ap, n_bootstrap=n_bootstrap, seed=seed + 1),
        "precision": bootstrap_ci(y_true, y_score, _precision, n_bootstrap=n_bootstrap, seed=seed + 2),
        "recall": bootstrap_ci(y_true, y_score, _recall, n_bootstrap=n_bootstrap, seed=seed + 3),
        "f1": bootstrap_ci(y_true, y_score, _f1, n_bootstrap=n_bootstrap, seed=seed + 4),
    }


def friedman_test(score_matrix: np.ndarray) -> dict[str, float]:
    """Friedman test over rows=datasets/folds, cols=models."""
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 2:
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import friedmanchisquare

        stat, p_value = friedmanchisquare(*[scores[:, i] for i in range(scores.shape[1])])
        return {"statistic": round(float(stat), 6), "p_value": round(float(p_value), 8)}
    except Exception:
        ranks = np.vstack([_midrank(row) for row in scores])
        n, k = scores.shape
        rank_sums = ranks.sum(axis=0)
        stat = 12.0 / (n * k * (k + 1.0)) * np.sum(rank_sums ** 2) - 3.0 * n * (k + 1.0)
        return {"statistic": round(float(stat), 6), "p_value": round(_chi2_df1_sf(stat), 8)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test statistical utilities.")
    parser.add_argument("--out", type=Path, default=Path("Backend/reports/statistical_tests_smoke.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    y = rng.binomial(1, 0.25, size=800)
    score_a = np.clip(0.25 + 0.55 * y + rng.normal(0, 0.22, size=len(y)), 0, 1)
    score_b = np.clip(0.22 + 0.45 * y + rng.normal(0, 0.26, size=len(y)), 0, 1)
    report = {
        "delong": delong_auc_test(y, score_a, score_b),
        "mcnemar": mcnemar_test(y, score_a >= 0.3, score_b >= 0.3),
        "ci": metric_ci_bundle(y, score_a, n_bootstrap=100, seed=args.seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
