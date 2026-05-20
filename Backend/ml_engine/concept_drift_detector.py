"""Concept drift monitoring utilities for TaxInspector models.

The module is intentionally dependency-light so it can run inside the online
feature pipeline without pulling River or scikit-multiflow into production.
It provides three complementary detectors:

* ADWIN-style adaptive windowing over score/confidence streams.
* Page-Hinkley sequential mean-shift detection.
* PSI distribution comparison for tabular feature drift.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np


DEFAULT_ALERT_PATH = Path(__file__).resolve().parents[1] / "data" / "logs" / "drift_alerts.jsonl"


@dataclass
class DriftAlert:
    model_key: str
    detector: str
    severity: str
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return payload


class ADWINDetector:
    """Small ADWIN-inspired change detector for bounded numeric streams.

    It compares the mean of two adjacent subwindows and shrinks the window when
    a statistically meaningful shift is detected. The implementation is not a
    byte-for-byte River clone, but it follows the same operational idea and is
    deterministic enough for online monitoring and tests.
    """

    def __init__(
        self,
        *,
        delta: float = 0.002,
        min_window: int = 20,
        max_window: int = 512,
        min_subwindow: int = 10,
    ):
        self.delta = float(delta)
        self.min_window = int(min_window)
        self.max_window = int(max_window)
        self.min_subwindow = int(min_subwindow)
        self.window: list[float] = []
        self.drift_detected = False
        self.last_change_index: int | None = None
        self.last_statistic: dict[str, float] = {}
        self.total_seen = 0

    def update(self, value: float) -> bool:
        x = float(value)
        if not math.isfinite(x):
            return False
        self.total_seen += 1
        self.window.append(x)
        if len(self.window) > self.max_window:
            self.window = self.window[-self.max_window :]

        self.drift_detected = False
        self.last_statistic = {}
        n = len(self.window)
        if n < self.min_window:
            return False

        best: tuple[float, int, float, float, float] | None = None
        start = self.min_subwindow
        stop = n - self.min_subwindow + 1
        step = max(1, n // 24)
        for cut in range(start, stop, step):
            left = self.window[:cut]
            right = self.window[cut:]
            n0 = len(left)
            n1 = len(right)
            m0 = mean(left)
            m1 = mean(right)
            diff = abs(m0 - m1)
            harmonic = 1.0 / ((1.0 / n0) + (1.0 / n1))
            epsilon = math.sqrt((1.0 / (2.0 * harmonic)) * math.log(4.0 / max(self.delta, 1e-12)))
            if diff > epsilon and (best is None or diff - epsilon > best[0] - best[4]):
                best = (diff, cut, m0, m1, epsilon)

        if best is None:
            return False

        diff, cut, m0, m1, epsilon = best
        self.drift_detected = True
        self.last_change_index = self.total_seen - n + cut
        self.last_statistic = {
            "mean_before": float(m0),
            "mean_after": float(m1),
            "difference": float(diff),
            "epsilon": float(epsilon),
            "cut": float(cut),
        }
        self.window = self.window[cut:]
        return True


class PageHinkleyDetector:
    """Page-Hinkley test for sequential mean shifts."""

    def __init__(
        self,
        *,
        delta: float = 0.005,
        threshold: float = 2.5,
        alpha: float = 0.999,
        min_instances: int = 30,
    ):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.min_instances = int(min_instances)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.maximum = 0.0
        self.last_statistic: dict[str, float] = {}

    def update(self, value: float) -> bool:
        x = float(value)
        if not math.isfinite(x):
            return False
        self.count += 1
        self.mean += (x - self.mean) / self.count
        self.cumulative = self.alpha * self.cumulative + x - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        self.maximum = max(self.maximum, self.cumulative)
        increase = self.cumulative - self.minimum
        decrease = self.maximum - self.cumulative
        stat = max(increase, decrease)
        self.last_statistic = {
            "mean": float(self.mean),
            "cumulative": float(self.cumulative),
            "statistic": float(stat),
            "threshold": float(self.threshold),
        }
        if self.count >= self.min_instances and stat > self.threshold:
            self.reset()
            return True
        return False


def population_stability_index(
    expected: list[float] | np.ndarray,
    actual: list[float] | np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI between expected and actual distributions."""

    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if exp.size < 2 or act.size < 2:
        return 0.0

    quantiles = np.linspace(0, 100, int(bins) + 1)
    edges = np.percentile(exp, quantiles)
    edges = np.unique(edges)
    if edges.size <= 2:
        lo = min(float(exp.min()), float(act.min()))
        hi = max(float(exp.max()), float(act.max()))
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, int(bins) + 1)

    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)
    exp_pct = exp_counts / max(1, exp_counts.sum())
    act_pct = act_counts / max(1, act_counts.sum())
    exp_pct = np.clip(exp_pct, epsilon, None)
    act_pct = np.clip(act_pct, epsilon, None)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi))


def compute_feature_psi(
    expected: dict[str, list[float]],
    actual: dict[str, list[float]],
    *,
    bins: int = 10,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, exp_values in expected.items():
        if key in actual:
            out[key] = population_stability_index(exp_values, actual[key], bins=bins)
    return out


class DriftMonitor:
    """Coordinates online drift detectors and alert persistence."""

    def __init__(
        self,
        *,
        model_key: str = "fraud-hybrid-v2",
        alert_path: Path | str = DEFAULT_ALERT_PATH,
        adwin_delta: float = 0.002,
        page_hinkley_threshold: float = 2.5,
        psi_threshold: float = 0.25,
        warmup: int = 120,
        feature_window: int = 512,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.model_key = model_key
        self.adwin = ADWINDetector(delta=adwin_delta)
        self.page_hinkley = PageHinkleyDetector(threshold=page_hinkley_threshold)
        self.psi_threshold = float(psi_threshold)
        self.warmup = int(warmup)
        self.feature_window = int(feature_window)
        self.alert_path = Path(alert_path)
        self.event_callback = event_callback
        self.prediction_count = 0
        self.alert_count = 0
        self.last_alert: dict[str, Any] | None = None
        self._baseline_features: dict[str, list[float]] = {}
        self._recent_features: dict[str, list[float]] = {}

    def update(
        self,
        *,
        prediction_confidence: float,
        features: dict[str, Any] | None = None,
        model_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = model_key or self.model_key
        conf = float(prediction_confidence)
        if not math.isfinite(conf):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        self.prediction_count += 1
        alerts: list[DriftAlert] = []

        if self.adwin.update(conf):
            alerts.append(
                DriftAlert(
                    model_key=key,
                    detector="adwin",
                    severity="high",
                    metric="prediction_confidence",
                    value=float(self.adwin.last_statistic.get("difference", 0.0)),
                    threshold=float(self.adwin.last_statistic.get("epsilon", 0.0)),
                    message="ADWIN detected a confidence distribution change.",
                    metadata={**(metadata or {}), **self.adwin.last_statistic},
                )
            )

        if self.page_hinkley.update(conf):
            alerts.append(
                DriftAlert(
                    model_key=key,
                    detector="page_hinkley",
                    severity="medium",
                    metric="prediction_confidence",
                    value=float(self.page_hinkley.last_statistic.get("statistic", 0.0)),
                    threshold=float(self.page_hinkley.last_statistic.get("threshold", 0.0)),
                    message="Page-Hinkley detected a sequential mean shift.",
                    metadata=metadata or {},
                )
            )

        psi_scores = self._update_feature_windows(features or {})
        flagged_psi = {k: v for k, v in psi_scores.items() if v > self.psi_threshold}
        if flagged_psi:
            worst_feature, worst_value = max(flagged_psi.items(), key=lambda item: item[1])
            alerts.append(
                DriftAlert(
                    model_key=key,
                    detector="psi",
                    severity="high" if worst_value >= 0.35 else "medium",
                    metric=worst_feature,
                    value=float(worst_value),
                    threshold=self.psi_threshold,
                    message="Feature PSI exceeded the retraining threshold.",
                    metadata={"psi_scores": flagged_psi, **(metadata or {})},
                )
            )

        for alert in alerts:
            self._emit_alert(alert)

        return {
            "model_key": key,
            "prediction_count": self.prediction_count,
            "drift_detected": bool(alerts),
            "retraining_recommended": bool(alerts),
            "alerts": [a.to_dict() for a in alerts],
            "psi_scores": psi_scores,
            "last_alert": self.last_alert,
        }

    def _update_feature_windows(self, features: dict[str, Any]) -> dict[str, float]:
        psi_scores: dict[str, float] = {}
        for name, raw_value in features.items():
            if name.startswith("_"):
                continue
            try:
                value = float(raw_value)
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            baseline = self._baseline_features.setdefault(name, [])
            recent = self._recent_features.setdefault(name, [])
            if len(baseline) < self.warmup:
                baseline.append(value)
                continue
            recent.append(value)
            if len(recent) > self.feature_window:
                del recent[: len(recent) - self.feature_window]
            if len(recent) >= max(30, min(80, self.warmup // 2)):
                psi_scores[name] = population_stability_index(baseline, recent)
        return psi_scores

    def _emit_alert(self, alert: DriftAlert) -> None:
        payload = alert.to_dict()
        self.alert_count += 1
        self.last_alert = payload
        try:
            self.alert_path.parent.mkdir(parents=True, exist_ok=True)
            with self.alert_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        if self.event_callback:
            try:
                self.event_callback(payload)
            except Exception:
                pass

    def get_status(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "prediction_count": self.prediction_count,
            "alert_count": self.alert_count,
            "last_alert": self.last_alert,
            "adwin_window_size": len(self.adwin.window),
            "baseline_feature_count": {k: len(v) for k, v in self._baseline_features.items()},
            "recent_feature_count": {k: len(v) for k, v in self._recent_features.items()},
            "psi_threshold": self.psi_threshold,
        }
