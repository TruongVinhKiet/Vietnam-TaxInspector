from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.concept_drift_detector import (
    ADWINDetector,
    DriftMonitor,
    PageHinkleyDetector,
    population_stability_index,
)


def test_adwin_detects_injected_shift():
    detector = ADWINDetector(delta=0.01, min_window=24, min_subwindow=8, max_window=160)
    detected = False
    for value in [0.82] * 70 + [0.30] * 70:
        detected = detector.update(value) or detected
    assert detected is True
    assert detector.last_change_index is not None


def test_page_hinkley_detects_mean_shift():
    detector = PageHinkleyDetector(delta=0.001, threshold=0.9, min_instances=20)
    detected = False
    for value in [0.20] * 60 + [0.68] * 60:
        detected = detector.update(value) or detected
    assert detected is True


def test_population_stability_index_stable_vs_shifted():
    rng = np.random.default_rng(42)
    expected = rng.normal(0, 1, 600)
    stable = rng.normal(0.03, 1, 600)
    shifted = rng.normal(1.4, 1.1, 600)

    stable_psi = population_stability_index(expected, stable)
    shifted_psi = population_stability_index(expected, shifted)

    assert stable_psi < 0.1
    assert shifted_psi > 0.25


def test_drift_monitor_emits_alert_file(tmp_path: Path):
    alert_path = tmp_path / "drift.jsonl"
    monitor = DriftMonitor(
        model_key="test-model",
        alert_path=alert_path,
        adwin_delta=0.01,
        page_hinkley_threshold=0.9,
        warmup=20,
        feature_window=80,
    )

    statuses = []
    for idx, value in enumerate([0.86] * 80 + [0.22] * 80):
        statuses.append(
            monitor.update(
                prediction_confidence=value,
                features={"amount_log": float(idx % 11), "invoice_count": float(idx)},
            )
        )

    assert any(status["drift_detected"] for status in statuses)
    assert alert_path.exists()
    assert "test-model" in alert_path.read_text(encoding="utf-8")
    assert monitor.get_status()["alert_count"] >= 1
