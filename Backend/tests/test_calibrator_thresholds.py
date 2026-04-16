import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.metrics_engine import ProbabilityCalibrator


def test_thresholds_are_persisted_and_loaded(tmp_path):
    calibrator = ProbabilityCalibrator()

    node_probs = np.array([0.05, 0.12, 0.33, 0.51, 0.72, 0.91], dtype=float)
    node_labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    edge_probs = np.array([0.02, 0.08, 0.4, 0.67, 0.88], dtype=float)
    edge_labels = np.array([0, 0, 1, 1, 1], dtype=int)

    info = calibrator.optimize_decision_thresholds(
        node_probs=node_probs,
        node_labels=node_labels,
        edge_probs=edge_probs,
        edge_labels=edge_labels,
        metric="f1",
    )

    calibrator.save(str(tmp_path))

    loaded = ProbabilityCalibrator()
    assert loaded.load(str(tmp_path)) is True
    assert loaded.node_threshold == pytest.approx(info["node_threshold"])
    assert loaded.edge_threshold == pytest.approx(info["edge_threshold"])
    assert loaded.threshold_meta.get("metric") == "f1"


def test_load_old_calibrator_artifact_uses_default_thresholds(tmp_path):
    legacy_payload = {
        "node_calibrator": None,
        "edge_calibrator": None,
        "fitted": False,
    }
    with open(tmp_path / "calibrator.pkl", "wb") as f:
        pickle.dump(legacy_payload, f)

    calibrator = ProbabilityCalibrator()
    assert calibrator.load(str(tmp_path)) is True
    assert calibrator.node_threshold == pytest.approx(0.5)
    assert calibrator.edge_threshold == pytest.approx(0.5)
