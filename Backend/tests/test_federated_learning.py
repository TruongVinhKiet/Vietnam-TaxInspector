from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.federated_learning_poc import (
    _sigmoid,
    build_feature_matrix,
    non_iid_two_node_split,
    run_federated_learning_poc,
)


def test_sigmoid_is_bounded():
    values = _sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert values[1] == 0.5


def test_non_iid_split_returns_two_non_empty_nodes():
    frame = pd.DataFrame(
        {
            "province": ["HCM"] * 40 + ["Da Nang"] * 40,
            "industry": ["finance"] * 20 + ["retail"] * 60,
            "fraud_label": [0, 1] * 40,
        }
    )
    idx_a, idx_b = non_iid_two_node_split(frame, seed=7)
    assert len(idx_a) > 0
    assert len(idx_b) > 0
    assert len(set(idx_a) & set(idx_b)) == 0


def test_build_feature_matrix_uses_expected_columns():
    frame = pd.DataFrame(
        {
            "revenue": [1.0, 2.0],
            "cost_of_goods": [0.4, 0.8],
            "operating_expenses": [0.1, 0.2],
            "vat_input": [0.05, 0.08],
            "vat_output": [0.06, 0.09],
            "num_employees": [5, 8],
            "registered_capital": [10, 20],
            "vat_graph_out_pr_ratio": [0.1, 0.9],
            "cycle_participation_score": [0.2, 0.8],
            "graph_neighbor_risk": [0.1, 0.7],
            "graph_centrality_delta": [0.0, 1.0],
            "fraud_label": [0, 1],
        }
    )
    X, y, cols = build_feature_matrix(frame)
    assert X.shape == (2, len(cols))
    assert y.tolist() == [0, 1]
    assert "revenue" in cols


def test_federated_learning_poc_smoke():
    result = run_federated_learning_poc(rows=800, rounds=2, seed=123, dp_noise=0.0001)
    assert 0.0 <= result["federated_auc"] <= 1.0
    assert 0.0 <= result["centralized_auc"] <= 1.0
    assert result["communication_bytes_per_round"] > 0
    assert len(result["node_results"]) >= 2
    assert len(result["convergence_curve"]) == 2
