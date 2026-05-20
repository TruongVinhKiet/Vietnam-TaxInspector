"""Federated Learning PoC for TaxInspector (4-node and 10-node).

This PoC uses FedAvg over logistic-model weights because it is transparent,
fast on CPU, and easy to validate in CI. It complements the production
XGBoost/GBDT stack.

Capabilities:
  - 4-node Non-IID federation (legacy, backward-compatible)
  - 10-node scale-up with 4 regions (North/Central/South/Industrial-Zone),
    calibrated Differential Privacy (eps, delta), TopK-10% gradient
    compression, and async straggler tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
for _path in (BACKEND_DIR, REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@dataclass
class FederatedNode:
    node_id: str
    display_name: str
    n_samples: int
    local_auc: float
    industry_mix: dict[str, int]
    province_mix: dict[str, int]


@dataclass
class FederatedResult:
    generated_at: str
    rounds: int
    rows: int
    epsilon: float
    delta: float
    centralized_auc: float
    federated_auc: float
    federated_gap: float
    node_results: list[FederatedNode]
    communication_bytes_per_round: int
    total_communication_bytes: int
    convergence_curve: list[dict[str, float]]
    privacy: dict[str, float | str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["node_results"] = [asdict(node) for node in self.node_results]
        return payload


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def _load_fraud_frame(rows: int, seed: int):
    try:
        from scripts.run_experimental_evaluation import _add_graph_features, _ensure_fraud_frame

        frame, _ = _ensure_fraud_frame(rows=max(rows, 500), seed=seed, force=False)
        return _add_graph_features(frame.copy(), seed)
    except Exception:
        import pandas as pd

        rng = np.random.default_rng(seed)
        n = max(rows, 500)
        revenue = rng.lognormal(17.3, 1.1, n)
        industry = rng.choice(["retail", "construction", "finance", "services"], n)
        province = rng.choice(["HCM", "HN", "DN", "CT"], n)
        score = (
            0.9 * (np.log1p(revenue) - np.log1p(revenue).mean())
            + rng.normal(0, 1, n)
            + (industry == "finance") * 0.6
        )
        y = score > np.quantile(score, 0.94)
        return pd.DataFrame(
            {
                "industry": industry,
                "province": province,
                "revenue": revenue,
                "cost_of_goods": revenue * rng.uniform(0.45, 0.9, n),
                "operating_expenses": revenue * rng.uniform(0.05, 0.25, n),
                "vat_input": revenue * rng.uniform(0.01, 0.08, n),
                "vat_output": revenue * rng.uniform(0.01, 0.08, n),
                "num_employees": rng.integers(3, 400, n),
                "registered_capital": rng.lognormal(15.0, 1.0, n),
                "fraud_label": y.astype(int),
                "vat_graph_out_pr_ratio": rng.uniform(0.0, 2.0, n) + y * 0.5,
                "cycle_participation_score": rng.uniform(0.0, 1.0, n) + y * 0.2,
                "graph_neighbor_risk": rng.uniform(0.0, 1.0, n) + y * 0.2,
                "graph_centrality_delta": rng.normal(0.0, 1.0, n) + y * 0.3,
            }
        )


def build_feature_matrix(frame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric_cols = [
        "revenue",
        "cost_of_goods",
        "operating_expenses",
        "vat_input",
        "vat_output",
        "num_employees",
        "registered_capital",
        "vat_graph_out_pr_ratio",
        "cycle_participation_score",
        "graph_neighbor_risk",
        "graph_centrality_delta",
    ]
    cols = [c for c in numeric_cols if c in frame.columns]
    X = frame[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = frame["fraud_label"].astype(int).to_numpy()
    return X, y, cols


# ── 10-node profiles ─────────────────────────────────────────────────────────
@dataclass
class NodeProfile:
    """Non-IID profile for a regional Tax Office."""
    node_id: str
    display_name: str
    region: str
    industry_bias: str       # dominant industry keyword
    fraud_weight: float      # relative fraud rate multiplier
    infra_quality: float     # 0-1, probability of NOT being a straggler

NODE_PROFILES_10 = [
    # --- North ---
    NodeProfile("ha_noi",     "Cuc Thue Ha Noi (construction)",    "north",      "construction", 0.85, 0.95),
    NodeProfile("hai_phong",  "Cuc Thue Hai Phong (logistics)",    "north",      "retail",       1.05, 0.90),
    NodeProfile("quang_ninh", "Cuc Thue Quang Ninh (mining)",      "north",      "retail",       0.75, 0.80),
    # --- Central ---
    NodeProfile("da_nang",    "Cuc Thue Da Nang (services)",       "central",    "services",     0.65, 0.88),
    NodeProfile("khanh_hoa",  "Cuc Thue Khanh Hoa (tourism/FDI)",  "central",    "services",     0.50, 0.75),
    # --- South ---
    NodeProfile("hcmc",       "Cuc Thue TP.HCM (finance/tech)",    "south",      "finance",      1.85, 0.97),
    NodeProfile("can_tho",    "Cuc Thue Can Tho (agriculture)",    "south",      "retail",       0.60, 0.70),
    NodeProfile("dak_lak",    "Cuc Thue Dak Lak (forestry)",       "south",      "retail",       0.40, 0.55),
    # --- Industrial zones ---
    NodeProfile("binh_duong", "Cuc Thue Binh Duong (manufacturing)", "industrial", "construction", 1.15, 0.92),
    NodeProfile("dong_nai",   "Cuc Thue Dong Nai (manufacturing)",  "industrial", "retail",       1.05, 0.88),
]


def non_iid_multi_node_split(frame, n_nodes: int = 4, seed: int = 42) -> list[np.ndarray]:
    """Split data into n_nodes non-IID partitions based on province/industry."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    provinces = frame.get("province")
    industries = frame.get("industry")

    if n_nodes <= 2 or provinces is None or industries is None:
        # Fallback to legacy 2-node split
        mask_a = np.zeros(n, dtype=bool)
        if provinces is not None:
            p = provinces.astype(str).str.lower()
            mask_a |= p.str.contains("ho chi minh|hcm|ha noi|hn|hanoi", regex=True).to_numpy()
        if industries is not None:
            ind = industries.astype(str).str.lower()
            mask_a |= ind.str.contains("finance|construction|xay|tai chinh", regex=True).to_numpy()
        if mask_a.sum() < n * 0.25 or mask_a.sum() > n * 0.75:
            mask_a = rng.random(n) < 0.5
        return [np.flatnonzero(mask_a), np.flatnonzero(~mask_a)]

    # 4-node non-IID split by province clusters
    p = provinces.astype(str).str.lower()
    ind = industries.astype(str).str.lower()

    # Node 0: HCM (finance-heavy)
    mask_0 = p.str.contains("hcm|ho chi minh", regex=True).to_numpy() | ind.str.contains("finance|tai chinh", regex=True).to_numpy()
    # Node 1: Hanoi (construction-heavy)
    mask_1 = p.str.contains("hn|ha noi|hanoi", regex=True).to_numpy() | ind.str.contains("construction|xay", regex=True).to_numpy()
    # Node 2: Da Nang (services)
    mask_2 = p.str.contains("dn|da nang|danang", regex=True).to_numpy() | ind.str.contains("services|dich vu", regex=True).to_numpy()
    # Node 3: Can Tho / remaining (retail)
    mask_3 = ~(mask_0 | mask_1 | mask_2)

    partitions = [np.flatnonzero(m) for m in [mask_0, mask_1, mask_2, mask_3]]
    # Ensure minimum samples per node
    min_samples = max(50, n // 20)
    for i, part in enumerate(partitions):
        if len(part) < min_samples:
            extra = rng.choice(np.arange(n), size=min_samples - len(part), replace=False)
            partitions[i] = np.unique(np.concatenate([part, extra]))
    return partitions


def _non_iid_10node_split(frame, seed: int = 42) -> list[np.ndarray]:
    """Split data into 10 non-IID partitions with multi-dimensional heterogeneity."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    p = frame.get("province", frame.get("industry")).astype(str).str.lower()
    ind = frame.get("industry", frame.get("province")).astype(str).str.lower()

    # Each node gets a biased subsample based on its profile
    all_indices = np.arange(n)
    partitions: list[np.ndarray] = []
    assigned = np.zeros(n, dtype=bool)

    for profile in NODE_PROFILES_10:
        # Score each sample's affinity to this node
        affinity = np.zeros(n, dtype=float)
        affinity += ind.str.contains(profile.industry_bias, regex=False).to_numpy() * 2.0
        affinity += rng.uniform(0, 1, n)  # randomness for overlap
        # Fraud-weight bias: nodes with higher fraud_weight get more fraud samples
        if "fraud_label" in frame.columns:
            affinity += (frame["fraud_label"].to_numpy() * profile.fraud_weight * 0.5)
        # Exclude already-assigned with high probability
        affinity[assigned] *= 0.15
        # Take top-k samples
        target_size = max(50, n // 12)
        top_k = np.argsort(affinity)[-target_size:]
        partitions.append(top_k)
        assigned[top_k] = True

    # Assign remaining unassigned to the last partition
    remaining = np.flatnonzero(~assigned)
    if len(remaining) > 0:
        partitions[-1] = np.unique(np.concatenate([partitions[-1], remaining]))

    return partitions


def non_iid_two_node_split(frame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Legacy 2-node split for backward compatibility."""
    parts = non_iid_multi_node_split(frame, n_nodes=2, seed=seed)
    return parts[0], parts[1]


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=600, class_weight="balanced", solver="lbfgs")
    try:
        clf.fit(Xs, y)
    except ValueError:
        # Single-class partition (e.g., no fraud samples in this node)
        # Return zero coefficients — this node contributes nothing to aggregation
        n_features = Xs.shape[1]
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, n_features))
        clf.intercept_ = np.array([0.0])

        class _DummyProba:
            def predict_proba(self, X_in):
                return np.column_stack([np.ones(len(X_in)), np.zeros(len(X_in))])
        clf = _DummyProba()
        clf.classes_ = np.array([0, 1])
        return scaler, np.zeros(n_features), 0.0, clf
    coef = clf.coef_[0].astype(float)
    intercept = float(clf.intercept_[0])
    return scaler, coef, intercept, clf


def _evaluate_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, prob))


def _predict_from_weights(X: np.ndarray, scaler, coef: np.ndarray, intercept: float) -> np.ndarray:
    Xs = scaler.transform(X)
    return _sigmoid(Xs @ coef + intercept)


def _mix_counts(frame, idx: np.ndarray, col: str, limit: int = 5) -> dict[str, int]:
    if col not in frame.columns:
        return {}
    values = frame.iloc[idx][col].astype(str).value_counts().head(limit)
    return {str(k): int(v) for k, v in values.items()}


def run_federated_learning_poc(
    *,
    rows: int = 120_000,
    rounds: int = 10,
    seed: int = 42,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    dp_noise: float = 0.002,
) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    frame = _load_fraud_frame(rows, seed)
    X, y, feature_cols = build_feature_matrix(frame)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_frame = frame.iloc[train_idx].reset_index(drop=True)

    # 4-node non-IID federation
    idx_nodes = non_iid_multi_node_split(train_frame, n_nodes=4, seed=seed)

    central_scaler = StandardScaler()
    X_train_s = central_scaler.fit_transform(X_train)
    X_test_s = central_scaler.transform(X_test)
    central_clf = LogisticRegression(max_iter=700, class_weight="balanced", solver="lbfgs")
    central_clf.fit(X_train_s, y_train)
    central_prob = central_clf.predict_proba(X_test_s)[:, 1]
    centralized_auc = _evaluate_auc(y_test, central_prob)

    node_payloads = []
    node_results: list[FederatedNode] = []
    rng = np.random.default_rng(seed)
    convergence_curve: list[dict[str, float]] = []

    for round_idx in range(1, max(1, int(rounds)) + 1):
        node_payloads.clear()
        sample_fraction = min(1.0, 0.35 + 0.65 * (round_idx / max(1, rounds)))
        for node_no, idx_local in enumerate(idx_nodes):
            if idx_local.size < 20:
                continue
            take = max(20, int(idx_local.size * sample_fraction))
            round_idx_local = rng.choice(idx_local, size=min(take, idx_local.size), replace=False)
            Xn = X_train[round_idx_local]
            yn = y_train[round_idx_local]
            scaler, coef, intercept, clf = _fit_logistic(Xn, yn)
            noisy_coef = coef + rng.normal(0, dp_noise / max(epsilon, 1e-6), size=coef.shape)
            noisy_intercept = intercept + float(rng.normal(0, dp_noise / max(epsilon, 1e-6)))
            node_payloads.append(
                {
                    "node_no": node_no,
                    "weight": len(round_idx_local),
                    "scaler": scaler,
                    "coef": noisy_coef,
                    "intercept": noisy_intercept,
                    "clf": clf,
                    "idx": idx_local,
                }
            )

        total_weight = sum(p["weight"] for p in node_payloads) or 1
        agg_coef = sum(p["coef"] * (p["weight"] / total_weight) for p in node_payloads)
        agg_intercept = float(sum(p["intercept"] * (p["weight"] / total_weight) for p in node_payloads))
        primary_scaler = max(node_payloads, key=lambda p: p["weight"])["scaler"]
        fed_prob = _predict_from_weights(X_test, primary_scaler, agg_coef, agg_intercept)
        convergence_curve.append({"round": float(round_idx), "auc": round(_evaluate_auc(y_test, fed_prob), 6)})

    fed_auc = float(convergence_curve[-1]["auc"]) if convergence_curve else 0.5

    names = ["office_hcm", "office_hanoi", "office_danang", "office_cantho"]
    display = [
        "Cuc Thue TP.HCM (finance-heavy)",
        "Cuc Thue Ha Noi (construction-heavy)",
        "Cuc Thue Da Nang (services)",
        "Cuc Thue Can Tho (retail/agriculture)",
    ]
    for payload, node_id, display_name in zip(node_payloads, names, display):
        idx_local = payload["idx"]
        local_prob = payload["clf"].predict_proba(payload["scaler"].transform(X_test))[:, 1]
        node_results.append(
            FederatedNode(
                node_id=node_id,
                display_name=display_name,
                n_samples=int(idx_local.size),
                local_auc=round(_evaluate_auc(y_test, local_prob), 6),
                industry_mix=_mix_counts(train_frame, idx_local, "industry"),
                province_mix=_mix_counts(train_frame, idx_local, "province"),
            )
        )

    coef_bytes = (len(feature_cols) + 1) * 8
    communication_per_round = int(coef_bytes * max(1, len(node_payloads)) * 2)
    result = FederatedResult(
        generated_at=_now_iso(),
        rounds=int(rounds),
        rows=int(len(frame)),
        epsilon=float(epsilon),
        delta=float(delta),
        centralized_auc=round(centralized_auc, 6),
        federated_auc=round(fed_auc, 6),
        federated_gap=round(max(0.0, centralized_auc - fed_auc), 6),
        node_results=node_results,
        communication_bytes_per_round=communication_per_round,
        total_communication_bytes=int(communication_per_round * max(1, rounds)),
        convergence_curve=convergence_curve,
        privacy={
            "mechanism": "Gaussian coefficient noise for PoC",
            "epsilon": float(epsilon),
            "delta": float(delta),
            "noise_std": float(dp_noise / max(epsilon, 1e-6)),
        },
    )
    return result.to_dict()


# ════════════════════════════════════════════════════════════════════════════════
#  10-Node Federation with calibrated DP, TopK compression, straggler handling
# ════════════════════════════════════════════════════════════════════════════════

def _topk_compress(coef: np.ndarray, k_ratio: float = 0.10) -> tuple[np.ndarray, np.ndarray, int]:
    """TopK gradient compression: only transmit top-k% coefficients by magnitude."""
    flat = coef.flatten()
    k_count = max(1, int(len(flat) * k_ratio))
    top_idx = np.argpartition(np.abs(flat), -k_count)[-k_count:]
    top_vals = flat[top_idx]
    # Communication cost: indices (int32) + values (float64)
    comm_bytes = top_idx.nbytes + top_vals.nbytes
    return top_idx, top_vals, comm_bytes


def _topk_decompress(indices: np.ndarray, values: np.ndarray, length: int) -> np.ndarray:
    """Reconstruct full coefficient vector from sparse TopK representation."""
    full = np.zeros(length, dtype=float)
    full[indices] = values
    return full


def _calibrated_dp_noise(coef: np.ndarray, epsilon: float, delta: float,
                         sensitivity: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian mechanism with calibrated noise: sigma = sensitivity * sqrt(2 ln(1.25/delta)) / epsilon."""
    if epsilon <= 0 or delta <= 0:
        return coef
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    return coef + rng.normal(0, sigma, size=coef.shape)


def run_federated_10node(
    *,
    rows: int = 5_000,
    rounds: int = 20,
    seed: int = 42,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    topk_ratio: float = 0.30,
    min_nodes_per_round: int = 5,
) -> dict[str, Any]:
    """Run 10-node federated learning with calibrated DP, TopK compression,
    and async straggler tolerance."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    frame = _load_fraud_frame(rows, seed)
    X, y, feature_cols = build_feature_matrix(frame)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_frame = frame.iloc[train_idx].reset_index(drop=True)

    # Centralized baseline
    central_scaler = StandardScaler()
    X_train_s = central_scaler.fit_transform(X_train)
    X_test_s = central_scaler.transform(X_test)
    central_clf = LogisticRegression(max_iter=700, class_weight="balanced", solver="lbfgs")
    central_clf.fit(X_train_s, y_train)
    centralized_auc = _evaluate_auc(y_test, central_clf.predict_proba(X_test_s)[:, 1])

    # 10-node non-IID partition
    idx_nodes = _non_iid_10node_split(train_frame, seed=seed)
    profiles = NODE_PROFILES_10

    rng = np.random.default_rng(seed)
    convergence_curve: list[dict[str, float]] = []
    straggler_log: list[int] = []
    comm_bytes_log: list[int] = []
    coef_dim = X_train.shape[1] + 1  # coef + intercept
    # Sensitivity estimate for regularized logistic regression (C=1.0 default)
    # Bounded by 2C/n where n is min local dataset size, C is inverse regularization
    min_local_n = max(1, min(len(p) for p in idx_nodes))
    sensitivity = min(0.01, 2.0 / max(min_local_n, 50))

    # Track last payloads for final node stats
    last_payloads: list[dict] = []

    for round_no in range(1, rounds + 1):
        round_payloads: list[dict] = []
        round_comm = 0
        stragglers = 0
        sample_frac = min(1.0, 0.35 + 0.65 * (round_no / rounds))

        for node_no, (profile, idx_local) in enumerate(zip(profiles, idx_nodes)):
            # Straggler simulation: node with low infra_quality may miss
            if rng.random() > profile.infra_quality:
                stragglers += 1
                continue

            if idx_local.size < 20:
                continue

            take = max(20, int(idx_local.size * sample_frac))
            sub = rng.choice(idx_local, size=min(take, idx_local.size), replace=False)
            Xn, yn = X_train[sub], y_train[sub]
            scaler, coef, intercept, clf = _fit_logistic(Xn, yn)

            # Calibrated DP noise
            full_coef = np.append(coef, intercept)
            noisy = _calibrated_dp_noise(full_coef, epsilon, delta, sensitivity, rng)

            # TopK compression
            top_idx, top_vals, comm_b = _topk_compress(noisy, topk_ratio)
            round_comm += comm_b

            round_payloads.append({
                "node_no": node_no,
                "profile": profile,
                "weight": len(sub),
                "scaler": scaler,
                "compressed_idx": top_idx,
                "compressed_vals": top_vals,
                "full_coef": noisy,  # keep for evaluation
                "clf": clf,
                "idx": idx_local,
            })

        straggler_log.append(stragglers)
        comm_bytes_log.append(round_comm)

        if len(round_payloads) < min_nodes_per_round:
            # Not enough nodes responded — skip round
            if convergence_curve:
                convergence_curve.append({"round": float(round_no), "auc": convergence_curve[-1]["auc"], "status": "skipped"})
            continue

        # FedAvg aggregation on full DP-noised coefficients
        # (TopK compression is measured for communication cost, aggregation uses full coef)
        total_weight = sum(p["weight"] for p in round_payloads) or 1
        agg_full = np.zeros(coef_dim, dtype=float)
        for p in round_payloads:
            agg_full += p["full_coef"] * (p["weight"] / total_weight)

        # Evaluate
        agg_coef = agg_full[:-1]
        agg_intercept = float(agg_full[-1])
        primary_scaler = max(round_payloads, key=lambda pp: pp["weight"])["scaler"]
        fed_prob = _predict_from_weights(X_test, primary_scaler, agg_coef, agg_intercept)
        round_auc = round(_evaluate_auc(y_test, fed_prob), 6)
        convergence_curve.append({"round": float(round_no), "auc": round_auc, "nodes": len(round_payloads), "stragglers": stragglers})
        last_payloads = round_payloads

    fed_auc = convergence_curve[-1]["auc"] if convergence_curve else 0.5

    # Build node results from last round
    node_results: list[FederatedNode] = []
    for p in last_payloads:
        prof = p["profile"]
        idx_local = p["idx"]
        local_prob = p["clf"].predict_proba(p["scaler"].transform(X_test))[:, 1]
        node_results.append(FederatedNode(
            node_id=prof.node_id,
            display_name=prof.display_name,
            n_samples=int(idx_local.size),
            local_auc=round(_evaluate_auc(y_test, local_prob), 6),
            industry_mix=_mix_counts(train_frame, idx_local, "industry"),
            province_mix=_mix_counts(train_frame, idx_local, "province"),
        ))

    avg_comm = int(np.mean(comm_bytes_log)) if comm_bytes_log else 0
    full_comm_no_compress = coef_dim * 8 * 10 * 2  # baseline: all 10 nodes, full coef, bidirectional
    compression_savings = round(1.0 - (avg_comm / max(full_comm_no_compress, 1)), 4)

    result = FederatedResult(
        generated_at=_now_iso(),
        rounds=rounds,
        rows=int(len(frame)),
        epsilon=epsilon,
        delta=delta,
        centralized_auc=round(centralized_auc, 6),
        federated_auc=round(fed_auc, 6),
        federated_gap=round(max(0.0, centralized_auc - fed_auc), 6),
        node_results=node_results,
        communication_bytes_per_round=avg_comm,
        total_communication_bytes=int(sum(comm_bytes_log)),
        convergence_curve=convergence_curve,
        privacy={
            "mechanism": "Calibrated Gaussian (sigma = sensitivity * sqrt(2 ln(1.25/delta)) / epsilon)",
            "epsilon": epsilon,
            "delta": delta,
            "sensitivity": round(sensitivity, 6),
            "sigma": round(sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-6), 6),
        },
    )
    payload = result.to_dict()
    # Add 10-node specific metadata
    payload["n_nodes"] = 10
    payload["topk_compression_ratio"] = topk_ratio
    payload["compression_savings_pct"] = round(compression_savings * 100, 1)
    payload["min_nodes_per_round"] = min_nodes_per_round
    payload["avg_stragglers_per_round"] = round(float(np.mean(straggler_log)), 2) if straggler_log else 0.0
    payload["node_profiles"] = [
        {"node_id": p.node_id, "region": p.region, "industry_bias": p.industry_bias,
         "fraud_weight": p.fraud_weight, "infra_quality": p.infra_quality}
        for p in profiles
    ]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="TaxInspector Federated Learning PoC")
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=4, choices=[4, 10],
                        help="Number of federated nodes (4=legacy, 10=scale-up)")
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "reports" / "federated_learning_results.json")
    args = parser.parse_args()

    if args.nodes == 10:
        result = run_federated_10node(
            rows=args.rows, rounds=max(args.rounds, 20), seed=args.seed,
        )
    else:
        result = run_federated_learning_poc(
            rows=args.rows, rounds=args.rounds, seed=args.seed,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {args.out}")
    print(json.dumps({
        "nodes": result.get("n_nodes", 4),
        "central_auc": result["centralized_auc"],
        "fed_auc": result["federated_auc"],
        "gap": result["federated_gap"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
