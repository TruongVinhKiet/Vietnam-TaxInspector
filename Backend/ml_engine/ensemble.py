"""
ensemble.py – Intelligent Ensemble: GNN + Rules + Anomaly Detector
===================================================================
Architecture:
    GNN_Score ──┐
    Rule_Score ──┼── Meta-Model (LogisticRegression) ──→ Final Risk
    Anomaly_Score┘

Components:
    1. HeuristicRuleScorer: deterministic business rules (age, cycles, reciprocal)
    2. AnomalyDetector: Isolation Forest on node/edge feature space
    3. EnsembleModel: Meta-learner combining all signal sources

Design:
    - Each component outputs a [0,1] probability independently
    - Meta-model learns optimal weights from validation data
    - Fallback: if any component fails, others still produce a score
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import date, timedelta
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ════════════════════════════════════════════════════════════════
#  1. Heuristic Rule Scorer
# ════════════════════════════════════════════════════════════════

class HeuristicRuleScorer:
    """
    Deterministic business rules for fraud scoring.
    
    Rules encode domain knowledge from Vietnamese tax inspectors:
    - Young companies (< 1 year) are suspicious
    - Companies in detected cycles are suspicious  
    - High reciprocal trade ratio is suspicious
    - Very high invoice amounts relative to company age
    """

    def score_node(self, company: dict, cycle_nodes: set, 
                   in_amount: float, out_amount: float, degree: int) -> float:
        """Score a single company node. Returns [0, 1]."""
        score = 0.0

        # Rule 1: Company age tiers (newer companies are riskier)
        reg = company.get("registration_date")
        if reg:
            if isinstance(reg, str):
                reg = date.fromisoformat(reg)
            age_days = (date.today() - reg).days
            if age_days < 180:
                score += 0.35
            elif age_days < 365:
                score += 0.25

        # Rule 2: In a detected cycle → +0.30
        if company["tax_code"] in cycle_nodes:
            score += 0.30

        # Rule 3: Suspicious in/out ratio (near 1:1 with high volume)
        total = in_amount + out_amount
        if total > 0:
            ratio = min(in_amount, out_amount) / max(1, max(in_amount, out_amount))
            if ratio > 0.8 and total > 1_000_000_000:  # > 1 billion VND
                score += 0.20

        # Rule 4: Low degree + high volume → shell indicator
        if degree <= 3 and total > 2_000_000_000:
            score += 0.15

        return min(1.0, score)

    def score_edge(self, invoice: dict, cycle_edges: set, 
                   reciprocal_pairs: set) -> float:
        """Score a single edge/invoice. Returns [0, 1]."""
        score = 0.0
        s, b = invoice["seller_tax_code"], invoice["buyer_tax_code"]

        # Rule 1: Part of a detected cycle → +0.40
        if (s, b) in cycle_edges:
            score += 0.40

        # Rule 2: Has reciprocal trade → +0.15
        if (b, s) in reciprocal_pairs:
            score += 0.15

        # Rule 3: Round amount (potential fabrication)
        amount = float(invoice["amount"])
        if amount > 100_000_000 and amount % 1_000_000 == 0:
            score += 0.10

        # Rule 4: Large amount → +scaled
        if amount > 1_000_000_000:
            score += 0.15
        elif amount > 500_000_000:
            score += 0.10

        return min(1.0, score)


# ════════════════════════════════════════════════════════════════
#  2. Anomaly Detector (Isolation Forest)
# ════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Isolation Forest for unsupervised anomaly detection.
    
    Trained on node features to identify statistical outliers
    that don't fit normal company profiles, regardless of labels.
    """

    def __init__(self):
        self.node_forest = None
        self.edge_forest = None
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self._fitted = False

    def fit(self, node_features: np.ndarray, edge_features: np.ndarray):
        """Fit isolation forests on feature matrices."""
        # Node anomaly detector
        if len(node_features) >= 10:
            self.node_scaler.fit(node_features)
            scaled_nodes = self.node_scaler.transform(node_features)
            self.node_forest = IsolationForest(
                n_estimators=100, contamination=0.15,
                random_state=42, n_jobs=-1
            )
            self.node_forest.fit(scaled_nodes)

        # Edge anomaly detector
        if len(edge_features) >= 10:
            self.edge_scaler.fit(edge_features)
            scaled_edges = self.edge_scaler.transform(edge_features)
            self.edge_forest = IsolationForest(
                n_estimators=100, contamination=0.10,
                random_state=42, n_jobs=-1
            )
            self.edge_forest.fit(scaled_edges)

        self._fitted = True

    def score_nodes(self, node_features: np.ndarray) -> np.ndarray:
        """Return anomaly scores [0, 1] for each node. Higher = more anomalous."""
        if not self._fitted or self.node_forest is None:
            return np.full(len(node_features), 0.5)
        
        scaled = self.node_scaler.transform(node_features)
        # decision_function returns negative for anomalies
        raw_scores = -self.node_forest.decision_function(scaled)
        # Normalize to [0, 1]
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            return (raw_scores - min_s) / (max_s - min_s)
        return np.full(len(node_features), 0.5)

    def score_edges(self, edge_features: np.ndarray) -> np.ndarray:
        """Return anomaly scores [0, 1] for each edge. Higher = more anomalous."""
        if not self._fitted or self.edge_forest is None:
            return np.full(len(edge_features), 0.5)
        
        scaled = self.edge_scaler.transform(edge_features)
        raw_scores = -self.edge_forest.decision_function(scaled)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            return (raw_scores - min_s) / (max_s - min_s)
        return np.full(len(edge_features), 0.5)

    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else MODEL_DIR
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "anomaly_detector.pkl", "wb") as f:
            pickle.dump({
                "node_forest": self.node_forest,
                "edge_forest": self.edge_forest,
                "node_scaler": self.node_scaler,
                "edge_scaler": self.edge_scaler,
                "fitted": self._fitted,
            }, f)
        print(f"[OK] Anomaly detector saved to {save_path / 'anomaly_detector.pkl'}")

    def load(self, path: Optional[str] = None) -> bool:
        load_path = Path(path) if path else MODEL_DIR
        pkl_path = load_path / "anomaly_detector.pkl"
        if not pkl_path.exists():
            return False
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.node_forest = data["node_forest"]
            self.edge_forest = data["edge_forest"]
            self.node_scaler = data["node_scaler"]
            self.edge_scaler = data["edge_scaler"]
            self._fitted = data["fitted"]
            print(f"[OK] Anomaly detector loaded from {pkl_path}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load anomaly detector: {e}")
            return False


# ════════════════════════════════════════════════════════════════
#  3. Meta-Model Ensemble
# ════════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    Meta-learner that combines GNN, Rule, and Anomaly scores.
    
    Uses Logistic Regression to learn optimal weights:
    Final_Risk = σ(w1·GNN + w2·Rules + w3·Anomaly + b)
    
    Trained on validation set to avoid overfitting to training data.
    """

    def __init__(self):
        self.node_meta = LogisticRegression(max_iter=1000, random_state=42)
        self.edge_meta = LogisticRegression(max_iter=1000, random_state=42)
        self._fitted = False

    def fit(self, node_scores: np.ndarray, node_labels: np.ndarray,
            edge_scores: np.ndarray, edge_labels: np.ndarray):
        """
        Fit meta-models.
        
        Args:
            node_scores: shape (n_nodes, 3) — [gnn, rule, anomaly] per node
            node_labels: shape (n_nodes,) — binary
            edge_scores: shape (n_edges, 3) — [gnn, rule, anomaly] per edge
            edge_labels: shape (n_edges,) — binary
        """
        # Node meta-model
        if len(np.unique(node_labels)) >= 2 and len(node_labels) >= 5:
            self.node_meta.fit(node_scores, node_labels)
        
        # Edge meta-model
        if len(np.unique(edge_labels)) >= 2 and len(edge_labels) >= 5:
            self.edge_meta.fit(edge_scores, edge_labels)

        self._fitted = True

    def predict_node(self, gnn_score: float, rule_score: float, anomaly_score: float) -> float:
        """Predict final node risk. Returns [0, 1]."""
        if not self._fitted:
            return gnn_score * 0.5 + rule_score * 0.3 + anomaly_score * 0.2
        X = np.array([[gnn_score, rule_score, anomaly_score]])
        return float(self.node_meta.predict_proba(X)[0, 1])

    def predict_edge(self, gnn_score: float, rule_score: float, anomaly_score: float) -> float:
        """Predict final edge risk. Returns [0, 1]."""
        if not self._fitted:
            return gnn_score * 0.5 + rule_score * 0.3 + anomaly_score * 0.2
        X = np.array([[gnn_score, rule_score, anomaly_score]])
        return float(self.edge_meta.predict_proba(X)[0, 1])

    def predict_nodes_batch(self, scores: np.ndarray) -> np.ndarray:
        """Batch predict for nodes. scores shape: (n, 3)."""
        if not self._fitted:
            return scores[:, 0] * 0.5 + scores[:, 1] * 0.3 + scores[:, 2] * 0.2
        return self.node_meta.predict_proba(scores)[:, 1]

    def predict_edges_batch(self, scores: np.ndarray) -> np.ndarray:
        """Batch predict for edges. scores shape: (n, 3)."""
        if not self._fitted:
            return scores[:, 0] * 0.5 + scores[:, 1] * 0.3 + scores[:, 2] * 0.2
        return self.edge_meta.predict_proba(scores)[:, 1]

    def get_weights(self) -> dict:
        """Extract learned feature importances."""
        result = {}
        if hasattr(self.node_meta, 'coef_'):
            w = self.node_meta.coef_[0]
            result["node_weights"] = {
                "gnn": round(float(w[0]), 4),
                "rules": round(float(w[1]), 4),
                "anomaly": round(float(w[2]), 4),
            }
        if hasattr(self.edge_meta, 'coef_'):
            w = self.edge_meta.coef_[0]
            result["edge_weights"] = {
                "gnn": round(float(w[0]), 4),
                "rules": round(float(w[1]), 4),
                "anomaly": round(float(w[2]), 4),
            }
        return result

    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else MODEL_DIR
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "ensemble_meta.pkl", "wb") as f:
            pickle.dump({
                "node_meta": self.node_meta,
                "edge_meta": self.edge_meta,
                "fitted": self._fitted,
            }, f)
        print(f"[OK] Ensemble meta-model saved to {save_path / 'ensemble_meta.pkl'}")

    def load(self, path: Optional[str] = None) -> bool:
        load_path = Path(path) if path else MODEL_DIR
        pkl_path = load_path / "ensemble_meta.pkl"
        if not pkl_path.exists():
            return False
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.node_meta = data["node_meta"]
            self.edge_meta = data["edge_meta"]
            self._fitted = data["fitted"]
            print(f"[OK] Ensemble meta-model loaded from {pkl_path}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load ensemble: {e}")
            return False


# ════════════════════════════════════════════════════════════════
#  4. Path-Level Evidence Extractor
# ════════════════════════════════════════════════════════════════

class PathEvidenceExtractor:
    """
    Extract connected fraud chains as structured investigation evidence.
    
    Instead of flagging individual nodes/edges, this produces:
    "A sold 5B to B on Jan 11 → B sold 5B to C on Jan 12 → C sold back to A on Jan 13"
    
    Output: List of path objects with per-hop details.
    """

    def extract_paths(self, cycles: list, invoices: list, 
                      edge_probs: dict, node_probs: dict) -> list:
        """
        Extract path-level evidence from detected cycles.
        
        Returns:
            List of evidence paths, each containing:
            - path_id: unique identifier
            - companies: ordered list of tax_codes in the ring
            - hops: detailed per-hop information
            - total_amount: sum of all hop amounts
            - avg_probability: mean fraud probability across hops
            - time_span_hours: total time from first to last hop
            - risk_level: "critical" / "high" / "medium"
        """
        # Build invoice lookup: (seller, buyer) -> list of invoices
        pair_invoices = {}
        for inv in invoices:
            key = (inv["seller_tax_code"], inv["buyer_tax_code"])
            pair_invoices.setdefault(key, []).append(inv)

        evidence_paths = []

        for path_idx, cycle in enumerate(cycles):
            hops = []
            total_amount = 0
            all_dates = []
            hop_probs = []

            for i in range(len(cycle)):
                seller = cycle[i]
                buyer = cycle[(i + 1) % len(cycle)]

                # Find the most suspicious invoice for this hop
                candidates = pair_invoices.get((seller, buyer), [])
                if not candidates:
                    continue

                # Pick the one with highest fraud probability
                best_inv = None
                best_prob = -1
                for inv in candidates:
                    inv_num = inv.get("invoice_number", "")
                    prob = edge_probs.get(inv_num, 0.0)
                    if prob > best_prob:
                        best_prob = prob
                        best_inv = inv

                if best_inv is None:
                    continue

                inv_date = best_inv["date"]
                if isinstance(inv_date, str):
                    inv_date = date.fromisoformat(inv_date)
                all_dates.append(inv_date)

                amount = float(best_inv["amount"])
                total_amount += amount
                hop_probs.append(best_prob)

                hops.append({
                    "from": seller,
                    "to": buyer,
                    "invoice_number": best_inv.get("invoice_number", ""),
                    "amount": amount,
                    "amount_formatted": f"{amount:,.0f} VNĐ",
                    "date": str(inv_date),
                    "fraud_probability": round(best_prob, 4),
                    "seller_shell_prob": round(node_probs.get(seller, 0), 4),
                    "buyer_shell_prob": round(node_probs.get(buyer, 0), 4),
                })

            if not hops:
                continue

            # Compute path-level metrics
            avg_prob = sum(hop_probs) / len(hop_probs) if hop_probs else 0
            if len(all_dates) >= 2:
                time_span = (max(all_dates) - min(all_dates)).days * 24
            else:
                time_span = 0

            risk_level = "critical" if avg_prob > 0.7 else ("high" if avg_prob > 0.4 else "medium")

            evidence_paths.append({
                "path_id": f"RING-{path_idx + 1:03d}",
                "companies": cycle,
                "num_hops": len(hops),
                "hops": hops,
                "total_amount": total_amount,
                "total_amount_formatted": f"{total_amount:,.0f} VNĐ",
                "avg_probability": round(avg_prob, 4),
                "time_span_hours": time_span,
                "risk_level": risk_level,
                "summary": self._format_summary(cycle, hops, time_span),
            })

        # Sort by risk (most dangerous first)
        evidence_paths.sort(key=lambda p: -p["avg_probability"])
        return evidence_paths

    def _format_summary(self, cycle, hops, time_span_hours) -> str:
        """Generate human-readable Vietnamese summary."""
        chain = " → ".join(cycle[:4])
        if len(cycle) > 4:
            chain += f" → ... → {cycle[0]}"
        else:
            chain += f" → {cycle[0]}"

        total = sum(h["amount"] for h in hops)
        days = time_span_hours / 24 if time_span_hours > 0 else 0

        return (
            f"Chuỗi xoay vòng {len(cycle)} chủ thể: {chain}. "
            f"Tổng giá trị: {total:,.0f} VNĐ. "
            f"Diễn ra trong {days:.0f} ngày."
        )
