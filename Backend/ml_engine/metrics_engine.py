"""
metrics_engine.py – Enterprise Evaluation & Calibration Engine
===============================================================
Responsibilities:
    1. Temporal Mask Generation (Train / Val / Test by invoice date)
    2. Multi-metric Evaluation (F1, PR-AUC, Recall@K, FPR-by-industry)
    3. Probability Calibration (Isotonic Regression on Val set)
    4. Evaluation Report Generation

Design Principles:
    - ALL evaluation runs in model.eval() mode (no dropout noise)
    - Edge masks are derived from invoice dates, not random splits
    - Node masks are derived from their participation window
    - Calibrator is persisted alongside the model for inference use
"""

import json
import pickle
import os
import numpy as np
import torch
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.isotonic import IsotonicRegression

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ════════════════════════════════════════════════════════════════
#  1. Temporal Mask Generator
# ════════════════════════════════════════════════════════════════

class TemporalMaskGenerator:
    """
    Generate train/val/test masks based on temporal cutoffs.

    Strategy:
        - Edges (invoices) are always split by invoice date.
        - Nodes can use either:
            * stratified random split (default, transductive-friendly)
            * strict temporal first-seen split
        - Strict temporal node masks are always generated for transparent reporting.

    Default split (configurable):
        Train:  T1–T8   (Jan–Aug)
        Val:    T9       (Sep)
        Test:   T10–T12  (Oct–Dec)
    """

    def __init__(
        self,
        train_end_month: int = 8,   # inclusive
        val_end_month: int = 9,     # inclusive
        base_year: int = 2024,
        node_split_strategy: str = "stratified",
    ):
        self.base_year = base_year
        self.train_cutoff = date(base_year, train_end_month + 1, 1)  # first day of month after train
        self.val_cutoff = date(base_year, val_end_month + 1, 1)      # first day of month after val
        strategy = node_split_strategy.strip().lower()
        if strategy not in {"stratified", "temporal"}:
            raise ValueError("node_split_strategy must be one of: stratified, temporal")
        self.node_split_strategy = strategy

    @staticmethod
    def _to_date(value):
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        raise ValueError(f"Unsupported date value: {value!r}")

    def _generate_stratified_node_masks(self, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Default node split for transductive GNN: random 60/20/20.
        Ratios can be overridden by env vars:
            - GNN_NODE_TRAIN_RATIO
            - GNN_NODE_VAL_RATIO
        """
        from sklearn.model_selection import train_test_split

        node_train = torch.zeros(n_nodes, dtype=torch.bool)
        node_val = torch.zeros(n_nodes, dtype=torch.bool)
        node_test = torch.zeros(n_nodes, dtype=torch.bool)

        if n_nodes == 0:
            return node_train, node_val, node_test

        train_ratio = float(os.getenv("GNN_NODE_TRAIN_RATIO", "0.60"))
        val_ratio = float(os.getenv("GNN_NODE_VAL_RATIO", "0.20"))
        # Keep ratios sane even with misconfigured env.
        train_ratio = min(max(train_ratio, 0.50), 0.85)
        val_ratio = min(max(val_ratio, 0.05), 0.35)
        if train_ratio + val_ratio >= 0.95:
            val_ratio = 0.95 - train_ratio
        test_ratio = max(0.05, 1.0 - train_ratio - val_ratio)

        all_indices = np.arange(n_nodes)
        try:
            train_idx, temp_idx = train_test_split(
                all_indices, test_size=(1.0 - train_ratio), random_state=42
            )
            val_portion_in_temp = val_ratio / max(1e-6, (val_ratio + test_ratio))
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=(1.0 - val_portion_in_temp), random_state=42
            )
        except ValueError:
            train_end = int(n_nodes * train_ratio)
            val_end = int(n_nodes * (train_ratio + val_ratio))
            train_idx = all_indices[:train_end]
            val_idx = all_indices[train_end:val_end]
            test_idx = all_indices[val_end:]

        node_train[train_idx] = True
        node_val[val_idx] = True
        node_test[test_idx] = True
        return node_train, node_val, node_test

    def _generate_temporal_node_masks(self, invoices: list[dict], tc_to_idx: dict[str, int], n_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Strict temporal split by first-seen timestamp of each node in invoice stream.
        """
        node_train = torch.zeros(n_nodes, dtype=torch.bool)
        node_val = torch.zeros(n_nodes, dtype=torch.bool)
        node_test = torch.zeros(n_nodes, dtype=torch.bool)

        first_seen: dict[str, date] = {}
        for inv in invoices:
            try:
                d = self._to_date(inv.get("date"))
            except Exception:
                continue

            seller = inv.get("seller_tax_code")
            buyer = inv.get("buyer_tax_code")
            if seller in tc_to_idx and seller not in first_seen:
                first_seen[seller] = d
            if buyer in tc_to_idx and buyer not in first_seen:
                first_seen[buyer] = d

        for tc, idx in tc_to_idx.items():
            first_d = first_seen.get(tc)
            if first_d is None:
                # Unseen nodes are treated as historical/background entities.
                node_train[idx] = True
            elif first_d < self.train_cutoff:
                node_train[idx] = True
            elif first_d < self.val_cutoff:
                node_val[idx] = True
            else:
                node_test[idx] = True

        # Keep train non-empty for stable optimization in edge cases.
        if n_nodes > 0 and int(node_train.sum()) == 0:
            node_train[0] = True
            node_val[0] = False
            node_test[0] = False

        return node_train, node_val, node_test

    def generate_masks(
        self,
        invoices: list[dict],
        companies: list[dict],
        tc_to_idx: dict[str, int],
        n_nodes: int,
        n_edges: int,
    ) -> dict:
        """
        Generate temporal masks for nodes and edges.

        Returns:
            {
                "node_train_mask": Tensor[bool],
                "node_val_mask": Tensor[bool],
                "node_test_mask": Tensor[bool],
                "edge_train_mask": Tensor[bool],
                "edge_val_mask": Tensor[bool],
                "edge_test_mask": Tensor[bool],
                "split_info": {...}
            }
        """
        # ── Edge masks by date ──
        edge_train = torch.zeros(n_edges, dtype=torch.bool)
        edge_val = torch.zeros(n_edges, dtype=torch.bool)
        edge_test = torch.zeros(n_edges, dtype=torch.bool)

        for i, inv in enumerate(invoices):
            if i >= n_edges:
                break
            d = inv["date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            if d < self.train_cutoff:
                edge_train[i] = True
            elif d < self.val_cutoff:
                edge_val[i] = True
            else:
                edge_test[i] = True

        # ── Node masks ──
        # We always generate strict temporal masks for honest time-based reporting.
        node_temporal_train, node_temporal_val, node_temporal_test = self._generate_temporal_node_masks(
            invoices=invoices,
            tc_to_idx=tc_to_idx,
            n_nodes=n_nodes,
        )

        # Selected operational strategy (default: stratified for transductive training).
        if self.node_split_strategy == "temporal":
            node_train, node_val, node_test = (
                node_temporal_train,
                node_temporal_val,
                node_temporal_test,
            )
        else:
            node_train, node_val, node_test = self._generate_stratified_node_masks(n_nodes)

        split_info = {
            "edge_train": int(edge_train.sum()),
            "edge_val": int(edge_val.sum()),
            "edge_test": int(edge_test.sum()),
            "node_train": int(node_train.sum()),
            "node_val": int(node_val.sum()),
            "node_test": int(node_test.sum()),
            "node_temporal_train": int(node_temporal_train.sum()),
            "node_temporal_val": int(node_temporal_val.sum()),
            "node_temporal_test": int(node_temporal_test.sum()),
            "node_split_strategy": self.node_split_strategy,
            "train_cutoff": str(self.train_cutoff),
            "val_cutoff": str(self.val_cutoff),
        }

        return {
            "node_train_mask": node_train,
            "node_val_mask": node_val,
            "node_test_mask": node_test,
            "node_temporal_train_mask": node_temporal_train,
            "node_temporal_val_mask": node_temporal_val,
            "node_temporal_test_mask": node_temporal_test,
            "edge_train_mask": edge_train,
            "edge_val_mask": edge_val,
            "edge_test_mask": edge_test,
            "split_info": split_info,
        }


# ════════════════════════════════════════════════════════════════
#  2. Comprehensive Evaluator
# ════════════════════════════════════════════════════════════════

class GNNEvaluator:
    """
    Evaluate a GNN model with production-grade metrics.

    ALL evaluation runs with model.eval() to disable dropout/batchnorm noise.
    """

    def __init__(self, model, data, node_labels, edge_labels, companies=None, tc_to_idx=None):
        self.model = model
        self.data = data
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.companies = companies or []
        self.tc_to_idx = tc_to_idx or {}

    @torch.no_grad()
    def evaluate(
        self,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        split_name: str = "test",
        top_k: int = 20,
    ) -> dict:
        """
        Run full evaluation on a masked subset.

        Returns dict with:
            - node_f1, node_pr_auc, node_roc_auc
            - edge_f1, edge_pr_auc, edge_roc_auc
            - recall_at_k (top-K node alerts)
            - fpr_by_industry
            - raw probabilities for calibration
        """
        self.model.eval()  # CRITICAL: disable dropout

        out = self.model(self.data)
        node_probs = torch.sigmoid(out["node_logits"]).numpy()
        edge_probs = torch.sigmoid(out["edge_logits"]).numpy()

        # ── Node metrics ──
        nm = node_mask.numpy()
        nl = self.node_labels[node_mask].numpy()
        np_masked = node_probs[nm]
        n_pred = (np_masked > 0.5).astype(int)

        if len(nl) > 0 and len(np.unique(nl)) >= 2:
            node_metrics = self._compute_metrics(nl, np_masked, n_pred, "node")
        else:
            node_metrics = {f"node_{m}": 0.0 for m in ["f1","precision","recall","pr_auc","roc_auc","fpr"]}

        # ── Edge metrics ──
        em = edge_mask.numpy()
        el = self.edge_labels[edge_mask].numpy()
        ep_masked = edge_probs[em]
        e_pred = (ep_masked > 0.5).astype(int)

        if len(el) > 0 and len(np.unique(el)) >= 2:
            edge_metrics = self._compute_metrics(el, ep_masked, e_pred, "edge")
        else:
            edge_metrics = {f"edge_{m}": 0.0 for m in ["f1","precision","recall","pr_auc","roc_auc","fpr"]}

        # ── Recall@K (top-K suspicious nodes) ──
        recall_at_k = self._recall_at_k(
            self.node_labels.numpy(), node_probs, top_k
        )

        # ── FPR by industry ──
        fpr_by_industry = self._fpr_by_industry(node_probs)

        result = {
            "split": split_name,
            **node_metrics,
            **edge_metrics,
            "recall_at_k": recall_at_k,
            "top_k": top_k,
            "fpr_by_industry": fpr_by_industry,
            # Raw outputs for calibration
            "_node_probs": np_masked,
            "_node_labels": nl,
            "_edge_probs": ep_masked,
            "_edge_labels": el,
        }

        self.model.train()  # restore training mode
        return result

    def _compute_metrics(self, labels, probs, preds, prefix: str) -> dict:
        """Compute F1, PR-AUC, ROC-AUC for a binary task."""
        metrics = {}

        # F1
        metrics[f"{prefix}_f1"] = float(f1_score(labels, preds, zero_division=0))
        metrics[f"{prefix}_precision"] = float(precision_score(labels, preds, zero_division=0))
        metrics[f"{prefix}_recall"] = float(recall_score(labels, preds, zero_division=0))

        # PR-AUC (most important for imbalanced data)
        try:
            metrics[f"{prefix}_pr_auc"] = float(average_precision_score(labels, probs))
        except ValueError:
            metrics[f"{prefix}_pr_auc"] = 0.0

        # ROC-AUC
        try:
            metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            metrics[f"{prefix}_roc_auc"] = 0.0

        # Confusion matrix
        if len(np.unique(labels)) >= 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            metrics[f"{prefix}_tp"] = int(tp)
            metrics[f"{prefix}_fp"] = int(fp)
            metrics[f"{prefix}_fn"] = int(fn)
            metrics[f"{prefix}_tn"] = int(tn)
            metrics[f"{prefix}_fpr"] = float(fp / max(1, fp + tn))
        else:
            metrics[f"{prefix}_fpr"] = 0.0

        return metrics

    def _recall_at_k(self, all_labels, all_probs, k: int) -> float:
        """
        Among the top-K highest-risk nodes, what fraction are actual frauds?
        This is the metric that matters most to investigators.
        """
        if k <= 0 or len(all_labels) == 0:
            return 0.0

        top_k_indices = np.argsort(all_probs)[-k:]
        actual_positives_in_top_k = all_labels[top_k_indices].sum()
        total_positives = max(1, all_labels.sum())
        return float(actual_positives_in_top_k / total_positives)

    def _fpr_by_industry(self, node_probs) -> dict:
        """
        Compute False Positive Rate per industry sector.
        Ensures the model doesn't unfairly flag one sector over another.
        """
        industry_fpr = {}
        idx_to_tc = {v: k for k, v in self.tc_to_idx.items()}
        company_map = {c["tax_code"]: c for c in self.companies}

        industry_groups = {}  # industry -> [(true_label, predicted_prob)]
        for idx in range(len(self.node_labels)):
            tc = idx_to_tc.get(idx, "")
            comp = company_map.get(tc, {})
            industry = comp.get("industry", "Unknown")
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append((
                int(self.node_labels[idx].item()),
                float(node_probs[idx])
            ))

        for industry, pairs in industry_groups.items():
            trues = np.array([p[0] for p in pairs])
            preds = (np.array([p[1] for p in pairs]) > 0.5).astype(int)
            n_neg = (trues == 0).sum()
            n_fp = ((preds == 1) & (trues == 0)).sum()
            industry_fpr[industry] = round(float(n_fp / max(1, n_neg)), 4)

        return industry_fpr


# ════════════════════════════════════════════════════════════════
#  3. Probability Calibrator
# ════════════════════════════════════════════════════════════════

class ProbabilityCalibrator:
    """
    Calibrate raw sigmoid outputs using Isotonic Regression.

    After calibration, a predicted probability of 70% means that
    approximately 70% of cases with that score are indeed fraudulent.
    """

    def __init__(self):
        self.node_calibrator = None
        self.edge_calibrator = None
        self.node_threshold = 0.5
        self.edge_threshold = 0.5
        self.threshold_meta = {}
        self._fitted = False

    @staticmethod
    def _optimize_threshold(
        probs: np.ndarray,
        labels: np.ndarray,
        metric: str = "f1",
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        num_steps: int = 91,
    ) -> float:
        probs = np.asarray(probs)
        labels = np.asarray(labels)

        if len(labels) == 0 or len(np.unique(labels)) < 2:
            return 0.5

        metric_key = metric.strip().lower()
        if metric_key not in {"f1", "recall", "precision"}:
            metric_key = "f1"

        best_threshold = 0.5
        best_score = -1.0
        best_tie_recall = -1.0

        for t in np.linspace(min_threshold, max_threshold, num_steps):
            preds = (probs >= t).astype(int)
            if metric_key == "recall":
                score = float(recall_score(labels, preds, zero_division=0))
            elif metric_key == "precision":
                score = float(precision_score(labels, preds, zero_division=0))
            else:
                score = float(f1_score(labels, preds, zero_division=0))

            tie_recall = float(recall_score(labels, preds, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(t)
                best_tie_recall = tie_recall
            elif np.isclose(score, best_score):
                # Prefer better recall, then threshold closer to 0.5 for stability.
                if tie_recall > best_tie_recall:
                    best_threshold = float(t)
                    best_tie_recall = tie_recall
                elif np.isclose(tie_recall, best_tie_recall):
                    if abs(t - 0.5) < abs(best_threshold - 0.5):
                        best_threshold = float(t)

        return float(best_threshold)

    @staticmethod
    def _evaluate_threshold_metric(labels: np.ndarray, preds: np.ndarray, metric: str) -> float:
        metric_key = metric.strip().lower()
        if metric_key == "recall":
            return float(recall_score(labels, preds, zero_division=0))
        if metric_key == "precision":
            return float(precision_score(labels, preds, zero_division=0))
        return float(f1_score(labels, preds, zero_division=0))

    @staticmethod
    def _optimize_cold_start_delta(
        probs: np.ndarray,
        labels: np.ndarray,
        node_degrees: Optional[np.ndarray],
        base_threshold: float,
        metric: str,
        degree_threshold: int,
        max_delta: float,
    ) -> float:
        if node_degrees is None:
            return 0.0

        degrees = np.asarray(node_degrees)
        if len(degrees) != len(labels):
            return 0.0

        low_mask = degrees <= degree_threshold
        if int(low_mask.sum()) < 5:
            return 0.0

        # Need positive samples in low-degree region to tune recall meaningfully.
        if int(np.asarray(labels)[low_mask].sum()) == 0:
            return 0.0

        best_delta = 0.0
        best_score = -1.0
        best_low_recall = -1.0

        for delta in np.linspace(0.0, max_delta, 8):
            effective_thresholds = np.full(len(labels), base_threshold, dtype=float)
            effective_thresholds[low_mask] = np.maximum(0.05, base_threshold - float(delta))
            preds = (probs >= effective_thresholds).astype(int)

            score = ProbabilityCalibrator._evaluate_threshold_metric(labels, preds, metric)
            low_recall = float(recall_score(np.asarray(labels)[low_mask], preds[low_mask], zero_division=0))

            if score > best_score:
                best_score = score
                best_delta = float(delta)
                best_low_recall = low_recall
            elif np.isclose(score, best_score):
                # With same global metric, prefer higher low-degree recall.
                if low_recall > best_low_recall:
                    best_delta = float(delta)
                    best_low_recall = low_recall

        return float(best_delta)

    def optimize_decision_thresholds(
        self,
        node_probs: np.ndarray,
        node_labels: np.ndarray,
        node_degrees: Optional[np.ndarray] = None,
        edge_probs: Optional[np.ndarray] = None,
        edge_labels: Optional[np.ndarray] = None,
        metric: str = "f1",
        cold_start_degree_threshold: int = 2,
        max_cold_start_delta: float = 0.12,
        min_edge_threshold: Optional[float] = None,
        default_cold_start_delta: Optional[float] = None,
    ) -> dict:
        """
        Learn decision thresholds from validation scores.

        Thresholds are later applied in serving inference to avoid a hard-coded 0.5 cutoff.
        """
        self.node_threshold = self._optimize_threshold(node_probs, node_labels, metric=metric)
        cold_start_delta = self._optimize_cold_start_delta(
            probs=np.asarray(node_probs),
            labels=np.asarray(node_labels),
            node_degrees=node_degrees,
            base_threshold=float(self.node_threshold),
            metric=metric,
            degree_threshold=int(cold_start_degree_threshold),
            max_delta=float(max_cold_start_delta),
        )

        if default_cold_start_delta is None:
            default_cold_start_delta = float(os.getenv("GNN_DEFAULT_COLD_START_THRESHOLD_DELTA", "0.06"))
        default_cold_start_delta = float(max(0.0, min(default_cold_start_delta, max_cold_start_delta)))

        delta_source = "optimized"
        if cold_start_delta <= 0.0 and default_cold_start_delta > 0.0:
            cold_start_delta = default_cold_start_delta
            delta_source = "fallback_default"

        if min_edge_threshold is None:
            min_edge_threshold = float(os.getenv("GNN_MIN_EDGE_DECISION_THRESHOLD", "0.45"))

        if edge_probs is not None and edge_labels is not None and len(edge_labels) > 0:
            raw_edge_threshold = self._optimize_threshold(edge_probs, edge_labels, metric=metric)
            self.edge_threshold = max(float(min_edge_threshold), float(raw_edge_threshold))
        else:
            self.edge_threshold = 0.5

        self.threshold_meta = {
            "metric": metric,
            "node_validation_support": int(len(node_labels)),
            "edge_validation_support": int(len(edge_labels)) if edge_labels is not None else 0,
            "cold_start_degree_threshold": int(cold_start_degree_threshold),
            "cold_start_threshold_delta": float(cold_start_delta),
            "cold_start_delta_source": delta_source,
            "min_edge_threshold": float(min_edge_threshold),
        }

        return {
            "node_threshold": float(self.node_threshold),
            "edge_threshold": float(self.edge_threshold),
            "metric": metric,
            "cold_start_threshold_delta": float(cold_start_delta),
            "cold_start_degree_threshold": int(cold_start_degree_threshold),
            "cold_start_delta_source": delta_source,
            "min_edge_threshold": float(min_edge_threshold),
        }

    def fit(self, node_probs: np.ndarray, node_labels: np.ndarray,
            edge_probs: np.ndarray, edge_labels: np.ndarray):
        """
        Fit calibrators on VALIDATION set (never train set!).
        """
        # Node calibrator
        if len(np.unique(node_labels)) >= 2 and len(node_labels) >= 5:
            self.node_calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.node_calibrator.fit(node_probs, node_labels)
        else:
            self.node_calibrator = None

        # Edge calibrator
        if len(np.unique(edge_labels)) >= 2 and len(edge_labels) >= 5:
            self.edge_calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.edge_calibrator.fit(edge_probs, edge_labels)
        else:
            self.edge_calibrator = None

        self._fitted = True

    def calibrate_node(self, raw_prob: float) -> float:
        """Calibrate a single node probability."""
        if not self._fitted or self.node_calibrator is None:
            return raw_prob
        return float(self.node_calibrator.predict([raw_prob])[0])

    def calibrate_edge(self, raw_prob: float) -> float:
        """Calibrate a single edge probability."""
        if not self._fitted or self.edge_calibrator is None:
            return raw_prob
        return float(self.edge_calibrator.predict([raw_prob])[0])

    def calibrate_nodes_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate a batch of node probabilities."""
        if not self._fitted or self.node_calibrator is None:
            return raw_probs
        return self.node_calibrator.predict(raw_probs)

    def calibrate_edges_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate a batch of edge probabilities."""
        if not self._fitted or self.edge_calibrator is None:
            return raw_probs
        return self.edge_calibrator.predict(raw_probs)

    def save(self, path: Optional[str] = None):
        """Persist calibrators to disk."""
        save_path = Path(path) if path else MODEL_DIR
        save_path.mkdir(parents=True, exist_ok=True)
        calibrator_path = save_path / "calibrator.pkl"
        with open(calibrator_path, "wb") as f:
            pickle.dump({
                "node_calibrator": self.node_calibrator,
                "edge_calibrator": self.edge_calibrator,
                "fitted": self._fitted,
                "node_threshold": float(self.node_threshold),
                "edge_threshold": float(self.edge_threshold),
                "threshold_meta": self.threshold_meta,
            }, f)
        print(f"[OK] Calibrator saved to {calibrator_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load calibrators from disk."""
        load_path = Path(path) if path else MODEL_DIR
        calibrator_path = load_path / "calibrator.pkl"
        if not calibrator_path.exists():
            print("[WARN] No calibrator found. Using raw probabilities.")
            return False
        try:
            with open(calibrator_path, "rb") as f:
                data = pickle.load(f)
            self.node_calibrator = data.get("node_calibrator")
            self.edge_calibrator = data.get("edge_calibrator")
            self._fitted = data.get("fitted", False)
            self.node_threshold = float(data.get("node_threshold", 0.5))
            self.edge_threshold = float(data.get("edge_threshold", 0.5))
            self.threshold_meta = data.get("threshold_meta", {})
            print(f"[OK] Calibrator loaded from {calibrator_path}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load calibrator: {e}")
            return False


# ════════════════════════════════════════════════════════════════
#  4. Report Formatter
# ════════════════════════════════════════════════════════════════

def format_evaluation_report(eval_result: dict, calibrated_result: dict = None) -> str:
    """
    Format evaluation results into a human-readable report.
    """
    lines = []
    split = eval_result.get("split", "unknown")
    lines.append(f"\n{'='*70}")
    lines.append(f"  EVALUATION REPORT — {split.upper()} SET")
    lines.append(f"{'='*70}")

    # Node metrics
    lines.append(f"\n  ▸ NODE CLASSIFICATION (Shell Company Detection)")
    lines.append(f"    F1-Score:    {eval_result.get('node_f1', 0):.4f}")
    lines.append(f"    Precision:   {eval_result.get('node_precision', 0):.4f}")
    lines.append(f"    Recall:      {eval_result.get('node_recall', 0):.4f}")
    lines.append(f"    PR-AUC:      {eval_result.get('node_pr_auc', 0):.4f}")
    lines.append(f"    ROC-AUC:     {eval_result.get('node_roc_auc', 0):.4f}")
    lines.append(f"    FPR:         {eval_result.get('node_fpr', 0):.4f}")
    if 'node_tp' in eval_result:
        lines.append(f"    TP={eval_result['node_tp']}  FP={eval_result['node_fp']}  "
                     f"FN={eval_result['node_fn']}  TN={eval_result['node_tn']}")

    # Edge metrics
    lines.append(f"\n  ▸ EDGE CLASSIFICATION (Circular Invoice Detection)")
    lines.append(f"    F1-Score:    {eval_result.get('edge_f1', 0):.4f}")
    lines.append(f"    Precision:   {eval_result.get('edge_precision', 0):.4f}")
    lines.append(f"    Recall:      {eval_result.get('edge_recall', 0):.4f}")
    lines.append(f"    PR-AUC:      {eval_result.get('edge_pr_auc', 0):.4f}")
    lines.append(f"    ROC-AUC:     {eval_result.get('edge_roc_auc', 0):.4f}")
    lines.append(f"    FPR:         {eval_result.get('edge_fpr', 0):.4f}")
    if 'edge_tp' in eval_result:
        lines.append(f"    TP={eval_result['edge_tp']}  FP={eval_result['edge_fp']}  "
                     f"FN={eval_result['edge_fn']}  TN={eval_result['edge_tn']}")

    # Recall@K
    k = eval_result.get("top_k", 20)
    lines.append(f"\n  ▸ OPERATIONAL METRICS")
    lines.append(f"    Recall@{k}:   {eval_result.get('recall_at_k', 0):.4f}  "
                 f"(of top-{k} alerts, how many are real fraud)")

    # FPR by industry
    fpr_ind = eval_result.get("fpr_by_industry", {})
    if fpr_ind:
        lines.append(f"\n  ▸ FALSE POSITIVE RATE BY INDUSTRY")
        for ind, fpr in sorted(fpr_ind.items(), key=lambda x: -x[1]):
            bar = "█" * int(fpr * 50)
            lines.append(f"    {ind:.<30s} {fpr:.4f} {bar}")

    # Calibration info
    if calibrated_result:
        lines.append(f"\n  ▸ POST-CALIBRATION METRICS")
        lines.append(f"    Node F1 (calibrated):  {calibrated_result.get('node_f1', 0):.4f}")
        lines.append(f"    Edge F1 (calibrated):  {calibrated_result.get('edge_f1', 0):.4f}")

    lines.append(f"\n{'='*70}\n")
    return "\n".join(lines)
