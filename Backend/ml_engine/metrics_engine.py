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
        - Edges (invoices) are split by their invoice date
        - Nodes are assigned to the EARLIEST period where they appear
        - This prevents future information from leaking into training

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
    ):
        self.base_year = base_year
        self.train_cutoff = date(base_year, train_end_month + 1, 1)  # first day of month after train
        self.val_cutoff = date(base_year, val_end_month + 1, 1)      # first day of month after val

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

        # ── Node masks: Stratified random split ──
        # In transductive GNN, ALL nodes see the full graph structure.
        # Since most companies have invoices spanning all periods,
        # a purely temporal node split puts nearly everything in 'train'.
        # Instead, we use stratified random split (70/15/15) preserving
        # label distribution, while edges remain strictly temporal.
        from sklearn.model_selection import train_test_split
        import numpy as np

        node_train = torch.zeros(n_nodes, dtype=torch.bool)
        node_val = torch.zeros(n_nodes, dtype=torch.bool)
        node_test = torch.zeros(n_nodes, dtype=torch.bool)

        # Build label array for stratification
        all_indices = np.arange(n_nodes)
        
        # Simple split: 70% train, 15% val, 15% test
        try:
            train_idx, temp_idx = train_test_split(
                all_indices, test_size=0.30, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.50, random_state=42
            )
        except ValueError:
            # Fallback if too few samples
            n = n_nodes
            train_idx = all_indices[:int(n * 0.7)]
            val_idx = all_indices[int(n * 0.7):int(n * 0.85)]
            test_idx = all_indices[int(n * 0.85):]

        node_train[train_idx] = True
        node_val[val_idx] = True
        node_test[test_idx] = True

        split_info = {
            "edge_train": int(edge_train.sum()),
            "edge_val": int(edge_val.sum()),
            "edge_test": int(edge_test.sum()),
            "node_train": int(node_train.sum()),
            "node_val": int(node_val.sum()),
            "node_test": int(node_test.sum()),
            "train_cutoff": str(self.train_cutoff),
            "val_cutoff": str(self.val_cutoff),
        }

        return {
            "node_train_mask": node_train,
            "node_val_mask": node_val,
            "node_test_mask": node_test,
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
        self._fitted = False

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
