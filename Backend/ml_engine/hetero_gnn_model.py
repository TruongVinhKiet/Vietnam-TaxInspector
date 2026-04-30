"""
hetero_gnn_model.py – Heterogeneous Graph Transformer for OSINT Fraud Detection
=================================================================================
Architecture:
    - HGTConv (Heterogeneous Graph Transformer) — SOTA for heterogeneous graphs
    - Multi-entity node classification: Company, Person, Offshore Entity
    - 5 edge types: owns, controls, alias, phoenix_successor, issued_invoice_to
    - Per-node-type linear heads for risk classification

Node Types & Features (15 dims each, standardised):
    - company:          risk_score, country_risk, industry_bucket, status, age, degrees...
    - offshore_entity:  risk_score, country_risk, jurisdiction_opacity, shell indicators...
    - person:           risk_score, role_flags, connection_density...

Edge Types:
    - (company,  owns,               company)
    - (person,   controls,           company)
    - (company,  alias,              company)
    - (company,  phoenix_successor,  company)
    - (company,  issued_invoice_to,  company)

Design:
    - 2 HGTConv layers with 4 attention heads
    - Per-type classification heads (binary: is_suspicious)
    - Built-in graceful degradation if PyG is unavailable
    - Isotonic calibration support for deployment

References:
    Hu et al., "Heterogeneous Graph Transformer", WWW 2020
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import HGTConv, Linear
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("torch_geometric not available. HeteroGNN features disabled.")

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# ════════════════════════════════════════════════════════════════
#  Feature Constants (must match osint_heterograph_dataset.py)
# ════════════════════════════════════════════════════════════════

NODE_TYPES = ("company", "offshore_entity", "person")
EDGE_TYPES = [
    ("company", "owns", "company"),
    ("person", "controls", "company"),
    ("company", "alias", "company"),
    ("company", "phoenix_successor", "company"),
    ("company", "issued_invoice_to", "company"),
]
NODE_FEATURE_DIM = 15


# ════════════════════════════════════════════════════════════════
#  1. HGT Model Definition
# ════════════════════════════════════════════════════════════════

class TaxFraudHGT(nn.Module):
    """
    Heterogeneous Graph Transformer for multi-entity fraud detection.
    
    Uses HGTConv layers that learn type-specific attention patterns
    across heterogeneous node/edge types. Each node type gets its
    own classification head for binary risk prediction.
    """

    def __init__(
        self,
        node_feature_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        node_types: tuple[str, ...] = NODE_TYPES,
        edge_types: list[tuple[str, str, str]] | None = None,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types or EDGE_TYPES
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        metadata = (list(self.node_types), self.edge_types)

        # Per-type input projections
        self.input_projections = nn.ModuleDict()
        for ntype in self.node_types:
            self.input_projections[ntype] = Linear(node_feature_dim, hidden_dim)

        # HGT Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
            )

        # Per-type classification heads
        self.classifiers = nn.ModuleDict()
        for ntype in self.node_types:
            self.classifiers[ntype] = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

    def forward(self, data: "HeteroData") -> dict[str, torch.Tensor]:
        """
        Forward pass through HGT layers.
        
        Returns dict mapping node_type -> logits tensor.
        """
        # Project each node type into shared hidden space
        x_dict = {}
        for ntype in self.node_types:
            if ntype in data.node_types and hasattr(data[ntype], "x"):
                x_dict[ntype] = self.input_projections[ntype](data[ntype].x)

        if not x_dict:
            return {}

        # Build edge_index_dict from data
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type in data.edge_types:
                edge_index_dict[edge_type] = data[edge_type].edge_index

        # HGT message passing
        # NOTE: HGTConv may drop node types that receive no incoming messages.
        # We re-inject them from the previous layer to keep all types alive.
        for conv in self.convs:
            prev_x_dict = {k: v.clone() for k, v in x_dict.items()}
            out_dict = conv(x_dict, edge_index_dict)
            out_dict = {
                ntype: F.elu(F.dropout(x, p=self.dropout, training=self.training))
                for ntype, x in out_dict.items()
            }
            # Re-inject node types that were dropped (no incoming edges)
            for ntype in prev_x_dict:
                if ntype not in out_dict:
                    out_dict[ntype] = prev_x_dict[ntype]
            x_dict = out_dict

        # Per-type classification
        logits = {}
        for ntype in self.node_types:
            if ntype in x_dict:
                logits[ntype] = self.classifiers[ntype](x_dict[ntype]).squeeze(-1)

        return logits


# ════════════════════════════════════════════════════════════════
#  2. Training Pipeline
# ════════════════════════════════════════════════════════════════

class HeteroGNNTrainer:
    """Train and persist the HGT-based heterogeneous fraud classifier."""

    def __init__(
        self,
        node_feature_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.003,
        weight_decay: float = 5e-4,
    ):
        self.config = {
            "node_feature_dim": node_feature_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.model = TaxFraudHGT(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(
        self,
        data: "HeteroData",
        labels: dict[str, torch.Tensor],
        train_masks: dict[str, torch.Tensor] | None = None,
        val_masks: dict[str, torch.Tensor] | None = None,
        epochs: int = 200,
    ) -> dict[str, Any]:
        """
        Train the HGT model with per-type labels and masks.
        
        Returns final validation metrics dict.
        """
        from sklearn.metrics import f1_score, average_precision_score

        # Default masks: all nodes
        if train_masks is None:
            train_masks = {
                ntype: torch.ones(labels[ntype].shape[0], dtype=torch.bool)
                for ntype in labels
            }
        if val_masks is None:
            val_masks = train_masks

        # Compute per-type pos_weight for class imbalance
        pos_weights = {}
        for ntype in labels:
            mask = train_masks[ntype]
            n_pos = max(1, labels[ntype][mask].sum().item())
            n_neg = max(1, mask.sum().item() - n_pos)
            pos_weights[ntype] = torch.tensor([n_neg / n_pos])

        best_metrics = {}
        for epoch in range(epochs):
            # ── TRAIN ──
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(data)

            total_loss = torch.tensor(0.0)
            for ntype in labels:
                if ntype not in logits:
                    continue
                mask = train_masks[ntype]
                loss = F.binary_cross_entropy_with_logits(
                    logits[ntype][mask],
                    labels[ntype][mask].float(),
                    pos_weight=pos_weights[ntype],
                )
                total_loss = total_loss + loss

            total_loss.backward()
            self.optimizer.step()

            # ── EVAL every 50 epochs ──
            if (epoch + 1) % 50 == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_logits = self.model(data)
                    metrics_str_parts = []
                    for ntype in labels:
                        if ntype not in eval_logits:
                            continue
                        mask = val_masks[ntype]
                        probs = torch.sigmoid(eval_logits[ntype][mask]).numpy()
                        preds = (probs > 0.5).astype(int)
                        y_true = labels[ntype][mask].numpy()

                        f1 = f1_score(y_true, preds, zero_division=0)
                        try:
                            pr_auc = average_precision_score(y_true, probs)
                        except ValueError:
                            pr_auc = 0.0

                        best_metrics[ntype] = {"f1": f1, "pr_auc": pr_auc}
                        metrics_str_parts.append(
                            f"{ntype}(F1={f1:.3f}, PR-AUC={pr_auc:.3f})"
                        )

                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Loss={total_loss.item():.4f} | "
                    + " | ".join(metrics_str_parts)
                )
                print(
                    f"  Epoch {epoch+1}/{epochs} | Loss={total_loss.item():.4f} | "
                    + " | ".join(metrics_str_parts)
                )

        return best_metrics

    def save(self, path: str | Path | None = None) -> None:
        save_dir = Path(path) if path else MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / "hgt_model.pt")
        with open(save_dir / "hgt_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"[OK] HGT model saved to {save_dir}")


# ════════════════════════════════════════════════════════════════
#  3. Inference Pipeline
# ════════════════════════════════════════════════════════════════

class HeteroGNNInference:
    """Load a trained HGT model and run inference on OSINT heterographs."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model: TaxFraudHGT | None = None
        self._loaded = False

    def load(self) -> bool:
        config_path = self.model_dir / "hgt_config.json"
        model_path = self.model_dir / "hgt_model.pt"

        if not config_path.exists() or not model_path.exists():
            logger.warning("HGT model not found. Heterogeneous GNN disabled.")
            return False

        with open(config_path) as f:
            config = json.load(f)

        self.model = TaxFraudHGT(**config)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"HGT model loaded from {self.model_dir}")
        return True

    def predict(self, data: "HeteroData") -> dict[str, list[dict[str, Any]]]:
        """
        Run inference on a HeteroData graph.

        Returns:
            {
                "company": [{"node_id": ..., "risk_score": ..., "is_suspicious": ...}, ...],
                "person": [...],
                "offshore_entity": [...],
            }
        """
        if not self._loaded or self.model is None:
            return {}

        results: dict[str, list[dict[str, Any]]] = {}

        with torch.no_grad():
            logits = self.model(data)

        for ntype in logits:
            probs = torch.sigmoid(logits[ntype]).numpy()
            node_results = []
            for i, prob in enumerate(probs):
                score = float(prob)
                node_results.append({
                    "index": i,
                    "risk_score": round(score * 100, 2),
                    "is_suspicious": bool(score > 0.5),
                    "confidence": round(max(score, 1 - score) * 100, 2),
                })
            results[ntype] = node_results

        return results
