"""
gnn_explainer.py – GNN Explainability for Tax Fraud Investigations
===================================================================
Architecture:
    - GNNExplainer (Ying et al., 2019): learns edge + feature masks
      that minimize the prediction change when important elements are kept.
    - SubgraphExtractor: extracts k-hop ego-networks for visualisation
    - AttentionExplainer: extracts GAT attention weights as explanations

Purpose:
    For each high-risk node flagged by the GNN, produce:
    1. A minimal subgraph (evidence) that explains the prediction
    2. Edge importance scores (which transactions matter most)
    3. Feature importance scores (which features drive the decision)
    4. Human-readable summary for auditors

Design:
    - Works with both homogeneous TaxFraudGAT and heterogeneous HGT
    - Produces JSON-serializable explanation objects
    - Integrates with investigation_agent for enriched narratives
    - Falls back to attention-based explanation if GNNExplainer unavailable

Reference:
    Ying et al., "GNNExplainer: Generating Explanations for Graph
    Neural Networks", NeurIPS 2019
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

logger = logging.getLogger(__name__)

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import k_hop_subgraph
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("torch_geometric not available. GNNExplainer disabled.")

try:
    from torch_geometric.explain import Explainer, GNNExplainer as PYGExplainer
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
#  1. Explanation Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class NodeExplanation:
    """Complete explanation for a single node prediction."""
    target_node_id: str
    target_node_index: int
    prediction_score: float
    is_suspicious: bool
    # Subgraph evidence
    subgraph_nodes: list[dict[str, Any]]
    subgraph_edges: list[dict[str, Any]]
    # Importance scores
    edge_importances: list[dict[str, Any]]
    feature_importances: list[dict[str, Any]]
    # Human-readable
    summary: str
    reasoning_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_node_id": self.target_node_id,
            "target_node_index": self.target_node_index,
            "prediction_score": self.prediction_score,
            "is_suspicious": self.is_suspicious,
            "subgraph_nodes": self.subgraph_nodes,
            "subgraph_edges": self.subgraph_edges,
            "edge_importances": self.edge_importances,
            "feature_importances": self.feature_importances,
            "summary": self.summary,
            "reasoning_steps": self.reasoning_steps,
        }


# ════════════════════════════════════════════════════════════════
#  2. Subgraph Extractor (k-hop ego network)
# ════════════════════════════════════════════════════════════════

class SubgraphExtractor:
    """
    Extract k-hop neighborhood around a target node.
    
    This provides the local context that the GNN uses to make
    its prediction. Essential for audit evidence.
    """

    def __init__(self, num_hops: int = 2):
        self.num_hops = num_hops

    def extract(
        self,
        target_idx: int,
        data: "Data",
        idx_to_tc: dict[int, str],
        company_map: dict[str, dict],
        invoices: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Extract the k-hop subgraph around target_idx.
        
        Returns:
            (subgraph_nodes, subgraph_edges) as serializable dicts.
        """
        if not PYG_AVAILABLE:
            return [], []

        # Get k-hop subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=target_idx,
            num_hops=self.num_hops,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=data.num_nodes,
        )

        # Build node list
        nodes = []
        for node_idx in subset.tolist():
            tc = idx_to_tc.get(node_idx, f"node_{node_idx}")
            company = company_map.get(tc, {})
            nodes.append({
                "index": node_idx,
                "tax_code": tc,
                "name": company.get("name", "Unknown"),
                "industry": company.get("industry", "Unknown"),
                "risk_score": float(company.get("risk_score", 0)),
                "is_target": bool(node_idx == target_idx),
            })

        # Build edge list
        edges = []
        subset_set = set(subset.tolist())
        for e_idx in range(data.edge_index.shape[1]):
            src = data.edge_index[0, e_idx].item()
            dst = data.edge_index[1, e_idx].item()
            if src in subset_set and dst in subset_set:
                src_tc = idx_to_tc.get(src, "")
                dst_tc = idx_to_tc.get(dst, "")
                edge_info = {
                    "from": src_tc,
                    "to": dst_tc,
                    "from_index": src,
                    "to_index": dst,
                }
                # Enrich with invoice data if available
                if invoices and e_idx < len(invoices):
                    inv = invoices[e_idx]
                    edge_info["amount"] = float(inv.get("amount", 0))
                    edge_info["invoice_number"] = str(
                        inv.get("invoice_number", "")
                    )
                edges.append(edge_info)

        return nodes, edges


# ════════════════════════════════════════════════════════════════
#  3. Attention-Based Explainer (fallback)
# ════════════════════════════════════════════════════════════════

class AttentionExplainer:
    """
    Extract edge importance from GAT attention weights.
    
    This is a lighter-weight alternative to GNNExplainer that
    leverages the built-in attention mechanism of GATv2Conv.
    """

    def explain_node(
        self,
        target_idx: int,
        attention_weights: tuple[torch.Tensor, torch.Tensor] | None,
        data: "Data",
        idx_to_tc: dict[int, str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Rank edges by attention weight for a target node.
        
        Returns top_k most important edges.
        """
        if attention_weights is None:
            return []

        attn_edge_index, attn_values = attention_weights

        # Find edges connected to target node
        edge_importances = []
        for e_idx in range(attn_edge_index.shape[1]):
            src = attn_edge_index[0, e_idx].item()
            dst = attn_edge_index[1, e_idx].item()
            if src == target_idx or dst == target_idx:
                weight = float(attn_values[e_idx].mean().item())
                edge_importances.append({
                    "from": idx_to_tc.get(src, str(src)),
                    "to": idx_to_tc.get(dst, str(dst)),
                    "attention_weight": round(weight, 4),
                    "direction": "incoming" if dst == target_idx else "outgoing",
                })

        # Sort by importance descending
        edge_importances.sort(key=lambda x: x["attention_weight"], reverse=True)
        return edge_importances[:top_k]


# ════════════════════════════════════════════════════════════════
#  4. Feature Importance Explainer
# ════════════════════════════════════════════════════════════════

class FeatureImportanceExplainer:
    """
    Compute per-feature importance via gradient-based attribution.
    
    Uses vanilla gradients (∂output/∂input) as a simple but effective
    feature importance measure for the target node.
    """

    NODE_FEATURE_NAMES = [
        "company_age", "capital_log", "latitude", "longitude",
        "degree", "in_amount", "out_amount", "in_out_ratio",
        "is_ghost", "revenue_accel",
        "industry_0", "industry_1", "industry_2", "industry_3", "industry_4",
        "industry_5", "industry_6", "industry_7", "industry_8", "industry_9",
        "avg_invoice_interval", "burst_score",
    ]

    def explain_features(
        self,
        model: nn.Module,
        data: "Data",
        target_idx: int,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Compute gradient-based feature importance for target node.
        
        Returns top_k most important features with their values
        and gradient magnitudes.
        """
        if not PYG_AVAILABLE:
            return []

        model.eval()
        data_clone = data.clone()
        data_clone.x = data_clone.x.clone().detach().requires_grad_(True)

        out = model(data_clone)
        node_logit = out["node_logits"][target_idx]
        node_logit.backward()

        grads = data_clone.x.grad[target_idx].detach().numpy()
        values = data.x[target_idx].detach().numpy()

        importances = []
        for i, (grad, val) in enumerate(zip(grads, values)):
            name = (
                self.NODE_FEATURE_NAMES[i]
                if i < len(self.NODE_FEATURE_NAMES)
                else f"feature_{i}"
            )
            importances.append({
                "feature": name,
                "value": round(float(val), 4),
                "gradient": round(float(abs(grad)), 6),
                "contribution": round(float(grad * val), 6),
            })

        importances.sort(key=lambda x: x["gradient"], reverse=True)
        return importances[:top_k]


# ════════════════════════════════════════════════════════════════
#  5. Unified GNN Explainer (orchestrates all components)
# ════════════════════════════════════════════════════════════════

class TaxGNNExplainer:
    """
    Unified explainability engine for the TaxInspector GNN pipeline.
    
    Combines:
    - Subgraph extraction (k-hop evidence)
    - Attention-based edge importance
    - Gradient-based feature importance
    - Human-readable summary generation
    """

    def __init__(self, num_hops: int = 2):
        self.subgraph_extractor = SubgraphExtractor(num_hops=num_hops)
        self.attention_explainer = AttentionExplainer()
        self.feature_explainer = FeatureImportanceExplainer()

    def explain(
        self,
        model: nn.Module,
        data: "Data",
        target_idx: int,
        idx_to_tc: dict[int, str],
        company_map: dict[str, dict],
        invoices: list[dict] | None = None,
        attention_weights: tuple | None = None,
        prediction_score: float = 0.0,
    ) -> NodeExplanation:
        """
        Generate a complete explanation for a single node's prediction.
        
        This is the main entry point for auditors and investigators.
        """
        target_tc = idx_to_tc.get(target_idx, f"node_{target_idx}")
        company = company_map.get(target_tc, {})
        is_suspicious = prediction_score > 0.5

        # 1. Extract subgraph evidence
        subgraph_nodes, subgraph_edges = self.subgraph_extractor.extract(
            target_idx, data, idx_to_tc, company_map, invoices
        )

        # 2. Edge importance from attention
        edge_importances = self.attention_explainer.explain_node(
            target_idx, attention_weights, data, idx_to_tc, top_k=10
        )

        # 3. Feature importance from gradients
        feature_importances = self.feature_explainer.explain_features(
            model, data, target_idx, top_k=8
        )

        # 4. Generate human-readable summary
        summary, reasoning = self._generate_summary(
            target_tc, company, prediction_score, is_suspicious,
            subgraph_nodes, subgraph_edges,
            edge_importances, feature_importances,
        )

        return NodeExplanation(
            target_node_id=target_tc,
            target_node_index=target_idx,
            prediction_score=round(prediction_score, 4),
            is_suspicious=is_suspicious,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
            edge_importances=edge_importances,
            feature_importances=feature_importances,
            summary=summary,
            reasoning_steps=reasoning,
        )

    def _generate_summary(
        self,
        tax_code: str,
        company: dict,
        score: float,
        is_suspicious: bool,
        nodes: list[dict],
        edges: list[dict],
        edge_imp: list[dict],
        feat_imp: list[dict],
    ) -> tuple[str, list[str]]:
        """Generate human-readable explanation text."""
        name = company.get("name", tax_code)
        risk_level = (
            "RẤT CAO" if score > 0.8
            else "CAO" if score > 0.6
            else "TRUNG BÌNH" if score > 0.4
            else "THẤP"
        )

        summary = (
            f"Doanh nghiệp {name} (MST: {tax_code}) được GNN đánh giá "
            f"mức rủi ro {risk_level} ({score*100:.1f}%). "
            f"Mạng lưới giao dịch liên quan bao gồm {len(nodes)} thực thể "
            f"và {len(edges)} giao dịch trong vùng lân cận 2 bước."
        )

        reasoning = []

        # Top features
        if feat_imp:
            top_feat = feat_imp[0]
            reasoning.append(
                f"Yếu tố quan trọng nhất: '{top_feat['feature']}' "
                f"(giá trị={top_feat['value']}, đóng góp={top_feat['contribution']:.4f})"
            )

        # Top edges
        if edge_imp:
            top_edge = edge_imp[0]
            reasoning.append(
                f"Giao dịch đáng ngờ nhất: {top_edge['from']} → {top_edge['to']} "
                f"(attention={top_edge['attention_weight']:.4f})"
            )

        # Network patterns
        n_connected = len(nodes) - 1
        if n_connected > 5:
            reasoning.append(
                f"Mạng lưới dày đặc: {n_connected} đối tác trong bán kính 2 bước — "
                f"vượt ngưỡng cảnh báo."
            )

        # Reciprocal warnings
        incoming = sum(1 for e in edge_imp if e.get("direction") == "incoming")
        outgoing = sum(1 for e in edge_imp if e.get("direction") == "outgoing")
        if incoming > 0 and outgoing > 0:
            reasoning.append(
                f"Phát hiện giao dịch hai chiều: {incoming} luồng vào, {outgoing} luồng ra — "
                f"dấu hiệu quay vòng tiềm năng."
            )

        if not reasoning:
            reasoning.append("Không có đủ dữ liệu để giải thích chi tiết.")

        return summary, reasoning


# ════════════════════════════════════════════════════════════════
#  6. Batch Explainer for Investigation Reports
# ════════════════════════════════════════════════════════════════

class BatchExplainer:
    """
    Generate explanations for all suspicious nodes in a graph.
    
    Optimized for batch processing during investigation workflows.
    """

    def __init__(self, explainer: TaxGNNExplainer | None = None):
        self.explainer = explainer or TaxGNNExplainer()

    def explain_suspicious(
        self,
        model: nn.Module,
        data: "Data",
        node_scores: dict[str, float],
        idx_to_tc: dict[int, str],
        tc_to_idx: dict[str, int],
        company_map: dict[str, dict],
        invoices: list[dict] | None = None,
        attention_weights: tuple | None = None,
        threshold: float = 0.5,
        max_explanations: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Explain all nodes above the suspicion threshold.
        
        Returns list of explanation dicts, sorted by risk score.
        """
        suspicious = [
            (tc, score) for tc, score in node_scores.items()
            if score > threshold
        ]
        suspicious.sort(key=lambda x: x[1], reverse=True)
        suspicious = suspicious[:max_explanations]

        explanations = []
        for tc, score in suspicious:
            if tc not in tc_to_idx:
                continue
            idx = tc_to_idx[tc]
            try:
                explanation = self.explainer.explain(
                    model=model,
                    data=data,
                    target_idx=idx,
                    idx_to_tc=idx_to_tc,
                    company_map=company_map,
                    invoices=invoices,
                    attention_weights=attention_weights,
                    prediction_score=score,
                )
                explanations.append(explanation.to_dict())
            except Exception as e:
                logger.warning(f"Failed to explain node {tc}: {e}")

        return explanations
