"""
gnn_model.py – Graph Attention Network (GAT) for VAT Fraud Detection
=====================================================================
Architecture:
    - 2-layer GAT with multi-head attention
    - Node Classification: Predict shell companies (is_shell)
    - Edge Classification: Predict circular fraud edges (is_circular)
    - Explainability: Extract attention weights to highlight suspicious flows

Node Features (20 dims per company):
    - company_age_days: tuổi đời DN
    - capital_log: log(vốn điều lệ) [blanked to prevent leakage]
    - lat, lng: toạ độ địa lý
    - degree: số cạnh giao dịch
    - in_amount_sum: tổng giá trị nhận vào
    - out_amount_sum: tổng giá trị xuất ra
    - in_out_ratio: tỷ lệ nhận/xuất
    - industry_encoded: mã ngành one-hot (10 dims)
    - avg_interval_days: khoảng cách trung bình giữa các hoá đơn xuất (temporal)
    - burst_score: số hoá đơn tối đa trong cửa sổ 48h (temporal)

Edge Features (10 dims per invoice):
    - amount_log: log(giá trị hoá đơn)
    - vat_rate: thuế suất
    - days_since_base: vị trí thời gian (normalized)
    - is_reciprocal: có giao dịch ngược lại hay không
    - delta_recip_norm: khoảng cách thời gian tới hoá đơn ngược gần nhất (temporal)
    - sin/cos day_of_week: encoding chu kỳ ngày trong tuần (temporal)
    - sin/cos month: encoding chu kỳ tháng trong năm (temporal)
    - seller_velocity: tần suất xuất hoá đơn trong 7 ngày (temporal)
"""

import os
import math
import json
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    from torch_geometric.data import Data
    PYGEOM_AVAILABLE = True
except ImportError:
    PYGEOM_AVAILABLE = False
    print("[WARN] torch_geometric not available. GNN features disabled.")

import networkx as nx

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ════════════════════════════════════════════════════════════════
#  1. GAT Model Definition
# ════════════════════════════════════════════════════════════════

class TaxFraudGAT(nn.Module):
    """
    Graph Attention Network v2 for combined Node + Edge classification.
    Uses multi-head attention → the attention weights serve as
    built-in explainability for which edges contribute most to risk.
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 hidden_dim: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # GAT layers (GATv2Conv supports edge_attr)
        self.conv1 = GATv2Conv(
            node_feat_dim, hidden_dim, heads=heads,
            edge_dim=edge_feat_dim, dropout=dropout, concat=True
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads, hidden_dim, heads=1,
            edge_dim=edge_feat_dim, dropout=dropout, concat=False
        )

        # Node classifier: is_shell (binary)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Edge classifier: is_circular (binary)
        # Input: concat of src_embedding + dst_embedding + edge_features
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: "Data", return_attention: bool = False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Layer 1 with attention
        x1, attn1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Layer 2 with attention
        x2, attn2 = self.conv2(x1, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x2 = F.elu(x2)

        # Node predictions
        node_logits = self.node_classifier(x2).squeeze(-1)

        # Edge predictions
        src_idx, dst_idx = edge_index[0], edge_index[1]
        edge_repr = torch.cat([x2[src_idx], x2[dst_idx], edge_attr], dim=-1)
        edge_logits = self.edge_classifier(edge_repr).squeeze(-1)

        result = {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "node_embeddings": x2,
        }

        if return_attention:
            result["attention_weights"] = attn2  # (edge_index, attention_coefficients)

        return result


# ════════════════════════════════════════════════════════════════
#  2. Graph Construction from DB Data
# ════════════════════════════════════════════════════════════════

class GraphConstructor:
    """Build PyG Data objects from raw database query results."""

    INDUSTRY_MAP = {
        "Sản xuất": 0, "Thương mại": 1, "Xây dựng": 2,
        "Vận tải & Logistics": 3, "Công nghệ thông tin": 4,
        "Thực phẩm & Đồ uống": 5, "Bất động sản": 6,
        "Dệt may": 7, "Nông nghiệp": 8, "Dịch vụ tài chính": 9,
    }
    NUM_INDUSTRIES = 10

    def __init__(self, amount_feature_mode: str = "legacy"):
        self.tax_code_to_idx = {}
        mode = str(amount_feature_mode or "legacy").strip().lower()
        if mode not in {"legacy", "robust"}:
            mode = "legacy"
        self.amount_feature_mode = mode
        self.amount_stats = {
            "mode": self.amount_feature_mode,
            "median_log_amount": 0.0,
            "iqr_log_amount": 1.0,
            "clip": 3.0,
        }

    @staticmethod
    def _compute_amount_stats(invoices: list[dict]) -> dict:
        logs = []
        for inv in invoices:
            try:
                logs.append(math.log1p(float(inv.get("amount", 0.0))))
            except (TypeError, ValueError):
                continue

        if not logs:
            return {
                "median_log_amount": 0.0,
                "iqr_log_amount": 1.0,
                "clip": 3.0,
            }

        arr = np.asarray(logs, dtype=float)
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        iqr = max(1e-6, q3 - q1)
        return {
            "median_log_amount": float(np.median(arr)),
            "iqr_log_amount": float(iqr),
            "clip": 3.0,
        }

    def _normalize_amount_feature(self, amount: float) -> float:
        amount = max(0.0, float(amount))
        if self.amount_feature_mode == "robust":
            log_amount = math.log1p(amount)
            median = float(self.amount_stats.get("median_log_amount", 0.0))
            iqr = max(1e-6, float(self.amount_stats.get("iqr_log_amount", 1.0)))
            clip = max(0.5, float(self.amount_stats.get("clip", 3.0)))
            z = (log_amount - median) / iqr
            z = max(-clip, min(clip, z))
            return (z + clip) / (2.0 * clip)
        return math.log1p(amount) / 25.0

    def build_graph(self, companies: list[dict], invoices: list[dict], ownership_links: list[dict] = None) -> "Data":
        """
        Build a PyG Data object from raw company + invoice + ownership records.
        """
        ownership_links = ownership_links or []
        
        # ── Inject Ghost Nodes ──
        existing_tcs = {c["tax_code"] for c in companies}
        all_referenced_tcs = set()
        for inv in invoices:
            all_referenced_tcs.add(inv["seller_tax_code"])
            all_referenced_tcs.add(inv["buyer_tax_code"])
        for own in ownership_links:
            all_referenced_tcs.add(own["parent_tax_code"])
            all_referenced_tcs.add(own["child_tax_code"])
            
        missing_tcs = all_referenced_tcs - existing_tcs
        for tc in missing_tcs:
            companies.append({
                "tax_code": tc,
                "name": f"Doanh nghiệp chưa định danh {tc}",
                "industry": "Chưa định danh",
                "registration_date": str(date.today()), "risk_score": 100, 
                "is_active": False, "lat": 0.0, "lng": 0.0, "is_ghost": True
            })

        # ── Map tax_code → index ──
        self.tax_code_to_idx = {c["tax_code"]: i for i, c in enumerate(companies)}
        n_nodes = len(companies)

        # ── Build adjacency info ──
        # Pre-compute per-node aggregates
        in_amount = {c["tax_code"]: 0.0 for c in companies}
        out_amount = {c["tax_code"]: 0.0 for c in companies}
        degree = {c["tax_code"]: 0 for c in companies}
        reciprocal_pairs = set()

        for inv in invoices:
            s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
            if s in self.tax_code_to_idx and b in self.tax_code_to_idx:
                out_amount[s] += float(inv["amount"])
                in_amount[b] += float(inv["amount"])
                degree[s] += 1
                degree[b] += 1
                reciprocal_pairs.add((s, b))

        today = date.today()

        # ── Node features ──
        node_features = []
        for c in companies:
            reg = c.get("registration_date")
            if reg:
                if isinstance(reg, str):
                    reg = date.fromisoformat(reg)
                age_days = (today - reg).days
            else:
                age_days = 1000

            tc = c["tax_code"]
            in_amt = in_amount.get(tc, 0)
            out_amt = out_amount.get(tc, 0)
            deg = degree.get(tc, 0)

            # Industry one-hot
            ind_idx = self.INDUSTRY_MAP.get(c.get("industry", ""), 0)
            ind_onehot = [0.0] * self.NUM_INDUSTRIES
            ind_onehot[ind_idx] = 1.0

            # New feature: is_ghost_or_inactive
            is_ghost = float(c.get("is_ghost", False) or not c.get("is_active", True))
            
            # New feature: Revenue Acceleration (dummy derivative: out / in+1)
            revenue_accel = out_amt / max(1.0, in_amt)
            revenue_accel_norm = min(1.0, revenue_accel / 10.0)

            feat = [
                age_days / 3650.0,                              # normalized age
                0.0,                                            # BLANKED out risk_score to prevent data leakage
                float(c.get("lat") or 0) / 90.0,                # normalized lat
                float(c.get("lng") or 0) / 180.0,               # normalized lng
                deg / max(1, max(degree.values()) or 1),        # normalized degree
                math.log1p(in_amt) / 25.0,                     # normalized in_amount
                math.log1p(out_amt) / 25.0,                    # normalized out_amount
                in_amt / max(1, in_amt + out_amt),              # in_out_ratio
                is_ghost,                                       # IS GHOST OR INACTIVE
                revenue_accel_norm,                             # HIT AND RUN REVENUE DERIVATIVE
            ] + ind_onehot

            node_features.append(feat)

        # ── Edge index and features ──
        src_list, dst_list = [], []
        edge_features = []

        # Sort invoices by invoice_number for deterministic edge ordering
        sorted_invoices = sorted(invoices, key=lambda inv: inv.get("invoice_number", ""))

        # Keep only invoices that become graph edges.
        graph_invoices = [
            inv for inv in sorted_invoices
            if inv.get("seller_tax_code") in self.tax_code_to_idx and inv.get("buyer_tax_code") in self.tax_code_to_idx
        ]

        # Compute robust amount statistics from the active graph only.
        amount_stats = self._compute_amount_stats(graph_invoices)
        self.amount_stats = {
            "mode": self.amount_feature_mode,
            **amount_stats,
        }

        # Find earliest invoice date for normalization
        all_dates = []
        for inv in graph_invoices:
            d = inv["date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            all_dates.append(d)
        base_date = min(all_dates) if all_dates else today
        date_span_days = max(1, (max(all_dates) - base_date).days) if all_dates else 365

        # ── Pre-compute temporal lookup structures ──
        # 1. Reciprocal time gap: for each (A,B) edge, find nearest (B,A) invoice date
        pair_dates = {}  # (seller, buyer) -> sorted list of dates
        for inv in graph_invoices:
            s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
            d = inv["date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            pair_dates.setdefault((s, b), []).append(d)

        # 2. Seller/buyer activity windows for velocity scoring
        seller_dates = {}  # tax_code -> sorted list of (date, amount)
        buyer_dates = {}
        for inv in graph_invoices:
            s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
            d = inv["date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            seller_dates.setdefault(s, []).append(d)
            buyer_dates.setdefault(b, []).append(d)

        # Sort all date lists for binary search
        for k in pair_dates:
            pair_dates[k].sort()
        for k in seller_dates:
            seller_dates[k].sort()
        for k in buyer_dates:
            buyer_dates[k].sort()

        # ── Pre-compute node temporal aggregates ──
        # avg_interval_days: mean gap between consecutive outgoing invoices
        # burst_score: max invoices in any 48h window
        node_avg_interval = {}
        node_burst_score = {}
        for tc in self.tax_code_to_idx:
            dates = seller_dates.get(tc, [])
            if len(dates) >= 2:
                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                node_avg_interval[tc] = sum(intervals) / len(intervals)
            else:
                node_avg_interval[tc] = 365.0  # no pattern

            # Burst: sliding 48h window
            max_burst = 0
            for i, d in enumerate(dates):
                count = sum(1 for d2 in dates[i:] if (d2 - d).days <= 2)
                max_burst = max(max_burst, count)
            node_burst_score[tc] = max_burst

        # Append temporal features to node feature vectors
        max_burst_global = max(1, max(node_burst_score.values())) if node_burst_score else 1
        for i, c in enumerate(companies):
            tc = c["tax_code"]
            node_features[i].append(min(1.0, node_avg_interval.get(tc, 365) / 365.0))   # avg_interval (normalized)
            node_features[i].append(node_burst_score.get(tc, 0) / max_burst_global)      # burst_score (normalized)

        # ── Build edge features with temporal signals ──
        import bisect

        for inv in graph_invoices:
            s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
            src_list.append(self.tax_code_to_idx[s])
            dst_list.append(self.tax_code_to_idx[b])

            d = inv["date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            days_since = (d - base_date).days

            is_recip = 1.0 if (b, s) in reciprocal_pairs else 0.0

            # ── NEW: Temporal feature 1 — Delta hours to nearest reciprocal ──
            reverse_dates = pair_dates.get((b, s), [])
            if reverse_dates:
                # Find nearest reverse invoice date using binary search
                idx = bisect.bisect_left(reverse_dates, d)
                candidates = []
                if idx > 0:
                    candidates.append(abs((d - reverse_dates[idx-1]).days))
                if idx < len(reverse_dates):
                    candidates.append(abs((d - reverse_dates[idx]).days))
                delta_days_recip = min(candidates) if candidates else 365
            else:
                delta_days_recip = 365  # no reciprocal found

            # Normalize: 0=same day (suspicious), 1=far apart (normal)
            delta_recip_norm = min(1.0, delta_days_recip / 30.0)

            # ── NEW: Temporal features 2-5 — Cyclical time encoding ──
            day_of_week = d.weekday()  # 0=Mon, 6=Sun
            month = d.month  # 1-12
            sin_dow = math.sin(2 * math.pi * day_of_week / 7.0)
            cos_dow = math.cos(2 * math.pi * day_of_week / 7.0)
            sin_month = math.sin(2 * math.pi * (month - 1) / 12.0)
            cos_month = math.cos(2 * math.pi * (month - 1) / 12.0)

            # ── NEW: Temporal feature 6 — Seller velocity (7-day window) ──
            s_dates = seller_dates.get(s, [])
            window_start = d - timedelta(days=7)
            left = bisect.bisect_left(s_dates, window_start)
            right = bisect.bisect_right(s_dates, d)
            seller_vel = (right - left) / max(1, len(s_dates))  # normalized

            amount_val = float(inv.get("amount", 0.0))
            is_smurfing = 1.0 if 19_000_000 <= amount_val <= 20_000_000 else 0.0

            # Geo-Speed approx
            s_lat = float(next((c["lat"] for c in companies if c["tax_code"] == s), 0))
            s_lng = float(next((c["lng"] for c in companies if c["tax_code"] == s), 0))
            b_lat = float(next((c["lat"] for c in companies if c["tax_code"] == b), 0))
            b_lng = float(next((c["lng"] for c in companies if c["tax_code"] == b), 0))
            dist_km = math.sqrt((s_lat - b_lat)**2 + (s_lng - b_lng)**2) * 111.0
            geo_speed = dist_km / max(1.0, days_since) # km per day since epoch
            geo_speed_norm = min(1.0, geo_speed / 1000.0) # normalize

            edge_features.append([
                self._normalize_amount_feature(amount_val),  # [0] normalized amount
                float(inv.get("vat_rate", 10.0)) / 100.0,  # [1] normalized vat
                days_since / max(1, date_span_days),        # [2] normalized time position
                is_recip,                                    # [3] reciprocal flag
                delta_recip_norm,                            # [4] NEW: time gap to nearest reciprocal
                sin_dow,                                     # [5] NEW: cyclical day-of-week (sin)
                cos_dow,                                     # [6] NEW: cyclical day-of-week (cos)
                sin_month,                                   # [7] NEW: cyclical month (sin)
                cos_month,                                   # [8] NEW: cyclical month (cos)
                seller_vel,                                  # [9] NEW: seller velocity (7-day)
                0.0,                                         # [10] is_ownership_edge (0 for invoice)
                is_smurfing,                                 # [11] is_near_limit_smurfing
                geo_speed_norm,                              # [12] geo_speed (teleportation)
            ])

        # ── MULTIPLEX: Append Ownership Links as Edges ──
        for own in ownership_links:
            s, b = own.get("parent_tax_code"), own.get("child_tax_code")
            if s in self.tax_code_to_idx and b in self.tax_code_to_idx:
                src_list.append(self.tax_code_to_idx[s])
                dst_list.append(self.tax_code_to_idx[b])
                
                # Ownership edges have blank financial parameters
                edge_features.append([
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, # is_ownership_edge
                    0.0, # smurfing
                    0.0, # geo_speed
                ])

        # Convert to tensors securely if PyGeom is available
        if PYGEOM_AVAILABLE:
            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) if src_list else torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros((0, 13), dtype=torch.float32)
    
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.num_nodes = n_nodes
        else:
            data = None

        # Store ONLY invoices that became edges (1:1 with edge_index columns)
        self._sorted_invoices = graph_invoices

        return data

    def detect_cycles_networkx(self, invoices: list[dict]) -> list[list[str]]:
        """Use NetworkX to find simple cycles (rings) in the invoice graph."""
        G = nx.DiGraph()
        for inv in invoices:
            s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
            G.add_edge(s, b, amount=float(inv["amount"]))

        cycles = []
        max_cycle_len = max(3, min(20, int(os.getenv("GNN_CYCLE_MAX_LEN", "10"))))
        max_cycle_count = max(1, min(5000, int(os.getenv("GNN_CYCLE_MAX_COUNT", "1200"))))
        try:
            for cycle in nx.simple_cycles(G, length_bound=max_cycle_len):
                if len(cycle) >= 3:
                    cycles.append(cycle)
                if len(cycles) >= max_cycle_count:
                    break
        except Exception:
            pass
        return cycles


# ════════════════════════════════════════════════════════════════
#  3. Training Pipeline
# ════════════════════════════════════════════════════════════════

class GNNTrainer:
    """Train and save the GAT model."""

    def __init__(self, node_feat_dim: int = 22, edge_feat_dim: int = 13, amount_feature_mode: str = "legacy"):
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.amount_feature_mode = str(amount_feature_mode or "legacy").strip().lower()
        self.model = TaxFraudGAT(node_feat_dim, edge_feat_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)

    def train(self, data: "Data", node_labels: torch.Tensor,
              edge_labels: torch.Tensor, epochs: int = 200,
              node_train_mask: Optional[torch.Tensor] = None,
              node_val_mask: Optional[torch.Tensor] = None,
              edge_train_mask: Optional[torch.Tensor] = None,
              edge_val_mask: Optional[torch.Tensor] = None):
        """
        Train the GAT model with full temporal masks.

        CRITICAL: Evaluation runs in model.eval() to disable dropout.
        Training runs in model.train() to enable dropout.
        """
        from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
        
        if node_train_mask is None:
            node_train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        if node_val_mask is None:
            node_val_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        if edge_train_mask is None:
            edge_train_mask = torch.ones(edge_labels.shape[0], dtype=torch.bool)
        if edge_val_mask is None:
            edge_val_mask = torch.ones(edge_labels.shape[0], dtype=torch.bool)

        # Handle class imbalance with pos_weight based on train mask
        n_pos_node = max(1, node_labels[node_train_mask].sum().item())
        n_neg_node = max(1, node_train_mask.sum().item() - n_pos_node)
        node_pos_weight = torch.tensor([n_neg_node / n_pos_node])

        n_pos_edge = max(1, edge_labels[edge_train_mask].sum().item())
        n_neg_edge = max(1, edge_train_mask.sum().item() - n_pos_edge)
        edge_pos_weight = torch.tensor([n_neg_edge / n_pos_edge])

        for epoch in range(epochs):
            # ── TRAIN step (dropout ON) ──
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data)

            # Node loss on train_mask
            node_loss = F.binary_cross_entropy_with_logits(
                out["node_logits"][node_train_mask], node_labels[node_train_mask].float(),
                pos_weight=node_pos_weight
            )

            # Edge loss on train_mask
            edge_loss = F.binary_cross_entropy_with_logits(
                out["edge_logits"][edge_train_mask], edge_labels[edge_train_mask].float(),
                pos_weight=edge_pos_weight
            )

            loss = node_loss + edge_loss
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 50 == 0:
                # ── EVAL step (dropout OFF) — CRITICAL for honest metrics ──
                self.model.eval()
                with torch.no_grad():
                    eval_out = self.model(data)
                    node_probs = torch.sigmoid(eval_out["node_logits"][node_val_mask]).numpy()
                    edge_probs = torch.sigmoid(eval_out["edge_logits"][edge_val_mask]).numpy()
                    node_pred = (node_probs > 0.5).astype(int)
                    edge_pred = (edge_probs > 0.5).astype(int)
                    
                    nl = node_labels[node_val_mask].numpy()
                    el = edge_labels[edge_val_mask].numpy()
                    
                    node_f1 = f1_score(nl, node_pred, zero_division=0)
                    edge_f1 = f1_score(el, edge_pred, zero_division=0)
                    
                    try:
                        node_pr_auc = average_precision_score(nl, node_probs)
                    except ValueError:
                        node_pr_auc = 0.0
                    try:
                        edge_pr_auc = average_precision_score(el, edge_probs)
                    except ValueError:
                        edge_pr_auc = 0.0

                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                      f"VAL F1(Node={node_f1:.3f}, Edge={edge_f1:.3f}) | "
                      f"PR-AUC(Node={node_pr_auc:.3f}, Edge={edge_pr_auc:.3f})")

    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else MODEL_DIR
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path / "gat_model.pt")
        # Save config
        config = {
            "node_feat_dim": self.node_feat_dim,
            "edge_feat_dim": self.edge_feat_dim,
            "amount_feature_mode": self.amount_feature_mode,
        }
        with open(save_path / "gat_config.json", "w") as f:
            json.dump(config, f)
        print(f"[OK] GAT model saved to {save_path}")


# ════════════════════════════════════════════════════════════════
#  4. Inference Pipeline
# ════════════════════════════════════════════════════════════════

class GNNInference:
    """Load a trained GAT model and run inference on subgraphs."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model = None
        default_amount_mode = os.getenv("GNN_AMOUNT_FEATURE_MODE", "legacy")
        self.graph_constructor = GraphConstructor(amount_feature_mode=default_amount_mode)
        self.calibrator = None
        self._loaded = False

    def load(self):
        config_path = self.model_dir / "gat_config.json"
        model_path = self.model_dir / "gat_model.pt"

        if not config_path.exists() or not model_path.exists():
            print("[WARN] GAT model not found. Will use heuristic fallback.")
            self._loaded = False
            return

        with open(config_path) as f:
            config = json.load(f)

        self.model = TaxFraudGAT(
            node_feat_dim=config["node_feat_dim"],
            edge_feat_dim=config["edge_feat_dim"],
        )
        self.graph_constructor = GraphConstructor(
            amount_feature_mode=config.get("amount_feature_mode", "legacy")
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()  # CRITICAL: inference always in eval mode
        self._loaded = True
        print(f"[OK] GAT model loaded from {self.model_dir}")

        # Load calibrator if available
        try:
            from ml_engine.metrics_engine import ProbabilityCalibrator
            self.calibrator = ProbabilityCalibrator()
            if not self.calibrator.load(str(self.model_dir)):
                self.calibrator = None
        except Exception as e:
            print(f"[WARN] Calibrator not loaded: {e}")
            self.calibrator = None

        # Load ensemble components
        try:
            from ml_engine.ensemble import AnomalyDetector, EnsembleModel, HeuristicRuleScorer, PathEvidenceExtractor
            self.anomaly_detector = AnomalyDetector()
            if not self.anomaly_detector.load(str(self.model_dir)):
                self.anomaly_detector = None
            self.ensemble_model = EnsembleModel()
            if not self.ensemble_model.load(str(self.model_dir)):
                self.ensemble_model = None
            self.rule_scorer = HeuristicRuleScorer()
            self.path_extractor = PathEvidenceExtractor()
        except Exception as e:
            print(f"[WARN] Ensemble components not loaded: {e}")
            self.anomaly_detector = None
            self.ensemble_model = None
            self.rule_scorer = None
            self.path_extractor = None

    def predict(self, companies: list[dict], invoices: list[dict], ownership_links: list[dict] = None) -> dict:
        """
        Run inference on a subgraph.

        Returns:
            {
                "nodes": [{"tax_code", "name", "risk_score", "is_shell", "shell_probability", ...}],
                "edges": [{"from", "to", "amount", "is_circular", "circular_probability", ...}],
                "cycles": [[tax_code, ...], ...],
                "total_suspicious_amount": float,
                "logs": [{"timestamp", "severity", "title", "description"}, ...],
                "attention_weights": [...],
            }
        """
        # Build graph
        data = self.graph_constructor.build_graph(companies, invoices, ownership_links)
        tc_to_idx = self.graph_constructor.tax_code_to_idx
        idx_to_tc = {v: k for k, v in tc_to_idx.items()}
        company_map = {c["tax_code"]: c for c in companies}

        # Detect cycles using NetworkX (deterministic, always works)
        cycles = self.graph_constructor.detect_cycles_networkx(invoices)
        cycle_edges = set()
        cycle_nodes = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                s, b = cycle[i], cycle[(i + 1) % len(cycle)]
                cycle_edges.add((s, b))
                cycle_nodes.add(s)
                cycle_nodes.add(b)

        # ── GNN Inference ──
        node_probs = {}
        edge_probs = {}
        attention_data = []

        if self._loaded and PYGEOM_AVAILABLE and self.model is not None and data is not None and data.num_nodes > 0 and data.edge_index.shape[1] > 0:
            with torch.no_grad():
                out = self.model(data, return_attention=True)
                node_sigmoid = torch.sigmoid(out["node_logits"]).numpy()
                edge_sigmoid = torch.sigmoid(out["edge_logits"]).numpy()

                # Apply calibration if available
                if self.calibrator:
                    node_sigmoid = self.calibrator.calibrate_nodes_batch(node_sigmoid)
                    edge_sigmoid = self.calibrator.calibrate_edges_batch(edge_sigmoid)

                for i in range(data.num_nodes):
                    tc = idx_to_tc.get(i, "")
                    node_probs[tc] = float(node_sigmoid[i])

                # Use sorted_invoices from GraphConstructor for stable mapping
                sorted_invs = self.graph_constructor._sorted_invoices
                for e_idx in range(data.edge_index.shape[1]):
                    if e_idx < len(sorted_invs):
                        inv_num = sorted_invs[e_idx].get("invoice_number", "")
                        edge_probs[inv_num] = float(edge_sigmoid[e_idx])

                # Extract attention weights
                if "attention_weights" in out:
                    attn_idx, attn_vals = out["attention_weights"]
                    for e_idx in range(attn_idx.shape[1]):
                        src = idx_to_tc.get(attn_idx[0, e_idx].item(), "")
                        dst = idx_to_tc.get(attn_idx[1, e_idx].item(), "")
                        weight = float(attn_vals[e_idx].mean().item())
                        attention_data.append({"from": src, "to": dst, "weight": round(weight, 4)})

        # ── Build output nodes (with ensemble scoring) ──
        result_nodes = []
        node_decision_threshold = 0.5
        edge_decision_threshold = 0.5
        cold_start_degree_threshold = 2
        cold_start_threshold_delta = 0.0
        node_blend_alpha_gnn = 0.0
        if self.calibrator is not None:
            node_decision_threshold = float(getattr(self.calibrator, "node_threshold", 0.5))
            edge_decision_threshold = float(getattr(self.calibrator, "edge_threshold", 0.5))
            threshold_meta = getattr(self.calibrator, "threshold_meta", {}) or {}
            cold_start_degree_threshold = int(threshold_meta.get("cold_start_degree_threshold", 2))
            cold_start_threshold_delta = float(threshold_meta.get("cold_start_threshold_delta", 0.0))
            node_blend_alpha_gnn = float(threshold_meta.get("node_blend_alpha_gnn", 0.0))
        
        # Pre-compute aggregates for rule scorer
        in_amount_map = {}
        out_amount_map = {}
        degree_map = {}
        reciprocal_pairs_set = set()
        for inv in invoices:
            s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
            out_amount_map[s] = out_amount_map.get(s, 0) + float(inv["amount"])
            in_amount_map[b] = in_amount_map.get(b, 0) + float(inv["amount"])
            degree_map[s] = degree_map.get(s, 0) + 1
            degree_map[b] = degree_map.get(b, 0) + 1
            reciprocal_pairs_set.add((s, b))

        # Get anomaly scores if available
        node_anomaly = {}
        if self.anomaly_detector and data is not None and data.num_nodes > 0:
            try:
                a_scores = self.anomaly_detector.score_nodes(data.x.numpy())
                for i in range(data.num_nodes):
                    node_anomaly[idx_to_tc.get(i, "")] = float(a_scores[i])
            except Exception:
                pass

        for c in companies:
            tc = c["tax_code"]
            gnn_score = node_probs.get(tc, 0.0)
            inferred_country_raw = str(c.get("country_inferred") or "").strip()
            inferred_country = inferred_country_raw or "Vietnam"
            is_within_vietnam = c.get("is_within_vietnam")
            is_offshore = (
                is_within_vietnam is False
                or (inferred_country_raw and inferred_country_raw.lower() not in {"vietnam", "việt nam"})
                or str(c.get("industry") or "").strip().lower() == "offshore entity"
            )
            
            # Rule score
            rule_score = 0.0
            if self.rule_scorer:
                rule_score = self.rule_scorer.score_node(
                    c, cycle_nodes,
                    in_amount_map.get(tc, 0), out_amount_map.get(tc, 0),
                    degree_map.get(tc, 0)
                )
            else:
                if tc in cycle_nodes:
                    rule_score = 0.4
                reg = c.get("registration_date")
                if reg:
                    if isinstance(reg, str):
                        reg = date.fromisoformat(reg)
                    age_days = (date.today() - reg).days
                    if age_days < 180:
                        rule_score += 0.35
                    elif age_days < 365:
                        rule_score += 0.25

            anomaly_score = node_anomaly.get(tc, 0.5)

            # Ensemble branch score
            if self.ensemble_model and self._loaded:
                ensemble_shell = self.ensemble_model.predict_node(gnn_score, rule_score, anomaly_score)
            elif self._loaded:
                ensemble_shell = min(1.0, gnn_score * 0.5 + rule_score * 0.3 + anomaly_score * 0.2)
            else:
                ensemble_shell = rule_score

            # Final node score can blend calibrated GNN ranking with ensemble robustness.
            if self._loaded:
                combined_shell = (
                    node_blend_alpha_gnn * gnn_score
                    + (1.0 - node_blend_alpha_gnn) * ensemble_shell
                )
            else:
                combined_shell = ensemble_shell

            tc_degree = int(degree_map.get(tc, 0))
            adaptive_threshold = node_decision_threshold
            if tc_degree <= cold_start_degree_threshold and cold_start_threshold_delta > 0:
                adaptive_threshold = max(0.05, node_decision_threshold - cold_start_threshold_delta)

            is_shell = combined_shell >= adaptive_threshold
            node_risk = round(combined_shell * 100, 1)

            result_nodes.append({
                "id": tc,
                "tax_code": tc,
                "label": c.get("name", tc),
                "industry": c.get("industry", ""),
                "registration_date": str(c.get("registration_date", "")),
                "lat": c.get("lat", 0),
                "lng": c.get("lng", 0),
                "country_inferred": inferred_country,
                "is_within_vietnam": is_within_vietnam,
                "is_offshore": is_offshore,
                "risk_score": node_risk,
                "is_shell": is_shell,
                "shell_probability": round(combined_shell, 4),
                "node_degree": tc_degree,
                "decision_threshold": round(float(adaptive_threshold), 4),
                "group": "shell" if is_shell else ("suspicious" if tc in cycle_nodes else "normal"),
            })

        # ── Build output edges (with ensemble scoring) ──
        result_edges = []
        total_suspicious_amount = 0.0

        # Get edge anomaly scores if available
        edge_anomaly = {}
        if self.anomaly_detector and data is not None and data.edge_index.shape[1] > 0:
            try:
                sorted_invs = self.graph_constructor._sorted_invoices
                ea_scores = self.anomaly_detector.score_edges(data.edge_attr.numpy())
                for e_idx in range(len(sorted_invs)):
                    inv_num = sorted_invs[e_idx].get("invoice_number", "")
                    edge_anomaly[inv_num] = float(ea_scores[e_idx])
            except Exception:
                pass

        for i, inv in enumerate(invoices):
            s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
            if s not in tc_to_idx or b not in tc_to_idx:
                continue

            inv_num = inv.get("invoice_number", "")
            gnn_score = edge_probs.get(inv_num, 0.0)

            # Rule score
            rule_score = 0.0
            if self.rule_scorer:
                rule_score = self.rule_scorer.score_edge(inv, cycle_edges, reciprocal_pairs_set)
            else:
                if (s, b) in cycle_edges:
                    rule_score = 0.8

            anomaly_score = edge_anomaly.get(inv_num, 0.5)

            # Ensemble final score
            if self.ensemble_model and self._loaded:
                combined_circ = self.ensemble_model.predict_edge(gnn_score, rule_score, anomaly_score)
            elif self._loaded:
                combined_circ = min(1.0, gnn_score * 0.5 + rule_score * 0.3 + anomaly_score * 0.2)
            else:
                combined_circ = rule_score

            is_circular = combined_circ >= edge_decision_threshold
            amount = float(inv["amount"])
            if is_circular:
                total_suspicious_amount += amount

            result_edges.append({
                "from": s,
                "to": b,
                "amount": amount,
                "vat_rate": float(inv.get("vat_rate", 10.0)),
                "date": str(inv.get("date", "")),
                "invoice_number": inv.get("invoice_number", ""),
                "is_circular": is_circular,
                "circular_probability": round(combined_circ, 4),
                "group": "circular" if is_circular else "normal",
            })

        circular_edge_count = sum(1 for edge in result_edges if edge.get("is_circular"))
        cycle_backed_circular_edges = sum(
            1
            for edge in result_edges
            if edge.get("is_circular") and (edge.get("from"), edge.get("to")) in cycle_edges
        )
        circular_edge_cycle_coverage = (
            float(cycle_backed_circular_edges) / float(max(1, circular_edge_count))
        )

        # ── Extract path-level evidence ──
        evidence_paths = []
        if self.path_extractor and cycles:
            evidence_paths = self.path_extractor.extract_paths(
                cycles, invoices, edge_probs, node_probs
            )

        # ── Generate investigation logs ──
        logs = self._generate_logs(result_nodes, result_edges, cycles, total_suspicious_amount)

        return {
            "nodes": result_nodes,
            "edges": result_edges,
            "cycles": cycles,
            "evidence_paths": evidence_paths,
            "total_suspicious_amount": round(total_suspicious_amount, 0),
            "total_suspicious_invoices": sum(1 for e in result_edges if e["is_circular"]),
            "total_companies": len(result_nodes),
            "total_invoices": len(result_edges),
            "logs": logs,
            "attention_weights": attention_data[:50],
            "forensic_metrics": {
                "cycle_count": len(cycles),
                "circular_edge_count": circular_edge_count,
                "cycle_backed_circular_edge_count": cycle_backed_circular_edges,
                "circular_edge_cycle_coverage": round(circular_edge_cycle_coverage, 4),
                "ghost_node_count": sum(1 for c in companies if c.get("is_ghost")),
                "smurfing_count": sum(1 for e in invoices if 19_000_000 <= float(e.get("amount", 0)) <= 20_000_000),
                "teleportation_count": int(torch.sum(data.edge_attr[:, 12] > 0.5).item()) if data is not None and data.edge_attr is not None and data.edge_attr.shape[1] > 12 else 0,
            },
            "model_loaded": self._loaded,
            "ensemble_active": self.ensemble_model is not None,
            "decision_thresholds": {
                "node": round(node_decision_threshold, 4),
                "edge": round(edge_decision_threshold, 4),
                "policy": {
                    "cold_start_degree_threshold": int(cold_start_degree_threshold),
                    "cold_start_threshold_delta": round(float(cold_start_threshold_delta), 4),
                    "node_blend_alpha_gnn": round(float(node_blend_alpha_gnn), 4),
                },
            },
        }

    def _generate_logs(self, nodes, edges, cycles, total_amount) -> list[dict]:
        """Generate investigation trail logs (Nhật ký truy vết)."""
        from datetime import datetime
        logs = []
        now = datetime.now()
        max_cycle_logs = max(1, min(50, int(os.getenv("GNN_CYCLE_LOG_MAX_ITEMS", "10"))))

        # Log 1: Summary
        shell_nodes = [n for n in nodes if n["is_shell"]]
        circ_edges = [e for e in edges if e["is_circular"]]

        if cycles:
            logs.append({
                "timestamp": (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "critical",
                "title": f"Phát hiện {len(cycles)} vòng lặp giao dịch tuần hoàn",
                "description": f"Hệ thống AI (GAT) phát hiện {len(cycles)} chuỗi giao dịch xoay vòng "
                               f"với tổng giá trị {total_amount:,.0f} VNĐ. "
                               f"Các vòng lặp bao gồm {len(set().union(*cycles))} chủ thể.",
            })

        if shell_nodes:
            names = ", ".join([n["label"][:20] for n in shell_nodes[:3]])
            logs.append({
                "timestamp": (now - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "high",
                "title": f"Nhận diện {len(shell_nodes)} chủ thể vỏ bọc (Shell Corp)",
                "description": f"Danh sách nổi bật theo xác suất shell: {names}...",
            })

        for cycle in cycles[:max_cycle_logs]:
            cycle_str = " → ".join(cycle[:4]) + f" → {cycle[0]}"
            logs.append({
                "timestamp": (now - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "critical",
                "title": f"Xoay vòng: {cycle_str}",
                "description": f"Chu trình có {len(cycle)} chủ thể, được trích xuất trực tiếp từ đồ thị giao dịch có hướng.",
            })

        if len(cycles) > max_cycle_logs:
            logs.append({
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "medium",
                "title": f"Hiển thị {max_cycle_logs}/{len(cycles)} vòng giao dịch",
                "description": "Nhật ký đang hiển thị mẫu các vòng có mức ưu tiên cao nhất.",
            })

        if circ_edges:
            # Find the largest circular invoice
            largest = max(circ_edges, key=lambda e: e["amount"])
            logs.append({
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "medium",
                "title": f"Hoá đơn lớn nhất: {largest['invoice_number']}",
                "description": f"Giá trị {largest['amount']:,.0f} VNĐ từ {largest['from']} → {largest['to']}. "
                               f"Xác suất gian lận xoay vòng: {largest['circular_probability']*100:.0f}%.",
            })

        if not logs:
            logs.append({
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "low",
                "title": "Không phát hiện bất thường",
                "description": "Mạng lưới giao dịch hiện tại không có dấu hiệu gian lận tuần hoàn.",
            })

        return logs
