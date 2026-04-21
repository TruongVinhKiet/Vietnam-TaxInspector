"""
evaluate_gnn_stress.py - Harder-data evaluation suite for VAT GNN.

Scenarios:
  1) Class imbalance shift
  2) Covariate shift (invoice amount)
  3) Unseen nodes
  4) Unseen edges (test-time novel seller-buyer pairs)
  5) Temporal drift (date projection)

Usage:
  python -m app.scripts.evaluate_gnn_stress
"""

import argparse
import copy
import json
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

from ml_engine.ensemble import AnomalyDetector, EnsembleModel, HeuristicRuleScorer
from ml_engine.gnn_model import GraphConstructor, TaxFraudGAT
from ml_engine.metrics_engine import ProbabilityCalibrator, TemporalMaskGenerator

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "TaxInspector")


def _parse_date(value):
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    return None


def _add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = ((d.month - 1 + months) % 12) + 1
    day = min(d.day, 28)
    return date(year, month, day)


def _safe_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int) if len(probs) else np.array([], dtype=int)

    result = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pr_auc": 0.0,
        "roc_auc": 0.0,
        "support": int(len(labels)),
        "positive_support": int(labels.sum()) if len(labels) else 0,
        "threshold": float(threshold),
    }

    if len(labels) == 0:
        return result

    result["f1"] = float(f1_score(labels, preds, zero_division=0))
    result["precision"] = float(precision_score(labels, preds, zero_division=0))
    result["recall"] = float(recall_score(labels, preds, zero_division=0))

    unique = np.unique(labels)
    if len(unique) == 1:
        result["pr_auc"] = 1.0 if int(unique[0]) == 1 else 0.0
    else:
        result["pr_auc"] = float(average_precision_score(labels, probs))
        result["roc_auc"] = float(roc_auc_score(labels, probs))

    return result


def _safe_binary_metrics_from_preds(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = np.asarray(preds)

    result = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pr_auc": 0.0,
        "roc_auc": 0.0,
        "support": int(len(labels)),
        "positive_support": int(labels.sum()) if len(labels) else 0,
        "threshold": float(threshold),
    }

    if len(labels) == 0:
        return result

    result["f1"] = float(f1_score(labels, preds, zero_division=0))
    result["precision"] = float(precision_score(labels, preds, zero_division=0))
    result["recall"] = float(recall_score(labels, preds, zero_division=0))

    unique = np.unique(labels)
    if len(unique) == 1:
        result["pr_auc"] = 1.0 if int(unique[0]) == 1 else 0.0
    else:
        result["pr_auc"] = float(average_precision_score(labels, probs))
        result["roc_auc"] = float(roc_auc_score(labels, probs))

    return result


def _apply_node_threshold_policy(
    probs: np.ndarray,
    node_degrees: np.ndarray,
    base_threshold: float,
    cold_start_degree_threshold: int,
    cold_start_threshold_delta: float,
):
    probs = np.asarray(probs)
    node_degrees = np.asarray(node_degrees)
    thresholds = np.full(len(probs), float(base_threshold), dtype=float)

    if len(node_degrees) == len(probs) and cold_start_threshold_delta > 0:
        low_degree_mask = node_degrees <= int(cold_start_degree_threshold)
        thresholds[low_degree_mask] = np.maximum(0.05, base_threshold - float(cold_start_threshold_delta))

    preds = (probs >= thresholds).astype(int)
    return preds, thresholds


def _recall_at_k(labels: np.ndarray, probs: np.ndarray, k: int = 20) -> float:
    if len(labels) == 0:
        return 0.0
    k = max(1, min(k, len(labels)))
    top_idx = np.argsort(probs)[-k:]
    positives_in_top = float(labels[top_idx].sum())
    total_positives = float(max(1, labels.sum()))
    return positives_in_top / total_positives


class StressEvaluator:
    def __init__(self, model_dir: Path, seed: int = 42):
        self.model_dir = Path(model_dir)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.model = None
        self.calibrator = ProbabilityCalibrator()
        self.anomaly = AnomalyDetector()
        self.ensemble = EnsembleModel()
        self.rule_scorer = HeuristicRuleScorer()

        self.node_threshold = 0.5
        self.edge_threshold = 0.5
        self.cold_start_degree_threshold = 2
        self.cold_start_threshold_delta = 0.0
        self.node_blend_alpha_gnn = 0.0
        self.amount_feature_mode = "legacy"

        self._load_artifacts()

    def _load_artifacts(self):
        config_path = self.model_dir / "gat_config.json"
        model_path = self.model_dir / "gat_model.pt"
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError("Missing model artifacts (gat_config.json or gat_model.pt).")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.model = TaxFraudGAT(
            node_feat_dim=int(cfg["node_feat_dim"]),
            edge_feat_dim=int(cfg["edge_feat_dim"]),
        )
        self.amount_feature_mode = str(cfg.get("amount_feature_mode", "legacy"))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

        self.calibrator.load(str(self.model_dir))
        self.anomaly.load(str(self.model_dir))
        self.ensemble.load(str(self.model_dir))

        self.node_threshold = float(getattr(self.calibrator, "node_threshold", 0.5))
        self.edge_threshold = float(getattr(self.calibrator, "edge_threshold", 0.5))
        threshold_meta = getattr(self.calibrator, "threshold_meta", {}) or {}
        self.cold_start_degree_threshold = int(threshold_meta.get("cold_start_degree_threshold", 2))
        self.cold_start_threshold_delta = float(threshold_meta.get("cold_start_threshold_delta", 0.0))
        self.node_blend_alpha_gnn = float(threshold_meta.get("node_blend_alpha_gnn", 0.0))

    def _build_labels(self, constructor: GraphConstructor, data, companies: list[dict]):
        node_labels = np.zeros(data.num_nodes, dtype=np.int64)
        tc_to_idx = constructor.tax_code_to_idx

        for c in companies:
            tc = c.get("tax_code", "")
            idx = tc_to_idx.get(tc)
            if idx is None:
                continue
            try:
                if int(str(tc)[2:7]) >= 900:
                    node_labels[idx] = 1
            except Exception:
                pass

        sorted_invoices = constructor._sorted_invoices
        edge_labels = np.zeros(len(sorted_invoices), dtype=np.int64)
        edge_dates = []
        edge_pairs = []
        for i, inv in enumerate(sorted_invoices):
            inv_num = str(inv.get("invoice_number", ""))
            if inv_num.startswith("CIRC-"):
                edge_labels[i] = 1
            d = _parse_date(inv.get("date"))
            edge_dates.append(d)
            edge_pairs.append((inv.get("seller_tax_code"), inv.get("buyer_tax_code")))

        return node_labels, edge_labels, sorted_invoices, edge_dates, edge_pairs

    def _compute_rule_features(self, companies: list[dict], sorted_invoices: list[dict], tc_to_idx: dict):
        in_amount = {}
        out_amount = {}
        degree = {}
        reciprocal = set()

        for inv in sorted_invoices:
            s = inv.get("seller_tax_code")
            b = inv.get("buyer_tax_code")
            amt = float(inv.get("amount", 0.0))
            out_amount[s] = out_amount.get(s, 0.0) + amt
            in_amount[b] = in_amount.get(b, 0.0) + amt
            degree[s] = degree.get(s, 0) + 1
            degree[b] = degree.get(b, 0) + 1
            reciprocal.add((s, b))

        cycle_nodes = set()
        cycle_edges = set()

        # Fast default cycle heuristic for stress runs: explicit synthetic circular invoices.
        # Optional full NetworkX cycle extraction can be enabled for deeper analysis.
        for inv in sorted_invoices:
            inv_num = str(inv.get("invoice_number", ""))
            if inv_num.startswith("CIRC-"):
                s = inv.get("seller_tax_code")
                b = inv.get("buyer_tax_code")
                cycle_nodes.add(s)
                cycle_nodes.add(b)
                cycle_edges.add((s, b))

        use_networkx_cycles = os.getenv("STRESS_USE_NETWORKX_CYCLES", "0").strip().lower() in {"1", "true", "yes"}
        if use_networkx_cycles:
            try:
                constructor = GraphConstructor()
                cycles = constructor.detect_cycles_networkx(sorted_invoices)
                for cycle in cycles:
                    for i in range(len(cycle)):
                        s = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        cycle_nodes.add(s)
                        cycle_nodes.add(b)
                        cycle_edges.add((s, b))
            except Exception:
                pass

        node_rule = np.zeros(len(tc_to_idx), dtype=float)
        for c in companies:
            tc = c.get("tax_code")
            idx = tc_to_idx.get(tc)
            if idx is None:
                continue
            node_rule[idx] = self.rule_scorer.score_node(
                c,
                cycle_nodes,
                in_amount.get(tc, 0.0),
                out_amount.get(tc, 0.0),
                degree.get(tc, 0),
            )

        edge_rule = np.zeros(len(sorted_invoices), dtype=float)
        for i, inv in enumerate(sorted_invoices):
            edge_rule[i] = self.rule_scorer.score_edge(inv, cycle_edges, reciprocal)

        return node_rule, edge_rule

    def evaluate_dataset(self, companies: list[dict], invoices: list[dict], scenario: str):
        constructor = GraphConstructor(amount_feature_mode=self.amount_feature_mode)
        data = constructor.build_graph(companies, invoices)
        if data is None or data.num_nodes == 0 or data.edge_index.shape[1] == 0:
            raise RuntimeError(f"Scenario {scenario}: graph build failed or empty graph")

        node_labels, edge_labels, sorted_invoices, edge_dates, edge_pairs = self._build_labels(
            constructor, data, companies
        )

        with torch.no_grad():
            out = self.model(data)
            raw_node_probs = torch.sigmoid(out["node_logits"]).numpy()
            raw_edge_probs = torch.sigmoid(out["edge_logits"]).numpy()

        cal_node_probs = self.calibrator.calibrate_nodes_batch(raw_node_probs)
        cal_edge_probs = self.calibrator.calibrate_edges_batch(raw_edge_probs)

        node_rule, edge_rule = self._compute_rule_features(companies, sorted_invoices, constructor.tax_code_to_idx)

        if getattr(self.anomaly, "_fitted", False):
            node_anomaly = self.anomaly.score_nodes(data.x.numpy())
            edge_anomaly = self.anomaly.score_edges(data.edge_attr.numpy())
        else:
            node_anomaly = np.full(len(node_labels), 0.5)
            edge_anomaly = np.full(len(edge_labels), 0.5)

        node_meta = np.column_stack([cal_node_probs, node_rule, node_anomaly])
        edge_meta = np.column_stack([cal_edge_probs, edge_rule, edge_anomaly])

        if getattr(self.ensemble, "_fitted", False):
            ensemble_node_probs = self.ensemble.predict_nodes_batch(node_meta)
            serving_edge_probs = self.ensemble.predict_edges_batch(edge_meta)
        else:
            ensemble_node_probs = node_meta[:, 0] * 0.5 + node_meta[:, 1] * 0.3 + node_meta[:, 2] * 0.2
            serving_edge_probs = edge_meta[:, 0] * 0.5 + edge_meta[:, 1] * 0.3 + edge_meta[:, 2] * 0.2

        serving_node_probs = (
            self.node_blend_alpha_gnn * cal_node_probs
            + (1.0 - self.node_blend_alpha_gnn) * ensemble_node_probs
        )

        idx_to_tc = [""] * len(constructor.tax_code_to_idx)
        for tc, idx in constructor.tax_code_to_idx.items():
            idx_to_tc[idx] = tc

        node_degree_map = {}
        for inv in sorted_invoices:
            s = inv.get("seller_tax_code")
            b = inv.get("buyer_tax_code")
            node_degree_map[s] = node_degree_map.get(s, 0) + 1
            node_degree_map[b] = node_degree_map.get(b, 0) + 1
        node_degrees = np.array([int(node_degree_map.get(tc, 0)) for tc in idx_to_tc], dtype=int)

        serving_node_preds, serving_node_thresholds = _apply_node_threshold_policy(
            probs=serving_node_probs,
            node_degrees=node_degrees,
            base_threshold=self.node_threshold,
            cold_start_degree_threshold=self.cold_start_degree_threshold,
            cold_start_threshold_delta=self.cold_start_threshold_delta,
        )

        result = {
            "scenario": scenario,
            "raw": {
                "node": _safe_binary_metrics(node_labels, raw_node_probs, threshold=0.5),
                "edge": _safe_binary_metrics(edge_labels, raw_edge_probs, threshold=0.5),
            },
            "serving": {
                "node": _safe_binary_metrics_from_preds(
                    node_labels,
                    serving_node_probs,
                    serving_node_preds,
                    threshold=self.node_threshold,
                ),
                "edge": _safe_binary_metrics(edge_labels, serving_edge_probs, threshold=self.edge_threshold),
                "node_recall_at_k": _recall_at_k(node_labels, serving_node_probs, k=20),
            },
            "thresholds": {
                "node": self.node_threshold,
                "edge": self.edge_threshold,
                "policy": {
                    "cold_start_degree_threshold": self.cold_start_degree_threshold,
                    "cold_start_threshold_delta": self.cold_start_threshold_delta,
                    "node_blend_alpha_gnn": self.node_blend_alpha_gnn,
                },
            },
            "context": {
                "node_labels": node_labels,
                "edge_labels": edge_labels,
                "node_probs": serving_node_probs,
                "node_preds": serving_node_preds,
                "node_thresholds": serving_node_thresholds,
                "node_degrees": node_degrees,
                "edge_probs": serving_edge_probs,
                "node_tax_codes": idx_to_tc,
                "edge_dates": edge_dates,
                "edge_pairs": edge_pairs,
                "sorted_invoices": sorted_invoices,
            },
        }
        return result


class StressScenarios:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _sample_for_ratio(pos_idx, neg_idx, ratio: float, rng):
        if ratio <= 0 or ratio >= 1 or len(pos_idx) == 0 or len(neg_idx) == 0:
            return np.array([], dtype=int)

        p = len(pos_idx)
        n = len(neg_idx)
        candidates = []

        neg_for_all_pos = int(round(p * (1 - ratio) / ratio))
        if 0 < neg_for_all_pos <= n:
            candidates.append((p + neg_for_all_pos, p, neg_for_all_pos))

        pos_for_all_neg = int(round(n * ratio / (1 - ratio)))
        if 0 < pos_for_all_neg <= p:
            candidates.append((n + pos_for_all_neg, pos_for_all_neg, n))

        if not candidates:
            pos_take = max(1, min(p, int(round(n * ratio / (1 - ratio)))))
            neg_take = max(1, min(n, int(round(pos_take * (1 - ratio) / ratio))))
        else:
            _, pos_take, neg_take = max(candidates, key=lambda x: x[0])

        sel_pos = rng.choice(pos_idx, size=pos_take, replace=False)
        sel_neg = rng.choice(neg_idx, size=neg_take, replace=False)
        return np.concatenate([sel_pos, sel_neg])

    def class_imbalance_shift(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        threshold: float,
        node_degrees: Optional[np.ndarray] = None,
        cold_start_degree_threshold: int = 2,
        cold_start_threshold_delta: float = 0.0,
    ):
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        ratios = [0.05, 0.2, 0.5, 0.8]
        report = {}

        for r in ratios:
            idx = self._sample_for_ratio(pos_idx, neg_idx, r, self.rng)
            key = f"ratio_{str(r).replace('.', '_')}"
            if len(idx) < 10:
                report[key] = {
                    "status": "insufficient_data",
                    "target_ratio": r,
                    "support": int(len(idx)),
                }
                continue

            y = labels[idx]
            p = probs[idx]
            if node_degrees is not None:
                d = np.asarray(node_degrees)[idx]
                preds, _ = _apply_node_threshold_policy(
                    probs=p,
                    node_degrees=d,
                    base_threshold=threshold,
                    cold_start_degree_threshold=cold_start_degree_threshold,
                    cold_start_threshold_delta=cold_start_threshold_delta,
                )
                m = _safe_binary_metrics_from_preds(y, p, preds, threshold=threshold)
            else:
                m = _safe_binary_metrics(y, p, threshold=threshold)
            m["actual_ratio"] = float(y.mean()) if len(y) else 0.0
            report[key] = m

        return report

    @staticmethod
    def _shift_amounts(invoices: list[dict], mode: str):
        shifted = []
        for i, inv in enumerate(invoices):
            x = copy.deepcopy(inv)
            amt = float(x.get("amount", 0.0))
            if mode == "micro":
                amt = max(10_000.0, min(amt * 0.08, 50_000_000.0))
            elif mode == "mega":
                amt = max(1_000_000_000.0, min(amt * 8.0, 10_000_000_000.0))
            elif mode == "bimodal":
                if i % 2 == 0:
                    amt = max(10_000.0, min(amt * 0.08, 50_000_000.0))
                else:
                    amt = max(1_000_000_000.0, min(amt * 8.0, 10_000_000_000.0))
            x["amount"] = round(float(amt), 2)
            shifted.append(x)
        return shifted

    @staticmethod
    def _log_amount_stats(invoices: list[dict]):
        vals = np.array([np.log1p(float(inv.get("amount", 0.0))) for inv in invoices], dtype=float)
        if len(vals) == 0:
            return {"mean": 0.0, "std": 0.0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    def covariate_shift(self, evaluator: StressEvaluator, companies: list[dict], invoices: list[dict], baseline):
        base_stats = self._log_amount_stats(baseline["context"]["sorted_invoices"])
        report = {}

        for mode in ["micro", "mega", "bimodal"]:
            shifted_invoices = self._shift_amounts(invoices, mode)
            res = evaluator.evaluate_dataset(companies, shifted_invoices, scenario=f"covariate_{mode}")
            stats = self._log_amount_stats(res["context"]["sorted_invoices"])
            report[mode] = {
                "serving": res["serving"],
                "raw": res["raw"],
                "amount_log_shift": {
                    "baseline_mean": base_stats["mean"],
                    "scenario_mean": stats["mean"],
                    "delta_mean": float(stats["mean"] - base_stats["mean"]),
                },
                "delta_vs_baseline": {
                    "node_f1": float(res["serving"]["node"]["f1"] - baseline["serving"]["node"]["f1"]),
                    "edge_f1": float(res["serving"]["edge"]["f1"] - baseline["serving"]["edge"]["f1"]),
                    "node_pr_auc": float(res["serving"]["node"]["pr_auc"] - baseline["serving"]["node"]["pr_auc"]),
                    "edge_pr_auc": float(res["serving"]["edge"]["pr_auc"] - baseline["serving"]["edge"]["pr_auc"]),
                },
            }
        return report

    def unseen_nodes(self, evaluator: StressEvaluator, companies: list[dict], invoices: list[dict], num_unseen: int = 20):
        existing_codes = {c["tax_code"] for c in companies}
        unseen_codes = set()
        new_companies = []

        base_lats = [float(c.get("lat", 0.0)) for c in companies if c.get("lat") is not None]
        base_lngs = [float(c.get("lng", 0.0)) for c in companies if c.get("lng") is not None]
        mean_lat = float(np.mean(base_lats)) if base_lats else 16.0
        mean_lng = float(np.mean(base_lngs)) if base_lngs else 106.0

        unseen_industries = [
            "Crypto Services",
            "Green Energy Trading",
            "Maritime Brokerage",
            "Digital Assets",
        ]

        for i in range(num_unseen):
            is_shell = i < max(3, num_unseen // 3)
            prefix = random.choice(["01", "03", "08", "79"])
            # Keep unseen cohort class-balanced under the existing legacy label rule
            # (tax_code[2:7] >= 900 marks shell), while preserving hard unseen shells.
            mid = 95000 + i if is_shell else 100 + i
            suffix = random.randint(10, 99)
            tax_code = f"{prefix}{mid:05d}{suffix}"
            while tax_code in existing_codes or tax_code in unseen_codes:
                suffix = random.randint(10, 99)
                tax_code = f"{prefix}{mid:05d}{suffix}"

            unseen_codes.add(tax_code)
            reg_days = random.randint(30, 220) if is_shell else random.randint(450, 3200)
            new_companies.append({
                "tax_code": tax_code,
                "name": f"Stress Unseen Co {i+1}",
                "industry": random.choice(unseen_industries),
                "registration_date": date.today() - timedelta(days=reg_days),
                "risk_score": float(random.uniform(10, 90)),
                "is_active": True,
                "lat": round(mean_lat + random.uniform(-0.5, 0.5), 6),
                "lng": round(mean_lng + random.uniform(-0.5, 0.5), 6),
            })

        extended_companies = companies + new_companies
        extended_invoices = list(invoices)

        shells = [c for c in new_companies if int(str(c["tax_code"])[2:7]) >= 900]
        if len(shells) >= 3:
            base_day = date(2024, 10, 1)
            for i in range(len(shells)):
                s = shells[i]
                b = shells[(i + 1) % len(shells)]
                for j in range(3):
                    extended_invoices.append({
                        "seller_tax_code": s["tax_code"],
                        "buyer_tax_code": b["tax_code"],
                        "amount": round(random.uniform(800_000_000, 4_000_000_000), 2),
                        "vat_rate": 10.0,
                        "date": base_day + timedelta(days=i * 2 + j),
                        "invoice_number": f"CIRC-UNS-{i+1:02d}-{j+1:02d}",
                    })

        existing_sample = random.sample(companies, k=min(30, len(companies)))
        for i, u in enumerate(new_companies):
            peer = random.choice(existing_sample)
            for j in range(2):
                extended_invoices.append({
                    "seller_tax_code": u["tax_code"],
                    "buyer_tax_code": peer["tax_code"],
                    "amount": round(random.uniform(30_000_000, 1_200_000_000), 2),
                    "vat_rate": random.choice([8.0, 10.0]),
                    "date": date(2024, random.randint(10, 12), random.randint(1, 28)),
                    "invoice_number": f"UNS-N-{i+1:03d}-{j+1:02d}",
                })

        res = evaluator.evaluate_dataset(extended_companies, extended_invoices, scenario="unseen_nodes")

        node_labels = res["context"]["node_labels"]
        node_probs = res["context"]["node_probs"]
        node_degrees = res["context"].get("node_degrees")
        node_tax_codes = np.array(res["context"]["node_tax_codes"])

        unseen_mask = np.isin(node_tax_codes, list(unseen_codes))
        seen_mask = ~unseen_mask

        if node_degrees is not None:
            node_degrees = np.asarray(node_degrees)
            unseen_preds, _ = _apply_node_threshold_policy(
                probs=node_probs[unseen_mask],
                node_degrees=node_degrees[unseen_mask],
                base_threshold=evaluator.node_threshold,
                cold_start_degree_threshold=evaluator.cold_start_degree_threshold,
                cold_start_threshold_delta=evaluator.cold_start_threshold_delta,
            )
            seen_preds, _ = _apply_node_threshold_policy(
                probs=node_probs[seen_mask],
                node_degrees=node_degrees[seen_mask],
                base_threshold=evaluator.node_threshold,
                cold_start_degree_threshold=evaluator.cold_start_degree_threshold,
                cold_start_threshold_delta=evaluator.cold_start_threshold_delta,
            )
            unseen_metrics = _safe_binary_metrics_from_preds(
                node_labels[unseen_mask],
                node_probs[unseen_mask],
                unseen_preds,
                threshold=evaluator.node_threshold,
            )
            seen_metrics = _safe_binary_metrics_from_preds(
                node_labels[seen_mask],
                node_probs[seen_mask],
                seen_preds,
                threshold=evaluator.node_threshold,
            )
        else:
            unseen_metrics = _safe_binary_metrics(
                node_labels[unseen_mask],
                node_probs[unseen_mask],
                threshold=evaluator.node_threshold,
            )
            seen_metrics = _safe_binary_metrics(
                node_labels[seen_mask],
                node_probs[seen_mask],
                threshold=evaluator.node_threshold,
            )

        seen_support = int(seen_metrics.get("support", 0))
        seen_pos = int(seen_metrics.get("positive_support", 0))
        unseen_support = int(unseen_metrics.get("support", 0))
        unseen_pos = int(unseen_metrics.get("positive_support", 0))

        seen_has_both_classes = seen_pos > 0 and (seen_support - seen_pos) > 0
        unseen_has_both_classes = unseen_pos > 0 and (unseen_support - unseen_pos) > 0
        if seen_has_both_classes and unseen_has_both_classes:
            generalization_gap_f1 = float(seen_metrics["f1"] - unseen_metrics["f1"])
        else:
            generalization_gap_f1 = None

        return {
            "overall": res["serving"],
            "seen_nodes": seen_metrics,
            "unseen_nodes": unseen_metrics,
            "generalization_gap_f1": generalization_gap_f1,
            "unseen_node_count": int(unseen_mask.sum()),
        }

    @staticmethod
    def unseen_edges(baseline_ctx: dict, evaluator: StressEvaluator, train_end_month: int = 8, val_end_month: int = 9):
        edge_labels = baseline_ctx["edge_labels"]
        edge_probs = baseline_ctx["edge_probs"]
        edge_dates = baseline_ctx["edge_dates"]
        edge_pairs = baseline_ctx["edge_pairs"]

        mask_gen = TemporalMaskGenerator(train_end_month=train_end_month, val_end_month=val_end_month, base_year=2024)
        train_cutoff = mask_gen.train_cutoff
        val_cutoff = mask_gen.val_cutoff

        first_seen = {}
        for pair, d in zip(edge_pairs, edge_dates):
            if d is None:
                continue
            if pair not in first_seen or d < first_seen[pair]:
                first_seen[pair] = d

        unseen_idx = []
        seen_idx = []
        for i, (pair, d) in enumerate(zip(edge_pairs, edge_dates)):
            if d is None or d < val_cutoff:
                continue
            first = first_seen.get(pair)
            if first is None:
                continue
            if first >= train_cutoff:
                unseen_idx.append(i)
            else:
                seen_idx.append(i)

        unseen_idx = np.array(unseen_idx, dtype=int)
        seen_idx = np.array(seen_idx, dtype=int)

        report = {
            "train_cutoff": str(train_cutoff),
            "test_start": str(val_cutoff),
            "unseen_edge_support": int(len(unseen_idx)),
            "seen_edge_support": int(len(seen_idx)),
        }

        report["unseen_edges"] = _safe_binary_metrics(
            edge_labels[unseen_idx] if len(unseen_idx) else np.array([]),
            edge_probs[unseen_idx] if len(unseen_idx) else np.array([]),
            threshold=evaluator.edge_threshold,
        )
        report["seen_edges"] = _safe_binary_metrics(
            edge_labels[seen_idx] if len(seen_idx) else np.array([]),
            edge_probs[seen_idx] if len(seen_idx) else np.array([]),
            threshold=evaluator.edge_threshold,
        )

        if report["seen_edges"]["positive_support"] > 0 and report["unseen_edges"]["positive_support"] > 0:
            report["generalization_gap_f1"] = float(report["seen_edges"]["f1"] - report["unseen_edges"]["f1"])
        else:
            report["generalization_gap_f1"] = None
        return report

    @staticmethod
    def temporal_drift(evaluator: StressEvaluator, companies: list[dict], invoices: list[dict], baseline):
        report = {}
        for months in [3, 12]:
            shifted = []
            for inv in invoices:
                x = copy.deepcopy(inv)
                d = _parse_date(x.get("date"))
                if d is not None:
                    x["date"] = _add_months(d, months)
                shifted.append(x)

            res = evaluator.evaluate_dataset(companies, shifted, scenario=f"temporal_plus_{months}m")
            key = f"plus_{months}m"
            report[key] = {
                "serving": res["serving"],
                "raw": res["raw"],
                "delta_vs_baseline": {
                    "node_f1": float(res["serving"]["node"]["f1"] - baseline["serving"]["node"]["f1"]),
                    "edge_f1": float(res["serving"]["edge"]["f1"] - baseline["serving"]["edge"]["f1"]),
                    "node_pr_auc": float(res["serving"]["node"]["pr_auc"] - baseline["serving"]["node"]["pr_auc"]),
                    "edge_pr_auc": float(res["serving"]["edge"]["pr_auc"] - baseline["serving"]["edge"]["pr_auc"]),
                },
            }
        return report


def _load_db_data():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                       COALESCE(ST_Y(geom), 0.0) as lat, COALESCE(ST_X(geom), 0.0) as lng
                FROM companies
                """
            )
        except Exception:
            conn.rollback()
            cur.execute(
                """
                SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                       0.0 as lat, 0.0 as lng
                FROM companies
                """
            )
        cols = [d[0] for d in cur.description]
        companies = [dict(zip(cols, row)) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number
            FROM invoices
            ORDER BY invoice_number
            """
        )
        cols = [d[0] for d in cur.description]
        invoices = [dict(zip(cols, row)) for row in cur.fetchall()]
        return companies, invoices
    finally:
        conn.close()


def _to_json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_ready(v) for v in obj]
    return obj


def _build_summary(stress_report: dict):
    degradations = []

    for scenario_name, payload in stress_report.items():
        if isinstance(payload, dict):
            if "delta_vs_baseline" in payload:
                degradations.append((scenario_name, payload["delta_vs_baseline"].get("node_f1", 0.0)))
            else:
                for sub_name, sub_payload in payload.items():
                    if isinstance(sub_payload, dict) and "delta_vs_baseline" in sub_payload:
                        degradations.append((f"{scenario_name}/{sub_name}", sub_payload["delta_vs_baseline"].get("node_f1", 0.0)))

    if degradations:
        worst = min(degradations, key=lambda x: x[1])
        mean_drop = float(np.mean([d for _, d in degradations]))
    else:
        worst = ("none", 0.0)
        mean_drop = 0.0

    return {
        "worst_node_f1_scenario": worst[0],
        "worst_node_f1_delta": float(worst[1]),
        "mean_node_f1_delta": mean_drop,
    }


def _build_stress_acceptance_gates(stress_report: dict, summary: dict) -> dict:
    max_worst_node_f1_drop = float(os.getenv("GNN_STRESS_MAX_WORST_NODE_F1_DROP", "0.20"))
    max_unseen_node_gap = float(os.getenv("GNN_STRESS_MAX_UNSEEN_NODE_GAP", "0.45"))
    max_plus3m_edge_f1_drop = float(os.getenv("GNN_STRESS_MAX_PLUS3M_EDGE_F1_DROP", "0.10"))
    max_plus3m_edge_prauc_drop = float(os.getenv("GNN_STRESS_MAX_PLUS3M_EDGE_PRAUC_DROP", "0.06"))

    worst_node_delta = float(summary.get("worst_node_f1_delta", 0.0))
    unseen_gap = stress_report.get("unseen_nodes", {}).get("generalization_gap_f1")
    plus3m_delta = stress_report.get("temporal_drift", {}).get("plus_3m", {}).get("delta_vs_baseline", {})
    plus3m_edge_f1_drop = float(abs(min(0.0, plus3m_delta.get("edge_f1", 0.0))))
    plus3m_edge_prauc_drop = float(abs(min(0.0, plus3m_delta.get("edge_pr_auc", 0.0))))

    unseen_gap_pass = True if unseen_gap is None else float(unseen_gap) <= max_unseen_node_gap

    criteria = {
        "worst_node_f1_drop": {
            "pass": worst_node_delta >= -max_worst_node_f1_drop,
            "actual": worst_node_delta,
            "threshold": -max_worst_node_f1_drop,
        },
        "unseen_node_generalization_gap": {
            "pass": unseen_gap_pass,
            "actual": unseen_gap,
            "threshold": max_unseen_node_gap,
            "note": "insufficient_class_variation" if unseen_gap is None else None,
        },
        "temporal_plus3m_edge_f1_drop": {
            "pass": plus3m_edge_f1_drop <= max_plus3m_edge_f1_drop,
            "actual": plus3m_edge_f1_drop,
            "threshold": max_plus3m_edge_f1_drop,
        },
        "temporal_plus3m_edge_prauc_drop": {
            "pass": plus3m_edge_prauc_drop <= max_plus3m_edge_prauc_drop,
            "actual": plus3m_edge_prauc_drop,
            "threshold": max_plus3m_edge_prauc_drop,
        },
    }

    return {
        "overall_pass": all(v.get("pass", False) for v in criteria.values()),
        "criteria": criteria,
    }


def run_stress_suite(
    companies: list[dict],
    invoices: list[dict],
    model_dir: Path,
    output_path: Path,
    seed: int = 42,
    unseen_nodes: int = 20,
    merge_serving_report: bool = True,
):
    model_dir = Path(model_dir)
    output_path = Path(output_path)

    evaluator = StressEvaluator(model_dir=model_dir, seed=seed)
    scenarios = StressScenarios(seed=seed)

    baseline = evaluator.evaluate_dataset(companies, invoices, scenario="baseline")

    stress = {}
    stress["class_imbalance_shift"] = scenarios.class_imbalance_shift(
        labels=baseline["context"]["node_labels"],
        probs=baseline["context"]["node_probs"],
        threshold=evaluator.node_threshold,
        node_degrees=baseline["context"].get("node_degrees"),
        cold_start_degree_threshold=evaluator.cold_start_degree_threshold,
        cold_start_threshold_delta=evaluator.cold_start_threshold_delta,
    )
    stress["covariate_shift"] = scenarios.covariate_shift(evaluator, companies, invoices, baseline)
    stress["unseen_nodes"] = scenarios.unseen_nodes(evaluator, companies, invoices, num_unseen=unseen_nodes)
    stress["unseen_edges"] = scenarios.unseen_edges(baseline["context"], evaluator)
    stress["temporal_drift"] = scenarios.temporal_drift(evaluator, companies, invoices, baseline)
    stress_summary = _build_summary(stress)
    stress_acceptance_gates = _build_stress_acceptance_gates(stress, stress_summary)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_dir": str(model_dir),
        "seed": seed,
        "baseline": {
            "raw": baseline["raw"],
            "serving": baseline["serving"],
            "thresholds": baseline["thresholds"],
        },
        "stress_evaluations": _to_json_ready(stress),
        "stress_summary": stress_summary,
        "stress_acceptance_gates": _to_json_ready(stress_acceptance_gates),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if merge_serving_report:
        serving_path = model_dir / "serving_e2e_report.json"
        if serving_path.exists():
            try:
                with open(serving_path, "r", encoding="utf-8") as f:
                    serving_report = json.load(f)
            except Exception:
                serving_report = {}
            serving_report["stress_evaluations"] = report["stress_evaluations"]
            serving_report["stress_summary"] = report["stress_summary"]
            serving_report["stress_acceptance_gates"] = report["stress_acceptance_gates"]
            serving_report["stress_generated_at"] = report["generated_at"]
            with open(serving_path, "w", encoding="utf-8") as f:
                json.dump(serving_report, f, indent=2)

    print("[DONE] Stress evaluation report written:", output_path)
    print("[INFO] Worst node F1 delta:", report["stress_summary"]["worst_node_f1_delta"])
    print("[INFO] Stress gates overall pass:", report["stress_acceptance_gates"]["overall_pass"])
    return report


def run(args):
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    companies, invoices = _load_db_data()
    run_stress_suite(
        companies=companies,
        invoices=invoices,
        model_dir=model_dir,
        output_path=output_path,
        seed=args.seed,
        unseen_nodes=args.unseen_nodes,
        merge_serving_report=args.merge_serving_report,
    )


def parse_args():
    default_model_dir = BACKEND_DIR / "data" / "models"
    default_output = default_model_dir / "stress_evaluation_report.json"

    parser = argparse.ArgumentParser(description="VAT GNN harder-data evaluation suite")
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir))
    parser.add_argument("--output", type=str, default=str(default_output))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unseen-nodes", type=int, default=20)
    parser.add_argument(
        "--merge-serving-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge stress results into data/models/serving_e2e_report.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
