"""
train_gnn.py – Enterprise GAT Training Pipeline
==================================================
Usage: python -m app.scripts.train_gnn

Pipeline:
    1. Load companies and invoices from PostgreSQL
    2. Build PyG graph (deterministic edge ordering by invoice_number)
    3. Generate ground-truth labels (shell nodes, circular edges)
    4. Create temporal masks (Train T1-T8, Val T9, Test T10-T12)
    5. Train GAT model with eval-mode metrics
    6. Fit probability calibrator on validation set
    7. Evaluate on held-out test set with full report
    8. Save model + calibrator to data/models/
"""

import os, sys
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import psycopg2
import torch
import numpy as np
from datetime import date, datetime
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score

from ml_engine.gnn_model import GraphConstructor, GNNTrainer
from ml_engine.metrics_engine import (
    TemporalMaskGenerator,
    GNNEvaluator,
    ProbabilityCalibrator,
    format_evaluation_report,
)
from ml_engine.ensemble import (
    HeuristicRuleScorer,
    AnomalyDetector,
    EnsembleModel,
    PathEvidenceExtractor,
)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "TaxInspector")


def _safe_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int) if len(probs) else np.array([], dtype=int)

    metrics = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pr_auc": 0.0,
        "roc_auc": 0.0,
        "threshold": float(threshold),
        "support": int(len(labels)),
    }

    if len(labels) == 0:
        return metrics

    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
    metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
    metrics["recall"] = float(recall_score(labels, preds, zero_division=0))

    if len(np.unique(labels)) >= 2:
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))

    return metrics


def _safe_binary_metrics_from_preds(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = np.asarray(preds)

    metrics = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pr_auc": 0.0,
        "roc_auc": 0.0,
        "threshold": float(threshold),
        "support": int(len(labels)),
    }

    if len(labels) == 0:
        return metrics

    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
    metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
    metrics["recall"] = float(recall_score(labels, preds, zero_division=0))

    if len(np.unique(labels)) >= 2:
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))

    return metrics


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


def _optimize_node_blend_alpha(
    gnn_probs: np.ndarray,
    ensemble_probs: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Pick blend alpha for node serving score:
        blended = alpha * gnn_probs + (1 - alpha) * ensemble_probs

    Objective defaults to PR-AUC to reduce ranking regressions vs raw GNN.
    """
    gnn_probs = np.asarray(gnn_probs)
    ensemble_probs = np.asarray(ensemble_probs)
    labels = np.asarray(labels)

    objective = os.getenv("GNN_NODE_BLEND_OBJECTIVE", "pr_auc").strip().lower()
    if objective not in {"pr_auc", "f1"}:
        objective = "pr_auc"

    min_alpha = float(os.getenv("GNN_NODE_BLEND_MIN_ALPHA", "0.0"))
    max_alpha = float(os.getenv("GNN_NODE_BLEND_MAX_ALPHA", "1.0"))
    min_alpha = min(max(min_alpha, 0.0), 1.0)
    max_alpha = min(max(max_alpha, min_alpha), 1.0)

    grid = np.linspace(min_alpha, max_alpha, 21)

    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {
            "alpha_gnn": float(max_alpha),
            "objective": objective,
            "best_objective": 0.0,
            "best_pr_auc": 0.0,
            "best_f1": 0.0,
            "ensemble_pr_auc": 0.0,
            "gnn_pr_auc": 0.0,
        }

    gnn_pr_auc = float(average_precision_score(labels, gnn_probs))
    ensemble_pr_auc = float(average_precision_score(labels, ensemble_probs))

    best_alpha = float(max_alpha)
    best_objective = -1.0
    best_pr_auc = -1.0
    best_f1 = -1.0

    for alpha in grid:
        blended = alpha * gnn_probs + (1.0 - alpha) * ensemble_probs
        pr_auc = float(average_precision_score(labels, blended))
        f1 = float(f1_score(labels, (blended >= 0.5).astype(int), zero_division=0))
        score = pr_auc if objective == "pr_auc" else f1

        if score > best_objective:
            best_objective = score
            best_alpha = float(alpha)
            best_pr_auc = pr_auc
            best_f1 = f1
        elif np.isclose(score, best_objective):
            # Tie-breaker: prefer higher PR-AUC then slightly more GNN weight for stability.
            if pr_auc > best_pr_auc:
                best_alpha = float(alpha)
                best_pr_auc = pr_auc
                best_f1 = f1
            elif np.isclose(pr_auc, best_pr_auc) and alpha > best_alpha:
                best_alpha = float(alpha)
                best_f1 = f1

    return {
        "alpha_gnn": float(best_alpha),
        "objective": objective,
        "best_objective": float(best_objective),
        "best_pr_auc": float(best_pr_auc),
        "best_f1": float(best_f1),
        "ensemble_pr_auc": ensemble_pr_auc,
        "gnn_pr_auc": gnn_pr_auc,
    }


def _build_feature_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "q0": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q100": 0.0,
        }
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q0": float(np.quantile(arr, 0.0)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q100": float(np.quantile(arr, 1.0)),
    }


def main():
    print("=" * 70)
    print("  TaxInspector – Enterprise GNN (GAT) Training Pipeline")
    print("=" * 70)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        dbname=DB_NAME
    )
    cur = conn.cursor()

    # ── 1. Load companies ──
    # Keep training data aligned with serving path: include all companies and
    # only fallback to zeroed coordinates when PostGIS geom is missing.
    try:
        cur.execute("""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   COALESCE(ST_Y(geom), 0.0) as lat, COALESCE(ST_X(geom), 0.0) as lng
            FROM companies
        """)
        geom_mode = "postgis_optional"
    except Exception:
        conn.rollback()
        cur.execute("""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   0.0 as lat, 0.0 as lng
            FROM companies
        """)
        geom_mode = "geom_fallback_zero"

    columns = [desc[0] for desc in cur.description]
    companies = [dict(zip(columns, row)) for row in cur.fetchall()]
    print(f"[OK] Loaded {len(companies)} companies ({geom_mode})")

    if not companies:
        print("[ERROR] No companies found. Seed companies before training GNN.")
        return

    # ── 2. Load invoices (sorted by invoice_number for deterministic ordering) ──
    cur.execute("""
        SELECT seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number
        FROM invoices
        ORDER BY invoice_number
    """)
    columns = [desc[0] for desc in cur.description]
    invoices = [dict(zip(columns, row)) for row in cur.fetchall()]
    print(f"[OK] Loaded {len(invoices)} invoices")

    conn.close()

    if not invoices:
        print("[ERROR] No invoices found. Run generate_graph_mock_data.py first!")
        return

    # ── 3. Build graph ──
    amount_feature_mode = os.getenv("GNN_AMOUNT_FEATURE_MODE", "robust").strip().lower()
    constructor = GraphConstructor(amount_feature_mode=amount_feature_mode)
    data = constructor.build_graph(companies, invoices)
    sorted_invoices = constructor._sorted_invoices  # stable ordering used by graph
    print(f"[OK] Graph built: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"     Node features: {data.x.shape[1]} dims")
    print(f"     Edge features: {data.edge_attr.shape[1]} dims")
    print(f"     Amount feature mode: {constructor.amount_feature_mode}")

    tc_to_idx = constructor.tax_code_to_idx

    # ── 4. Generate ground-truth labels ──
    # Node labels: shell companies have tax_code index 900+ (from mock data generator)
    node_labels = torch.zeros(data.num_nodes)
    for c in companies:
        idx = tc_to_idx.get(c["tax_code"])
        if idx is None:
            continue
        try:
            if int(c["tax_code"][2:7]) >= 900:
                node_labels[idx] = 1.0
        except (ValueError, IndexError):
            pass

    # Edge labels: matched 1-to-1 with sorted_invoices
    edge_labels = torch.zeros(data.edge_index.shape[1])
    for e_idx in range(min(len(sorted_invoices), data.edge_index.shape[1])):
        if sorted_invoices[e_idx].get("invoice_number", "").startswith("CIRC-"):
            edge_labels[e_idx] = 1.0

    n_shell = int(node_labels.sum().item())
    n_circ = int(edge_labels.sum().item())
    print(f"[OK] Labels: {n_shell} shell nodes / {data.num_nodes} total | "
          f"{n_circ} circular edges / {data.edge_index.shape[1]} total")

    # ── 5. Generate temporal masks ──
    node_split_strategy = os.getenv("GNN_NODE_SPLIT_STRATEGY", "stratified")
    mask_gen = TemporalMaskGenerator(
        train_end_month=8,
        val_end_month=9,
        base_year=2024,
        node_split_strategy=node_split_strategy,
    )
    masks = mask_gen.generate_masks(
        invoices=sorted_invoices,
        companies=companies,
        tc_to_idx=tc_to_idx,
        n_nodes=data.num_nodes,
        n_edges=data.edge_index.shape[1],
    )

    split_info = masks["split_info"]
    print(f"\n[TEMPORAL SPLIT]")
    print(f"  Node strategy: {split_info['node_split_strategy']}")
    print(f"  Nodes:  Train={split_info['node_train']}  Val={split_info['node_val']}  Test={split_info['node_test']}")
    print(f"  Nodes (strict temporal): Train={split_info['node_temporal_train']}  Val={split_info['node_temporal_val']}  Test={split_info['node_temporal_test']}")
    print(f"  Edges:  Train={split_info['edge_train']}  Val={split_info['edge_val']}  Test={split_info['edge_test']}")
    print(f"  Cutoffs: Train<{split_info['train_cutoff']}  Val<{split_info['val_cutoff']}")

    # ── 6. Train ──
    node_feat_dim = data.x.shape[1]
    edge_feat_dim = data.edge_attr.shape[1]
    print(f"\n[TRAIN] Starting GAT training (node_feat={node_feat_dim}, edge_feat={edge_feat_dim})...")

    trainer = GNNTrainer(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        amount_feature_mode=constructor.amount_feature_mode,
    )
    trainer.train(
        data, node_labels, edge_labels, epochs=300,
        node_train_mask=masks["node_train_mask"],
        node_val_mask=masks["node_val_mask"],
        edge_train_mask=masks["edge_train_mask"],
        edge_val_mask=masks["edge_val_mask"],
    )

    # ── 7. Fit Probability Calibrator on VALIDATION set ──
    print("\n[CALIBRATION] Fitting Isotonic Regression on validation set...")
    trainer.model.eval()
    with torch.no_grad():
        val_out = trainer.model(data)
        val_node_probs = torch.sigmoid(val_out["node_logits"][masks["node_val_mask"]]).numpy()
        val_edge_probs = torch.sigmoid(val_out["edge_logits"][masks["edge_val_mask"]]).numpy()
        val_node_labels = node_labels[masks["node_val_mask"]].numpy()
        val_edge_labels = edge_labels[masks["edge_val_mask"]].numpy()

    calibrator = ProbabilityCalibrator()
    calibrator.fit(val_node_probs, val_node_labels, val_edge_probs, val_edge_labels)

    # ── 8. Evaluate on HELD-OUT TEST set ──
    print("\n[EVALUATION] Running comprehensive test evaluation...")
    evaluator = GNNEvaluator(
        model=trainer.model,
        data=data,
        node_labels=node_labels,
        edge_labels=edge_labels,
        companies=companies,
        tc_to_idx=tc_to_idx,
    )

    # Test evaluation (raw)
    test_result = evaluator.evaluate(
        node_mask=masks["node_test_mask"],
        edge_mask=masks["edge_test_mask"],
        split_name="test",
        top_k=20,
    )

    # Also run on validation for comparison
    val_result = evaluator.evaluate(
        node_mask=masks["node_val_mask"],
        edge_mask=masks["edge_val_mask"],
        split_name="validation",
        top_k=20,
    )

    # Print reports
    print(format_evaluation_report(val_result))
    print(format_evaluation_report(test_result))

    # Strict temporal node evaluation for production-time realism.
    strict_temporal_result = evaluator.evaluate(
        node_mask=masks["node_temporal_test_mask"],
        edge_mask=masks["edge_test_mask"],
        split_name="strict_temporal_test",
        top_k=20,
    )
    print(format_evaluation_report(strict_temporal_result))

    # ── 9. Train Ensemble Components ──
    print("\n[ENSEMBLE] Training anomaly detector + meta-model...")

    # 9a. Anomaly Detector (unsupervised, on ALL data)
    anomaly = AnomalyDetector()
    anomaly.fit(data.x.numpy(), data.edge_attr.numpy())
    node_anomaly_scores = anomaly.score_nodes(data.x.numpy())
    edge_anomaly_scores = anomaly.score_edges(data.edge_attr.numpy())

    # 9b. Heuristic Rule Scores
    rule_scorer = HeuristicRuleScorer()
    cycles = constructor.detect_cycles_networkx(sorted_invoices)
    cycle_edges_set = set()
    cycle_nodes_set = set()
    for cycle in cycles:
        for ci in range(len(cycle)):
            s, b = cycle[ci], cycle[(ci + 1) % len(cycle)]
            cycle_edges_set.add((s, b))
            cycle_nodes_set.add(s)
            cycle_nodes_set.add(b)

    # Compute per-node aggregates for rules
    in_amount_map = {}
    out_amount_map = {}
    degree_map = {}
    reciprocal_set = set()
    for inv in sorted_invoices:
        s, b = inv["seller_tax_code"], inv["buyer_tax_code"]
        out_amount_map[s] = out_amount_map.get(s, 0) + float(inv["amount"])
        in_amount_map[b] = in_amount_map.get(b, 0) + float(inv["amount"])
        degree_map[s] = degree_map.get(s, 0) + 1
        degree_map[b] = degree_map.get(b, 0) + 1
        reciprocal_set.add((s, b))

    node_rule_scores = np.zeros(data.num_nodes)
    for c in companies:
        idx = tc_to_idx.get(c["tax_code"])
        if idx is not None:
            node_rule_scores[idx] = rule_scorer.score_node(
                c, cycle_nodes_set,
                in_amount_map.get(c["tax_code"], 0),
                out_amount_map.get(c["tax_code"], 0),
                degree_map.get(c["tax_code"], 0),
            )

    edge_rule_scores = np.zeros(data.edge_index.shape[1])
    for e_idx, inv in enumerate(sorted_invoices):
        if e_idx >= data.edge_index.shape[1]:
            break
        edge_rule_scores[e_idx] = rule_scorer.score_edge(inv, cycle_edges_set, reciprocal_set)

    # 9c. Get GNN scores (in eval mode)
    trainer.model.eval()
    with torch.no_grad():
        gnn_out = trainer.model(data)
        gnn_node_scores_raw = torch.sigmoid(gnn_out["node_logits"]).numpy()
        gnn_edge_scores_raw = torch.sigmoid(gnn_out["edge_logits"]).numpy()

    # Use the same calibrated GNN signal for ensemble training and serving.
    gnn_node_scores = calibrator.calibrate_nodes_batch(gnn_node_scores_raw)
    gnn_edge_scores = calibrator.calibrate_edges_batch(gnn_edge_scores_raw)

    # 9d. Stack features for meta-model: [gnn, rules, anomaly]
    node_meta_features = np.column_stack([gnn_node_scores, node_rule_scores, node_anomaly_scores])
    edge_meta_features = np.column_stack([gnn_edge_scores, edge_rule_scores, edge_anomaly_scores])

    node_degree_all = np.zeros(data.num_nodes, dtype=int)
    for c in companies:
        idx = tc_to_idx.get(c["tax_code"])
        if idx is not None:
            node_degree_all[idx] = int(degree_map.get(c["tax_code"], 0))

    # Train meta-model on VALIDATION set only
    val_nm = masks["node_val_mask"].numpy()
    val_em = masks["edge_val_mask"].numpy()

    ensemble = EnsembleModel()
    ensemble.fit(
        node_meta_features[val_nm], node_labels[masks["node_val_mask"]].numpy(),
        edge_meta_features[val_em], edge_labels[masks["edge_val_mask"]].numpy(),
    )

    weights = ensemble.get_weights()
    if weights:
        print(f"  Meta-model weights: {weights}")

    # Learn serving decision thresholds on validation serving-path outputs.
    val_node_ensemble_probs = ensemble.predict_nodes_batch(node_meta_features[val_nm])
    val_node_gnn_probs = gnn_node_scores[val_nm]
    val_node_labels_np = node_labels[masks["node_val_mask"]].numpy()

    blend_info = _optimize_node_blend_alpha(
        gnn_probs=val_node_gnn_probs,
        ensemble_probs=val_node_ensemble_probs,
        labels=val_node_labels_np,
    )
    node_blend_alpha_gnn = float(blend_info["alpha_gnn"])
    val_node_serving_probs = (
        node_blend_alpha_gnn * val_node_gnn_probs
        + (1.0 - node_blend_alpha_gnn) * val_node_ensemble_probs
    )

    val_edge_serving_probs = ensemble.predict_edges_batch(edge_meta_features[val_em])
    threshold_metric = os.getenv("GNN_THRESHOLD_METRIC", "f1")
    cold_start_degree_threshold = int(os.getenv("GNN_COLD_START_DEGREE_THRESHOLD", "5"))
    max_cold_start_delta = float(os.getenv("GNN_MAX_COLD_START_THRESHOLD_DELTA", "0.12"))
    default_cold_start_delta = float(os.getenv("GNN_DEFAULT_COLD_START_THRESHOLD_DELTA", "0.06"))
    min_edge_threshold = float(os.getenv("GNN_MIN_EDGE_DECISION_THRESHOLD", "0.45"))
    threshold_info = calibrator.optimize_decision_thresholds(
        node_probs=val_node_serving_probs,
        node_labels=val_node_labels_np,
        node_degrees=node_degree_all[val_nm],
        edge_probs=val_edge_serving_probs,
        edge_labels=edge_labels[masks["edge_val_mask"]].numpy(),
        metric=threshold_metric,
        cold_start_degree_threshold=cold_start_degree_threshold,
        max_cold_start_delta=max_cold_start_delta,
        min_edge_threshold=min_edge_threshold,
        default_cold_start_delta=default_cold_start_delta,
    )
    threshold_meta = calibrator.threshold_meta if isinstance(calibrator.threshold_meta, dict) else {}
    threshold_meta.update(
        {
            "node_blend_alpha_gnn": node_blend_alpha_gnn,
            "node_blend_objective": blend_info["objective"],
            "node_blend_best_pr_auc": blend_info["best_pr_auc"],
            "node_blend_ensemble_pr_auc": blend_info["ensemble_pr_auc"],
            "node_blend_gnn_pr_auc": blend_info["gnn_pr_auc"],
        }
    )
    calibrator.threshold_meta = threshold_meta

    print(
        "  Decision thresholds (metric={}): node={:.3f}, edge={:.3f}, cold_start_delta={:.3f}, edge_floor={:.3f}, node_blend_alpha={:.2f}".format(
            threshold_info["metric"],
            threshold_info["node_threshold"],
            threshold_info["edge_threshold"],
            threshold_info.get("cold_start_threshold_delta", 0.0),
            threshold_info.get("min_edge_threshold", min_edge_threshold),
            node_blend_alpha_gnn,
        )
    )

    # ── 10. Evaluate full serving path (GNN -> Calibrator -> Ensemble) ──
    print("\n[E2E EVALUATION] Running serving-path test evaluation...")
    test_nm = masks["node_test_mask"].numpy()
    test_em = masks["edge_test_mask"].numpy()

    # Ensemble inputs: calibrated GNN + rules + anomaly (same as serving)
    node_test_meta_features = np.column_stack([
        gnn_node_scores[test_nm],
        node_rule_scores[test_nm],
        node_anomaly_scores[test_nm],
    ])
    edge_test_meta_features = np.column_stack([
        gnn_edge_scores[test_em],
        edge_rule_scores[test_em],
        edge_anomaly_scores[test_em],
    ])

    node_e2e_ensemble_probs = ensemble.predict_nodes_batch(node_test_meta_features)
    node_e2e_gnn_probs = gnn_node_scores[test_nm]
    threshold_meta = calibrator.threshold_meta if isinstance(calibrator.threshold_meta, dict) else {}
    node_blend_alpha_gnn = float(threshold_meta.get("node_blend_alpha_gnn", node_blend_alpha_gnn))
    node_e2e_probs = (
        node_blend_alpha_gnn * node_e2e_gnn_probs
        + (1.0 - node_blend_alpha_gnn) * node_e2e_ensemble_probs
    )
    edge_e2e_probs = ensemble.predict_edges_batch(edge_test_meta_features)

    node_test_labels = node_labels[masks["node_test_mask"]].numpy()
    edge_test_labels = edge_labels[masks["edge_test_mask"]].numpy()

    node_threshold = float(calibrator.node_threshold)
    edge_threshold = float(calibrator.edge_threshold)
    threshold_meta = calibrator.threshold_meta if isinstance(calibrator.threshold_meta, dict) else {}
    cold_start_degree_threshold = int(threshold_meta.get("cold_start_degree_threshold", cold_start_degree_threshold))
    cold_start_threshold_delta = float(threshold_meta.get("cold_start_threshold_delta", 0.0))

    test_node_degrees = node_degree_all[test_nm]
    node_e2e_preds, node_e2e_thresholds = _apply_node_threshold_policy(
        probs=node_e2e_probs,
        node_degrees=test_node_degrees,
        base_threshold=node_threshold,
        cold_start_degree_threshold=cold_start_degree_threshold,
        cold_start_threshold_delta=cold_start_threshold_delta,
    )

    node_e2e_metrics = _safe_binary_metrics_from_preds(
        node_test_labels,
        node_e2e_probs,
        node_e2e_preds,
        threshold=node_threshold,
    )
    edge_e2e_metrics = _safe_binary_metrics(edge_test_labels, edge_e2e_probs, threshold=edge_threshold)

    print("  Node  F1: {:.4f}  PR-AUC: {:.4f}  (raw F1: {:.4f}, raw PR-AUC: {:.4f}, thr={:.3f})".format(
        node_e2e_metrics["f1"],
        node_e2e_metrics["pr_auc"],
        test_result.get("node_f1", 0.0),
        test_result.get("node_pr_auc", 0.0),
        node_threshold,
    ))
    print("  Edge  F1: {:.4f}  PR-AUC: {:.4f}  (raw F1: {:.4f}, raw PR-AUC: {:.4f}, thr={:.3f})".format(
        edge_e2e_metrics["f1"],
        edge_e2e_metrics["pr_auc"],
        test_result.get("edge_f1", 0.0),
        test_result.get("edge_pr_auc", 0.0),
        edge_threshold,
    ))

    raw_node_f1 = float(test_result.get("node_f1", 0.0))
    raw_edge_f1 = float(test_result.get("edge_f1", 0.0))
    raw_node_pr_auc = float(test_result.get("node_pr_auc", 0.0))
    raw_edge_pr_auc = float(test_result.get("edge_pr_auc", 0.0))

    node_f1_ratio = float(node_e2e_metrics["f1"] / raw_node_f1) if raw_node_f1 > 0 else 1.0
    edge_f1_ratio = float(edge_e2e_metrics["f1"] / raw_edge_f1) if raw_edge_f1 > 0 else 1.0
    node_pr_auc_drop = float(raw_node_pr_auc - node_e2e_metrics["pr_auc"])
    edge_pr_auc_drop = float(raw_edge_pr_auc - edge_e2e_metrics["pr_auc"])

    min_node_f1_ratio = float(os.getenv("GNN_MIN_NODE_F1_RATIO", "0.95"))
    min_edge_f1_ratio = float(os.getenv("GNN_MIN_EDGE_F1_RATIO", "0.95"))
    max_node_pr_auc_drop = float(os.getenv("GNN_MAX_NODE_PRAUC_DROP", "0.02"))
    max_edge_pr_auc_drop = float(os.getenv("GNN_MAX_EDGE_PRAUC_DROP", "0.02"))
    min_node_support = int(os.getenv("GNN_MIN_NODE_TEST_SUPPORT", "20"))
    min_node_val_support = int(os.getenv("GNN_MIN_NODE_VAL_SUPPORT", "20"))
    relaxed_node_support_cutoff = int(os.getenv("GNN_NODE_PRAUC_RELAX_SUPPORT", "40"))
    relaxed_node_pr_auc_drop = float(os.getenv("GNN_MAX_NODE_PRAUC_DROP_RELAXED", "0.08"))

    node_support_actual = int(node_e2e_metrics["support"])
    effective_node_pr_auc_drop_threshold = max_node_pr_auc_drop
    if node_support_actual < relaxed_node_support_cutoff:
        effective_node_pr_auc_drop_threshold = max(max_node_pr_auc_drop, relaxed_node_pr_auc_drop)

    acceptance_gates = {
        "node_f1_parity": {
            "pass": node_f1_ratio >= min_node_f1_ratio,
            "actual": round(node_f1_ratio, 6),
            "threshold": min_node_f1_ratio,
        },
        "edge_f1_parity": {
            "pass": edge_f1_ratio >= min_edge_f1_ratio,
            "actual": round(edge_f1_ratio, 6),
            "threshold": min_edge_f1_ratio,
        },
        "node_pr_auc_drop": {
            "pass": node_pr_auc_drop <= effective_node_pr_auc_drop_threshold,
            "actual": round(node_pr_auc_drop, 6),
            "threshold": effective_node_pr_auc_drop_threshold,
        },
        "edge_pr_auc_drop": {
            "pass": edge_pr_auc_drop <= max_edge_pr_auc_drop,
            "actual": round(edge_pr_auc_drop, 6),
            "threshold": max_edge_pr_auc_drop,
        },
        "node_test_support": {
            "pass": node_support_actual >= min_node_support,
            "actual": node_support_actual,
            "threshold": min_node_support,
        },
        "node_validation_support": {
            "pass": int(threshold_meta.get("node_validation_support", 0)) >= min_node_val_support,
            "actual": int(threshold_meta.get("node_validation_support", 0)),
            "threshold": min_node_val_support,
        },
    }
    overall_pass = all(item["pass"] for item in acceptance_gates.values())

    serving_report = {
        "node_split_strategy": split_info["node_split_strategy"],
        "raw_test": {
            "node_f1": float(test_result.get("node_f1", 0.0)),
            "node_pr_auc": float(test_result.get("node_pr_auc", 0.0)),
            "edge_f1": float(test_result.get("edge_f1", 0.0)),
            "edge_pr_auc": float(test_result.get("edge_pr_auc", 0.0)),
        },
        "serving_path_test": {
            "node": node_e2e_metrics,
            "edge": edge_e2e_metrics,
        },
        "decision_thresholds": {
            "node": node_threshold,
            "edge": edge_threshold,
            "metric": threshold_metric,
            "metadata": calibrator.threshold_meta,
            "policy": {
                "cold_start_degree_threshold": cold_start_degree_threshold,
                "cold_start_threshold_delta": cold_start_threshold_delta,
                "node_blend_alpha_gnn": node_blend_alpha_gnn,
            },
        },
        "comparison": {
            "node_f1_delta": round(float(node_e2e_metrics["f1"] - raw_node_f1), 6),
            "node_pr_auc_delta": round(float(node_e2e_metrics["pr_auc"] - raw_node_pr_auc), 6),
            "edge_f1_delta": round(float(edge_e2e_metrics["f1"] - raw_edge_f1), 6),
            "edge_pr_auc_delta": round(float(edge_e2e_metrics["pr_auc"] - raw_edge_pr_auc), 6),
        },
        "acceptance_gates": {
            "overall_pass": overall_pass,
            "criteria": acceptance_gates,
            "policy": {
                "node_pr_auc_relax_support_cutoff": relaxed_node_support_cutoff,
                "node_pr_auc_relaxed_threshold": relaxed_node_pr_auc_drop,
                "node_pr_auc_effective_threshold": effective_node_pr_auc_drop_threshold,
            },
        },
        "strict_temporal_test": {
            "node_f1": float(strict_temporal_result.get("node_f1", 0.0)),
            "node_pr_auc": float(strict_temporal_result.get("node_pr_auc", 0.0)),
            "edge_f1": float(strict_temporal_result.get("edge_f1", 0.0)),
            "edge_pr_auc": float(strict_temporal_result.get("edge_pr_auc", 0.0)),
        },
    }

    report_path = BACKEND_DIR / "data" / "models" / "serving_e2e_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serving_report, f, indent=2)
    print(f"  E2E report saved: {report_path}")

    # Drift baseline stats used by /api/monitoring/drift_report.
    today = date.today()
    company_age_days = []
    for c in companies:
        reg = c.get("registration_date")
        if isinstance(reg, str):
            reg = date.fromisoformat(reg)
        if isinstance(reg, date):
            company_age_days.append(float((today - reg).days))

    invoice_amount_log = [float(np.log1p(float(inv.get("amount", 0.0)))) for inv in sorted_invoices]
    pair_set = {
        (inv.get("seller_tax_code"), inv.get("buyer_tax_code"))
        for inv in sorted_invoices
    }
    is_reciprocal_ratio = [
        1.0 if (inv.get("buyer_tax_code"), inv.get("seller_tax_code")) in pair_set else 0.0
        for inv in sorted_invoices
    ]

    drift_baseline = {
        "generated_at": datetime.utcnow().isoformat(),
        "features": {
            "company_age_days": _build_feature_stats(company_age_days),
            "invoice_amount_log": _build_feature_stats(invoice_amount_log),
            "is_reciprocal_ratio": _build_feature_stats(is_reciprocal_ratio),
        },
    }
    drift_baseline_path = BACKEND_DIR / "data" / "models" / "drift_baseline.json"
    with open(drift_baseline_path, "w", encoding="utf-8") as f:
        json.dump(drift_baseline, f, indent=2)
    print(f"  Drift baseline saved: {drift_baseline_path}")

    # ── 11. Save ALL artifacts ──
    trainer.save()
    calibrator.save()
    anomaly.save()
    ensemble.save()

    run_stress_eval = os.getenv("GNN_RUN_STRESS_EVAL", "1").strip().lower() not in {"0", "false", "no"}
    if run_stress_eval:
        try:
            from app.scripts.evaluate_gnn_stress import run_stress_suite

            stress_seed = int(os.getenv("GNN_STRESS_SEED", "42"))
            stress_unseen_nodes = int(os.getenv("GNN_STRESS_UNSEEN_NODES", "20"))
            stress_output = BACKEND_DIR / "data" / "models" / "stress_evaluation_report.json"
            stress_report = run_stress_suite(
                companies=companies,
                invoices=invoices,
                model_dir=BACKEND_DIR / "data" / "models",
                output_path=stress_output,
                seed=stress_seed,
                unseen_nodes=stress_unseen_nodes,
                merge_serving_report=True,
            )
            print(f"  Stress report: {stress_output}")
            print(
                "  Stress worst node F1 delta: {:.4f} ({})".format(
                    float(stress_report["stress_summary"]["worst_node_f1_delta"]),
                    stress_report["stress_summary"]["worst_node_f1_scenario"],
                )
            )
        except Exception as exc:
            print(f"[WARN] Stress evaluation step failed: {exc}")

    print("\n[DONE] Enterprise GNN + Ensemble training complete!")
    print("  Model:       data/models/gat_model.pt")
    print("  Config:      data/models/gat_config.json")
    print("  Calibrator:  data/models/calibrator.pkl")
    print("  Anomaly:     data/models/anomaly_detector.pkl")
    print("  Ensemble:    data/models/ensemble_meta.pkl")
    print("  Stress:      data/models/stress_evaluation_report.json")


if __name__ == "__main__":
    main()
