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
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import psycopg2
import torch
import numpy as np
from datetime import date

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
    cur.execute("""
        SELECT tax_code, name, industry, registration_date, risk_score, is_active,
               ST_Y(geom) as lat, ST_X(geom) as lng
        FROM companies
        WHERE geom IS NOT NULL
    """)
    columns = [desc[0] for desc in cur.description]
    companies = [dict(zip(columns, row)) for row in cur.fetchall()]
    print(f"[OK] Loaded {len(companies)} companies with geom")

    if not companies:
        print("[ERROR] No companies with geom found. Run generate_graph_mock_data.py first!")
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
    constructor = GraphConstructor()
    data = constructor.build_graph(companies, invoices)
    sorted_invoices = constructor._sorted_invoices  # stable ordering used by graph
    print(f"[OK] Graph built: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"     Node features: {data.x.shape[1]} dims")
    print(f"     Edge features: {data.edge_attr.shape[1]} dims")

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
    mask_gen = TemporalMaskGenerator(train_end_month=8, val_end_month=9, base_year=2024)
    masks = mask_gen.generate_masks(
        invoices=sorted_invoices,
        companies=companies,
        tc_to_idx=tc_to_idx,
        n_nodes=data.num_nodes,
        n_edges=data.edge_index.shape[1],
    )

    split_info = masks["split_info"]
    print(f"\n[TEMPORAL SPLIT]")
    print(f"  Nodes:  Train={split_info['node_train']}  Val={split_info['node_val']}  Test={split_info['node_test']}")
    print(f"  Edges:  Train={split_info['edge_train']}  Val={split_info['edge_val']}  Test={split_info['edge_test']}")
    print(f"  Cutoffs: Train<{split_info['train_cutoff']}  Val<{split_info['val_cutoff']}")

    # ── 6. Train ──
    node_feat_dim = data.x.shape[1]
    edge_feat_dim = data.edge_attr.shape[1]
    print(f"\n[TRAIN] Starting GAT training (node_feat={node_feat_dim}, edge_feat={edge_feat_dim})...")

    trainer = GNNTrainer(node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim)
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
        gnn_node_scores = torch.sigmoid(gnn_out["node_logits"]).numpy()
        gnn_edge_scores = torch.sigmoid(gnn_out["edge_logits"]).numpy()

    # 9d. Stack features for meta-model: [gnn, rules, anomaly]
    node_meta_features = np.column_stack([gnn_node_scores, node_rule_scores, node_anomaly_scores])
    edge_meta_features = np.column_stack([gnn_edge_scores, edge_rule_scores, edge_anomaly_scores])

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

    # ── 10. Save ALL artifacts ──
    trainer.save()
    calibrator.save()
    anomaly.save()
    ensemble.save()
    print("\n[DONE] Enterprise GNN + Ensemble training complete!")
    print("  Model:       data/models/gat_model.pt")
    print("  Config:      data/models/gat_config.json")
    print("  Calibrator:  data/models/calibrator.pkl")
    print("  Anomaly:     data/models/anomaly_detector.pkl")
    print("  Ensemble:    data/models/ensemble_meta.pkl")


if __name__ == "__main__":
    main()
