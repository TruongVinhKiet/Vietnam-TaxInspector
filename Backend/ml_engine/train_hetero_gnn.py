"""
train_hetero_gnn.py – Training Script for Heterogeneous Graph Transformer
===========================================================================
Loads OSINT heterograph data from the database, builds HeteroData,
and trains the HGTConv-based model for multi-entity risk classification.

Usage:
    python -m ml_engine.train_hetero_gnn
    python ml_engine/train_hetero_gnn.py --epochs 300

Artifacts:
    - data/models/hgt_model.pt
    - data/models/hgt_config.json
    - data/models/hgt_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Ensure backend root importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

MODEL_DIR = BACKEND_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_REQUIRED_SAMPLES = max(200, int(os.getenv("HETERO_GNN_MIN_SAMPLES", "200")))


def main(epochs: int = 200, min_samples: int = MIN_REQUIRED_SAMPLES) -> int:
    print("=" * 64)
    print("  HETEROGENEOUS GRAPH TRANSFORMER TRAINING (OSINT)")
    print("=" * 64)

    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        print("[ABORT] torch_geometric is required. Install: pip install torch-geometric")
        return 1

    from ml_engine.osint_heterograph_dataset import OsintHeteroGraphBuilder
    from ml_engine.hetero_gnn_model import HeteroGNNTrainer, NODE_TYPES

    # ── Step 1: Build heterograph from database ──
    print("\n[1/4] Building heterograph from database...")
    from app.database import SessionLocal

    try:
        with SessionLocal() as db:
            builder = OsintHeteroGraphBuilder()
            data, meta = builder.build_from_db(db)
    except Exception as e:
        print(f"[ABORT] Failed to build heterograph: {e}")
        return 2

    # Validate minimum graph size
    total_nodes = sum(
        data[ntype].x.shape[0]
        for ntype in NODE_TYPES
        if ntype in data.node_types and hasattr(data[ntype], "x")
    )
    print(f"       Total nodes: {total_nodes}")
    if total_nodes < min_samples:
        print(f"[ABORT] Need >= {min_samples} nodes, got {total_nodes}")
        return 2

    for ntype in NODE_TYPES:
        if ntype in data.node_types and hasattr(data[ntype], "x"):
            print(f"       {ntype}: {data[ntype].x.shape[0]} nodes")

    for etype in data.edge_types:
        print(f"       Edge {etype}: {data[etype].edge_index.shape[1]}")

    # ── Step 2: Generate labels ──
    print("\n[2/4] Generating training labels...")
    labels = {}
    train_masks = {}
    val_masks = {}

    for ntype in NODE_TYPES:
        if ntype not in data.node_types or not hasattr(data[ntype], "x"):
            continue
        n = data[ntype].x.shape[0]

        # Use risk_score_norm (feature index 0) as weak supervision
        risk_scores = data[ntype].x[:, 0].numpy()
        threshold = np.percentile(risk_scores, 75)
        lbl = torch.tensor((risk_scores > threshold).astype(int), dtype=torch.long)
        labels[ntype] = lbl

        # Train/val split (80/20)
        perm = torch.randperm(n)
        n_train = int(n * 0.8)
        tmask = torch.zeros(n, dtype=torch.bool)
        vmask = torch.zeros(n, dtype=torch.bool)
        tmask[perm[:n_train]] = True
        vmask[perm[n_train:]] = True
        train_masks[ntype] = tmask
        val_masks[ntype] = vmask

        pos = int(lbl.sum().item())
        print(f"       {ntype}: {n} nodes, {pos} positive ({pos/n*100:.1f}%)")

    # ── Step 3: Train ──
    print(f"\n[3/4] Training HGT model ({epochs} epochs)...")
    trainer = HeteroGNNTrainer(
        node_feature_dim=data[NODE_TYPES[0]].x.shape[1] if NODE_TYPES[0] in data.node_types else 15,
    )
    metrics = trainer.train(
        data=data,
        labels=labels,
        train_masks=train_masks,
        val_masks=val_masks,
        epochs=epochs,
    )

    # ── Step 4: Save ──
    print("\n[4/4] Saving artifacts...")
    trainer.save(str(MODEL_DIR))

    # Quality report
    quality_report = {
        "model_name": "osint_hetero_gnn",
        "model_version": "hgt-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "total_nodes": total_nodes,
            "node_types": {
                ntype: int(data[ntype].x.shape[0])
                for ntype in NODE_TYPES
                if ntype in data.node_types and hasattr(data[ntype], "x")
            },
        },
        "metrics": {
            ntype: {k: round(v, 4) for k, v in vals.items()}
            for ntype, vals in metrics.items()
        },
    }
    with open(MODEL_DIR / "hgt_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    print(f"       [OK] Model saved to {MODEL_DIR}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HGT OSINT model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min-samples", type=int, default=MIN_REQUIRED_SAMPLES)
    args = parser.parse_args()
    sys.exit(main(epochs=args.epochs, min_samples=args.min_samples))
