"""
train_temporal_transformer.py – Training Script for Delinquency Transformer
=============================================================================
Loads payment history sequences from the database and trains the
TransformerEncoder-based model for multi-horizon delinquency prediction.

Usage:
    python -m ml_engine.train_temporal_transformer
    python ml_engine/train_temporal_transformer.py --epochs 200

Artifacts:
    - data/models/temporal_transformer.pt
    - data/models/temporal_transformer_config.json
    - data/models/temporal_transformer_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

MODEL_DIR = BACKEND_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_REQUIRED_SAMPLES = max(500, int(os.getenv("TEMPORAL_MIN_SAMPLES", "500")))


def _load_payment_sequences(db_url: str, lookback_days: int = 730) -> tuple:
    """
    Load payment history and build sequences grouped by tax_code.
    
    Returns:
        sequences: (N, 24, 8) tensor
        padding_masks: (N, 24) tensor
        labels_30d, labels_60d, labels_90d: (N,) tensors
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from ml_engine.temporal_transformer import PaymentSequenceBuilder

    conn = psycopg2.connect(db_url)
    builder = PaymentSequenceBuilder()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Load all payments
            cur.execute("""
                SELECT tax_code, amount, payment_date, tax_period,
                       status, days_overdue, penalty_amount
                FROM tax_payments
                WHERE payment_date >= (NOW() - (%s || ' days')::interval)
                ORDER BY tax_code, payment_date
            """, (lookback_days,))
            payments = cur.fetchall()

            # Load tax returns for revenue features
            cur.execute("""
                SELECT tax_code, quarter, revenue
                FROM tax_returns
                WHERE filing_date >= (NOW() - (%s || ' days')::interval)
                ORDER BY tax_code, filing_date
            """, (lookback_days,))
            returns = cur.fetchall()

            # Load delinquency predictions for labels
            cur.execute("""
                SELECT DISTINCT ON (tax_code)
                    tax_code, prob_30d, prob_60d, prob_90d
                FROM delinquency_predictions
                ORDER BY tax_code, created_at DESC
            """)
            predictions = cur.fetchall()
    finally:
        conn.close()

    # Group by tax_code
    payments_by_tc: dict[str, list] = {}
    for p in payments:
        tc = p["tax_code"]
        payments_by_tc.setdefault(tc, []).append(p)

    returns_by_tc: dict[str, list] = {}
    for r in returns:
        tc = r["tax_code"]
        returns_by_tc.setdefault(tc, []).append(r)

    # Build sequences and labels
    sequences = []
    masks = []
    l30, l60, l90 = [], [], []

    pred_map = {p["tax_code"]: p for p in predictions}

    for tc, tc_payments in payments_by_tc.items():
        if len(tc_payments) < 3:
            continue

        seq, mask = builder.build_sequence(tc_payments, returns_by_tc.get(tc))
        sequences.append(seq)
        masks.append(mask)

        pred = pred_map.get(tc, {})
        # Binary labels based on thresholds
        l30.append(1 if float(pred.get("prob_30d", 0) or 0) > 0.5 else 0)
        l60.append(1 if float(pred.get("prob_60d", 0) or 0) > 0.5 else 0)
        l90.append(1 if float(pred.get("prob_90d", 0) or 0) > 0.5 else 0)

    if not sequences:
        return None, None, None, None, None

    return (
        torch.stack(sequences),
        torch.stack(masks),
        torch.tensor(l30, dtype=torch.long),
        torch.tensor(l60, dtype=torch.long),
        torch.tensor(l90, dtype=torch.long),
    )


def _resolve_db_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "TaxInspector")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def main(epochs: int = 200, min_samples: int = MIN_REQUIRED_SAMPLES) -> int:
    print("=" * 64)
    print("  TEMPORAL TRANSFORMER TRAINING (Delinquency Sequences)")
    print("=" * 64)

    from ml_engine.temporal_transformer import TemporalTransformerTrainer

    # ── Step 1: Load data ──
    print("\n[1/3] Loading payment sequences from database...")
    db_url = _resolve_db_url()
    result = _load_payment_sequences(db_url)

    if result[0] is None:
        print("[ABORT] No payment sequences found.")
        return 2

    sequences, padding_masks, labels_30d, labels_60d, labels_90d = result
    n = sequences.shape[0]
    print(f"       Loaded {n} company sequences ({sequences.shape})")

    if n < min_samples:
        print(f"[ABORT] Need >= {min_samples} sequences, got {n}")
        return 2

    print(f"       Labels — 30d: {labels_30d.sum().item()}/{n} | "
          f"60d: {labels_60d.sum().item()}/{n} | "
          f"90d: {labels_90d.sum().item()}/{n}")

    # ── Step 2: Train ──
    print(f"\n[2/3] Training Temporal Transformer ({epochs} epochs)...")
    trainer = TemporalTransformerTrainer()
    metrics = trainer.train(
        sequences=sequences,
        padding_masks=padding_masks,
        labels_30d=labels_30d,
        labels_60d=labels_60d,
        labels_90d=labels_90d,
        epochs=epochs,
        batch_size=64,
    )

    # ── Step 3: Save ──
    print("\n[3/3] Saving artifacts...")
    trainer.save(str(MODEL_DIR))

    quality_report = {
        "model_name": "delinquency_transformer",
        "model_version": "temporal-transformer-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {"total_sequences": n, "seq_len": 24, "feature_dim": 8},
        "metrics": {
            horizon: {k: round(v, 4) for k, v in vals.items()}
            for horizon, vals in metrics.items()
        },
    }
    with open(MODEL_DIR / "temporal_transformer_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    print(f"       [OK] All artifacts saved to {MODEL_DIR}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal Transformer")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min-samples", type=int, default=MIN_REQUIRED_SAMPLES)
    args = parser.parse_args()
    sys.exit(main(epochs=args.epochs, min_samples=args.min_samples))
