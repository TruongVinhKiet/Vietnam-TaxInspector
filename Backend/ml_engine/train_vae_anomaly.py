"""
train_vae_anomaly.py – Training Script for VAE Transaction Anomaly Detector
==============================================================================
Loads invoice/transaction data from the database and trains the
Variational Autoencoder for unsupervised anomaly detection.

Usage:
    python -m ml_engine.train_vae_anomaly
    python ml_engine/train_vae_anomaly.py --epochs 150

Artifacts:
    - data/models/vae_anomaly.pt
    - data/models/vae_anomaly_config.json
    - data/models/vae_anomaly_quality_report.json
    - data/models/vae_anomaly_scaler.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

MODEL_DIR = BACKEND_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_REQUIRED_SAMPLES = max(1000, int(os.getenv("VAE_MIN_SAMPLES", "1000")))


def _load_transaction_features(db_url: str, max_rows: int = 100000) -> np.ndarray | None:
    """Load invoice data and build feature matrix for VAE."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from ml_engine.vae_anomaly import TransactionFeatureBuilder

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT i.invoice_number, i.seller_tax_code, i.buyer_tax_code,
                       i.amount, i.vat_rate, i.date
                FROM invoices i
                ORDER BY i.date DESC
                LIMIT %s
            """, (max_rows,))
            invoices = cur.fetchall()

            if not invoices:
                return None

            # Load company info
            tax_codes = set()
            for inv in invoices:
                tax_codes.add(inv["seller_tax_code"])
                tax_codes.add(inv["buyer_tax_code"])

            cur.execute("""
                SELECT tax_code, name, industry, registration_date,
                       risk_score, is_active
                FROM companies
                WHERE tax_code = ANY(%s)
            """, (list(tax_codes),))
            companies = cur.fetchall()
    finally:
        conn.close()

    company_map = {c["tax_code"]: c for c in companies}
    builder = TransactionFeatureBuilder()
    X = builder.build_features(invoices, company_map)

    return X if X.shape[0] > 0 else None


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


def main(epochs: int = 150, min_samples: int = MIN_REQUIRED_SAMPLES) -> int:
    print("=" * 64)
    print("  VAE ANOMALY DETECTOR TRAINING (Transactions)")
    print("=" * 64)

    from ml_engine.vae_anomaly import VAEAnomalyTrainer, TransactionFeatureBuilder

    # ── Step 1: Load data ──
    print("\n[1/4] Loading transaction features from database...")
    db_url = _resolve_db_url()
    X = _load_transaction_features(db_url)

    if X is None or X.shape[0] == 0:
        print("[ABORT] No transaction data found.")
        return 2

    n = X.shape[0]
    print(f"       Loaded {n} transactions ({X.shape})")

    if n < min_samples:
        print(f"[ABORT] Need >= {min_samples} samples, got {n}")
        return 2

    # ── Step 2: Normalize ──
    print("\n[2/4] Normalizing features...")
    builder = TransactionFeatureBuilder()
    builder.fit_scaler(X)
    X_norm = builder.transform(X)

    # Save scaler parameters
    scaler_data = {
        "means": builder.means.tolist() if builder.means is not None else [],
        "stds": builder.stds.tolist() if builder.stds is not None else [],
    }
    with open(MODEL_DIR / "vae_anomaly_scaler.json", "w") as f:
        json.dump(scaler_data, f, indent=2)
    print("       Scaler fitted and saved.")

    # ── Step 3: Train ──
    print(f"\n[3/4] Training VAE Anomaly Detector ({epochs} epochs)...")
    trainer = VAEAnomalyTrainer(input_dim=X_norm.shape[1])
    metrics = trainer.train(X_norm, epochs=epochs)

    # ── Step 4: Save ──
    print("\n[4/4] Saving artifacts...")
    trainer.save(str(MODEL_DIR))

    quality_report = {
        "model_name": "vae_anomaly",
        "model_version": "vae-anomaly-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {"total_transactions": n, "feature_dim": int(X_norm.shape[1])},
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "acceptance_gates": {
            "min_samples": {"threshold": min_samples, "actual": n, "pass": n >= min_samples},
        },
    }
    with open(MODEL_DIR / "vae_anomaly_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    print(f"       [OK] All artifacts saved to {MODEL_DIR}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE Anomaly Detector")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--min-samples", type=int, default=MIN_REQUIRED_SAMPLES)
    args = parser.parse_args()
    sys.exit(main(epochs=args.epochs, min_samples=args.min_samples))
