"""
train_all_new_models.py – Master Training Script for 4 New DL/ML Models
=========================================================================
Trains all new models using data from the TaxInspector database:
    1. Temporal Transformer  — from tax_payments (120,000 records)
    2. Causal Uplift T-Learner — from collection_actions/outcomes (4,000 records)
    3. VAE Anomaly Detector — from invoices (66,005 records)
    4. HeteroGNN — from companies + invoices (synthetic hetero graph)

Note: GNNExplainer (Module 5) does not require separate training.

Usage:
    python train_all_new_models.py
"""

import json
import math
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

MODEL_DIR = BACKEND_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_db_url():
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "TaxInspector")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


# ═══════════════════════════════════════════════════════════════
#  MODEL 1: Temporal Transformer (Delinquency Sequences)
# ═══════════════════════════════════════════════════════════════

def train_temporal_transformer(db_url: str):
    print("\n" + "=" * 70)
    print("  MODEL 1: TEMPORAL TRANSFORMER (Delinquency Sequences)")
    print("=" * 70)
    t0 = time.time()

    import psycopg2
    from psycopg2.extras import RealDictCursor
    from ml_engine.temporal_transformer import (
        DelinquencyTransformer, PaymentSequenceBuilder,
        TemporalTransformerTrainer, SEQ_LEN, FEATURE_DIM,
    )

    # ── Load payment data ──
    print("\n[1/4] Loading payment data...")
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT tax_code,
                       COALESCE(amount_paid, amount_due, 0) AS amount,
                       actual_payment_date AS payment_date,
                       tax_period,
                       status,
                       CASE
                           WHEN actual_payment_date IS NOT NULL AND due_date IS NOT NULL
                           THEN GREATEST(0, EXTRACT(DAY FROM (actual_payment_date::timestamp - due_date::timestamp)))
                           WHEN status IN ('overdue','partial') THEN 30
                           ELSE 0
                       END AS days_overdue,
                       COALESCE(penalty_amount, 0) AS penalty_amount
                FROM tax_payments
                WHERE actual_payment_date IS NOT NULL
                ORDER BY tax_code, actual_payment_date
            """)
            payments = cur.fetchall()
            print(f"       Loaded {len(payments):,} payment records")

            cur.execute("""
                SELECT DISTINCT ON (tax_code)
                    tax_code, prob_30d, prob_60d, prob_90d
                FROM delinquency_predictions
                ORDER BY tax_code, created_at DESC
            """)
            predictions = cur.fetchall()
            print(f"       Loaded {len(predictions):,} delinquency predictions")
    finally:
        conn.close()

    # ── Group by tax_code and build sequences ──
    print("\n[2/4] Building payment sequences...")
    payments_by_tc = {}
    for p in payments:
        tc = p["tax_code"]
        payments_by_tc.setdefault(tc, []).append(p)

    pred_map = {p["tax_code"]: p for p in predictions}
    builder = PaymentSequenceBuilder()

    sequences = []
    masks = []
    l30, l60, l90 = [], [], []

    for tc, tc_payments in payments_by_tc.items():
        if len(tc_payments) < 3:
            continue
        seq, mask = builder.build_sequence(tc_payments, [])
        sequences.append(seq)
        masks.append(mask)

        pred = pred_map.get(tc, {})
        l30.append(1 if float(pred.get("prob_30d", 0) or 0) > 0.5 else 0)
        l60.append(1 if float(pred.get("prob_60d", 0) or 0) > 0.5 else 0)
        l90.append(1 if float(pred.get("prob_90d", 0) or 0) > 0.5 else 0)

    sequences_t = torch.stack(sequences)
    masks_t = torch.stack(masks)
    labels_30d = torch.tensor(l30, dtype=torch.long)
    labels_60d = torch.tensor(l60, dtype=torch.long)
    labels_90d = torch.tensor(l90, dtype=torch.long)

    n = sequences_t.shape[0]
    print(f"       Built {n:,} sequences ({sequences_t.shape})")
    print(f"       Labels 30d={labels_30d.sum()}/{n} | 60d={labels_60d.sum()}/{n} | 90d={labels_90d.sum()}/{n}")

    # ── Train ──
    print("\n[3/4] Training Temporal Transformer (150 epochs)...")
    trainer = TemporalTransformerTrainer(lr=0.001)
    metrics = trainer.train(
        sequences=sequences_t,
        padding_masks=masks_t,
        labels_30d=labels_30d,
        labels_60d=labels_60d,
        labels_90d=labels_90d,
        epochs=150,
        batch_size=64,
    )

    # ── Save ──
    print("\n[4/4] Saving artifacts...")
    trainer.save(str(MODEL_DIR))

    report = {
        "model_name": "delinquency_transformer",
        "model_version": "temporal-transformer-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "training_time_seconds": round(time.time() - t0, 1),
        "dataset": {"total_sequences": n, "seq_len": SEQ_LEN, "feature_dim": FEATURE_DIM},
        "metrics": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in metrics.items()},
    }
    with open(MODEL_DIR / "temporal_transformer_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"       [OK] Done in {time.time()-t0:.1f}s")
    return report


# ═══════════════════════════════════════════════════════════════
#  MODEL 2: Causal Uplift T-Learner (Collections)
# ═══════════════════════════════════════════════════════════════

def train_causal_uplift(db_url: str):
    print("\n" + "=" * 70)
    print("  MODEL 2: CAUSAL UPLIFT T-LEARNER (Collection Optimization)")
    print("=" * 70)
    t0 = time.time()

    import psycopg2
    from psycopg2.extras import RealDictCursor
    from ml_engine.causal_uplift_model import TLearnerUplift, compute_qini_coefficient

    # ── Load data ──
    print("\n[1/4] Loading collection data...")
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    c.tax_code,
                    COALESCE(c.risk_score, 0) AS fraud_score,
                    0.55 AS fraud_confidence,
                    COALESCE(dp.prob_90d, 0) AS delinquency_90d,
                    COALESCE(vrp.risk_score, 0) AS vat_refund_score,
                    COALESCE(asp.priority_score, 0) AS prior_priority,
                    COALESCE(ca_stats.n_actions, 0) AS n_past_actions,
                    COALESCE(ca_stats.success_rate, 0) AS past_success_rate,
                    COALESCE(EXTRACT(YEAR FROM AGE(NOW(), c.registration_date)), 3) AS company_age_years,
                    LN(GREATEST(COALESCE(c.risk_score, 1)::numeric * 1000, 1)) AS revenue_log,
                    0.08 AS industry_risk,
                    CASE WHEN ca_stats.n_actions > 0 THEN 1 ELSE 0 END AS treatment,
                    CASE WHEN COALESCE(co.amount_collected, 0) > 0 THEN 1 ELSE 0 END AS outcome
                FROM companies c
                LEFT JOIN LATERAL (
                    SELECT prob_90d FROM delinquency_predictions
                    WHERE tax_code = c.tax_code ORDER BY created_at DESC LIMIT 1
                ) dp ON TRUE
                LEFT JOIN LATERAL (
                    SELECT risk_score FROM vat_refund_predictions
                    WHERE tax_code = c.tax_code ORDER BY created_at DESC LIMIT 1
                ) vrp ON TRUE
                LEFT JOIN LATERAL (
                    SELECT priority_score FROM audit_selection_predictions
                    WHERE tax_code = c.tax_code ORDER BY created_at DESC LIMIT 1
                ) asp ON TRUE
                LEFT JOIN LATERAL (
                    SELECT COUNT(*) AS n_actions,
                           AVG(CASE WHEN result='success' THEN 1.0
                                    WHEN result='partial' THEN 0.5 ELSE 0.0 END) AS success_rate
                    FROM collection_actions WHERE tax_code = c.tax_code
                ) ca_stats ON TRUE
                LEFT JOIN LATERAL (
                    SELECT SUM(amount_collected) AS amount_collected
                    FROM collection_outcomes WHERE tax_code = c.tax_code
                ) co ON TRUE
            """)
            rows = cur.fetchall()
            print(f"       Loaded {len(rows):,} company records")
    finally:
        conn.close()

    # ── Build feature matrix ──
    print("\n[2/4] Building feature matrix...")
    feature_keys = [
        "fraud_score", "fraud_confidence", "delinquency_90d", "vat_refund_score",
        "prior_priority", "n_past_actions", "past_success_rate",
        "company_age_years", "revenue_log", "industry_risk",
    ]
    X = np.array([[float(r.get(k, 0) or 0) for k in feature_keys] for r in rows], dtype=float)
    treatment = np.array([int(r.get("treatment", 0) or 0) for r in rows], dtype=int)

    # Build a nuanced outcome combining multiple signals:
    # - collection success (if treated)
    # - risk-based proxy (for untreated companies, high-risk ones likely *would* have yielded)
    outcome = np.zeros(len(rows), dtype=int)
    for i, r in enumerate(rows):
        has_collection = float(r.get("outcome", 0) or 0) > 0
        fraud_score = float(r.get("fraud_score", 0) or 0)
        delinq = float(r.get("delinquency_90d", 0) or 0)

        if treatment[i] == 1:
            # Treated: outcome from actual collection + risk signal
            outcome[i] = 1 if (has_collection or fraud_score > 60) else 0
        else:
            # Control: counterfactual proxy — high-risk untreated companies
            # would likely have yielded if acted upon
            outcome[i] = 1 if (fraud_score > 70 and delinq > 0.6) else 0

    n_treated = int(treatment.sum())
    n_control = int((1 - treatment).sum())
    print(f"       Treated: {n_treated:,} | Control: {n_control:,}")
    print(f"       Outcome positive: {outcome.sum():,}/{len(outcome):,}")
    print(f"       Treated+positive: {int((treatment * outcome).sum()):,} | Control+positive: {int(((1-treatment) * outcome).sum()):,}")

    # Ensure both groups have both classes
    t_classes = np.unique(outcome[treatment == 1])
    c_classes = np.unique(outcome[treatment == 0])
    if len(t_classes) < 2 or len(c_classes) < 2:
        print("       [WARN] Injecting counterfactual noise for class balance...")
        rng = np.random.default_rng(42)
        # Flip 5% of each group to ensure both classes
        for group_mask in [treatment == 1, treatment == 0]:
            group_idx = np.where(group_mask)[0]
            n_flip = max(10, len(group_idx) // 20)
            flip_idx = rng.choice(group_idx, size=n_flip, replace=False)
            outcome[flip_idx] = 1 - outcome[flip_idx]
        print(f"       After balancing: T+={int((treatment * outcome).sum()):,} | C+={int(((1-treatment) * outcome).sum()):,}")

    # ── Train ──
    print("\n[3/4] Training T-Learner Uplift model...")
    uplift = TLearnerUplift()
    metrics = uplift.fit(X, treatment, outcome, n_estimators=300, max_depth=6)

    cate_all = uplift.predict(X)
    qini = compute_qini_coefficient(cate_all, treatment, outcome)
    metrics["qini_coefficient"] = qini

    print(f"       Avg CATE: {metrics['avg_cate']}")
    print(f"       Treated AUC: {metrics['treated_auc']}")
    print(f"       Control AUC: {metrics['control_auc']}")
    print(f"       Qini Coefficient: {qini}")
    print(f"       % Positive Uplift: {metrics['pct_positive_uplift']}%")

    # ── Save ──
    print("\n[4/4] Saving artifacts...")
    uplift.save(str(MODEL_DIR))

    report = {
        "model_name": "causal_uplift_t_learner",
        "model_version": "uplift-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "training_time_seconds": round(time.time() - t0, 1),
        "dataset": {"total_samples": len(X), "n_treated": n_treated, "n_control": n_control},
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }
    with open(MODEL_DIR / "causal_uplift_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"       [OK] Done in {time.time()-t0:.1f}s")
    return report


# ═══════════════════════════════════════════════════════════════
#  MODEL 3: VAE Anomaly Detector (Invoices)
# ═══════════════════════════════════════════════════════════════

def train_vae_anomaly(db_url: str):
    print("\n" + "=" * 70)
    print("  MODEL 3: VAE ANOMALY DETECTOR (Invoice Transactions)")
    print("=" * 70)
    t0 = time.time()

    import psycopg2
    from psycopg2.extras import RealDictCursor
    from ml_engine.vae_anomaly import TransactionVAE, VAEAnomalyTrainer, TransactionFeatureBuilder

    # ── Load invoice data ──
    print("\n[1/4] Loading invoice data...")
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT i.invoice_number, i.seller_tax_code, i.buyer_tax_code,
                       i.amount, i.vat_rate, i.date
                FROM invoices i
                ORDER BY i.date DESC
                LIMIT 66000
            """)
            invoices = cur.fetchall()
            print(f"       Loaded {len(invoices):,} invoices")

            # Load company info
            tax_codes = set()
            for inv in invoices:
                tax_codes.add(inv.get("seller_tax_code", ""))
                tax_codes.add(inv.get("buyer_tax_code", ""))
            tax_codes.discard("")
            tax_codes.discard(None)

            cur.execute("""
                SELECT tax_code, name, industry, registration_date,
                       risk_score, is_active
                FROM companies
                WHERE tax_code = ANY(%s)
            """, (list(tax_codes),))
            companies = cur.fetchall()
            print(f"       Loaded {len(companies):,} company records")
    finally:
        conn.close()

    # ── Build features ──
    print("\n[2/4] Building feature matrix...")
    company_map = {c["tax_code"]: c for c in companies}
    builder = TransactionFeatureBuilder()
    X = builder.build_features(invoices, company_map)
    print(f"       Feature matrix: {X.shape}")

    # Normalize
    builder.fit_scaler(X)
    X_norm = builder.transform(X)

    scaler_data = {
        "means": builder.means.tolist() if builder.means is not None else [],
        "stds": builder.stds.tolist() if builder.stds is not None else [],
    }
    with open(MODEL_DIR / "vae_anomaly_scaler.json", "w") as f:
        json.dump(scaler_data, f, indent=2)

    # ── Train ──
    print("\n[3/4] Training VAE Anomaly Detector (150 epochs)...")
    trainer = VAEAnomalyTrainer(input_dim=X_norm.shape[1])
    metrics = trainer.train(X_norm, epochs=150, batch_size=256)

    # ── Save ──
    print("\n[4/4] Saving artifacts...")
    trainer.save(str(MODEL_DIR))

    report = {
        "model_name": "vae_anomaly",
        "model_version": "vae-anomaly-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "training_time_seconds": round(time.time() - t0, 1),
        "dataset": {"total_invoices": int(X.shape[0]), "feature_dim": int(X.shape[1])},
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
    }
    with open(MODEL_DIR / "vae_anomaly_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"       [OK] Done in {time.time()-t0:.1f}s")
    return report


# ═══════════════════════════════════════════════════════════════
#  MODEL 4: HeteroGNN (Synthetic Heterogeneous Graph)
# ═══════════════════════════════════════════════════════════════

def train_hetero_gnn(db_url: str):
    print("\n" + "=" * 70)
    print("  MODEL 4: HETEROGENEOUS GRAPH TRANSFORMER (HGT)")
    print("=" * 70)
    t0 = time.time()

    import psycopg2
    from psycopg2.extras import RealDictCursor

    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        print("[SKIP] torch_geometric not installed. Skipping HeteroGNN.")
        return None

    from ml_engine.hetero_gnn_model import TaxFraudHGT, HeteroGNNTrainer, NODE_TYPES, EDGE_TYPES

    # ── Load company + invoice data for graph construction ──
    print("\n[1/5] Loading graph data from database...")
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT tax_code, name, industry, risk_score,
                       registration_date, is_active
                FROM companies
            """)
            companies = cur.fetchall()
            print(f"       Companies: {len(companies):,}")

            cur.execute("""
                SELECT seller_tax_code, buyer_tax_code, amount, date
                FROM invoices
                ORDER BY date DESC LIMIT 50000
            """)
            invoices = cur.fetchall()
            print(f"       Invoices: {len(invoices):,}")
    finally:
        conn.close()

    # ── Build heterogeneous graph ──
    print("\n[2/5] Constructing heterogeneous graph...")
    tc_list = [c["tax_code"] for c in companies]
    tc_to_idx = {tc: i for i, tc in enumerate(tc_list)}
    n_companies = len(tc_list)

    # Node features for companies (15 dims)
    company_features = []
    for c in companies:
        risk = float(c.get("risk_score", 0) or 0) / 100.0
        age_days = 1000
        if c.get("registration_date"):
            try:
                reg = c["registration_date"]
                if isinstance(reg, str):
                    reg = date.fromisoformat(reg)
                age_days = (date.today() - reg).days
            except Exception:
                pass
        age_norm = min(1.0, age_days / 3650.0)
        capital = math.log1p(float(risk * 1000)) / 30.0
        lat = 0.0
        lon = 0.0
        active = 1.0 if c.get("is_active", True) else 0.0

        # Industry one-hot (simplified)
        industry = str(c.get("industry", "") or "").lower()
        ind_features = [0.0] * 8
        ind_map = {"xây dựng": 0, "bất động sản": 1, "thương mại": 2,
                   "sản xuất": 3, "nông nghiệp": 4, "công nghệ": 5,
                   "tài chính": 6, "dịch vụ": 7}
        for key, idx in ind_map.items():
            if key in industry:
                ind_features[idx] = 1.0
                break

        feat = [risk, capital, age_norm, lat, lon, active, 0.0] + ind_features
        company_features.append(feat[:15])

    company_x = torch.tensor(company_features, dtype=torch.float32)

    # Synthetic person nodes (directors/shareholders — 5% of companies)
    n_persons = max(50, n_companies // 20)
    person_x = torch.randn(n_persons, 15) * 0.3 + 0.2
    person_x = person_x.clamp(0, 1)

    # Synthetic offshore entities (2% of companies)
    n_offshore = max(20, n_companies // 50)
    offshore_x = torch.randn(n_offshore, 15) * 0.3 + 0.6
    offshore_x = offshore_x.clamp(0, 1)

    print(f"       Nodes: company={n_companies}, person={n_persons}, offshore={n_offshore}")

    # Build edges from invoices
    invoice_edges_src = []
    invoice_edges_dst = []
    for inv in invoices:
        s = tc_to_idx.get(inv.get("seller_tax_code"))
        b = tc_to_idx.get(inv.get("buyer_tax_code"))
        if s is not None and b is not None and s != b:
            invoice_edges_src.append(s)
            invoice_edges_dst.append(b)

    # Synthetic edges for other edge types
    rng = np.random.default_rng(42)
    # Person controls company
    controls_src = rng.integers(0, n_persons, size=min(n_persons * 3, n_companies))
    controls_dst = rng.integers(0, n_companies, size=len(controls_src))
    # Company owns company (subsidiaries)
    n_owns = n_companies // 10
    owns_src = rng.integers(0, n_companies, size=n_owns)
    owns_dst = rng.integers(0, n_companies, size=n_owns)
    # Company alias
    n_alias = n_companies // 50
    alias_src = rng.integers(0, n_companies, size=n_alias)
    alias_dst = rng.integers(0, n_companies, size=n_alias)
    # Phoenix successor
    n_phoenix = n_companies // 100
    phoenix_src = rng.integers(0, n_companies, size=n_phoenix)
    phoenix_dst = rng.integers(0, n_companies, size=n_phoenix)

    # ── Assemble HeteroData ──
    print("\n[3/5] Assembling HeteroData object...")
    data = HeteroData()
    data["company"].x = company_x
    data["person"].x = person_x
    data["offshore_entity"].x = offshore_x

    data["company", "issued_invoice_to", "company"].edge_index = torch.tensor(
        [invoice_edges_src[:50000], invoice_edges_dst[:50000]], dtype=torch.long
    )
    data["person", "controls", "company"].edge_index = torch.tensor(
        [controls_src.tolist(), controls_dst.tolist()], dtype=torch.long
    )
    data["company", "owns", "company"].edge_index = torch.tensor(
        [owns_src.tolist(), owns_dst.tolist()], dtype=torch.long
    )
    data["company", "alias", "company"].edge_index = torch.tensor(
        [alias_src.tolist(), alias_dst.tolist()], dtype=torch.long
    )
    data["company", "phoenix_successor", "company"].edge_index = torch.tensor(
        [phoenix_src.tolist(), phoenix_dst.tolist()], dtype=torch.long
    )

    for etype in data.edge_types:
        print(f"       Edge {etype[1]:25s}: {data[etype].edge_index.shape[1]:,}")

    # ── Generate labels (weak supervision from risk_score) ──
    print("\n[4/5] Training HGT model (200 epochs)...")
    labels = {}
    train_masks = {}
    val_masks = {}

    for ntype in NODE_TYPES:
        if ntype not in data.node_types:
            continue
        n = data[ntype].x.shape[0]
        risk = data[ntype].x[:, 0].numpy()
        threshold = np.percentile(risk, 75)
        lbl = torch.tensor((risk > threshold).astype(int), dtype=torch.long)
        labels[ntype] = lbl

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

    trainer = HeteroGNNTrainer(node_feature_dim=15)
    metrics = trainer.train(
        data=data, labels=labels,
        train_masks=train_masks, val_masks=val_masks,
        epochs=200,
    )

    # ── Save ──
    print("\n[5/5] Saving artifacts...")
    trainer.save(str(MODEL_DIR))

    report = {
        "model_name": "osint_hetero_gnn",
        "model_version": "hgt-v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "training_time_seconds": round(time.time() - t0, 1),
        "dataset": {
            "company_nodes": n_companies,
            "person_nodes": n_persons,
            "offshore_nodes": n_offshore,
            "invoice_edges": len(invoice_edges_src),
        },
        "metrics": {
            ntype: {k: round(v, 4) for k, v in vals.items()}
            for ntype, vals in metrics.items()
        },
    }
    with open(MODEL_DIR / "hgt_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"       [OK] Done in {time.time()-t0:.1f}s")
    return report


# ═══════════════════════════════════════════════════════════════
#  MAIN: Train All
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  TAXINSPECTOR -- MASTER TRAINING: 4 NEW DL/ML MODELS")
    print("=" * 70)

    db_url = get_db_url()
    total_start = time.time()
    results = {}

    # ── 1. Temporal Transformer ──
    try:
        results["temporal_transformer"] = train_temporal_transformer(db_url)
    except Exception as e:
        print(f"\n[ERROR] Temporal Transformer failed: {e}")
        import traceback; traceback.print_exc()
        results["temporal_transformer"] = {"status": "FAILED", "error": str(e)}

    # ── 2. Causal Uplift ──
    try:
        results["causal_uplift"] = train_causal_uplift(db_url)
    except Exception as e:
        print(f"\n[ERROR] Causal Uplift failed: {e}")
        import traceback; traceback.print_exc()
        results["causal_uplift"] = {"status": "FAILED", "error": str(e)}

    # ── 3. VAE Anomaly ──
    try:
        results["vae_anomaly"] = train_vae_anomaly(db_url)
    except Exception as e:
        print(f"\n[ERROR] VAE Anomaly failed: {e}")
        import traceback; traceback.print_exc()
        results["vae_anomaly"] = {"status": "FAILED", "error": str(e)}

    # ── 4. HeteroGNN ──
    try:
        results["hetero_gnn"] = train_hetero_gnn(db_url)
    except Exception as e:
        print(f"\n[ERROR] HeteroGNN failed: {e}")
        import traceback; traceback.print_exc()
        results["hetero_gnn"] = {"status": "FAILED", "error": str(e)}

    # ── Summary ──
    total_time = time.time() - total_start
    print("\n")
    print("=" * 70)
    print("  TRAINING SUMMARY")
    print("-" * 70)
    for name, res in results.items():
        if res is None:
            status = "SKIPPED"
        elif isinstance(res, dict) and res.get("status") == "FAILED":
            status = f"FAILED: {res.get('error', '')[:40]}"
        else:
            status = "OK"
        print(f"  {name:30s}  {status}")
    print("-" * 70)
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)

    # Save master report
    with open(MODEL_DIR / "training_master_report.json", "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_time_seconds": round(total_time, 1),
            "results": {
                k: v if v is not None else "skipped"
                for k, v in results.items()
            },
        }, f, indent=2, default=str)
