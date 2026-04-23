"""
train_osint.py – Training Script for Offshore Risk Classifier (OSINT)
=====================================================================
Extracts structural and financial features of Offshore Entities from the
database (ownership_links, invoices, companies), generates pseudo-labels
based on complex risk rules, and trains an XGBoost classifier.

Outputs:
    - data/models/osint_risk_model.joblib
    - data/models/osint_config.json
    - data/models/osint_quality_report.json
    - data/models/osint_drift_baseline.json
"""

import os
import sys
import json
import joblib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Ensure parent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "osint-classifier-v1"
FEATURE_NAMES = [
    "n_dom_subs",
    "n_rel_types",
    "max_own_pct",
    "inv_in_bn",
    "inv_out_bn",
    "max_dom_risk",
    "avg_dom_risk",
    "juris_risk",
]
OSINT_ACCEPTANCE_AUC_MIN = 0.60
OSINT_ACCEPTANCE_PR_AUC_MIN = 0.35
OSINT_MIN_TRAINING_SAMPLES = int(os.environ.get("OSINT_MIN_REQUIRED_SAMPLES", "10000"))

JURISDICTION_RISK = {
    "Cayman Islands": 5.0,
    "British Virgin Islands (BVI)": 5.0,
    "Panama": 5.0,
    "Seychelles": 4.0,
    "Bahamas": 4.0,
    "Cyprus": 3.0,
    "Hong Kong": 2.0,
    "Singapore": 2.0,
}


def _build_feature_stats(matrix: np.ndarray, feature_names: list[str]) -> dict[str, dict]:
    if matrix.size == 0:
        return {}

    stats: dict[str, dict] = {}
    for idx, name in enumerate(feature_names):
        values = np.asarray(matrix[:, idx], dtype=float)
        if values.size == 0:
            continue
        stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q0": float(np.min(values)),
            "q5": float(np.percentile(values, 5)),
            "q25": float(np.percentile(values, 25)),
            "q50": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75)),
            "q95": float(np.percentile(values, 95)),
            "q100": float(np.max(values)),
        }
    return stats

def fetch_offshore_data(db_url: str):
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("      Fetching offshore companies...")
    cur.execute("SELECT tax_code, province FROM companies WHERE industry = 'Offshore Entity'")
    offshore_rows = cur.fetchall()
    
    print("      Fetching ownership links & domestic companies risk...")
    cur.execute("""
        SELECT 
            ol.parent_tax_code as offshore_tax_code,
            ol.child_tax_code as domestic_tax_code,
            ol.relationship_type,
            ol.ownership_percent,
            COALESCE(c.risk_score, 0) as risk_score
        FROM ownership_links ol
        JOIN companies c ON c.tax_code = ol.child_tax_code
        WHERE ol.parent_tax_code IN (SELECT tax_code FROM companies WHERE industry = 'Offshore Entity')
    """)
    link_rows = cur.fetchall()
    
    print("      Fetching invoices...")
    cur.execute("""
        SELECT seller_tax_code, buyer_tax_code, amount, vat_rate 
        FROM invoices
    """)
    invoice_rows = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return offshore_rows, link_rows, invoice_rows

def build_features(offshore_rows, link_rows, invoice_rows):
    # Prepare invoice dicts
    invoice_in = {}  # key: domestic_tax_code, val: total_amount_in
    invoice_out = {} # key: domestic_tax_code, val: total_amount_out
    
    for inv in invoice_rows:
        seller = inv["seller_tax_code"]
        buyer = inv["buyer_tax_code"]
        amt = float(inv["amount"])
        
        invoice_out[seller] = invoice_out.get(seller, 0) + amt
        invoice_in[buyer] = invoice_in.get(buyer, 0) + amt

    # Prepare links grouping
    links_by_parent = {}
    for link in link_rows:
        parent = link["offshore_tax_code"]
        if parent not in links_by_parent:
            links_by_parent[parent] = []
        links_by_parent[parent].append(link)
        
    features = []
    labels = []
    tax_codes = []
    
    for off in offshore_rows:
        tc = off["tax_code"]
        country = off["province"]
        
        juris_risk = JURISDICTION_RISK.get(country, 1.0)
        
        links = links_by_parent.get(tc, [])
        n_dom_subs = len(links)
        n_rel_types = len(set(l["relationship_type"] for l in links))
        max_own = float(max((l["ownership_percent"] for l in links), default=0.0))
        
        tot_inv_in = 0.0
        tot_inv_out = 0.0
        risks = []
        
        for l in links:
            child = l["domestic_tax_code"]
            tot_inv_in += invoice_in.get(child, 0.0)
            tot_inv_out += invoice_out.get(child, 0.0)
            risks.append(float(l["risk_score"]))
            
        max_risk = max(risks) if risks else 0.0
        avg_risk = sum(risks)/len(risks) if risks else 0.0
        
        # Pseudo-labeling logic
        # 1. Very high domestic risk
        target = 0
        if max_risk > 0.7:
            target = 1
        # 2. High invoice volume (> 5B) AND suspicious jurisdiction
        elif (tot_inv_in + tot_inv_out) > 5_000_000_000 and juris_risk >= 4.0:
            target = 1
        # 3. High complex network (many shell subsidiaries) + avg risk > 0.4
        elif n_dom_subs >= 3 and n_rel_types >= 2 and avg_risk > 0.4:
            target = 1
            
        # Add random noise to pseudo-labels to prevent overfitting to the deterministic rule (10% flip)
        if np.random.rand() < 0.1:
            target = 1 - target
            
        features.append([
            float(n_dom_subs),
            float(n_rel_types),
            max_own,
            # scale invoice down to avoid numerical overflow (in billion VND)
            tot_inv_in / 1e9,
            tot_inv_out / 1e9,
            max_risk,
            avg_risk,
            juris_risk
        ])
        labels.append(target)
        tax_codes.append(tc)
        
    return np.array(features), np.array(labels), tax_codes

def train_model(db_url: str, seed: int = 42, min_samples: int = OSINT_MIN_TRAINING_SAMPLES):
    print("=" * 60)
    print("  OSINT OFFSHORE RISK MODEL TRAINING")
    print("=" * 60)

    np.random.seed(int(seed))
    
    env_url = os.environ.get("DATABASE_URL")
    if not env_url:
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "TaxInspector")
        env_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    if db_url == "default":
        db_url = env_url
        
    print("[1/4] Extracting raw data from PostgreSQL...")
    offshore_rows, link_rows, invoice_rows = fetch_offshore_data(db_url)
    print(f"      Got {len(offshore_rows)} offshore entities, {len(link_rows)} links, {len(invoice_rows)} invoices.")
    
    print("\n[2/4] Engineering graph & financial features...")
    X, y, tax_codes = build_features(offshore_rows, link_rows, invoice_rows)
    print(f"      Feature Matrix: {X.shape}")
    print(f"      High Risk Label Distribution: {int(y.sum())} / {len(y)}")

    required_samples = max(1, int(min_samples))
    if len(y) < required_samples:
        raise RuntimeError(
            f"OSINT training requires at least {required_samples:,} samples; got {len(y):,}."
        )

    if len(np.unique(y)) < 2:
        raise RuntimeError("OSINT training requires both positive and negative labels; current dataset has one class only.")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n[3/4] Training XGBoost Classifier...")
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=max(1, len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        model_type = "xgboost"
    except ImportError:
        print("      [WARN] XGBoost not available, falling back to sklearn.")
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        model_type = "sklearn_gbc"
        
    # Evaluate
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
        
    y_pred = (y_prob >= 0.5).astype(int)
    
    auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    
    print(f"      AUC-ROC: {auc:.4f}")
    print(f"      PR-AUC:  {pr_auc:.4f}")
    print("\n      Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
    
    feature_names = FEATURE_NAMES
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        for f, i in sorted(zip(feature_names, imps), key=lambda x: -x[1]):
            print(f"        {f:15s}: {i:.4f}")

    print("\n[4/4] Saving artifacts...")
    model_path = MODEL_DIR / "osint_risk_model.joblib"
    joblib.dump(model, model_path)
    
    config = {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "features": feature_names,
    }
    with open(MODEL_DIR / "osint_config.json", "w") as f:
        json.dump(config, f, indent=2)

    acceptance_criteria = {
        "training_samples_min": {
            "pass": bool(len(y) >= required_samples),
            "actual": int(len(y)),
            "threshold": int(required_samples),
        },
        "auc_min": {
            "pass": bool(auc >= OSINT_ACCEPTANCE_AUC_MIN),
            "actual": round(auc, 4),
            "threshold": OSINT_ACCEPTANCE_AUC_MIN,
        },
        "pr_auc_min": {
            "pass": bool(pr_auc >= OSINT_ACCEPTANCE_PR_AUC_MIN),
            "actual": round(pr_auc, 4),
            "threshold": OSINT_ACCEPTANCE_PR_AUC_MIN,
        },
    }
        
    quality = {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.utcnow().isoformat(),
        "metrics": {
            "auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4)
        },
        "acceptance_gates": {
            "overall_pass": bool(all(item.get("pass") for item in acceptance_criteria.values())),
            "criteria": acceptance_criteria,
        },
        "dataset": {
            "total_samples": len(y),
            "required_min_samples": int(required_samples),
            "high_risk_ratio": round(float(y.sum() / len(y)), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
    }
    with open(MODEL_DIR / "osint_quality_report.json", "w") as f:
        json.dump(quality, f, indent=2)

    drift_baseline = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": "osint_training_features",
        "sample_size": int(len(X)),
        "positive_ratio": round(float(y.sum() / len(y)), 6),
        "feature_names": feature_names,
        "features": _build_feature_stats(X, feature_names),
    }
    with open(MODEL_DIR / "osint_drift_baseline.json", "w", encoding="utf-8") as f:
        json.dump(drift_baseline, f, indent=2, ensure_ascii=False)

    print("      [OK] Saved artifacts to data/models/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=OSINT_MIN_TRAINING_SAMPLES)
    args = parser.parse_args()
    
    train_model(
        args.db_url,
        seed=args.seed,
        min_samples=max(1, int(args.min_samples)),
    )
