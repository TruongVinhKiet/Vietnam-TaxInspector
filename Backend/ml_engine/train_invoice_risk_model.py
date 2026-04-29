from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from ml_engine.invoice_risk_model import InvoiceRiskScorer

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_VERSION = "invoice-risk-learned-v1"


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    scorer = InvoiceRiskScorer()
    scorer.model = None
    scorer.model_config = {}
    scorer.model_version = "invoice-risk-heuristic-v1"
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    i.invoice_number, i.seller_tax_code, i.buyer_tax_code, i.amount, i.vat_rate, i.date,
                    i.payment_status, i.is_adjustment,
                    (SELECT COUNT(*) FROM invoice_events e WHERE e.invoice_number = i.invoice_number) AS event_count,
                    (SELECT COUNT(*) FROM invoice_fingerprints f
                        JOIN invoice_fingerprints f2 ON f.hash_near_dup = f2.hash_near_dup AND f2.invoice_number <> f.invoice_number
                     WHERE f.invoice_number = i.invoice_number AND f.hash_near_dup IS NOT NULL) AS near_dup_count,
                    (SELECT COUNT(*) FROM invoices i2
                     WHERE i2.date = i.date AND i2.seller_tax_code = i.seller_tax_code AND i2.buyer_tax_code = i.buyer_tax_code) AS same_day_pair_count,
                    (SELECT COALESCE(risk_score, 0) FROM ai_risk_assessments a WHERE a.tax_code = i.seller_tax_code ORDER BY created_at DESC LIMIT 1) AS seller_risk_score,
                    (SELECT COALESCE(risk_score, 0) FROM ai_risk_assessments a WHERE a.tax_code = i.buyer_tax_code ORDER BY created_at DESC LIMIT 1) AS buyer_risk_score
                FROM invoices i
                WHERE i.invoice_number IS NOT NULL
                ORDER BY i.date DESC
                LIMIT 50000
                """
            )
        ).mappings().all()

        X = []
        y = []
        for row in rows:
            invoice = dict(row)
            context = {
                "event_count": int(row["event_count"] or 0),
                "near_dup_count": int(row["near_dup_count"] or 0),
                "same_day_pair_count": int(row["same_day_pair_count"] or 0),
                "seller_risk_score": float(row["seller_risk_score"] or 0.0),
                "buyer_risk_score": float(row["buyer_risk_score"] or 0.0),
                "linked_invoice_ids": [],
            }
            features = scorer._build_feature_dict(invoice, context)
            teacher = scorer.score(invoice, context)
            pseudo_label = 1 if (teacher.risk_score >= 60 or context["near_dup_count"] > 0 or context["event_count"] >= 2) else 0
            X.append([features.get(col, 0.0) for col in scorer.FEATURE_COLS])
            y.append(pseudo_label)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise RuntimeError("Invoice risk training requires both classes.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, prob))
        pr_auc = float(average_precision_score(y_test, prob))

        joblib.dump(model, MODEL_DIR / "invoice_risk_model.joblib")
        with open(MODEL_DIR / "invoice_risk_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_version": MODEL_VERSION,
                    "model_type": "random_forest",
                    "features": list(scorer.FEATURE_COLS),
                },
                f,
                indent=2,
            )
        with open(MODEL_DIR / "invoice_risk_quality_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_version": MODEL_VERSION,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "metrics": {"auc": round(auc, 4), "pr_auc": round(pr_auc, 4)},
                    "dataset": {"samples": int(len(X)), "positive_ratio": round(float(y.mean()), 4)},
                    "acceptance_gates": {
                        "overall_pass": bool(auc >= 0.60 and pr_auc >= 0.30),
                        "criteria": {
                            "auc_min": {"pass": bool(auc >= 0.60), "actual": round(auc, 4), "threshold": 0.60},
                            "pr_auc_min": {"pass": bool(pr_auc >= 0.30), "actual": round(pr_auc, 4), "threshold": 0.30},
                        },
                    },
                },
                f,
                indent=2,
            )
        print(f"[OK] Trained invoice risk learned model auc={auc:.4f} pr_auc={pr_auc:.4f}")


if __name__ == "__main__":
    main()
