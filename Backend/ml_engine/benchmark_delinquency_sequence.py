from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from ml_engine.delinquency_model import DelinquencyFeatureEngineer, DelinquencyPipeline
from ml_engine.model_registry import ModelRegistryService

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _build_label(df: pd.DataFrame) -> int:
    late = 0
    for _, row in df.tail(4).iterrows():
        due = pd.to_datetime(row.get("due_date"), errors="coerce")
        actual = pd.to_datetime(row.get("actual_payment_date"), errors="coerce")
        status = str(row.get("status") or "")
        if pd.notna(actual) and pd.notna(due) and actual > due:
            late = 1
        elif status == "overdue":
            late = 1
    return late


def main() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for delinquency sequence benchmark.") from exc

    class SequenceClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 24):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return self.fc(h[-1]).squeeze(-1)

    from sklearn.metrics import average_precision_score, roc_auc_score

    max_len = 24
    with SessionLocal() as db:
        company_rows = db.execute(
            text(
                """
                SELECT DISTINCT tp.tax_code
                FROM tax_payments tp
                ORDER BY tp.tax_code
                LIMIT 6000
                """
            )
        ).fetchall()
        companies = [str(r[0]) for r in company_rows]
        seqs = []
        labels = []
        baseline_probs = []
        pipeline = DelinquencyPipeline()
        pipeline.load_models()
        engineer = DelinquencyFeatureEngineer()

        for tax_code in companies:
            payments = db.execute(
                text(
                    """
                    SELECT due_date, actual_payment_date, amount_due, amount_paid, penalty_amount, status
                    FROM tax_payments
                    WHERE tax_code = :tax_code
                    ORDER BY due_date
                    """
                ),
                {"tax_code": tax_code},
            ).mappings().all()
            if len(payments) < 4:
                continue
            df = pd.DataFrame(payments)
            label = _build_label(df)
            labels.append(label)

            rows = []
            for _, row in df.tail(max_len).iterrows():
                due = pd.to_datetime(row.get("due_date"), errors="coerce")
                actual = pd.to_datetime(row.get("actual_payment_date"), errors="coerce")
                overdue_days = 0.0
                if pd.notna(actual) and pd.notna(due):
                    overdue_days = max(0.0, float((actual - due).days))
                rows.append(
                    [
                        float(np.log1p(float(row.get("amount_due") or 0.0))),
                        float(np.log1p(float(row.get("amount_paid") or 0.0))),
                        float(np.log1p(float(row.get("penalty_amount") or 0.0))),
                        overdue_days / 120.0,
                        1.0 if str(row.get("status") or "") == "overdue" else 0.0,
                    ]
                )
            if len(rows) < max_len:
                rows = [[0.0] * 5 for _ in range(max_len - len(rows))] + rows
            seqs.append(rows[-max_len:])

            features = engineer.compute_features(df, None, None)
            base_prob = min(1.0, features.get("late_ratio_1yr", 0.0) * 0.5 + min(features.get("unpaid_count", 0) * 0.1, 0.3))
            if pipeline._loaded and pipeline.model is not None:
                pred = pipeline.predict_single(df, None, None)
                base_prob = float(pred.get("prob_90d") or base_prob)
            baseline_probs.append(base_prob)

        X = torch.tensor(np.asarray(seqs, dtype=np.float32))
        y = torch.tensor(np.asarray(labels, dtype=np.float32))
        if len(np.unique(np.asarray(labels))) < 2:
            raise RuntimeError("Sequence benchmark requires both classes.")

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        baseline_test = np.asarray(baseline_probs[split:], dtype=float)

        model = SequenceClassifier(input_dim=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(8):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test)).cpu().numpy()
        y_true = y_test.cpu().numpy()
        dl_auc = float(roc_auc_score(y_true, probs))
        dl_pr = float(average_precision_score(y_true, probs))
        baseline_auc = float(roc_auc_score(y_true, baseline_test))
        baseline_pr = float(average_precision_score(y_true, baseline_test))

        benchmark = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_version": "delinquency-sequence-benchmark-v1",
            "baseline_version": pipeline.config.get("model_version") or os.getenv("DELINQUENCY_MODEL_VERSION", "delinquency-temporal-v1"),
            "metrics": {
                "dl_auc": round(dl_auc, 4),
                "dl_pr_auc": round(dl_pr, 4),
                "baseline_auc": round(baseline_auc, 4),
                "baseline_pr_auc": round(baseline_pr, 4),
                "auc_delta": round(dl_auc - baseline_auc, 4),
                "pr_auc_delta": round(dl_pr - baseline_pr, 4),
            },
            "dataset": {"samples": int(len(X)), "train_size": int(len(X_train)), "test_size": int(len(X_test)), "sequence_length": max_len},
            "decision": "promote_challenger" if dl_pr >= baseline_pr and dl_auc >= baseline_auc else "keep_baseline",
        }
        with open(MODEL_DIR / "delinquency_sequence_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=2)

        registry = ModelRegistryService(db)
        registry.upsert_registry_entry(
            model_name="delinquency_sequence_benchmark",
            model_version="delinquency-sequence-benchmark-v1",
            artifact_path=str(MODEL_DIR / "delinquency_sequence_benchmark.json"),
            metrics=benchmark["metrics"],
            gates={"overall_pass": benchmark["decision"] == "promote_challenger"},
            status="staging",
        )
        db.execute(text("DELETE FROM champion_challenger_results WHERE model_name = 'delinquency_sequence'"))
        db.execute(
            text(
                """
                INSERT INTO champion_challenger_results
                (model_name, champion_version, challenger_version, decision, metric_summary, notes)
                VALUES
                ('delinquency_sequence', :champion_version, :challenger_version, :decision, CAST(:metric_summary AS jsonb), :notes)
                """
            ),
            {
                "champion_version": benchmark["baseline_version"],
                "challenger_version": benchmark["model_version"],
                "decision": benchmark["decision"],
                "metric_summary": json.dumps(benchmark["metrics"]),
                "notes": "Phase 90 sequence DL benchmark against current delinquency baseline.",
            },
        )
        db.commit()
        print(f"[OK] delinquency sequence benchmark auc_delta={benchmark['metrics']['auc_delta']:.4f} pr_delta={benchmark['metrics']['pr_auc_delta']:.4f}")


if __name__ == "__main__":
    main()
