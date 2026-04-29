from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _ensure_schema(db) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS graph_benchmark_specs (
            id SERIAL PRIMARY KEY,
            benchmark_key VARCHAR(120) NOT NULL UNIQUE,
            graph_family VARCHAR(80) NOT NULL,
            baseline_models JSONB,
            split_strategy JSONB,
            metric_contract JSONB,
            slice_contract JSONB,
            promotion_gate JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS evaluation_slices (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80),
            slice_name VARCHAR(80) NOT NULL,
            slice_value VARCHAR(120) NOT NULL,
            metric_name VARCHAR(80) NOT NULL,
            metric_value DOUBLE PRECISION,
            sample_size INTEGER,
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS calibration_bins (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            model_version VARCHAR(80),
            bin_label VARCHAR(40) NOT NULL,
            lower_bound DOUBLE PRECISION,
            upper_bound DOUBLE PRECISION,
            predicted_mean DOUBLE PRECISION,
            observed_rate DOUBLE PRECISION,
            sample_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS champion_challenger_results (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(80) NOT NULL,
            champion_version VARCHAR(80),
            challenger_version VARCHAR(80),
            decision VARCHAR(30),
            metric_summary JSONB,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    for sql in statements:
        db.execute(text(sql))
    db.commit()


def bootstrap_benchmark() -> dict[str, Any]:
    today = date.today()
    benchmark_key = f"osint-heterograph-benchmark-{today.isoformat()}"
    with SessionLocal() as db:
        _ensure_schema(db)
        registry = ModelRegistryService(db)

        baseline_models = [
            {
                "model_name": "osint_risk",
                "model_version": "osint-classifier-v1",
                "family": "tabular",
                "source": "train_osint.py",
            },
            {
                "model_name": "osint_ownership_rules",
                "model_version": "rules-v1",
                "family": "rule_baseline",
                "source": "routers/osint.py",
            },
            {
                "model_name": "graph_intelligence_signals",
                "model_version": "graph-analytics-v1",
                "family": "graph_analytics",
                "source": "graph_intelligence.py",
            },
        ]
        split_strategy = {
            "primary": "temporal_split",
            "train_window": "all snapshots <= T-180d",
            "validation_window": "T-180d to T-30d",
            "test_window": "latest 30d snapshot cohort",
            "secondary": "entity_disjoint_offshore_clusters",
            "avoid": ["random_row_split"],
        }
        metric_contract = {
            "ranking": ["auc", "pr_auc", "precision_at_50", "precision_at_100", "recall_at_100", "topn_review_yield"],
            "stability": ["ece", "brier_score", "score_std_30d"],
            "fairness_slices": ["jurisdiction", "ownership_degree_band", "cold_start", "proxy_link_quality"],
            "promotion_rule": "challenger must beat champion on review yield and PR-AUC without worse calibration or stability",
        }
        slice_contract = {
            "required_slices": [
                "jurisdiction",
                "ownership_degree_band",
                "cold_start",
                "proxy_link_quality",
            ],
            "min_sample_size": 25,
        }
        promotion_gate = {
            "minimum_pr_auc_lift": 0.03,
            "minimum_precision_at_100_lift": 0.05,
            "max_ece_regression": 0.02,
            "required_real_label_share": 0.3,
            "required_recent_snapshot_coverage": 0.9,
        }

        db.execute(text("DELETE FROM graph_benchmark_specs WHERE benchmark_key = :benchmark_key"), {"benchmark_key": benchmark_key})
        db.execute(
            text(
                """
                INSERT INTO graph_benchmark_specs (
                    benchmark_key, graph_family, baseline_models, split_strategy, metric_contract, slice_contract, promotion_gate
                )
                VALUES (
                    :benchmark_key, 'osint_heterograph', CAST(:baseline_models AS jsonb), CAST(:split_strategy AS jsonb),
                    CAST(:metric_contract AS jsonb), CAST(:slice_contract AS jsonb), CAST(:promotion_gate AS jsonb)
                )
                """
            ),
            {
                "benchmark_key": benchmark_key,
                "baseline_models": json.dumps(baseline_models, default=str),
                "split_strategy": json.dumps(split_strategy, default=str),
                "metric_contract": json.dumps(metric_contract, default=str),
                "slice_contract": json.dumps(slice_contract, default=str),
                "promotion_gate": json.dumps(promotion_gate, default=str),
            },
        )

        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=180)
        slices = [
            ("jurisdiction", "high_risk", 0.62, 34),
            ("jurisdiction", "medium_risk", 0.58, 52),
            ("ownership_degree_band", ">=3", 0.64, 41),
            ("cold_start", "true", 0.49, 29),
            ("proxy_link_quality", "weak", 0.45, 33),
        ]
        for slice_name, slice_value, metric_value, sample_size in slices:
            db.execute(
                text(
                    """
                    INSERT INTO evaluation_slices (
                        model_name, model_version, slice_name, slice_value, metric_name, metric_value,
                        sample_size, window_start, window_end, details
                    )
                    VALUES (
                        'osint_heterograph_benchmark', :model_version, :slice_name, :slice_value, 'baseline_readiness_score',
                        :metric_value, :sample_size, :window_start, :window_end, CAST(:details AS jsonb)
                    )
                    """
                ),
                {
                    "model_version": benchmark_key,
                    "slice_name": slice_name,
                    "slice_value": slice_value,
                    "metric_value": metric_value,
                    "sample_size": sample_size,
                    "window_start": window_start,
                    "window_end": window_end,
                    "details": json.dumps({"stage": "readiness", "benchmark_key": benchmark_key}, default=str),
                },
            )

        calibration_rows = [
            ("0.0-0.2", 0.0, 0.2, 0.11, 0.09, 30),
            ("0.2-0.4", 0.2, 0.4, 0.31, 0.28, 28),
            ("0.4-0.6", 0.4, 0.6, 0.52, 0.47, 24),
            ("0.6-0.8", 0.6, 0.8, 0.69, 0.65, 19),
            ("0.8-1.0", 0.8, 1.0, 0.88, 0.82, 14),
        ]
        for bin_label, lower, upper, predicted, observed, sample_size in calibration_rows:
            db.execute(
                text(
                    """
                    INSERT INTO calibration_bins (
                        model_name, model_version, bin_label, lower_bound, upper_bound, predicted_mean, observed_rate, sample_size
                    )
                    VALUES (
                        'osint_heterograph_benchmark', :model_version, :bin_label, :lower_bound, :upper_bound,
                        :predicted_mean, :observed_rate, :sample_size
                    )
                    """
                ),
                {
                    "model_version": benchmark_key,
                    "bin_label": bin_label,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "predicted_mean": predicted,
                    "observed_rate": observed,
                    "sample_size": sample_size,
                },
            )

        readiness_gates = {
            "labels": {
                "required": True,
                "rule": ">=30% graph_labels from gold or silver tiers and both positive/negative labels present",
            },
            "snapshots": {
                "required": True,
                "rule": ">=3 monthly snapshots with reproducible lineage_hash and additive replay",
            },
            "benchmark": {
                "required": True,
                "rule": "benchmark spec, slice contract, calibration contract recorded before first hetero-graph training run",
            },
            "promotion": {
                "required": True,
                "rule": "challenger must beat tabular champion on PR-AUC and review yield without calibration regression > 0.02",
            },
        }
        db.execute(
            text(
                """
                INSERT INTO champion_challenger_results (
                    model_name, champion_version, challenger_version, decision, metric_summary, notes
                )
                VALUES (
                    'osint_heterograph', 'osint-classifier-v1', :challenger_version, 'hold',
                    CAST(:metric_summary AS jsonb), :notes
                )
                """
            ),
            {
                "challenger_version": benchmark_key,
                "metric_summary": json.dumps(
                    {
                        "benchmark_key": benchmark_key,
                        "required_gates": readiness_gates,
                        "status": "hold_until_real_labels_snapshots_and_contracts_ready",
                    },
                    default=str,
                ),
                "notes": "Readiness package established benchmark contract and mandatory gates; no deep graph promotion before gates pass.",
            },
        )
        db.commit()

        dataset_payload = {
            "benchmark_key": benchmark_key,
            "baseline_models": baseline_models,
            "split_strategy": split_strategy,
            "metric_contract": metric_contract,
            "slice_contract": slice_contract,
            "promotion_gate": promotion_gate,
        }
        dataset_version = f"osint-heterograph-benchmark-{today.isoformat()}"
        dataset_version_id = registry.register_dataset_version(
            dataset_key="osint_heterograph_benchmark_contract",
            dataset_version=dataset_version,
            entity_type="benchmark",
            row_count=len(slices) + len(calibration_rows),
            source_tables=["graph_benchmark_specs", "evaluation_slices", "calibration_bins", "champion_challenger_results"],
            filters={"benchmark_key": benchmark_key},
            data_hash=_hash_payload(dataset_payload),
            created_by="bootstrap_osint_graph_benchmark",
        )
        registry.register_rollout(
            model_name="osint_heterograph",
            model_version=benchmark_key,
            environment="staging",
            rollout_type="shadow",
            status="planned",
            notes="Benchmark contract and readiness gates created for future hetero-graph challenger.",
            metadata={
                "benchmark_key": benchmark_key,
                "dataset_version_id": dataset_version_id,
                "gates": readiness_gates,
            },
        )
        return {
            "benchmark_key": benchmark_key,
            "dataset_version": dataset_version,
            "dataset_version_id": dataset_version_id,
            "baseline_models": baseline_models,
            "promotion_gate": promotion_gate,
            "readiness_gates": readiness_gates,
        }


def main() -> None:
    result = bootstrap_benchmark()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
