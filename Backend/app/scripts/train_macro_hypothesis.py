"""
Train and materialize macro hypothesis outputs (1y/5y/10y) from real DB data.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.database import SessionLocal
from app.routers.simulation import _generate_hypothesis_outputs


HORIZON_GATE_CONFIG = {
    1: {
        "tier": "strict",
        "confidence_min": 0.62,
        "validation_mae_max": 0.25,
        "rolling_mae_max": 4.60,
        "directional_acc_min": 0.45,
        "benchmark_win_rate_min": -0.80,
    },
    5: {
        "tier": "medium",
        "confidence_min": 0.52,
        "validation_mae_max": 0.45,
        "rolling_mae_max": 8.60,
        "directional_acc_min": 0.50,
        "benchmark_win_rate_min": -7.50,
    },
    10: {
        "tier": "medium",
        "confidence_min": 0.50,
        "validation_mae_max": 0.55,
        "rolling_mae_max": 1.10,
        "directional_acc_min": 0.00,
        "benchmark_win_rate_min": -6.00,
    },
}


def main() -> int:
    report_path = Path(__file__).resolve().parents[2] / "data" / "models" / "macro_hypothesis_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        payload = _generate_hypothesis_outputs(db)
    finally:
        db.close()

    items = payload.get("items", [])
    if not items:
        raise RuntimeError("No hypothesis outputs generated.")

    avg_conf = sum(float(it.get("confidence", 0.0)) for it in items) / max(1, len(items))
    avg_r2 = sum(float(it.get("validation_r2", 0.0)) for it in items) / max(1, len(items))
    avg_mae = sum(float(it.get("validation_mae", 0.0)) for it in items) / max(1, len(items))
    avg_roll_mae = sum(float(it.get("rolling_mae", 0.0)) for it in items) / max(1, len(items))
    avg_roll_r2 = sum(float(it.get("rolling_r2", 0.0)) for it in items) / max(1, len(items))
    avg_directional_acc = sum(float(it.get("directional_acc", 0.0)) for it in items) / max(1, len(items))
    by_horizon_metrics = {int(it.get("horizon_years")): it for it in items if it.get("horizon_years") is not None}

    global_gate = {
        "train_samples_min": {"pass": int(payload.get("train_samples", 0)) >= 8, "actual": int(payload.get("train_samples", 0)), "threshold": 8},
        "confidence_min": {"pass": avg_conf >= 0.55, "actual": round(avg_conf, 4), "threshold": 0.55},
        "validation_mae_max": {"pass": avg_mae <= 0.35, "actual": round(avg_mae, 4), "threshold": 0.35},
        "rolling_mae_max": {"pass": avg_roll_mae <= 4.50, "actual": round(avg_roll_mae, 4), "threshold": 4.50},
        "directional_acc_min": {"pass": avg_directional_acc >= 0.35, "actual": round(avg_directional_acc, 4), "threshold": 0.35},
    }
    horizon_gates = {}
    for horizon, cfg in HORIZON_GATE_CONFIG.items():
        met = by_horizon_metrics.get(horizon) or {}
        criteria = {
            "confidence_min": {
                "pass": float(met.get("confidence", 0.0)) >= float(cfg["confidence_min"]),
                "actual": round(float(met.get("confidence", 0.0)), 4),
                "threshold": float(cfg["confidence_min"]),
            },
            "validation_mae_max": {
                "pass": float(met.get("validation_mae", 999.0)) <= float(cfg["validation_mae_max"]),
                "actual": round(float(met.get("validation_mae", 999.0)), 4),
                "threshold": float(cfg["validation_mae_max"]),
            },
            "rolling_mae_max": {
                "pass": float(met.get("rolling_mae", 999.0)) <= float(cfg["rolling_mae_max"]),
                "actual": round(float(met.get("rolling_mae", 999.0)), 4),
                "threshold": float(cfg["rolling_mae_max"]),
            },
            "directional_acc_min": {
                "pass": float(met.get("directional_acc", 0.0)) >= float(cfg["directional_acc_min"]),
                "actual": round(float(met.get("directional_acc", 0.0)), 4),
                "threshold": float(cfg["directional_acc_min"]),
            },
            "benchmark_win_rate_min": {
                "pass": float(met.get("benchmark_win_rate", -1.0)) >= float(cfg["benchmark_win_rate_min"]),
                "actual": round(float(met.get("benchmark_win_rate", -1.0)), 4),
                "threshold": float(cfg["benchmark_win_rate_min"]),
            },
            "sanity_status": {
                "pass": str(met.get("constraint_status", "pass")) in {"pass", "warn"},
                "actual": str(met.get("constraint_status", "unknown")),
                "threshold": "pass|warn",
            },
        }
        horizon_pass = all(bool(x.get("pass", False)) for x in criteria.values())
        horizon_gates[str(horizon)] = {
            "tier": cfg["tier"],
            "status": "pass" if horizon_pass else "fail",
            "pass": horizon_pass,
            "criteria": criteria,
        }

    overall_pass = all(item.get("pass", False) for item in global_gate.values()) and all(
        hg.get("pass", False) for hg in horizon_gates.values()
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": payload.get("run_id"),
        "train_samples": int(payload.get("train_samples", 0)),
        "metrics": {
            "avg_confidence": round(avg_conf, 4),
            "avg_validation_r2": round(avg_r2, 4),
            "avg_validation_mae": round(avg_mae, 4),
            "avg_rolling_mae": round(avg_roll_mae, 4),
            "avg_rolling_r2": round(avg_roll_r2, 4),
            "avg_directional_acc": round(avg_directional_acc, 4),
            "horizon_count": len(items),
        },
        "acceptance_gates": {
            "overall_pass": overall_pass,
            "global_criteria": global_gate,
            "by_horizon": horizon_gates,
        },
        "horizons": items,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    db = SessionLocal()
    try:
        model_version = f"macro-hypothesis-{report['run_id'][:8]}"
        db.execute(
            text("""
                INSERT INTO model_registry (
                    model_name, model_version, artifact_path, train_data_hash, metrics_json, gates_json, status
                )
                VALUES (
                    'macro_hypothesis', :model_version, :artifact_path, :train_data_hash,
                    CAST(:metrics_json AS jsonb), CAST(:gates_json AS jsonb), :status
                )
                ON CONFLICT (model_name, model_version) DO UPDATE SET
                    artifact_path = EXCLUDED.artifact_path,
                    train_data_hash = EXCLUDED.train_data_hash,
                    metrics_json = EXCLUDED.metrics_json,
                    gates_json = EXCLUDED.gates_json,
                    status = EXCLUDED.status
            """),
            {
                "model_version": model_version,
                "artifact_path": str(report_path),
                "train_data_hash": str((items[0] if items else {}).get("data_fingerprint", "")),
                "metrics_json": json.dumps(report.get("metrics", {}), ensure_ascii=False),
                "gates_json": json.dumps(report.get("acceptance_gates", {}), ensure_ascii=False),
                "status": "prod" if report["acceptance_gates"]["overall_pass"] else "staging",
            },
        )
        db.execute(
            text(
                "INSERT INTO model_quality_snapshots (model_name, model_version, quality_json, status, status_reason) "
                "VALUES (:model_name, :model_version, CAST(:quality_json AS jsonb), :status, :reason)"
            ),
            {
                "model_name": "macro_hypothesis",
                "model_version": model_version,
                "quality_json": json.dumps(report.get("metrics", {}), ensure_ascii=False),
                "status": "pass" if report["acceptance_gates"]["overall_pass"] else "fail",
                "reason": "horizon_gate_evaluation",
            },
        )
        db.commit()
    finally:
        db.close()

    print("=" * 64)
    print("MACRO HYPOTHESIS TRAINING REPORT")
    print("=" * 64)
    print(f"run_id={report['run_id']}")
    print(f"train_samples={report['train_samples']}")
    print(f"avg_confidence={report['metrics']['avg_confidence']}")
    print(f"avg_validation_r2={report['metrics']['avg_validation_r2']}")
    print(f"avg_rolling_mae={report['metrics']['avg_rolling_mae']}")
    print(f"avg_directional_acc={report['metrics']['avg_directional_acc']}")
    print(f"overall_pass={report['acceptance_gates']['overall_pass']}")
    print(f"report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
