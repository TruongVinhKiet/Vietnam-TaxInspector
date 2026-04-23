"""
run_specialized_go_no_go_review.py

Phase 5 go/no-go evaluator for specialized tracks.

This script reads:
- audit_value_quality_report.json
- vat_refund_quality_report.json
- specialized_pilot_report.json
- (optional) split-trigger status snapshot from monitoring helper

It then evaluates hard/soft gates and writes a decision report:
    data/models/specialized_go_no_go_report.json

It also appends compact run records into:
    data/models/specialized_go_no_go_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


MODELS_DIR = BACKEND_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_AUDIT_QUALITY_FILE = MODELS_DIR / "audit_value_quality_report.json"
DEFAULT_VAT_QUALITY_FILE = MODELS_DIR / "vat_refund_quality_report.json"
DEFAULT_PILOT_FILE = MODELS_DIR / "specialized_pilot_report.json"
DEFAULT_OUTPUT_FILE = MODELS_DIR / "specialized_go_no_go_report.json"
DEFAULT_HISTORY_FILE = MODELS_DIR / "specialized_go_no_go_history.jsonl"
MIN_REQUIRED_TRAINING_SAMPLES = max(10_000, int(os.environ.get("TRAINING_MIN_REQUIRED_SAMPLES", "10000")))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except Exception:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except Exception:
        return []
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _quality_gate_result(payload: dict[str, Any] | None, track_name: str, min_training_samples: int) -> dict[str, Any]:
    if not payload:
        return {
            "name": f"quality_{track_name}",
            "pass": False,
            "status": "missing_report",
            "message": f"Missing quality report for {track_name}.",
            "actual": None,
        }

    gates = payload.get("acceptance_gates") or {}
    quality_pass = bool(gates.get("overall_pass", False))
    dataset = payload.get("dataset") or {}
    total_size = _to_int(dataset.get("total_size"), 0)
    sample_pass = total_size >= max(1, int(min_training_samples))
    overall_pass = bool(quality_pass and sample_pass)

    failed_items: list[str] = []
    if not quality_pass:
        failed_items.append("quality_gate_failed")
    if not sample_pass:
        failed_items.append(f"samples<{min_training_samples}")

    return {
        "name": f"quality_{track_name}",
        "pass": overall_pass,
        "status": "pass" if overall_pass else "fail",
        "message": "Quality acceptance gates passed." if overall_pass else (", ".join(failed_items) or "Quality acceptance gates failed."),
        "actual": {
            "overall_pass": overall_pass,
            "quality_gate_pass": quality_pass,
            "dataset_total_size": total_size,
            "criteria": gates.get("criteria"),
        },
        "thresholds": {
            "min_training_samples": int(min_training_samples),
        },
    }


def _pilot_gate_result(
    pilot_payload: dict[str, Any] | None,
    *,
    track_name: str,
    min_samples: int,
    min_f1_delta: float,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    if not pilot_payload:
        return {
            "name": f"pilot_{track_name}",
            "pass": False,
            "status": "missing_report",
            "message": f"Missing pilot report for {track_name}.",
            "actual": None,
        }

    tracks = pilot_payload.get("tracks") or {}
    track = tracks.get(track_name)
    if not isinstance(track, dict):
        return {
            "name": f"pilot_{track_name}",
            "pass": False,
            "status": "track_missing",
            "message": f"Track {track_name} missing in pilot report.",
            "actual": None,
        }

    samples = _to_int(track.get("samples_evaluated"), 0)
    delta = track.get("delta_model_minus_heuristic") or {}
    model = track.get("model") or {}
    heuristic = track.get("heuristic") or {}

    f1_delta = _to_float(delta.get("f1_delta"), 0.0)
    model_acc = _to_float(model.get("accuracy"), 0.0)
    heuristic_acc = _to_float(heuristic.get("accuracy"), 0.0)
    accuracy_drop = max(0.0, heuristic_acc - model_acc)

    samples_pass = samples >= max(1, int(min_samples))
    f1_pass = f1_delta >= float(min_f1_delta)
    acc_pass = accuracy_drop <= float(max_accuracy_drop)
    gate_pass = bool(samples_pass and f1_pass and acc_pass)

    failed_items: list[str] = []
    if not samples_pass:
        failed_items.append(f"samples<{min_samples}")
    if not f1_pass:
        failed_items.append(f"f1_delta<{min_f1_delta}")
    if not acc_pass:
        failed_items.append(f"accuracy_drop>{max_accuracy_drop}")

    return {
        "name": f"pilot_{track_name}",
        "pass": gate_pass,
        "status": "pass" if gate_pass else "fail",
        "message": "Pilot gate passed." if gate_pass else (", ".join(failed_items) or "Pilot gate failed."),
        "actual": {
            "samples_evaluated": samples,
            "f1_delta": round(f1_delta, 6),
            "accuracy_drop": round(accuracy_drop, 6),
            "model_accuracy": round(model_acc, 6),
            "heuristic_accuracy": round(heuristic_acc, 6),
        },
        "thresholds": {
            "min_samples": int(min_samples),
            "min_f1_delta": float(min_f1_delta),
            "max_accuracy_drop": float(max_accuracy_drop),
        },
    }


def _get_split_trigger_snapshot() -> dict[str, Any] | None:
    try:
        from app.routers import monitoring

        payload = monitoring.get_split_trigger_status_snapshot(
            persist_snapshot=False,
            snapshot_source="phase5_go_no_go",
        )
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _split_gate_result(split_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not split_payload:
        return {
            "name": "split_trigger",
            "pass": False,
            "status": "unavailable",
            "message": "Split-trigger snapshot unavailable.",
            "actual": None,
        }

    schema_ready = bool(split_payload.get("schema_ready", False))
    ready = bool(split_payload.get("ready", False))
    readiness_score = _to_float(split_payload.get("readiness_score"), 0.0)
    gate_pass = bool(schema_ready and ready)

    return {
        "name": "split_trigger",
        "pass": gate_pass,
        "status": "pass" if gate_pass else "not_ready",
        "message": "Split-trigger ready." if gate_pass else str(split_payload.get("reason") or "Split-trigger not ready."),
        "actual": {
            "schema_ready": schema_ready,
            "ready": ready,
            "readiness_score": round(readiness_score, 3),
            "critical_tracks": split_payload.get("critical_tracks") or [],
            "totals": split_payload.get("totals") or {},
        },
    }


def _count_consecutive_hard_pass(history_rows: list[dict[str, Any]], current_hard_pass: bool) -> int:
    sequence = [bool(row.get("hard_gates_pass", False)) for row in history_rows]
    sequence.append(bool(current_hard_pass))

    streak = 0
    for passed in reversed(sequence):
        if passed:
            streak += 1
        else:
            break
    return streak


def _stability_gate_result(
    history_rows: list[dict[str, Any]],
    *,
    current_hard_pass: bool,
    min_consecutive_hard_pass: int,
) -> dict[str, Any]:
    streak = _count_consecutive_hard_pass(history_rows, current_hard_pass)
    gate_pass = streak >= int(min_consecutive_hard_pass)
    return {
        "name": "stability",
        "pass": gate_pass,
        "status": "pass" if gate_pass else "warming_up",
        "message": "Stability gate passed." if gate_pass else "Need more consecutive hard-pass runs.",
        "actual": {
            "consecutive_hard_pass_runs": streak,
        },
        "thresholds": {
            "min_consecutive_hard_pass_runs": int(min_consecutive_hard_pass),
        },
    }


def _decision_from_gates(
    *,
    hard_gates_pass: bool,
    split_gate_pass: bool,
    stability_gate_pass: bool,
) -> dict[str, Any]:
    if not hard_gates_pass:
        return {
            "status": "no_go_tune_models_or_data",
            "go_live_phase_d": False,
            "message": "Hard gates failed. Keep integrated-first and tune model/data.",
            "recommended_actions": [
                "Review failing quality/pilot gates.",
                "Tune thresholds, features, and data labeling quality.",
                "Re-run training and pilot before next go/no-go review.",
            ],
        }

    if hard_gates_pass and split_gate_pass and stability_gate_pass:
        return {
            "status": "go_phase_d_candidate",
            "go_live_phase_d": True,
            "message": "All hard/soft gates passed. Candidate to open Phase D split workbench.",
            "recommended_actions": [
                "Open Phase D implementation scope for split workbench.",
                "Keep pilot monitoring cadence weekly during rollout.",
                "Maintain rollback path to integrated-first until stable in production.",
            ],
        }

    return {
        "status": "conditional_go_continue_integrated_first",
        "go_live_phase_d": False,
        "message": "Hard gates passed but soft gates not stable yet.",
        "recommended_actions": [
            "Continue integrated-first execution.",
            "Collect additional pilot cycles for stability confirmation.",
            "Re-evaluate split-trigger readiness and pass-rate trend on next review.",
        ],
    }


def build_go_no_go_report(
    *,
    audit_quality_payload: dict[str, Any] | None,
    vat_quality_payload: dict[str, Any] | None,
    pilot_payload: dict[str, Any] | None,
    split_payload: dict[str, Any] | None,
    history_rows: list[dict[str, Any]],
    min_pilot_samples: int,
    audit_min_f1_delta: float,
    vat_min_f1_delta: float,
    max_accuracy_drop: float,
    min_training_samples: int,
    min_consecutive_hard_pass_runs: int,
) -> dict[str, Any]:
    quality_audit_gate = _quality_gate_result(audit_quality_payload, "audit_value", min_training_samples)
    quality_vat_gate = _quality_gate_result(vat_quality_payload, "vat_refund", min_training_samples)

    pilot_audit_gate = _pilot_gate_result(
        pilot_payload,
        track_name="audit_value",
        min_samples=min_pilot_samples,
        min_f1_delta=audit_min_f1_delta,
        max_accuracy_drop=max_accuracy_drop,
    )
    pilot_vat_gate = _pilot_gate_result(
        pilot_payload,
        track_name="vat_refund",
        min_samples=min_pilot_samples,
        min_f1_delta=vat_min_f1_delta,
        max_accuracy_drop=max_accuracy_drop,
    )

    split_gate = _split_gate_result(split_payload)

    hard_gates = [quality_audit_gate, quality_vat_gate, pilot_audit_gate, pilot_vat_gate]
    hard_gates_pass = all(bool(g.get("pass", False)) for g in hard_gates)

    stability_gate = _stability_gate_result(
        history_rows,
        current_hard_pass=hard_gates_pass,
        min_consecutive_hard_pass=min_consecutive_hard_pass_runs,
    )

    decision = _decision_from_gates(
        hard_gates_pass=hard_gates_pass,
        split_gate_pass=bool(split_gate.get("pass", False)),
        stability_gate_pass=bool(stability_gate.get("pass", False)),
    )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "thresholds": {
            "min_pilot_samples": int(min_pilot_samples),
            "min_training_samples": int(min_training_samples),
            "audit_min_f1_delta": float(audit_min_f1_delta),
            "vat_min_f1_delta": float(vat_min_f1_delta),
            "max_accuracy_drop": float(max_accuracy_drop),
            "min_consecutive_hard_pass_runs": int(min_consecutive_hard_pass_runs),
        },
        "gates": {
            "quality_audit_value": quality_audit_gate,
            "quality_vat_refund": quality_vat_gate,
            "pilot_audit_value": pilot_audit_gate,
            "pilot_vat_refund": pilot_vat_gate,
            "split_trigger": split_gate,
            "stability": stability_gate,
        },
        "summary": {
            "hard_gates_pass": hard_gates_pass,
            "split_gate_pass": bool(split_gate.get("pass", False)),
            "stability_gate_pass": bool(stability_gate.get("pass", False)),
        },
        "decision": decision,
    }


def _build_history_row(report: dict[str, Any]) -> dict[str, Any]:
    gates = report.get("gates") or {}
    pilot_audit = (gates.get("pilot_audit_value") or {}).get("actual") or {}
    pilot_vat = (gates.get("pilot_vat_refund") or {}).get("actual") or {}
    summary = report.get("summary") or {}
    decision = report.get("decision") or {}

    return {
        "generated_at": report.get("generated_at"),
        "decision_status": decision.get("status"),
        "go_live_phase_d": bool(decision.get("go_live_phase_d", False)),
        "hard_gates_pass": bool(summary.get("hard_gates_pass", False)),
        "split_gate_pass": bool(summary.get("split_gate_pass", False)),
        "stability_gate_pass": bool(summary.get("stability_gate_pass", False)),
        "audit_f1_delta": pilot_audit.get("f1_delta"),
        "audit_samples": pilot_audit.get("samples_evaluated"),
        "vat_f1_delta": pilot_vat.get("f1_delta"),
        "vat_samples": pilot_vat.get("samples_evaluated"),
    }


def _print_summary(report: dict[str, Any], output_path: Path) -> None:
    summary = report.get("summary") or {}
    decision = report.get("decision") or {}
    gates = report.get("gates") or {}
    pilot_audit = (gates.get("pilot_audit_value") or {}).get("actual") or {}
    pilot_vat = (gates.get("pilot_vat_refund") or {}).get("actual") or {}

    print("[GO-NO-GO] report generated")
    print(f"[GO-NO-GO] output={output_path}")
    print(f"[GO-NO-GO] hard_gates_pass={summary.get('hard_gates_pass')}")
    print(f"[GO-NO-GO] split_gate_pass={summary.get('split_gate_pass')}")
    print(f"[GO-NO-GO] stability_gate_pass={summary.get('stability_gate_pass')}")
    print(
        "[GO-NO-GO] audit: samples={} f1_delta={} | vat: samples={} f1_delta={}".format(
            pilot_audit.get("samples_evaluated"),
            pilot_audit.get("f1_delta"),
            pilot_vat.get("samples_evaluated"),
            pilot_vat.get("f1_delta"),
        )
    )
    print(f"[GO-NO-GO] decision={decision.get('status')} go_live_phase_d={decision.get('go_live_phase_d')}")
    print(f"[GO-NO-GO] message={decision.get('message')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate specialized go/no-go decision gates")
    parser.add_argument("--audit-quality-file", type=str, default=str(DEFAULT_AUDIT_QUALITY_FILE))
    parser.add_argument("--vat-quality-file", type=str, default=str(DEFAULT_VAT_QUALITY_FILE))
    parser.add_argument("--pilot-file", type=str, default=str(DEFAULT_PILOT_FILE))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--history-file", type=str, default=str(DEFAULT_HISTORY_FILE))
    parser.add_argument("--min-pilot-samples", type=int, default=200)
    parser.add_argument("--min-training-samples", type=int, default=MIN_REQUIRED_TRAINING_SAMPLES)
    parser.add_argument("--audit-min-f1-delta", type=float, default=0.05)
    parser.add_argument("--vat-min-f1-delta", type=float, default=-0.05)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--min-consecutive-hard-pass-runs", type=int, default=2)
    args = parser.parse_args()

    if args.min_pilot_samples < 1:
        print("[ERROR] --min-pilot-samples must be >= 1")
        return 1
    if args.min_training_samples < MIN_REQUIRED_TRAINING_SAMPLES:
        print(f"[ERROR] --min-training-samples must be >= {MIN_REQUIRED_TRAINING_SAMPLES}")
        return 1
    if args.max_accuracy_drop < 0:
        print("[ERROR] --max-accuracy-drop must be >= 0")
        return 1
    if args.min_consecutive_hard_pass_runs < 1:
        print("[ERROR] --min-consecutive-hard-pass-runs must be >= 1")
        return 1

    audit_quality_payload = _load_json(Path(args.audit_quality_file))
    vat_quality_payload = _load_json(Path(args.vat_quality_file))
    pilot_payload = _load_json(Path(args.pilot_file))
    split_payload = _get_split_trigger_snapshot()
    history_rows = _load_jsonl(Path(args.history_file))

    report = build_go_no_go_report(
        audit_quality_payload=audit_quality_payload,
        vat_quality_payload=vat_quality_payload,
        pilot_payload=pilot_payload,
        split_payload=split_payload,
        history_rows=history_rows,
        min_pilot_samples=int(args.min_pilot_samples),
        min_training_samples=int(args.min_training_samples),
        audit_min_f1_delta=float(args.audit_min_f1_delta),
        vat_min_f1_delta=float(args.vat_min_f1_delta),
        max_accuracy_drop=float(args.max_accuracy_drop),
        min_consecutive_hard_pass_runs=int(args.min_consecutive_hard_pass_runs),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _append_jsonl(Path(args.history_file), _build_history_row(report))
    _print_summary(report, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())