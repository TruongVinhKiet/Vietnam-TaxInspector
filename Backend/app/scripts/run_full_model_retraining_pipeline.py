"""
run_full_model_retraining_pipeline.py

Master retraining pipeline for 4 flagship models:
1) Fraud (ml_engine/train_model.py)
2) VAT GNN (app/scripts/train_gnn.py)
3) OSINT (ml_engine/train_osint.py)
4) Simulation (ml_engine/train_simulation.py)

Includes unified quality-gate evaluation and automatic retry with reseed policy.

Usage:
    cd Backend
    python app/scripts/run_full_model_retraining_pipeline.py

Dry run:
    python app/scripts/run_full_model_retraining_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REPORT_FILE = MODELS_DIR / "full_model_retraining_report.json"

FRAUD_QUALITY_FILE = MODELS_DIR / "fraud_quality_report.json"
GNN_QUALITY_FILE = MODELS_DIR / "serving_e2e_report.json"
OSINT_QUALITY_FILE = MODELS_DIR / "osint_quality_report.json"
SIM_QUALITY_FILE = MODELS_DIR / "simulation_quality_report.json"


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _run_command(command: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    command_text = " ".join(command)
    print(f"[RUN] {command_text}")

    if dry_run:
        print("[DRY-RUN] skipped execution")
        return {
            "command": command,
            "command_text": command_text,
            "exit_code": 0,
            "dry_run": True,
        }

    result = subprocess.run(command, cwd=str(BACKEND_DIR))
    return {
        "command": command,
        "command_text": command_text,
        "exit_code": int(result.returncode),
        "dry_run": False,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _collect_failed_criteria(criteria: dict[str, Any] | None) -> list[str]:
    if not isinstance(criteria, dict):
        return []

    failed = []
    for name, rule in criteria.items():
        if isinstance(rule, dict) and rule.get("pass") is False:
            failed.append(str(name))
    return failed


def _evaluate_fraud_gate() -> dict[str, Any]:
    payload = _load_json(FRAUD_QUALITY_FILE)
    if not payload:
        return {
            "track": "fraud",
            "pass": False,
            "status": "missing_report",
            "report_file": str(FRAUD_QUALITY_FILE),
            "message": "Missing fraud quality report.",
            "metrics": {},
        }

    gates = payload.get("acceptance_gates") or {}
    criteria = gates.get("criteria") or {}
    calibrated = ((payload.get("performance") or {}).get("calibrated") or {})
    passed = bool(gates.get("overall_pass", False))

    return {
        "track": "fraud",
        "pass": passed,
        "status": "pass" if passed else "fail",
        "report_file": str(FRAUD_QUALITY_FILE),
        "message": "Fraud gate passed." if passed else "Fraud gate failed.",
        "failed_criteria": _collect_failed_criteria(criteria),
        "metrics": {
            "auc_roc": calibrated.get("auc_roc"),
            "pr_auc": calibrated.get("pr_auc"),
            "brier": calibrated.get("brier"),
            "ece": calibrated.get("ece"),
        },
    }


def _evaluate_gnn_gate() -> dict[str, Any]:
    payload = _load_json(GNN_QUALITY_FILE)
    if not payload:
        return {
            "track": "vat_gnn",
            "pass": False,
            "status": "missing_report",
            "report_file": str(GNN_QUALITY_FILE),
            "message": "Missing GNN quality report.",
            "metrics": {},
        }

    gates = payload.get("acceptance_gates") or {}
    criteria = gates.get("criteria") or {}
    raw_test = payload.get("raw_test") or {}
    passed = bool(gates.get("overall_pass", False))

    return {
        "track": "vat_gnn",
        "pass": passed,
        "status": "pass" if passed else "fail",
        "report_file": str(GNN_QUALITY_FILE),
        "message": "GNN gate passed." if passed else "GNN gate failed.",
        "failed_criteria": _collect_failed_criteria(criteria),
        "metrics": {
            "node_f1": raw_test.get("node_f1"),
            "edge_f1": raw_test.get("edge_f1"),
            "node_pr_auc": raw_test.get("node_pr_auc"),
            "edge_pr_auc": raw_test.get("edge_pr_auc"),
        },
    }


def _evaluate_osint_gate(min_auc: float, min_pr_auc: float) -> dict[str, Any]:
    payload = _load_json(OSINT_QUALITY_FILE)
    if not payload:
        return {
            "track": "osint",
            "pass": False,
            "status": "missing_report",
            "report_file": str(OSINT_QUALITY_FILE),
            "message": "Missing OSINT quality report.",
            "thresholds": {"auc_min": min_auc, "pr_auc_min": min_pr_auc},
            "metrics": {},
        }

    metrics = payload.get("metrics") or {}
    auc = metrics.get("auc")
    pr_auc = metrics.get("pr_auc")

    pass_auc = isinstance(auc, (int, float)) and float(auc) >= float(min_auc)
    pass_pr = isinstance(pr_auc, (int, float)) and float(pr_auc) >= float(min_pr_auc)
    passed = bool(pass_auc and pass_pr)

    failed = []
    if not pass_auc:
        failed.append("auc")
    if not pass_pr:
        failed.append("pr_auc")

    return {
        "track": "osint",
        "pass": passed,
        "status": "pass" if passed else "fail",
        "report_file": str(OSINT_QUALITY_FILE),
        "message": "OSINT gate passed." if passed else "OSINT gate failed.",
        "failed_criteria": failed,
        "thresholds": {
            "auc_min": float(min_auc),
            "pr_auc_min": float(min_pr_auc),
        },
        "metrics": {
            "auc": auc,
            "pr_auc": pr_auc,
            "total_samples": (payload.get("dataset") or {}).get("total_samples"),
        },
    }


def _evaluate_simulation_gate(min_r2: float, max_rmse: float) -> dict[str, Any]:
    payload = _load_json(SIM_QUALITY_FILE)
    if not payload:
        return {
            "track": "simulation",
            "pass": False,
            "status": "missing_report",
            "report_file": str(SIM_QUALITY_FILE),
            "message": "Missing simulation quality report.",
            "thresholds": {"r2_min": min_r2, "rmse_max": max_rmse},
            "metrics": {},
        }

    metrics = payload.get("metrics") or {}
    r2 = metrics.get("r2")
    rmse = metrics.get("rmse")

    pass_r2 = isinstance(r2, (int, float)) and float(r2) >= float(min_r2)
    pass_rmse = isinstance(rmse, (int, float)) and float(rmse) <= float(max_rmse)
    passed = bool(pass_r2 and pass_rmse)

    failed = []
    if not pass_r2:
        failed.append("r2")
    if not pass_rmse:
        failed.append("rmse")

    return {
        "track": "simulation",
        "pass": passed,
        "status": "pass" if passed else "fail",
        "report_file": str(SIM_QUALITY_FILE),
        "message": "Simulation gate passed." if passed else "Simulation gate failed.",
        "failed_criteria": failed,
        "thresholds": {
            "r2_min": float(min_r2),
            "rmse_max": float(max_rmse),
        },
        "metrics": {
            "r2": r2,
            "rmse": rmse,
            "mae": metrics.get("mae"),
            "total_samples": (payload.get("dataset") or {}).get("total_samples"),
        },
    }


def _evaluate_all_gates(args: argparse.Namespace) -> dict[str, Any]:
    tracks = [
        _evaluate_fraud_gate(),
        _evaluate_gnn_gate(),
        _evaluate_osint_gate(min_auc=float(args.osint_min_auc), min_pr_auc=float(args.osint_min_pr_auc)),
        _evaluate_simulation_gate(min_r2=float(args.simulation_min_r2), max_rmse=float(args.simulation_max_rmse)),
    ]

    failed = [track["track"] for track in tracks if not bool(track.get("pass", False))]
    overall_pass = len(failed) == 0

    return {
        "generated_at": _utc_now(),
        "overall_pass": overall_pass,
        "failed_tracks": failed,
        "tracks": tracks,
    }


def _build_training_commands(py: str, simulation_sample_size: int, seed_variant: int) -> list[tuple[str, list[str]]]:
    return [
        ("fraud", [py, "ml_engine/train_model.py"]),
        ("vat_gnn", [py, "app/scripts/train_gnn.py"]),
        (
            "osint",
            [
                py,
                "ml_engine/train_osint.py",
                "--db-url",
                "default",
                "--seed",
                str(seed_variant),
            ],
        ),
        (
            "simulation",
            [
                py,
                "ml_engine/train_simulation.py",
                "--db-url",
                "default",
                "--sample-size",
                str(simulation_sample_size),
                "--seed",
                str(seed_variant),
            ],
        ),
    ]


def _build_reseed_command(py: str, args: argparse.Namespace, seed_value: int) -> list[str]:
    command = [
        py,
        "data/seed_db.py",
        "--csv-file",
        str(args.seed_csv_file),
        "--batch-size",
        str(args.seed_batch_size),
        "--seed",
        str(seed_value),
        "--invoice-target",
        str(args.seed_invoice_target),
        "--offshore-count",
        str(args.seed_offshore_count),
        "--offshore-links",
        str(args.seed_offshore_links),
    ]

    if bool(args.seed_skip_graph_seed):
        command.append("--skip-graph-seed")
    if bool(args.seed_skip_osint_seed):
        command.append("--skip-osint-seed")

    return command


def _run_single_attempt(args: argparse.Namespace, attempt: int, py: str) -> dict[str, Any]:
    seed_variant = int(args.base_seed) + (attempt - 1) * int(args.seed_stride)
    commands = _build_training_commands(
        py=py,
        simulation_sample_size=int(args.simulation_sample_size),
        seed_variant=seed_variant,
    )

    command_results: list[dict[str, Any]] = []
    training_success = True

    for track_name, command in commands:
        result = _run_command(command, dry_run=bool(args.dry_run))
        result["track"] = track_name
        command_results.append(result)

        if int(result.get("exit_code", 1)) != 0:
            training_success = False
            break

    gate_evaluation = None
    if training_success:
        gate_evaluation = _evaluate_all_gates(args)

    attempt_pass = bool(training_success and gate_evaluation and gate_evaluation.get("overall_pass"))

    return {
        "attempt": int(attempt),
        "seed_variant": int(seed_variant),
        "started_at": _utc_now(),
        "training_success": bool(training_success),
        "commands": command_results,
        "gate_evaluation": gate_evaluation,
        "status": "pass" if attempt_pass else "fail",
    }


def run_pipeline(args: argparse.Namespace) -> int:
    py = sys.executable

    print("=" * 78)
    print("  Full Model Retraining Pipeline (Fraud + VAT GNN + OSINT + Simulation)")
    print("=" * 78)
    print(f"[INFO] python={py}")
    print(f"[INFO] backend={BACKEND_DIR}")
    print(f"[INFO] max_attempts={args.max_attempts} retry_reseed={args.retry_reseed}")

    attempts: list[dict[str, Any]] = []
    success = False
    last_failure_reason = "unknown"

    for attempt in range(1, int(args.max_attempts) + 1):
        print(f"\n[ATTEMPT {attempt}/{args.max_attempts}] Starting model retraining sequence...")
        attempt_result = _run_single_attempt(args=args, attempt=attempt, py=py)
        attempts.append(attempt_result)

        if not bool(attempt_result.get("training_success", False)):
            last_failure_reason = "training_command_failed"
            print(f"[ATTEMPT {attempt}] Training command failed.")
        else:
            gate_eval = attempt_result.get("gate_evaluation") or {}
            if bool(gate_eval.get("overall_pass", False)):
                success = True
                print(f"[ATTEMPT {attempt}] All model gates passed.")
                break
            failed_tracks = gate_eval.get("failed_tracks") or []
            last_failure_reason = "quality_gate_failed"
            print(f"[ATTEMPT {attempt}] Gate failed: {', '.join(failed_tracks) if failed_tracks else 'unknown'}")

        should_retry = attempt < int(args.max_attempts)
        if not should_retry:
            break

        if bool(args.retry_reseed):
            next_seed = int(args.base_seed) + attempt * int(args.seed_stride)
            reseed_cmd = _build_reseed_command(py=py, args=args, seed_value=next_seed)
            print(f"[ATTEMPT {attempt}] Running reseed before retry with seed={next_seed}...")
            reseed_result = _run_command(reseed_cmd, dry_run=bool(args.dry_run))
            attempt_result["retry_reseed"] = {
                "seed": int(next_seed),
                "result": reseed_result,
            }

            if int(reseed_result.get("exit_code", 1)) != 0:
                print(f"[ATTEMPT {attempt}] Reseed failed, stopping pipeline.")
                last_failure_reason = "reseed_failed"
                break
        else:
            print(f"[ATTEMPT {attempt}] Retry requested without reseed.")

    final_status = "success_all_models_passed" if success else f"failed_{last_failure_reason}"

    report = {
        "generated_at": _utc_now(),
        "pipeline": "full_model_retraining",
        "max_attempts": int(args.max_attempts),
        "retry_reseed": bool(args.retry_reseed),
        "seed_policy": {
            "base_seed": int(args.base_seed),
            "seed_stride": int(args.seed_stride),
        },
        "quality_thresholds": {
            "osint_min_auc": float(args.osint_min_auc),
            "osint_min_pr_auc": float(args.osint_min_pr_auc),
            "simulation_min_r2": float(args.simulation_min_r2),
            "simulation_max_rmse": float(args.simulation_max_rmse),
        },
        "summary": {
            "status": final_status,
            "success": bool(success),
            "attempts_executed": len(attempts),
        },
        "attempts": attempts,
    }

    report_path = Path(args.report_file)
    if not report_path.is_absolute():
        report_path = BACKEND_DIR / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[REPORT] {report_path}")
    print(f"[SUMMARY] status={final_status} attempts={len(attempts)}")

    return 0 if success else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain all 4 flagship models with unified gates and retry policy")

    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum retraining attempts")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed used for reseed variations")
    parser.add_argument("--seed-stride", type=int, default=97, help="Seed increment between attempts")
    parser.add_argument("--retry-reseed", dest="retry_reseed", action="store_true", help="Enable reseed before each retry")
    parser.add_argument("--no-retry-reseed", dest="retry_reseed", action="store_false", help="Disable reseed between retries")
    parser.set_defaults(retry_reseed=True)

    parser.add_argument("--seed-csv-file", type=str, default="data/tax_data_mock.csv", help="CSV file used by seed_db.py")
    parser.add_argument("--seed-batch-size", type=int, default=2000)
    parser.add_argument("--seed-invoice-target", type=int, default=60000)
    parser.add_argument("--seed-offshore-count", type=int, default=1200)
    parser.add_argument("--seed-offshore-links", type=int, default=8000)
    parser.add_argument("--seed-skip-graph-seed", action="store_true")
    parser.add_argument("--seed-skip-osint-seed", action="store_true")

    parser.add_argument("--osint-min-auc", type=float, default=0.60)
    parser.add_argument("--osint-min-pr-auc", type=float, default=0.35)
    parser.add_argument("--simulation-min-r2", type=float, default=0.90)
    parser.add_argument("--simulation-max-rmse", type=float, default=0.08)

    parser.add_argument("--simulation-sample-size", type=int, default=20000)
    parser.add_argument("--report-file", type=str, default=str(DEFAULT_REPORT_FILE))
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if int(args.max_attempts) < 1:
        print("[ERROR] --max-attempts must be >= 1")
        return 1
    if int(args.seed_stride) < 1:
        print("[ERROR] --seed-stride must be >= 1")
        return 1
    if float(args.osint_min_auc) < 0.0 or float(args.osint_min_auc) > 1.0:
        print("[ERROR] --osint-min-auc must be in [0, 1]")
        return 1
    if float(args.osint_min_pr_auc) < 0.0 or float(args.osint_min_pr_auc) > 1.0:
        print("[ERROR] --osint-min-pr-auc must be in [0, 1]")
        return 1
    if float(args.simulation_min_r2) < -1.0 or float(args.simulation_min_r2) > 1.0:
        print("[ERROR] --simulation-min-r2 must be in [-1, 1]")
        return 1
    if float(args.simulation_max_rmse) <= 0.0:
        print("[ERROR] --simulation-max-rmse must be > 0")
        return 1
    if int(args.simulation_sample_size) < 1000:
        print("[ERROR] --simulation-sample-size must be >= 1000")
        return 1

    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
