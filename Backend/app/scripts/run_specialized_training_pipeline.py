"""
run_specialized_training_pipeline.py

One-command pipeline for specialized model lifecycle:
1) Optional large random seed generation for training tables
2) Train audit_value model
3) Train vat_refund model
4) Run pilot cohort comparison (model vs heuristic)
5) Run go/no-go review for Phase 5 gating
6) Print artifact + quality + pilot + go/no-go summary

Usage (from Backend directory):
    python app/scripts/run_specialized_training_pipeline.py

Skip seeding and only retrain models:
    python app/scripts/run_specialized_training_pipeline.py --skip-seed

Dry-run commands only:
    python app/scripts/run_specialized_training_pipeline.py --dry-run

Use only real/non-synthetic label origins during training:
    python app/scripts/run_specialized_training_pipeline.py --label-origin-policy exclude_synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_DIR / "data" / "models"
PILOT_REPORT_FILE = "specialized_pilot_report.json"
GO_NO_GO_REPORT_FILE = "specialized_go_no_go_report.json"
MIN_REQUIRED_TRAINING_SAMPLES = max(10_000, int(os.environ.get("TRAINING_MIN_REQUIRED_SAMPLES", "10000")))


def _run_command(command: list[str], dry_run: bool = False) -> int:
    cmd_text = " ".join(command)
    print(f"[RUN] {cmd_text}")
    if dry_run:
        print("[DRY-RUN] skipped execution")
        return 0

    result = subprocess.run(command, cwd=str(BACKEND_DIR))
    if result.returncode != 0:
        print(f"[ERROR] command failed with code {result.returncode}")
    return int(result.returncode)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] cannot parse JSON {path.name}: {exc}")
        return None


def _print_quality_summary(file_name: str, min_training_samples: int) -> None:
    payload = _load_json(MODELS_DIR / file_name)
    if not payload:
        print(f"[WARN] missing or invalid quality file: {file_name}")
        return

    gates = payload.get("acceptance_gates") or {}
    perf = (payload.get("performance") or {}).get("calibrated") or {}
    dataset = payload.get("dataset") or {}

    print(f"[QUALITY] {file_name}")
    print(f"          overall_pass={gates.get('overall_pass')}")
    print(
        "          samples={} required_min_samples={}".format(
            dataset.get("total_size"),
            dataset.get("required_min_samples", min_training_samples),
        )
    )
    print(f"          auc_roc={perf.get('auc_roc')} pr_auc={perf.get('pr_auc')} brier={perf.get('brier')} ece={perf.get('ece')}")


def _print_artifacts_summary() -> None:
    required_prefixes = ("audit_value_", "vat_refund_")
    files = [
        f for f in MODELS_DIR.glob("*")
        if f.is_file() and any(f.name.startswith(prefix) for prefix in required_prefixes)
    ]
    files.sort(key=lambda p: p.name)

    print("[ARTIFACTS]")
    if not files:
        print("  - no audit/vat artifacts found")
        return

    for f in files:
        size_kb = f.stat().st_size / 1024.0
        print(f"  - {f.name} ({size_kb:.1f} KB)")


def _print_pilot_summary(file_name: str) -> None:
    payload = _load_json(MODELS_DIR / file_name)
    if not payload:
        print(f"[WARN] missing or invalid pilot report: {file_name}")
        return

    tracks = payload.get("tracks") or {}
    print(f"[PILOT] {file_name}")
    for track_name in ("audit_value", "vat_refund"):
        track = tracks.get(track_name) or {}
        model = track.get("model") or {}
        heuristic = track.get("heuristic") or {}
        delta = track.get("delta_model_minus_heuristic") or {}
        print(
            "          {} evaluated={} model_f1={} heuristic_f1={} delta_f1={}".format(
                track_name,
                track.get("samples_evaluated"),
                model.get("f1"),
                heuristic.get("f1"),
                delta.get("f1_delta"),
            )
        )


def _print_go_no_go_summary(file_name: str) -> None:
    payload = _load_json(MODELS_DIR / file_name)
    if not payload:
        print(f"[WARN] missing or invalid go/no-go report: {file_name}")
        return

    summary = payload.get("summary") or {}
    decision = payload.get("decision") or {}

    print(f"[GO-NO-GO] {file_name}")
    print(
        "             hard_gates_pass={} split_gate_pass={} stability_gate_pass={}".format(
            summary.get("hard_gates_pass"),
            summary.get("split_gate_pass"),
            summary.get("stability_gate_pass"),
        )
    )
    print(
        "             decision={} go_live_phase_d={}".format(
            decision.get("status"),
            decision.get("go_live_phase_d"),
        )
    )


def run_pipeline(args: argparse.Namespace) -> int:
    py = sys.executable
    min_training_samples = max(MIN_REQUIRED_TRAINING_SAMPLES, int(args.min_training_samples))
    print("=" * 72)
    print("  Specialized Training Pipeline")
    print("=" * 72)
    print(f"[INFO] python={py}")
    print(f"[INFO] backend={BACKEND_DIR}")
    print(f"[INFO] min_training_samples={min_training_samples}")

    if not args.skip_seed:
        seed_cmd = [
            py,
            "data/seed_large_training_data.py",
            "--assessment-rows", str(args.assessment_rows),
            "--label-rows", str(args.label_rows),
            "--seed", str(args.seed),
        ]
        if args.reset_generated:
            seed_cmd.append("--reset-generated")
        if args.seed_lookback_days is not None:
            seed_cmd.extend(["--lookback-days", str(args.seed_lookback_days)])

        if _run_command(seed_cmd, dry_run=args.dry_run) != 0:
            return 2
    else:
        print("[SKIP] seeding step")

    audit_cmd = [
        py,
        "ml_engine/train_audit_value.py",
        "--min-samples", str(min_training_samples),
        "--label-origin-policy", str(args.label_origin_policy),
    ]
    if args.audit_lookback_days is not None:
        audit_cmd.extend(["--lookback-days", str(args.audit_lookback_days)])

    if _run_command(audit_cmd, dry_run=args.dry_run) != 0:
        return 3

    vat_cmd = [
        py,
        "ml_engine/train_vat_refund.py",
        "--min-samples", str(min_training_samples),
        "--label-origin-policy", str(args.label_origin_policy),
    ]
    if args.vat_lookback_days is not None:
        vat_cmd.extend(["--lookback-days", str(args.vat_lookback_days)])

    if _run_command(vat_cmd, dry_run=args.dry_run) != 0:
        return 4

    if not args.skip_pilot:
        pilot_cmd = [
            py,
            "app/scripts/run_specialized_pilot_cohort.py",
            "--lookback-days", str(args.pilot_lookback_days),
            "--per-track-limit", str(args.pilot_per_track_limit),
            "--model-threshold", str(args.pilot_model_threshold),
            "--output", str(MODELS_DIR / PILOT_REPORT_FILE),
        ]
        if _run_command(pilot_cmd, dry_run=args.dry_run) != 0:
            return 5

        if not args.skip_go_no_go:
            go_no_go_cmd = [
                py,
                "app/scripts/run_specialized_go_no_go_review.py",
                "--pilot-file", str(MODELS_DIR / PILOT_REPORT_FILE),
                "--output", str(MODELS_DIR / GO_NO_GO_REPORT_FILE),
                "--min-pilot-samples", str(args.go_no_go_min_pilot_samples),
                "--audit-min-f1-delta", str(args.go_no_go_audit_min_f1_delta),
                "--vat-min-f1-delta", str(args.go_no_go_vat_min_f1_delta),
                "--max-accuracy-drop", str(args.go_no_go_max_accuracy_drop),
                "--min-training-samples", str(min_training_samples),
                "--min-consecutive-hard-pass-runs", str(args.go_no_go_min_consecutive_hard_pass_runs),
            ]
            if _run_command(go_no_go_cmd, dry_run=args.dry_run) != 0:
                return 6
        else:
            print("[SKIP] go/no-go review step")
    else:
        print("[SKIP] pilot comparison step")

    if args.dry_run:
        print("[DONE] dry-run complete")
        return 0

    _print_artifacts_summary()
    _print_quality_summary("audit_value_quality_report.json", min_training_samples=min_training_samples)
    _print_quality_summary("vat_refund_quality_report.json", min_training_samples=min_training_samples)
    if not args.skip_pilot:
        _print_pilot_summary(PILOT_REPORT_FILE)
        if not args.skip_go_no_go:
            _print_go_no_go_summary(GO_NO_GO_REPORT_FILE)

    print("[DONE] pipeline completed successfully")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end specialized model pipeline")
    parser.add_argument("--skip-seed", action="store_true", help="Skip large seed generation")
    parser.add_argument("--reset-generated", action="store_true", help="Reset rows generated by seed_large_training_data before reseeding")
    parser.add_argument("--assessment-rows", type=int, default=12000, help="Rows inserted into ai_risk_assessments")
    parser.add_argument("--label-rows", type=int, default=20000, help="Rows inserted into inspector_labels")
    parser.add_argument("--min-training-samples", type=int, default=MIN_REQUIRED_TRAINING_SAMPLES, help="Minimum required training samples per specialized model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    parser.add_argument("--seed-lookback-days", type=int, default=540, help="Lookback days for generated timestamps")
    parser.add_argument("--audit-lookback-days", type=int, default=None, help="Override lookback days for audit training")
    parser.add_argument("--vat-lookback-days", type=int, default=None, help="Override lookback days for VAT training")
    parser.add_argument(
        "--label-origin-policy",
        type=str,
        choices=["exclude_synthetic", "real_only", "all"],
        default="exclude_synthetic",
        help="Filter policy for inspector label origins used in specialized training",
    )
    parser.add_argument("--skip-pilot", action="store_true", help="Skip pilot model-vs-heuristic comparison")
    parser.add_argument("--skip-go-no-go", action="store_true", help="Skip go/no-go review stage")
    parser.add_argument("--pilot-lookback-days", type=int, default=540, help="Lookback days used by pilot cohort")
    parser.add_argument("--pilot-per-track-limit", type=int, default=3000, help="Max evaluation rows per track in pilot")
    parser.add_argument("--pilot-model-threshold", type=float, default=0.5, help="Model probability threshold for pilot")
    parser.add_argument("--go-no-go-min-pilot-samples", type=int, default=200, help="Min evaluated samples per track for go/no-go")
    parser.add_argument("--go-no-go-audit-min-f1-delta", type=float, default=0.05, help="Min audit model-vs-heuristic F1 delta")
    parser.add_argument("--go-no-go-vat-min-f1-delta", type=float, default=-0.05, help="Min VAT model-vs-heuristic F1 delta")
    parser.add_argument("--go-no-go-max-accuracy-drop", type=float, default=0.05, help="Max allowed model accuracy drop vs heuristic")
    parser.add_argument("--go-no-go-min-consecutive-hard-pass-runs", type=int, default=2, help="Stability gate requirement")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    if args.assessment_rows < 0 or args.label_rows < 0:
        print("[ERROR] --assessment-rows and --label-rows must be >= 0")
        return 1
    if args.min_training_samples < MIN_REQUIRED_TRAINING_SAMPLES:
        print(f"[ERROR] --min-training-samples must be >= {MIN_REQUIRED_TRAINING_SAMPLES}")
        return 1
    if not args.skip_seed and args.assessment_rows < args.min_training_samples:
        print("[ERROR] --assessment-rows must be >= --min-training-samples")
        return 1
    if not args.skip_seed and args.label_rows < (args.min_training_samples * 2):
        print("[ERROR] --label-rows must be >= 2 * --min-training-samples (dual-track requirement)")
        return 1
    if args.pilot_lookback_days < 1:
        print("[ERROR] --pilot-lookback-days must be >= 1")
        return 1
    if args.pilot_per_track_limit < 50:
        print("[ERROR] --pilot-per-track-limit should be >= 50")
        return 1
    if args.pilot_model_threshold < 0.0 or args.pilot_model_threshold > 1.0:
        print("[ERROR] --pilot-model-threshold must be in [0, 1]")
        return 1
    if args.go_no_go_min_pilot_samples < 1:
        print("[ERROR] --go-no-go-min-pilot-samples must be >= 1")
        return 1
    if args.go_no_go_max_accuracy_drop < 0:
        print("[ERROR] --go-no-go-max-accuracy-drop must be >= 0")
        return 1
    if args.go_no_go_min_consecutive_hard_pass_runs < 1:
        print("[ERROR] --go-no-go-min-consecutive-hard-pass-runs must be >= 1")
        return 1

    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
