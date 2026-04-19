"""
import_real_inspector_labels.py

Bulk import real inspector labels from CSV into inspector_labels.

Goals:
- Accelerate ingestion of real/field labels (manual_inspector, field_verified, imported_casework)
- Reject synthetic markers/origins on this real-label import path
- Preserve lineage by resolving model_version from row or linked assessment

Usage (from Backend):
  python app/scripts/import_real_inspector_labels.py --input-csv data/uploads/real_labels.csv
  python app/scripts/import_real_inspector_labels.py --input-csv data/uploads/real_labels.csv --strict-mode
  python app/scripts/import_real_inspector_labels.py --input-csv data/uploads/real_labels.csv --dry-run
  python app/scripts/import_real_inspector_labels.py --input-csv data/uploads/real_labels.csv --reject-report data/uploads/real_labels_rejected.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Add Backend root to sys.path
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from app.models import AIRiskAssessment, InspectorLabel


REAL_LABEL_ORIGINS = {"manual_inspector", "field_verified", "imported_casework"}
BLOCKED_LABEL_ORIGINS = {"bootstrap_generated", "auto_seed"}
SYNTHETIC_LABEL_MARKERS = (
    "[AUTO-SEED-LARGE]",
    "Bootstrap label from assessment",
    "synthetic training label",
)
TERMINAL_STATUSES = {"recovered", "partial_recovered", "unrecoverable", "dismissed"}


@dataclass
class ParseResult:
    row_number: int
    payload: dict[str, Any] | None
    error: str | None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_label_origin(raw_origin: Any, default_origin: str) -> str:
    normalized = _normalize_text(raw_origin).lower() or default_origin
    if normalized in BLOCKED_LABEL_ORIGINS:
        return normalized
    if normalized in REAL_LABEL_ORIGINS:
        return normalized
    return default_origin


def _contains_synthetic_marker(text: str) -> bool:
    low = _normalize_text(text).lower()
    if not low:
        return False
    for marker in SYNTHETIC_LABEL_MARKERS:
        if _normalize_text(marker).lower() in low:
            return True
    return False


def _parse_int(raw: Any) -> int | None:
    text = _normalize_text(raw)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_float(raw: Any) -> float | None:
    text = _normalize_text(raw)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_bool(raw: Any, default: bool = False) -> bool:
    text = _normalize_text(raw).lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_date(raw: Any) -> date | None:
    text = _normalize_text(raw)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_datetime(raw: Any) -> datetime | None:
    text = _normalize_text(raw)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = [dict(row or {}) for row in reader]
    return rows


def _resolve_assessment(db, assessment_id: int | None) -> AIRiskAssessment | None:
    if assessment_id is None:
        return None
    return db.query(AIRiskAssessment).filter(AIRiskAssessment.id == assessment_id).first()


def _parse_row(
    db,
    row: dict[str, Any],
    row_number: int,
    *,
    default_label_origin: str,
    default_inspector_id: int | None,
) -> ParseResult:
    tax_code = _normalize_text(row.get("tax_code"))
    if not tax_code:
        return ParseResult(row_number=row_number, payload=None, error="tax_code is required")

    label_type = _normalize_text(row.get("label_type"))
    if not label_type:
        return ParseResult(row_number=row_number, payload=None, error="label_type is required")

    evidence_summary = _normalize_text(row.get("evidence_summary"))
    if len(evidence_summary) < 12:
        return ParseResult(row_number=row_number, payload=None, error="evidence_summary must be at least 12 chars")
    if _contains_synthetic_marker(evidence_summary):
        return ParseResult(row_number=row_number, payload=None, error="evidence_summary contains synthetic markers")

    assessment_id = _parse_int(row.get("assessment_id"))
    assessment = _resolve_assessment(db, assessment_id)
    if assessment_id is not None and assessment is None:
        return ParseResult(row_number=row_number, payload=None, error="assessment_id not found")
    if assessment is not None and _normalize_text(getattr(assessment, "tax_code", "")) != tax_code:
        return ParseResult(row_number=row_number, payload=None, error="assessment_id tax_code mismatch")

    label_origin = _normalize_label_origin(row.get("label_origin"), default_label_origin)
    if label_origin in BLOCKED_LABEL_ORIGINS:
        return ParseResult(row_number=row_number, payload=None, error="blocked label_origin on real import path")

    confidence = _normalize_text(row.get("confidence")).lower() or "medium"
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    decision = _normalize_text(row.get("decision")) or None
    intervention_attempted = _parse_bool(row.get("intervention_attempted"), default=bool(decision))

    outcome_status = _normalize_text(row.get("outcome_status")).lower() or None
    amount_recovered = _parse_float(row.get("amount_recovered"))
    if outcome_status is None:
        if (amount_recovered or 0.0) > 0:
            outcome_status = "recovered"
        elif intervention_attempted:
            outcome_status = "in_progress"
        else:
            outcome_status = "pending"

    outcome_recorded_at = _parse_datetime(row.get("outcome_recorded_at"))
    if outcome_recorded_at is None and outcome_status in TERMINAL_STATUSES:
        outcome_recorded_at = datetime.utcnow()

    inspector_id = _parse_int(row.get("inspector_id"))
    if inspector_id is None:
        inspector_id = default_inspector_id

    model_version = _normalize_text(row.get("model_version"))[:80] or None
    if model_version is None and assessment is not None:
        linked_version = _normalize_text(getattr(assessment, "model_version", ""))[:80]
        model_version = linked_version or None

    payload = {
        "tax_code": tax_code,
        "inspector_id": inspector_id,
        "label_type": label_type,
        "confidence": confidence,
        "label_origin": label_origin,
        "assessment_id": assessment_id,
        "model_version": model_version,
        "evidence_summary": evidence_summary,
        "decision": decision,
        "decision_date": _parse_date(row.get("decision_date")),
        "tax_period": _normalize_text(row.get("tax_period")) or None,
        "amount_recovered": amount_recovered,
        "intervention_action": _normalize_text(row.get("intervention_action")) or None,
        "intervention_attempted": intervention_attempted,
        "outcome_status": outcome_status,
        "predicted_collection_uplift": _parse_float(row.get("predicted_collection_uplift")),
        "expected_recovery": _parse_float(row.get("expected_recovery")),
        "expected_net_recovery": _parse_float(row.get("expected_net_recovery")),
        "estimated_audit_cost": _parse_float(row.get("estimated_audit_cost")),
        "actual_audit_cost": _parse_float(row.get("actual_audit_cost")),
        "actual_audit_hours": _parse_float(row.get("actual_audit_hours")),
        "outcome_recorded_at": outcome_recorded_at,
        "kpi_window_days": _parse_int(row.get("kpi_window_days")) or 90,
    }

    return ParseResult(row_number=row_number, payload=payload, error=None)


def _write_reject_report(path: Path, rejected_rows: list[dict[str, Any]]) -> None:
    if not rejected_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rejected_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejected_rows)


def main(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        print(f"[ERROR] Input CSV not found: {input_csv}")
        return 1

    default_label_origin = _normalize_label_origin(args.default_label_origin, "imported_casework")
    if default_label_origin in BLOCKED_LABEL_ORIGINS:
        print("[ERROR] --default-label-origin cannot be synthetic on real import path")
        return 1

    rows = _load_csv_rows(input_csv)
    if not rows:
        print("[WARN] No rows in CSV")
        return 0

    print("=" * 72)
    print("  Import Real Inspector Labels")
    print("=" * 72)
    print(f"[INFO] input_csv={input_csv}")
    print(f"[INFO] row_count={len(rows)}")
    print(f"[INFO] strict_mode={bool(args.strict_mode)} dry_run={bool(args.dry_run)}")

    db = SessionLocal()
    parsed_ok: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    try:
        for idx, row in enumerate(rows, start=2):  # header at row 1
            result = _parse_row(
                db,
                row,
                row_number=idx,
                default_label_origin=default_label_origin,
                default_inspector_id=args.default_inspector_id,
            )
            if result.error:
                rejected.append({
                    "row_number": idx,
                    "error": result.error,
                    **row,
                })
                continue
            parsed_ok.append(result.payload or {})

        if args.strict_mode and rejected:
            print(f"[ABORT] strict-mode rejected rows={len(rejected)}")
            if args.reject_report:
                _write_reject_report(Path(args.reject_report).expanduser().resolve(), rejected)
                print(f"[INFO] reject_report={args.reject_report}")
            return 2

        if args.dry_run:
            print(f"[DRY-RUN] valid_rows={len(parsed_ok)} rejected_rows={len(rejected)}")
            if parsed_ok:
                sample = dict(parsed_ok[0])
                print(f"[DRY-RUN] first_valid_payload={json.dumps(sample, ensure_ascii=False, default=str)}")
            if args.reject_report and rejected:
                _write_reject_report(Path(args.reject_report).expanduser().resolve(), rejected)
                print(f"[INFO] reject_report={args.reject_report}")
            return 0

        inserted = 0
        batch_size = max(1, int(args.batch_size))
        batch: list[InspectorLabel] = []

        for payload in parsed_ok:
            batch.append(InspectorLabel(**payload))
            if len(batch) >= batch_size:
                db.add_all(batch)
                db.commit()
                inserted += len(batch)
                batch = []

        if batch:
            db.add_all(batch)
            db.commit()
            inserted += len(batch)

        if args.reject_report and rejected:
            _write_reject_report(Path(args.reject_report).expanduser().resolve(), rejected)
            print(f"[INFO] reject_report={args.reject_report}")

        print(f"[OK] inserted={inserted} rejected={len(rejected)}")
        return 0
    except Exception as exc:
        db.rollback()
        print(f"[ERROR] import failed: {exc}")
        return 3
    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk import real inspector labels from CSV")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument(
        "--default-label-origin",
        type=str,
        default="imported_casework",
        choices=["manual_inspector", "field_verified", "imported_casework", "bootstrap_generated", "auto_seed"],
        help="Default label_origin if row value is missing",
    )
    parser.add_argument("--default-inspector-id", type=int, default=None, help="Fallback inspector_id if row field is empty")
    parser.add_argument("--batch-size", type=int, default=500, help="Commit batch size")
    parser.add_argument("--strict-mode", action="store_true", help="Abort import if any row is invalid")
    parser.add_argument("--dry-run", action="store_true", help="Validate rows without DB writes")
    parser.add_argument("--reject-report", type=str, default=None, help="Optional CSV path for rejected rows")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
