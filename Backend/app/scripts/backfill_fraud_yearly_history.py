"""
Backfill yearly_history for AIRiskAssessment records.

Usage (from Backend directory):
  python -m app.scripts.backfill_fraud_yearly_history --dry-run
  python -m app.scripts.backfill_fraud_yearly_history
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from app.database import SessionLocal
from app import models


def _normalize_tax_code(value: Any) -> str:
    raw = str(value).strip()
    if raw.lower() in {"", "nan", "none", "<na>"}:
        return ""
    # When pandas parsed MST as float, remove trailing ".0" artifacts.
    if raw.endswith(".0") and raw.replace(".", "").isdigit():
        raw = raw[:-2]
    return raw


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_history(raw_history: Any) -> list[dict]:
    if not isinstance(raw_history, list):
        return []

    by_year: dict[int, dict] = {}
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        try:
            year = int(item.get("year"))
        except (TypeError, ValueError):
            continue
        if year <= 0:
            continue
        by_year[year] = {
            "year": year,
            "revenue": round(_to_float(item.get("revenue"), 0.0), 2),
            "total_expenses": round(_to_float(item.get("total_expenses"), 0.0), 2),
        }

    return [by_year[y] for y in sorted(by_year.keys())]


def _build_history_from_frame(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []

    if "year" not in frame.columns or "revenue" not in frame.columns:
        return []

    expense_col = "total_expenses" if "total_expenses" in frame.columns else (
        "expenses" if "expenses" in frame.columns else None
    )
    if expense_col is None:
        return []

    work = frame.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce")
    work[expense_col] = pd.to_numeric(work[expense_col], errors="coerce")
    work = work.dropna(subset=["year", "revenue", expense_col])
    if work.empty:
        return []

    work["year"] = work["year"].astype(int)
    work = work[work["year"] > 0]
    if work.empty:
        return []

    grouped = (
        work.groupby("year", as_index=False)[["revenue", expense_col]]
        .sum()
        .sort_values("year")
    )

    return [
        {
            "year": int(row["year"]),
            "revenue": round(_to_float(row["revenue"], 0.0), 2),
            "total_expenses": round(_to_float(row[expense_col], 0.0), 2),
        }
        for _, row in grouped.iterrows()
    ]


def _load_batch_frame(
    batch_id: int,
    batch_cache: dict[int, models.AIAnalysisBatch | None],
    frame_cache: dict[int, pd.DataFrame | None],
) -> pd.DataFrame | None:
    if batch_id in frame_cache:
        return frame_cache[batch_id]

    batch = batch_cache.get(batch_id)
    if not batch or not batch.file_path:
        frame_cache[batch_id] = None
        return None

    csv_path = Path(batch.file_path)
    if not csv_path.exists():
        frame_cache[batch_id] = None
        return None

    try:
        frame = pd.read_csv(csv_path, dtype={"tax_code": "string"}, low_memory=False)
    except Exception:
        frame_cache[batch_id] = None
        return None

    if "tax_code" not in frame.columns:
        frame_cache[batch_id] = None
        return None

    frame["tax_code"] = frame["tax_code"].astype("string").map(_normalize_tax_code)
    frame = frame[frame["tax_code"] != ""]
    frame_cache[batch_id] = frame
    return frame


def _build_history_from_tax_returns(
    tax_code: str,
    tax_return_cache: dict[str, list[dict]],
    db,
) -> list[dict]:
    if tax_code in tax_return_cache:
        return tax_return_cache[tax_code]

    rows = (
        db.query(models.TaxReturn)
        .filter(models.TaxReturn.tax_code == tax_code)
        .all()
    )

    yearly: dict[int, dict] = {}
    for tr in rows:
        filing_date = getattr(tr, "filing_date", None)
        year = filing_date.year if filing_date else None
        if year is None:
            continue
        bucket = yearly.setdefault(year, {"year": year, "revenue": 0.0, "total_expenses": 0.0})
        bucket["revenue"] += _to_float(getattr(tr, "revenue", 0.0), 0.0)
        bucket["total_expenses"] += _to_float(getattr(tr, "expenses", 0.0), 0.0)

    result = [yearly[y] for y in sorted(yearly.keys())]
    result = _normalize_history(result)
    tax_return_cache[tax_code] = result
    return result


def _build_history_from_assessment_group(group: list[models.AIRiskAssessment]) -> list[dict]:
    merged: dict[int, dict] = {}

    for record in group:
        for point in _normalize_history(record.yearly_history):
            merged.setdefault(point["year"], point)

        try:
            year = int(record.year) if record.year is not None else None
        except (TypeError, ValueError):
            year = None

        if year is None or year <= 0:
            continue

        merged.setdefault(
            year,
            {
                "year": year,
                "revenue": round(_to_float(record.revenue, 0.0), 2),
                "total_expenses": round(_to_float(record.total_expenses, 0.0), 2),
            },
        )

    return [merged[y] for y in sorted(merged.keys())]


def run_backfill(dry_run: bool = True, verbose: bool = True) -> tuple[int, int]:
    db = SessionLocal()
    updated = 0

    try:
        all_records = db.query(models.AIRiskAssessment).order_by(models.AIRiskAssessment.id.asc()).all()

        batch_ids = {int(r.batch_id) for r in all_records if r.batch_id is not None}
        batches = (
            db.query(models.AIAnalysisBatch)
            .filter(models.AIAnalysisBatch.id.in_(batch_ids))
            .all()
            if batch_ids
            else []
        )
        batch_cache = {b.id: b for b in batches}

        by_tax_code: dict[str, list[models.AIRiskAssessment]] = {}
        for r in all_records:
            tc = _normalize_tax_code(r.tax_code)
            by_tax_code.setdefault(tc, []).append(r)

        frame_cache: dict[int, pd.DataFrame | None] = {}
        tax_return_cache: dict[str, list[dict]] = {}

        for record in all_records:
            tax_code = _normalize_tax_code(record.tax_code)
            current = _normalize_history(record.yearly_history)
            best = current
            source = "current"

            if record.batch_id is not None:
                batch_frame = _load_batch_frame(int(record.batch_id), batch_cache, frame_cache)
                if batch_frame is not None and not batch_frame.empty:
                    rows = batch_frame[batch_frame["tax_code"] == tax_code]
                    from_batch = _build_history_from_frame(rows)
                    if len(from_batch) > len(best):
                        best = from_batch
                        source = "batch_csv"

            if len(best) <= 1:
                from_tax_returns = _build_history_from_tax_returns(tax_code, tax_return_cache, db)
                if len(from_tax_returns) > len(best):
                    best = from_tax_returns
                    source = "tax_returns"

            if len(best) <= 1:
                from_assessments = _build_history_from_assessment_group(by_tax_code.get(tax_code, []))
                if len(from_assessments) > len(best):
                    best = from_assessments
                    source = "assessment_rows"

            if best != current:
                updated += 1
                if verbose:
                    print(
                        f"[UPDATE] id={record.id} tax_code={tax_code} "
                        f"{len(current)}y -> {len(best)}y source={source}"
                    )
                if not dry_run:
                    record.yearly_history = best

        if not dry_run and updated > 0:
            db.commit()

        return len(all_records), updated
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill yearly_history in AIRiskAssessment")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print updates only, without writing to DB.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final summary, skip per-record [UPDATE] logs.",
    )
    args = parser.parse_args()

    total, updated = run_backfill(dry_run=args.dry_run, verbose=not args.quiet)
    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"[{mode}] scanned={total}, updates={updated}")


if __name__ == "__main__":
    main()
