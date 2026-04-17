"""
Chunked Delinquency cache refresh helper.

Why this script exists:
- API contract enforces limit <= 2000 per batch request.
- Full refresh for large datasets must be split into smaller tax_code chunks.

Usage (from Backend root):
    python -m app.scripts.refresh_delinquency_cache --base-url http://127.0.0.1:8000 --chunk-size 500
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

from ..database import SessionLocal
from ..models import TaxPayment


def _load_tax_codes(max_tax_codes: int | None = None) -> list[str]:
    db = SessionLocal()
    try:
        query = (
            db.query(TaxPayment.tax_code)
            .filter(TaxPayment.tax_code.isnot(None))
            .distinct()
            .order_by(TaxPayment.tax_code.asc())
        )
        if max_tax_codes is not None:
            query = query.limit(max_tax_codes)
        rows = query.all()
        return [row[0] for row in rows if row and row[0]]
    finally:
        db.close()


def _post_batch(
    base_url: str,
    tax_codes: list[str],
    refresh_existing: bool,
    timeout_sec: int,
) -> dict:
    payload = {
        "tax_codes": tax_codes,
        "limit": min(2000, len(tax_codes) if tax_codes else 1),
        "refresh_existing": refresh_existing,
    }
    data = json.dumps(payload).encode("utf-8")

    endpoint = base_url.rstrip("/") + "/api/delinquency/predict-batch"
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def run_refresh(
    base_url: str,
    chunk_size: int,
    refresh_existing: bool,
    timeout_sec: int,
    max_tax_codes: int | None,
) -> int:
    if chunk_size < 1 or chunk_size > 2000:
        raise ValueError("chunk_size must be in range [1, 2000]")

    tax_codes = _load_tax_codes(max_tax_codes=max_tax_codes)
    total = len(tax_codes)

    if total == 0:
        print("[INFO] No tax codes found in tax_payments. Nothing to refresh.")
        return 0

    batches = (total + chunk_size - 1) // chunk_size
    print(f"[INFO] Refresh target tax codes: {total}")
    print(f"[INFO] Chunk size: {chunk_size} (batches: {batches})")

    agg = {
        "processed": 0,
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
    }

    for i in range(batches):
        start = i * chunk_size
        end = min(total, start + chunk_size)
        chunk = tax_codes[start:end]

        try:
            result = _post_batch(
                base_url=base_url,
                tax_codes=chunk,
                refresh_existing=refresh_existing,
                timeout_sec=timeout_sec,
            )

            processed = int(result.get("processed", 0) or 0)
            created = int(result.get("created", 0) or 0)
            updated = int(result.get("updated", 0) or 0)
            skipped = int(result.get("skipped", 0) or 0)
            failed = int(result.get("failed", 0) or 0)

            agg["processed"] += processed
            agg["created"] += created
            agg["updated"] += updated
            agg["skipped"] += skipped
            agg["failed"] += failed

            print(
                "[OK] batch {}/{} processed={} created={} updated={} skipped={} failed={}".format(
                    i + 1,
                    batches,
                    processed,
                    created,
                    updated,
                    skipped,
                    failed,
                )
            )
        except urllib.error.HTTPError as exc:
            agg["failed"] += len(chunk)
            body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            print(f"[ERROR] batch {i + 1}/{batches} HTTP {exc.code}: {body}")
        except Exception as exc:  # noqa: BLE001
            agg["failed"] += len(chunk)
            print(f"[ERROR] batch {i + 1}/{batches} failed: {exc}")

    print("\n[SUMMARY]")
    print(f"processed={agg['processed']}")
    print(f"created={agg['created']}")
    print(f"updated={agg['updated']}")
    print(f"skipped={agg['skipped']}")
    print(f"failed={agg['failed']}")

    return 0 if agg["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh delinquency cache in deterministic chunks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--chunk-size", type=int, default=500, help="Batch chunk size (<=2000)")
    parser.add_argument("--timeout-sec", type=int, default=300, help="HTTP timeout per chunk")
    parser.add_argument("--max-tax-codes", type=int, default=None, help="Optional cap for dry runs")
    parser.add_argument(
        "--refresh-existing",
        dest="refresh_existing",
        action="store_true",
        help="Delete today's existing predictions and recreate.",
    )
    parser.add_argument(
        "--no-refresh-existing",
        dest="refresh_existing",
        action="store_false",
        help="Skip candidates that already have today's predictions.",
    )
    parser.set_defaults(refresh_existing=True)

    args = parser.parse_args()

    try:
        return run_refresh(
            base_url=args.base_url,
            chunk_size=args.chunk_size,
            refresh_existing=args.refresh_existing,
            timeout_sec=args.timeout_sec,
            max_tax_codes=args.max_tax_codes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
