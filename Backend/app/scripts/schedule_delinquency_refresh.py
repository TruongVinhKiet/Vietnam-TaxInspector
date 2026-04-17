"""
Periodic scheduler for Delinquency cache refresh.

Why this script exists:
- `refresh_delinquency_cache` is chunk-safe but one-shot.
- Production operation usually needs recurring refresh cycles.

Usage examples:
    python -m app.scripts.schedule_delinquency_refresh --once --chunk-size 500 --refresh-existing
    python -m app.scripts.schedule_delinquency_refresh --interval-minutes 180 --chunk-size 500 --no-refresh-existing
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta

from .refresh_delinquency_cache import run_refresh

_STOP_REQUESTED = False


def _handle_stop_signal(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[INFO] Received signal {signum}. Stopping after current cycle.")


def _wait_for_next_cycle(interval_seconds: int):
    if interval_seconds <= 0:
        return

    next_run = datetime.now() + timedelta(seconds=interval_seconds)
    print(f"[INFO] Next cycle at {next_run.isoformat(timespec='seconds')}")

    remaining = interval_seconds
    while remaining > 0 and not _STOP_REQUESTED:
        sleep_seconds = 1 if remaining > 1 else remaining
        time.sleep(sleep_seconds)
        remaining -= sleep_seconds


def _run_cycle(cycle_no: int, args) -> int:
    started_at = datetime.now()
    print(f"\n[CYCLE {cycle_no}] Started at {started_at.isoformat(timespec='seconds')}")

    exit_code = run_refresh(
        base_url=args.base_url,
        chunk_size=args.chunk_size,
        refresh_existing=args.refresh_existing,
        timeout_sec=args.timeout_sec,
        max_tax_codes=args.max_tax_codes,
    )

    ended_at = datetime.now()
    elapsed = (ended_at - started_at).total_seconds()
    print(
        f"[CYCLE {cycle_no}] Finished at {ended_at.isoformat(timespec='seconds')} "
        f"(elapsed={elapsed:.1f}s, exit_code={exit_code})"
    )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Delinquency cache refresh on a recurring schedule.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--chunk-size", type=int, default=500, help="Batch chunk size (<=2000)")
    parser.add_argument("--timeout-sec", type=int, default=300, help="HTTP timeout per chunk")
    parser.add_argument("--max-tax-codes", type=int, default=None, help="Optional cap for dry runs")
    parser.add_argument("--interval-minutes", type=int, default=180, help="Minutes between refresh cycles")
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional cap on number of cycles")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit.",
    )
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
    parser.set_defaults(refresh_existing=False)

    args = parser.parse_args()

    if args.interval_minutes < 1:
        print("[FATAL] interval_minutes must be >= 1")
        return 1
    if args.max_cycles is not None and args.max_cycles < 1:
        print("[FATAL] max_cycles must be >= 1 when provided")
        return 1

    signal.signal(signal.SIGINT, _handle_stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop_signal)

    print("[INFO] Delinquency refresh scheduler started")
    print(
        "[INFO] Config: base_url={}, chunk_size={}, refresh_existing={}, interval_minutes={}, once={}, max_cycles={}".format(
            args.base_url,
            args.chunk_size,
            args.refresh_existing,
            args.interval_minutes,
            args.once,
            args.max_cycles,
        )
    )

    failures = 0
    cycle_no = 0
    interval_seconds = args.interval_minutes * 60

    while not _STOP_REQUESTED:
        if args.max_cycles is not None and cycle_no >= args.max_cycles:
            break

        cycle_no += 1
        exit_code = _run_cycle(cycle_no, args)
        if exit_code != 0:
            failures += 1

        if args.once:
            break
        if args.max_cycles is not None and cycle_no >= args.max_cycles:
            break

        _wait_for_next_cycle(interval_seconds)

    print(f"[INFO] Scheduler stopped after {cycle_no} cycle(s), failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
