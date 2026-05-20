"""
CLI for macro event ingest.

Example:
    python Backend/scripts/ingest_macro_events.py --input events.json --source-name "GSO RSS" --dry-run
    python Backend/scripts/ingest_macro_events.py --input events.jsonl --source-name "VNA crawler"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_event_ingest import (  # noqa: E402
    build_ingest_status,
    ingest_macro_event_candidates,
    load_candidates_from_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest macro-economic event candidates into review queue.")
    parser.add_argument("--input", type=Path, help="JSON/JSONL file containing event/news candidates.")
    parser.add_argument("--source-name", default=None, help="Override source name for all candidates.")
    parser.add_argument("--dry-run", action="store_true", help="Run deduplication without writing the queue.")
    parser.add_argument("--status", action="store_true", help="Print current review queue status.")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(build_ingest_status(), ensure_ascii=False, indent=2))
        return 0

    if not args.input:
        parser.error("--input is required unless --status is used")
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    candidates = load_candidates_from_json(args.input, source_name=args.source_name)
    stats = ingest_macro_event_candidates(candidates, dry_run=args.dry_run)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
