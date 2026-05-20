"""
Fetch trusted RSS feeds and queue macro-economic event candidates for review.

No candidate is promoted directly into model training data. Human review remains
the gate before events become canonical.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_event_ingest import ingest_macro_event_candidates  # noqa: E402
from ml_engine.news_crawler import crawl_all_feeds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl macro-economic news RSS feeds into review queue.")
    parser.add_argument("--max-per-feed", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-llm", action="store_true", help="Use GEMINI_API_KEY for article classification when available.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "") if args.use_llm else ""
    candidates = crawl_all_feeds(api_key=api_key, max_per_feed=args.max_per_feed)
    stats = ingest_macro_event_candidates(candidates, dry_run=args.dry_run)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
