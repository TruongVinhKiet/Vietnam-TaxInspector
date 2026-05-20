"""
Real-time economic news crawler for Vietnam Digital Twin.

Crawls RSS feeds from Vietnamese and international news sources,
classifies articles using Gemini AI, and feeds them into the
macro event ingest pipeline for review.

Usage:
    python Backend/scripts/crawl_economic_news.py
    python Backend/scripts/crawl_economic_news.py --dry-run
    python Backend/scripts/crawl_economic_news.py --source vnexpress --no-ai
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

from ml_engine.news_crawler import crawl_all_feeds, RSS_FEEDS
from ml_engine.macro_event_ingest import ingest_macro_event_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl economic news and ingest into review queue.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to queue.")
    parser.add_argument("--no-ai", action="store_true", help="Skip Gemini API, use keyword classification only.")
    parser.add_argument("--source", type=str, default=None, help="Filter to specific source (vnexpress, tuoitre, reuters, etc.)")
    parser.add_argument("--max-per-feed", type=int, default=15, help="Max articles per RSS feed.")
    parser.add_argument("--output", type=Path, default=None, help="Save crawled candidates to JSON file instead of ingesting.")
    args = parser.parse_args()

    # Filter feeds if --source specified
    feeds = RSS_FEEDS
    if args.source:
        feeds = [f for f in RSS_FEEDS if args.source.lower() in f["source_name"].lower() or args.source.lower() in f["name"].lower()]
        if not feeds:
            print(f"No feeds matching '{args.source}'. Available: {[f['source_name'] for f in RSS_FEEDS]}")
            return 1

    # Get API key
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass  # dotenv not installed; rely on environment variables directly
    
    api_key = "" if args.no_ai else os.environ.get("GEMINI_API_KEY", "")
    if not api_key and not args.no_ai:
        print("⚠️  GEMINI_API_KEY not set. Using keyword-based classification (--no-ai mode).")

    # Crawl
    print(f"🔍 Crawling {len(feeds)} RSS feeds (max {args.max_per_feed}/feed)...")
    candidates = crawl_all_feeds(
        api_key=api_key,
        feeds=feeds,
        max_per_feed=args.max_per_feed,
    )
    print(f"📰 Found {len(candidates)} economic news candidates.")

    if not candidates:
        print("No candidates found. Exiting.")
        return 0

    # Output to file or ingest
    if args.output:
        import dataclasses
        output_data = [dataclasses.asdict(c) for c in candidates]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Saved to {args.output}")
    else:
        # Ingest into review queue
        stats = ingest_macro_event_candidates(candidates, dry_run=args.dry_run)
        print(f"\n{'🧪 DRY RUN' if args.dry_run else '✅ INGESTED'}:")
        print(f"   Received:   {stats['received']}")
        print(f"   Queued:     {stats['queued']}")
        print(f"   Duplicates: {stats['duplicates']}")
        print(f"   Queue path: {stats['queue_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
