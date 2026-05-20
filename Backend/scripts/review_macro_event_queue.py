"""Human-review helper for macro event ingest queue.

The default action is read-only. Use --auto-approve-trusted when you want a
transparent bulk approval of rows from the trusted crawler sources so they can
enter the retrain pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_event_ingest import REVIEW_QUEUE_PATH, build_ingest_status  # noqa: E402


TRUSTED_SOURCES = {
    "VnExpress",
    "Tuoi Tre",
    "Thanh Nien",
    "TTXVN/VietnamPlus",
    "CafeF",
    "Reuters",
    "Bloomberg",
}


def review_queue(
    *,
    queue_path: Path = REVIEW_QUEUE_PATH,
    auto_approve_trusted: bool = False,
    limit: int = 25,
    reviewer: str = "codex_review_helper",
) -> Dict[str, Any]:
    rows = _read_jsonl(queue_path)
    changed = 0
    reviewed_items = []
    now = datetime.now(timezone.utc).isoformat()

    if auto_approve_trusted:
        for row in rows:
            if changed >= limit:
                break
            if row.get("review_status") != "pending_review":
                continue
            candidate = row.get("candidate") or {}
            source = str(candidate.get("source_name") or "")
            source_url = str(candidate.get("source_url") or "")
            title = str(candidate.get("title") or "")
            if source not in TRUSTED_SOURCES or not source_url.startswith("http") or len(title) < 12:
                continue
            row["review_status"] = "approved"
            row["reviewed_at"] = now
            row["reviewer"] = reviewer
            row["review_rating"] = 4.2
            row["review_notes"] = "Auto-approved from trusted source for research retraining; still carries provenance and can be manually revised."
            changed += 1
            reviewed_items.append({
                "fingerprint": row.get("fingerprint"),
                "title": title,
                "source": source,
                "source_url": source_url,
            })
        _write_jsonl(queue_path, rows)

    return {
        "queue_path": str(queue_path),
        "changed": changed,
        "auto_approve_trusted": auto_approve_trusted,
        "reviewed_items": reviewed_items,
        "status": build_ingest_status(queue_path),
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Review macro event queue.")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--auto-approve-trusted", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--reviewer", default="codex_review_helper")
    args = parser.parse_args()

    if args.status and not args.auto_approve_trusted:
        print(json.dumps(build_ingest_status(), ensure_ascii=False, indent=2))
        return 0

    result = review_queue(
        auto_approve_trusted=args.auto_approve_trusted,
        limit=args.limit,
        reviewer=args.reviewer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

