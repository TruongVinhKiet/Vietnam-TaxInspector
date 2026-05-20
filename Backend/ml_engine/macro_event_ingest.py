"""
Macro event ingest pipeline with provenance, deduplication, and review queue.

External news/API/crawler data must enter this queue first. The simulation
engine should only consume reviewed canonical events from
historical_economic_events.json or the production DB table.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"
CANONICAL_EVENTS_PATH = DATA_DIR / "historical_economic_events.json"
REVIEW_QUEUE_PATH = DATA_DIR / "macro_event_review_queue.jsonl"


@dataclass
class MacroEventCandidate:
    title: str
    description: str
    source_name: str
    source_url: str
    published_at: Optional[str] = None
    event_type: str = "unknown"
    affected_provinces: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    impact_hints: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def normalized_title(self) -> str:
        return _normalize_text(self.title)

    def fingerprint(self) -> str:
        basis = {
            "title": self.normalized_title(),
            "source_url": self.source_url.strip().lower(),
            "published_at": str(self.published_at or "")[:10],
        }
        return hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_candidates_from_json(path: Path, *, source_name: Optional[str] = None) -> List[MacroEventCandidate]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        rows = parsed if isinstance(parsed, list) else parsed.get("items", [])

    candidates: List[MacroEventCandidate] = []
    for row in rows:
        candidates.append(MacroEventCandidate(
            title=str(row.get("title") or row.get("event_name_vi") or row.get("event_name") or "").strip(),
            description=str(row.get("description") or row.get("description_vi") or row.get("summary") or "").strip(),
            source_name=str(source_name or row.get("source_name") or row.get("source") or "unknown_source").strip(),
            source_url=str(row.get("source_url") or row.get("url") or "").strip(),
            published_at=row.get("published_at") or row.get("start_date") or row.get("date"),
            event_type=str(row.get("event_type") or "unknown").strip(),
            affected_provinces=[str(x) for x in row.get("affected_provinces", [])],
            affected_sectors=[str(x) for x in row.get("affected_sectors", [])],
            impact_hints=dict(row.get("impact_hints") or {}),
            raw_payload=dict(row),
        ))
    return [c for c in candidates if c.title and c.source_url]


def ingest_macro_event_candidates(
    candidates: Iterable[MacroEventCandidate],
    *,
    queue_path: Path = REVIEW_QUEUE_PATH,
    dry_run: bool = False,
) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    canonical = _load_canonical_events()
    queued = _load_queue(queue_path)
    existing_fingerprints = {str(item.get("fingerprint")) for item in queued if item.get("fingerprint")}

    stats = {
        "batch_id": f"macro-ingest-{uuid.uuid4().hex[:12]}",
        "received": 0,
        "queued": 0,
        "duplicates": 0,
        "rejected": 0,
        "dry_run": dry_run,
        "queue_path": str(queue_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
    }

    queue_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        stats["received"] += 1
        fingerprint = candidate.fingerprint()
        duplicate = _find_duplicate(candidate, canonical, queued)
        if fingerprint in existing_fingerprints:
            duplicate = ("queue_fingerprint", fingerprint, 1.0)

        status = "pending_review"
        if duplicate:
            status = "duplicate_candidate"
            stats["duplicates"] += 1
        else:
            stats["queued"] += 1
            existing_fingerprints.add(fingerprint)

        row = {
            "batch_id": stats["batch_id"],
            "fingerprint": fingerprint,
            "review_status": status,
            "duplicate_of": duplicate[1] if duplicate else None,
            "duplicate_reason": duplicate[0] if duplicate else None,
            "duplicate_score": duplicate[2] if duplicate else None,
            "candidate": _candidate_payload(candidate),
            "created_at": stats["created_at"],
        }
        stats["items"].append({
            "fingerprint": fingerprint,
            "status": status,
            "title": candidate.title,
            "source_url": candidate.source_url,
            "duplicate_reason": row["duplicate_reason"],
        })
        if status == "pending_review":
            queue_rows.append(row)

    if queue_rows and not dry_run:
        with queue_path.open("a", encoding="utf-8") as fh:
            for row in queue_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


def build_ingest_status(queue_path: Path = REVIEW_QUEUE_PATH) -> Dict[str, Any]:
    rows = _load_queue(queue_path)
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("review_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "queue_path": str(queue_path),
        "total": len(rows),
        "counts": counts,
        "pending_review": counts.get("pending_review", 0),
        "last_created_at": rows[-1].get("created_at") if rows else None,
    }


def _candidate_payload(candidate: MacroEventCandidate) -> Dict[str, Any]:
    return {
        "title": candidate.title,
        "description": candidate.description,
        "source_name": candidate.source_name,
        "source_url": candidate.source_url,
        "published_at": candidate.published_at,
        "event_type": candidate.event_type,
        "affected_provinces": candidate.affected_provinces,
        "affected_sectors": candidate.affected_sectors,
        "impact_hints": candidate.impact_hints,
        "raw_payload": candidate.raw_payload,
    }


def _load_canonical_events() -> List[Dict[str, Any]]:
    if not CANONICAL_EVENTS_PATH.exists():
        return []
    try:
        return json.loads(CANONICAL_EVENTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_queue(path: Path) -> List[Dict[str, Any]]:
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


def _find_duplicate(
    candidate: MacroEventCandidate,
    canonical: List[Dict[str, Any]],
    queued: List[Dict[str, Any]],
) -> Optional[Tuple[str, str, float]]:
    title = candidate.normalized_title()
    source_url = candidate.source_url.strip().lower()
    for event in canonical:
        canonical_url = str(event.get("source_url") or "").strip().lower()
        if source_url and canonical_url and source_url == canonical_url:
            return ("canonical_source_url", str(event.get("event_key") or canonical_url), 1.0)
        score = SequenceMatcher(None, title, _normalize_text(event.get("event_name_vi") or event.get("event_name"))).ratio()
        if score >= 0.92:
            return ("canonical_title_similarity", str(event.get("event_key") or event.get("event_name")), round(score, 4))

    for row in queued:
        payload = row.get("candidate") or {}
        queued_url = str(payload.get("source_url") or "").strip().lower()
        if source_url and queued_url and source_url == queued_url:
            return ("queue_source_url", str(row.get("fingerprint") or queued_url), 1.0)
        score = SequenceMatcher(None, title, _normalize_text(payload.get("title"))).ratio()
        if score >= 0.94:
            return ("queue_title_similarity", str(row.get("fingerprint") or payload.get("title")), round(score, 4))
    return None


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
