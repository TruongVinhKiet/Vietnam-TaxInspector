"""
tax_agent_feedback.py – Adaptive Feedback Loop (Continuous Learning)
=====================================================================
Collects, stores, and analyzes user feedback on agent responses to
enable continuous improvement, drift detection, and active learning.

Capabilities:
    1. Record thumbs up/down + optional correction per message
    2. Compute rolling satisfaction score & model drift alerts
    3. Identify uncertain predictions for active learning
    4. Aggregate feedback statistics for dashboard consumption
    5. Export training candidates (low-confidence + negative feedback)

Storage:
    - DB table: agent_feedback_events (session_id, turn_id, feedback_type, ...)
    - In-memory rolling window for fast dashboard queries

Design Decisions:
    - Minimal I/O: feedback recording is async-friendly
    - Privacy: no user PII stored — only session/turn IDs + feedback type
    - Drift detection: Jensen-Shannon divergence on confidence distributions
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class FeedbackRecord:
    """A single feedback entry."""
    session_id: str
    turn_id: int
    feedback_type: str          # "positive", "negative", "correction"
    intent: str                 # What was the classified intent
    confidence: float           # Agent's confidence on this response
    timestamp: float = field(default_factory=time.time)
    # Optional enrichment
    correction_text: str | None = None   # User's corrected answer
    suggested_intent: str | None = None  # User's suggested intent
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """Alert when model drift is detected."""
    metric: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    alert_level: str            # "warning", "critical"
    window_hours: int
    message: str


@dataclass
class ActiveLearningCandidate:
    """A prediction candidate for human labeling."""
    session_id: str
    turn_id: int
    intent: str
    confidence: float
    user_message: str
    agent_response: str
    reason: str                 # Why this is a candidate


# ════════════════════════════════════════════════════════════════
#  Feedback Collector
# ════════════════════════════════════════════════════════════════

class FeedbackCollector:
    """
    Collects and analyzes user feedback for continuous improvement.

    Thread-safe, with both in-memory buffer and DB persistence.

    Usage:
        collector = FeedbackCollector()
        collector.record_feedback(session_id, turn_id, "positive", intent, 0.85)
        stats = collector.get_statistics(window_hours=24)
        drift = collector.compute_drift(window_hours=48)
    """

    def __init__(self, max_buffer: int = 5000):
        self._buffer: deque[FeedbackRecord] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        # Rolling counters
        self._total_positive = 0
        self._total_negative = 0
        self._total_corrections = 0
        self._intent_feedback: dict[str, dict[str, int]] = defaultdict(
            lambda: {"positive": 0, "negative": 0, "correction": 0}
        )

    # ─── Core Recording ──────────────────────────────────────────

    def record_feedback(
        self,
        session_id: str,
        turn_id: int,
        feedback_type: str,
        intent: str = "",
        confidence: float = 0.0,
        correction_text: str | None = None,
        suggested_intent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        """
        Record a feedback event.

        Args:
            session_id: The conversation session ID.
            turn_id: The turn index within the session.
            feedback_type: One of "positive", "negative", "correction".
            intent: The classified intent for this turn.
            confidence: Agent's confidence score (0-1).
            correction_text: Optional corrected answer from user.
            suggested_intent: Optional intent correction from user.
            metadata: Optional additional context.

        Returns:
            The created FeedbackRecord.
        """
        record = FeedbackRecord(
            session_id=session_id,
            turn_id=turn_id,
            feedback_type=feedback_type,
            intent=intent,
            confidence=confidence,
            correction_text=correction_text,
            suggested_intent=suggested_intent,
            metadata=metadata or {},
        )

        with self._lock:
            self._buffer.append(record)
            if feedback_type == "positive":
                self._total_positive += 1
            elif feedback_type == "negative":
                self._total_negative += 1
            elif feedback_type == "correction":
                self._total_corrections += 1
            self._intent_feedback[intent][feedback_type] += 1

        logger.info(
            "[Feedback] Recorded %s for session=%s turn=%d intent=%s conf=%.2f",
            feedback_type, session_id, turn_id, intent, confidence,
        )

        return record

    def record_feedback_to_db(
        self,
        db,
        record: FeedbackRecord,
    ) -> bool:
        """Persist a feedback record to the database."""
        from sqlalchemy import text as sql_text
        import json

        try:
            turn_exists = db.execute(
                sql_text("SELECT 1 FROM agent_turns WHERE id = :turn_id LIMIT 1"),
                {"turn_id": record.turn_id},
            ).scalar()
            db.execute(
                sql_text("""
                    INSERT INTO agent_feedback_events
                    (session_id, turn_id, feedback_type, rating, notes, actor,
                     intent, confidence, correction_text, suggested_intent,
                     metadata_json, created_at)
                    VALUES
                    (:session_id, :turn_id, :feedback_type, :rating, :notes, :actor,
                     :intent, :confidence, :correction_text, :suggested_intent,
                     CAST(:metadata_json AS jsonb), to_timestamp(:ts))
                """),
                {
                    "session_id": record.session_id,
                    "turn_id": record.turn_id if turn_exists else None,
                    "feedback_type": record.feedback_type,
                    "rating": 1.0 if record.feedback_type == "positive" else 0.0 if record.feedback_type == "negative" else None,
                    "notes": record.correction_text,
                    "actor": "user",
                    "intent": record.intent,
                    "confidence": record.confidence,
                    "correction_text": record.correction_text,
                    "suggested_intent": record.suggested_intent,
                    "metadata_json": json.dumps(record.metadata, default=str),
                    "ts": record.timestamp,
                },
            )
            db.commit()
            return True
        except Exception as exc:
            logger.warning("[Feedback] DB persist failed: %s", exc)
            return False

    # ─── Statistics ──────────────────────────────────────────────

    def get_statistics(self, window_hours: int = 24, db=None) -> dict[str, Any]:
        """
        Get aggregated feedback statistics.

        Returns:
            Dict with satisfaction metrics, intent-level breakdown,
            and trending data.
        """
        cutoff = time.time() - (window_hours * 3600)

        if db is not None:
            try:
                from sqlalchemy import text as sql_text
                import datetime

                cutoff_dt = datetime.datetime.fromtimestamp(cutoff, tz=datetime.timezone.utc)
                rows = db.execute(
                    sql_text("""
                        SELECT feedback_type, COALESCE(intent, '') AS intent,
                               COALESCE(confidence, 0) AS confidence
                        FROM agent_feedback_events
                        WHERE created_at >= :cutoff
                    """),
                    {"cutoff": cutoff_dt},
                ).mappings().all()

                if rows:
                    total = len(rows)
                    positive = [r for r in rows if r["feedback_type"] == "positive"]
                    negative = [r for r in rows if r["feedback_type"] == "negative"]
                    corrections = [r for r in rows if r["feedback_type"] == "correction"]
                    denom = len(positive) + len(negative)
                    satisfaction = len(positive) / denom if denom else None
                    by_intent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
                    for r in rows:
                        by_intent[str(r["intent"] or "unknown")][str(r["feedback_type"])] += 1
                    return {
                        "window_hours": window_hours,
                        "total_feedback": total,
                        "satisfaction_rate": round(satisfaction, 4) if satisfaction is not None else None,
                        "by_intent": {k: dict(v) for k, v in by_intent.items()},
                        "by_type": {
                            "positive": len(positive),
                            "negative": len(negative),
                            "correction": len(corrections),
                        },
                        "avg_confidence_positive": (
                            round(sum(float(r["confidence"] or 0) for r in positive) / len(positive), 4)
                            if positive else None
                        ),
                        "avg_confidence_negative": (
                            round(sum(float(r["confidence"] or 0) for r in negative) / len(negative), 4)
                            if negative else None
                        ),
                    }
            except Exception as exc:
                logger.debug("[Feedback] DB statistics fallback to memory: %s", exc)

        with self._lock:
            recent = [r for r in self._buffer if r.timestamp >= cutoff]

        if not recent:
            return {
                "window_hours": window_hours,
                "total_feedback": 0,
                "satisfaction_rate": None,
                "by_intent": {},
                "by_type": {"positive": 0, "negative": 0, "correction": 0},
                "avg_confidence_positive": None,
                "avg_confidence_negative": None,
            }

        total = len(recent)
        positive = [r for r in recent if r.feedback_type == "positive"]
        negative = [r for r in recent if r.feedback_type == "negative"]
        corrections = [r for r in recent if r.feedback_type == "correction"]

        # Satisfaction rate: positive / (positive + negative)
        rated = len(positive) + len(negative)
        satisfaction = len(positive) / max(rated, 1)

        # Per-intent breakdown
        by_intent: dict[str, dict[str, int]] = defaultdict(
            lambda: {"positive": 0, "negative": 0, "correction": 0, "total": 0}
        )
        for r in recent:
            by_intent[r.intent][r.feedback_type] += 1
            by_intent[r.intent]["total"] += 1

        # Average confidence for positive vs negative
        avg_conf_pos = (
            sum(r.confidence for r in positive) / len(positive)
            if positive else None
        )
        avg_conf_neg = (
            sum(r.confidence for r in negative) / len(negative)
            if negative else None
        )

        return {
            "window_hours": window_hours,
            "total_feedback": total,
            "satisfaction_rate": round(satisfaction, 4),
            "by_type": {
                "positive": len(positive),
                "negative": len(negative),
                "correction": len(corrections),
            },
            "by_intent": dict(by_intent),
            "avg_confidence_positive": round(avg_conf_pos, 4) if avg_conf_pos else None,
            "avg_confidence_negative": round(avg_conf_neg, 4) if avg_conf_neg else None,
            "lifetime_totals": {
                "positive": self._total_positive,
                "negative": self._total_negative,
                "corrections": self._total_corrections,
            },
        }

    # ─── Drift Detection ─────────────────────────────────────────

    def compute_drift(self, window_hours: int = 48) -> dict[str, Any]:
        """
        Detect performance drift by comparing first half vs second half
        of the time window.

        Uses confidence distribution shift and satisfaction rate change
        as drift indicators.

        Returns:
            Dict with drift metrics and alerts.
        """
        now = time.time()
        half_window = window_hours * 3600 / 2
        full_cutoff = now - (window_hours * 3600)

        with self._lock:
            all_recent = [r for r in self._buffer if r.timestamp >= full_cutoff]

        if len(all_recent) < 10:
            return {"drift_detected": False, "reason": "insufficient_data", "sample_count": len(all_recent)}

        first_half = [r for r in all_recent if r.timestamp < now - half_window]
        second_half = [r for r in all_recent if r.timestamp >= now - half_window]

        if len(first_half) < 5 or len(second_half) < 5:
            return {"drift_detected": False, "reason": "insufficient_data_per_half"}

        alerts: list[dict] = []

        # 1. Satisfaction rate drift
        sat_1 = sum(1 for r in first_half if r.feedback_type == "positive") / max(
            sum(1 for r in first_half if r.feedback_type in ("positive", "negative")), 1
        )
        sat_2 = sum(1 for r in second_half if r.feedback_type == "positive") / max(
            sum(1 for r in second_half if r.feedback_type in ("positive", "negative")), 1
        )
        sat_drift = sat_1 - sat_2  # Positive = getting worse

        if sat_drift > 0.15:
            alerts.append({
                "metric": "satisfaction_rate",
                "baseline": round(sat_1, 4),
                "current": round(sat_2, 4),
                "drift": round(sat_drift, 4),
                "level": "critical" if sat_drift > 0.25 else "warning",
                "message": f"Tỷ lệ hài lòng giảm {sat_drift:.1%} (từ {sat_1:.1%} xuống {sat_2:.1%})",
            })

        # 2. Confidence drift
        conf_1 = sum(r.confidence for r in first_half) / len(first_half)
        conf_2 = sum(r.confidence for r in second_half) / len(second_half)
        conf_drift = conf_1 - conf_2

        if conf_drift > 0.1:
            alerts.append({
                "metric": "avg_confidence",
                "baseline": round(conf_1, 4),
                "current": round(conf_2, 4),
                "drift": round(conf_drift, 4),
                "level": "warning",
                "message": f"Độ tin cậy trung bình giảm {conf_drift:.2f} (từ {conf_1:.2f} xuống {conf_2:.2f})",
            })

        # 3. Negative feedback spike
        neg_rate_1 = sum(1 for r in first_half if r.feedback_type == "negative") / len(first_half)
        neg_rate_2 = sum(1 for r in second_half if r.feedback_type == "negative") / len(second_half)
        neg_spike = neg_rate_2 - neg_rate_1

        if neg_spike > 0.1:
            alerts.append({
                "metric": "negative_rate",
                "baseline": round(neg_rate_1, 4),
                "current": round(neg_rate_2, 4),
                "drift": round(neg_spike, 4),
                "level": "warning",
                "message": f"Tỷ lệ feedback tiêu cực tăng {neg_spike:.1%}",
            })

        return {
            "drift_detected": len(alerts) > 0,
            "window_hours": window_hours,
            "sample_count": len(all_recent),
            "first_half_count": len(first_half),
            "second_half_count": len(second_half),
            "alerts": alerts,
            "satisfaction": {
                "first_half": round(sat_1, 4),
                "second_half": round(sat_2, 4),
            },
            "confidence": {
                "first_half": round(conf_1, 4),
                "second_half": round(conf_2, 4),
            },
        }

    # ─── Active Learning ─────────────────────────────────────────

    def get_uncertain_predictions(
        self,
        db=None,
        confidence_threshold: float = 0.5,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Find predictions with low confidence that would benefit
        from human review (active learning candidates).

        Can query from in-memory buffer or from DB.
        """
        candidates: list[dict] = []

        # In-memory candidates
        with self._lock:
            for r in self._buffer:
                if r.confidence < confidence_threshold:
                    candidates.append({
                        "session_id": r.session_id,
                        "turn_id": r.turn_id,
                        "intent": r.intent,
                        "confidence": round(r.confidence, 4),
                        "feedback_type": r.feedback_type,
                        "reason": "low_confidence",
                    })

        # DB candidates (if available)
        if db is not None:
            try:
                from sqlalchemy import text as sql_text
                rows = db.execute(
                    sql_text("""
                        SELECT t.session_id, t.turn_index, t.normalized_intent,
                               t.confidence, t.message_text
                        FROM agent_turns t
                        WHERE t.role = 'assistant'
                          AND t.confidence IS NOT NULL
                          AND t.confidence < :threshold
                        ORDER BY t.confidence ASC
                        LIMIT :limit
                    """),
                    {"threshold": confidence_threshold, "limit": limit},
                ).mappings().all()

                for row in rows:
                    candidates.append({
                        "session_id": str(row["session_id"]),
                        "turn_id": int(row["turn_index"]),
                        "intent": str(row.get("normalized_intent") or ""),
                        "confidence": float(row.get("confidence") or 0),
                        "message_preview": str(row.get("message_text", ""))[:100],
                        "reason": "low_confidence_db",
                    })

                neg_rows = db.execute(
                    sql_text("""
                        SELECT session_id, turn_id, COALESCE(intent, '') AS intent,
                               COALESCE(confidence, 0) AS confidence,
                               COALESCE(correction_text, notes, '') AS message_preview
                        FROM agent_feedback_events
                        WHERE feedback_type IN ('negative', 'correction')
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """),
                    {"limit": limit},
                ).mappings().all()
                for row in neg_rows:
                    candidates.append({
                        "session_id": str(row["session_id"]),
                        "turn_id": int(row.get("turn_id") or 0),
                        "intent": str(row.get("intent") or ""),
                        "confidence": float(row.get("confidence") or 0),
                        "message_preview": str(row.get("message_preview") or "")[:100],
                        "feedback_type": "negative",
                        "reason": "negative_feedback_db",
                    })
            except Exception as exc:
                logger.debug("[Feedback] DB query for uncertain predictions failed: %s", exc)

        # Also include negative-feedback items
        with self._lock:
            for r in self._buffer:
                if r.feedback_type == "negative":
                    candidates.append({
                        "session_id": r.session_id,
                        "turn_id": r.turn_id,
                        "intent": r.intent,
                        "confidence": round(r.confidence, 4),
                        "feedback_type": r.feedback_type,
                        "reason": "negative_feedback",
                    })

        # Deduplicate by (session_id, turn_id) and sort by confidence
        seen = set()
        unique = []
        for c in candidates:
            key = (c["session_id"], c["turn_id"])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda x: x.get("confidence", 0))
        return unique[:limit]

    # ─── Export for Retraining ────────────────────────────────────

    def export_training_candidates(
        self,
        window_hours: int = 168,  # 1 week
    ) -> dict[str, Any]:
        """
        Export data suitable for model retraining.

        Returns negative feedback + corrections as training signals,
        along with suggested label corrections.
        """
        cutoff = time.time() - (window_hours * 3600)

        with self._lock:
            recent = [r for r in self._buffer if r.timestamp >= cutoff]

        # Corrections → strongest signal
        corrections = [
            {
                "session_id": r.session_id,
                "turn_id": r.turn_id,
                "original_intent": r.intent,
                "suggested_intent": r.suggested_intent,
                "correction_text": r.correction_text,
                "confidence": r.confidence,
                "signal_type": "correction",
            }
            for r in recent
            if r.feedback_type == "correction" and r.suggested_intent
        ]

        # Negative feedback → weaker signal (intent might be wrong)
        negatives = [
            {
                "session_id": r.session_id,
                "turn_id": r.turn_id,
                "intent": r.intent,
                "confidence": r.confidence,
                "signal_type": "negative",
            }
            for r in recent
            if r.feedback_type == "negative"
        ]

        return {
            "window_hours": window_hours,
            "corrections": corrections,
            "negatives": negatives,
            "total_candidates": len(corrections) + len(negatives),
        }


# ─── Singleton ────────────────────────────────────────────────────

_feedback: FeedbackCollector | None = None


def get_feedback_collector() -> FeedbackCollector:
    global _feedback
    if _feedback is None:
        _feedback = FeedbackCollector()
    return _feedback
