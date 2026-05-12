"""
tax_agent_memory.py – Conversation Memory Manager (Phase 2)
=============================================================
Manages short-term, working, and long-term conversation memory
for the multi-agent tax intelligence system.

Architecture:
    - Short-term: Sliding window of recent N turns
    - Working Memory: Active entities (MST, tax period, doc type) tracked across turns
    - Long-term: Session summaries compressed via extractive summarization
    - Entity Tracking: Auto-extract & track MST, company names, tax periods

Designed for: Stateless API with DB-backed persistence.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
SLIDING_WINDOW_SIZE = 10       # Keep last N turns in context
MAX_CONTEXT_TOKENS = 2000      # Approximate token budget for context window
ENTITY_MEMORY_LIMIT = 50       # Max entities tracked per session
SNAPSHOT_TTL_SECONDS = 6 * 60 * 60


@dataclass
class TrackedEntity:
    """An entity extracted from conversation."""
    entity_type: str          # "tax_code", "company_name", "tax_period", "doc_type", "amount"
    value: str
    first_seen_turn: int
    last_seen_turn: int
    mention_count: int = 1
    confidence: float = 1.0


@dataclass
class ConversationContext:
    """
    Full conversation context assembled for the current turn.
    Passed to the planner/orchestrator for informed decision-making.
    """
    session_id: str
    current_turn_index: int
    # Recent conversation history
    recent_turns: list[dict[str, Any]]
    # Active entities being discussed
    active_entities: list[TrackedEntity]
    # Extracted key facts
    active_tax_code: str | None = None
    active_tax_period: str | None = None
    active_intent_history: list[str] = field(default_factory=list)
    # Session summary (long-term)
    session_summary: str | None = None
    # Session memory — batch data from uploaded files (persists within session)
    last_batch_data: dict[str, Any] | None = None
    last_vat_snapshot: dict[str, Any] | None = None
    last_attachment_summary: str | None = None
    prior_answer_facts: list[dict[str, Any]] = field(default_factory=list)
    active_artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    # Metadata
    context_token_estimate: int = 0


# ─── Entity Extraction ────────────────────────────────────────────────────────

# Vietnamese tax code pattern: 10 or 13 digits, optionally with dashes
MST_PATTERN = re.compile(
    r"\b(\d{10}(?:-\d{3})?)\b"
)

# Tax period patterns
TAX_PERIOD_PATTERNS = [
    re.compile(r"(?:kỳ|kỳ thuế|kỳ nộp|kỳ kê khai)\s*(\d{4}[/-]?Q[1-4])", re.IGNORECASE),
    re.compile(r"(?:quý|quy)\s*(\d)\s*/?\s*(\d{4})", re.IGNORECASE),
    re.compile(r"(?:tháng|thang)\s*(\d{1,2})\s*/?\s*(\d{4})", re.IGNORECASE),
    re.compile(r"(?:năm|nam)\s*(\d{4})", re.IGNORECASE),
    re.compile(r"(\d{4})[/-]Q([1-4])", re.IGNORECASE),
]

# Amount patterns (Vietnamese currency)
AMOUNT_PATTERN = re.compile(
    r"\b(\d{1,3}(?:[.,]\d{3})*(?:\s*(?:đồng|VND|vnđ|triệu|tỷ|nghìn)))\b",
    re.IGNORECASE,
)


def extract_entities(text: str, turn_index: int = 0) -> list[TrackedEntity]:
    """
    Extract tax-domain entities from a message.

    Extracts:
    - MST (mã số thuế): 10 or 13 digits
    - Tax periods: Q1/2025, tháng 03/2026, năm 2025
    - Amounts: 500 triệu, 1.2 tỷ
    """
    entities: list[TrackedEntity] = []

    # Extract MST
    for match in MST_PATTERN.finditer(text):
        mst = match.group(1)
        # Validate: not a phone number or other numeric sequence
        if len(mst.replace("-", "")) in (10, 13):
            entities.append(TrackedEntity(
                entity_type="tax_code",
                value=mst,
                first_seen_turn=turn_index,
                last_seen_turn=turn_index,
            ))

    # Extract tax periods
    for pattern in TAX_PERIOD_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) == 1:
                period = groups[0]
            elif len(groups) == 2:
                period = f"{groups[1]}-Q{groups[0]}" if len(groups[0]) <= 2 else f"{groups[0]}-{groups[1]}"
            else:
                period = match.group(0)
            entities.append(TrackedEntity(
                entity_type="tax_period",
                value=period.strip(),
                first_seen_turn=turn_index,
                last_seen_turn=turn_index,
            ))

    # Extract amounts
    for match in AMOUNT_PATTERN.finditer(text):
        entities.append(TrackedEntity(
            entity_type="amount",
            value=match.group(1).strip(),
            first_seen_turn=turn_index,
            last_seen_turn=turn_index,
            confidence=0.8,
        ))

    return entities


class ConversationMemory:
    """
    Manages conversation state across turns.
    DB-backed for persistence, with in-memory caching for performance.

    Usage:
        memory = ConversationMemory(db)
        context = memory.build_context(session_id, current_turn)
        # ... agent processes ...
        memory.update_entities(session_id, new_entities)
    """

    def __init__(self, db):
        self.db = db
        self._entity_cache: dict[str, list[TrackedEntity]] = {}
        # Session-scoped cache for batch analysis results (in-memory, lost on restart)
        self._batch_data_cache: dict[str, dict[str, Any]] = {}
        self._vat_snapshot_cache: dict[str, dict[str, Any]] = {}
        self._attachment_summary_cache: dict[str, str] = {}

    def build_context(
        self,
        session_id: str,
        current_turn_index: int,
        current_message: str = "",
        model_mode: str | None = None,
    ) -> ConversationContext:
        """
        Assemble full conversation context for the current turn.

        Retrieves:
        1. Recent N turns (sliding window)
        2. Tracked entities (from DB + current message extraction)
        3. Intent history
        4. Session summary (if available)
        """
        t0 = time.perf_counter()

        # 1. Get recent turns
        recent_turns = self._get_recent_turns(session_id, current_turn_index)

        # 2. Get tracked entities
        stored_entities = self._get_stored_entities(session_id)

        # 3. Extract entities from current message
        new_entities = extract_entities(current_message, current_turn_index)

        # 4. Merge entities (update existing, add new)
        all_entities = self._merge_entities(stored_entities, new_entities, current_turn_index)

        # 5. Determine active tax code (most recently mentioned)
        active_tax_code = None
        tax_codes = [
            e for e in all_entities
            if e.entity_type == "tax_code"
        ]
        if tax_codes:
            active_tax_code = max(tax_codes, key=lambda e: e.last_seen_turn).value

        # 6. Determine active tax period
        active_tax_period = None
        periods = [
            e for e in all_entities
            if e.entity_type == "tax_period"
        ]
        if periods:
            active_tax_period = max(periods, key=lambda e: e.last_seen_turn).value

        # 7. Get intent history
        intent_history = self._get_intent_history(session_id)

        # 8. Get session summary
        session_summary = self._get_session_summary(session_id)
        prior_answer_facts = self.get_prior_answer_facts(session_id, mode=model_mode)
        active_artifact_refs = self._build_active_artifact_refs(session_id)

        # Estimate context token count (rough: 1 token ≈ 3 chars for Vietnamese)
        context_chars = sum(
            len(t.get("message_text", ""))
            for t in recent_turns
        )
        token_estimate = context_chars // 3

        context = ConversationContext(
            session_id=session_id,
            current_turn_index=current_turn_index,
            recent_turns=recent_turns,
            active_entities=all_entities,
            active_tax_code=active_tax_code,
            active_tax_period=active_tax_period,
            active_intent_history=intent_history,
            session_summary=session_summary,
            last_batch_data=self.get_batch_data(session_id),
            last_vat_snapshot=self.get_vat_snapshot(session_id),
            last_attachment_summary=self.get_attachment_summary(session_id),
            prior_answer_facts=prior_answer_facts,
            active_artifact_refs=active_artifact_refs,
            context_token_estimate=token_estimate,
        )

        latency = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "[Memory] Context built for session=%s turn=%d in %.1fms "
            "(turns=%d, entities=%d, tax_code=%s)",
            session_id, current_turn_index, latency,
            len(recent_turns), len(all_entities), active_tax_code,
        )

        return context

    def get_report_context(self, session_id: str) -> dict[str, Any]:
        """Aggregate all session context for report generation."""
        summary = self._get_session_summary(session_id)
        batch = self.get_batch_data(session_id)
        vat = self.get_vat_snapshot(session_id)
        facts = self.get_prior_answer_facts(session_id, limit=50)
        recent_turns = self._get_recent_turns(session_id, 9999)
        
        # Calculate readiness: Do we have enough analytical payload?
        score = len(facts)
        if batch: score += 5
        if vat: score += 5
        
        # Try to find the most recent visualization data in recent turns
        viz_data = {}
        for turn in recent_turns:
            if isinstance(turn.get("visualization_data"), dict) and turn["visualization_data"]:
                viz_data = turn["visualization_data"]
                break

        return {
            "session_summary": summary,
            "batch_data": batch,
            "vat_snapshot": vat,
            "facts": facts,
            "recent_turns": recent_turns,
            "viz_data": viz_data,
            "report_readiness_score": score,
            "is_ready": score >= 3
        }

    def save_batch_data(self, session_id: str, batch_data: dict[str, Any]) -> None:
        """Save batch analysis results in session-scoped memory."""
        self._batch_data_cache[session_id] = batch_data
        self._save_snapshot(session_id, "risk_batch", batch_data)
        logger.debug("[Memory] Batch data saved for session=%s (%d companies)",
                     session_id, len(batch_data.get("companies", [])))

    def save_vat_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        """VAT graph / invoice snapshot for follow-up questions in the same session."""
        self._vat_snapshot_cache[session_id] = snapshot
        self._save_snapshot(session_id, "vat_snapshot", snapshot)
        logger.debug(
            "[Memory] VAT snapshot saved for session=%s (batch_id=%s)",
            session_id, snapshot.get("batch_id"),
        )

    def save_attachment_summary(self, session_id: str, summary: str) -> None:
        """Save a human-readable summary of the last uploaded attachment."""
        self._attachment_summary_cache[session_id] = summary
        self._save_snapshot(session_id, "attachment_summary", {"summary": summary})

    def get_batch_data(self, session_id: str) -> dict[str, Any] | None:
        cached = self._batch_data_cache.get(session_id)
        if cached:
            return cached
        payload = self._load_snapshot(session_id, "risk_batch")
        if isinstance(payload, dict):
            self._batch_data_cache[session_id] = payload
        return payload

    def get_vat_snapshot(self, session_id: str) -> dict[str, Any] | None:
        cached = self._vat_snapshot_cache.get(session_id)
        if cached:
            return cached
        payload = self._load_snapshot(session_id, "vat_snapshot")
        if isinstance(payload, dict):
            self._vat_snapshot_cache[session_id] = payload
        return payload

    def get_attachment_summary(self, session_id: str) -> str | None:
        cached = self._attachment_summary_cache.get(session_id)
        if cached:
            return cached
        payload = self._load_snapshot(session_id, "attachment_summary")
        if isinstance(payload, dict):
            summary = str(payload.get("summary") or "")
            if summary:
                self._attachment_summary_cache[session_id] = summary
                return summary
        return None

    def save_prior_answer_facts(
        self,
        session_id: str,
        *,
        turn_id: int | None,
        mode: str,
        intent: str,
        facts: list[dict[str, Any]],
        ttl_seconds: int = SNAPSHOT_TTL_SECONDS,
    ) -> None:
        """Persist structured facts extracted from assistant answers."""
        if not facts:
            return
        try:
            for fact in facts[:80]:
                claim = str(fact.get("claim_text") or fact.get("claim") or "").strip()
                if not claim:
                    continue
                self.db.execute(
                    sql_text(
                        """
                        INSERT INTO agent_prior_answer_facts
                        (session_id, turn_id, mode, intent, subject_key, fact_type,
                         claim_text, value_json, source_tool, confidence, expires_at)
                        VALUES
                        (:session_id, :turn_id, :mode, :intent, :subject_key, :fact_type,
                         :claim_text, CAST(:value_json AS jsonb), :source_tool,
                         :confidence, NOW() + (:ttl_seconds || ' seconds')::interval)
                        """
                    ),
                    {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "mode": mode or "full",
                        "intent": intent,
                        "subject_key": fact.get("subject_key"),
                        "fact_type": fact.get("fact_type") or "claim",
                        "claim_text": claim[:1000],
                        "value_json": json.dumps(fact.get("value_json") or fact.get("value") or {}, ensure_ascii=False, default=str),
                        "source_tool": fact.get("source_tool"),
                        "confidence": float(fact.get("confidence") or 0.75),
                        "ttl_seconds": int(ttl_seconds),
                    },
                )
            self.db.commit()
        except Exception as exc:
            try:
                self.db.rollback()
            except Exception:
                pass
            logger.debug("[Memory] Prior answer facts persist skipped: %s", exc)

    def get_prior_answer_facts(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        """Retrieve prior facts to ground follow-up questions."""
        try:
            rows = self.db.execute(
                sql_text(
                    """
                    SELECT id, turn_id, mode, intent, subject_key, fact_type,
                           claim_text, value_json, source_tool, confidence, created_at
                    FROM agent_prior_answer_facts
                    WHERE session_id = :session_id
                      AND (:mode IS NULL OR mode = :mode OR mode = 'full')
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY created_at DESC, id DESC
                    LIMIT :limit
                    """
                ),
                {"session_id": session_id, "mode": mode, "limit": int(limit)},
            ).mappings().all()
        except Exception:
            try:
                self.db.rollback()
            except Exception:
                pass
            return []
        facts: list[dict[str, Any]] = []
        for row in rows:
            facts.append({
                "id": int(row.get("id") or 0),
                "turn_id": row.get("turn_id"),
                "mode": row.get("mode"),
                "intent": row.get("intent"),
                "subject_key": row.get("subject_key"),
                "fact_type": row.get("fact_type"),
                "claim_text": row.get("claim_text"),
                "value_json": row.get("value_json") if isinstance(row.get("value_json"), dict) else {},
                "source_tool": row.get("source_tool"),
                "confidence": float(row.get("confidence") or 0.0),
            })
        return facts

    def _build_active_artifact_refs(self, session_id: str) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        batch = self.get_batch_data(session_id)
        if isinstance(batch, dict):
            refs.append({
                "scope": "risk_batch",
                "filename": batch.get("filename"),
                "total": batch.get("total") or len(batch.get("companies") or []),
                "canonical_batch_id": batch.get("canonical_batch_id"),
            })
        vat = self.get_vat_snapshot(session_id)
        if isinstance(vat, dict):
            refs.append({
                "scope": "vat_snapshot",
                "filename": vat.get("filename"),
                "batch_id": vat.get("batch_id"),
                "invoice_count": len(vat.get("top_invoice_risks") or []),
            })
        summary = self.get_attachment_summary(session_id)
        if summary:
            refs.append({"scope": "attachment_summary", "summary": summary[:400]})
        return refs

    def _save_snapshot(self, session_id: str, scope: str, payload: dict[str, Any]) -> None:
        try:
            self.db.execute(
                sql_text(
                    """
                    INSERT INTO agent_session_snapshots (session_id, scope, payload_json, updated_at, expires_at)
                    VALUES (:session_id, :scope, CAST(:payload AS JSONB), NOW(), NOW() + (:ttl_seconds || ' seconds')::interval)
                    ON CONFLICT (session_id, scope) DO UPDATE
                    SET payload_json = EXCLUDED.payload_json,
                        updated_at = NOW(),
                        expires_at = EXCLUDED.expires_at
                    """
                ),
                {
                    "session_id": session_id,
                    "scope": scope,
                    "payload": json.dumps(payload, ensure_ascii=False, default=str),
                    "ttl_seconds": int(SNAPSHOT_TTL_SECONDS),
                },
            )
            self.db.commit()
        except Exception:
            self.db.rollback()

    def _load_snapshot(self, session_id: str, scope: str) -> dict[str, Any] | None:
        try:
            row = self.db.execute(
                sql_text(
                    """
                    SELECT payload_json
                    FROM agent_session_snapshots
                    WHERE session_id = :session_id
                      AND scope = :scope
                      AND (expires_at IS NULL OR expires_at > NOW())
                    LIMIT 1
                    """
                ),
                {"session_id": session_id, "scope": scope},
            ).mappings().first()
        except Exception:
            self.db.rollback()
            return None
        if not row:
            return None
        payload = row.get("payload_json")
        return payload if isinstance(payload, dict) else None

    def _get_recent_turns(
        self,
        session_id: str,
        current_turn_index: int,
    ) -> list[dict[str, Any]]:
        """Get the most recent N turns from the conversation."""
        rows = self.db.execute(
            sql_text("""
                SELECT turn_index, role, message_text, normalized_intent, confidence
                FROM agent_turns
                WHERE session_id = :session_id
                  AND turn_index <= :current_turn
                ORDER BY turn_index DESC
                LIMIT :window_size
            """),
            {
                "session_id": session_id,
                "current_turn": current_turn_index,
                "window_size": SLIDING_WINDOW_SIZE,
            },
        ).mappings().all()

        turns = []
        for row in reversed(list(rows)):  # Reverse to chronological order
            turns.append({
                "turn_index": int(row["turn_index"]),
                "role": str(row["role"]),
                "message_text": str(row.get("message_text") or ""),
                "intent": str(row.get("normalized_intent") or ""),
                "confidence": float(row.get("confidence") or 0.0),
            })

        return turns

    def _get_stored_entities(self, session_id: str) -> list[TrackedEntity]:
        """Get entities stored from previous turns."""
        # Check cache first
        if session_id in self._entity_cache:
            return self._entity_cache[session_id]

        try:
            rows = self.db.execute(
                sql_text("""
                    SELECT entity_type, entity_value, first_seen_turn,
                           last_seen_turn, mention_count, confidence
                    FROM agent_entity_memory
                    WHERE session_id = :session_id
                    ORDER BY last_seen_turn DESC
                    LIMIT :limit
                """),
                {"session_id": session_id, "limit": ENTITY_MEMORY_LIMIT},
            ).mappings().all()
        except Exception:
            # Table might not exist yet
            return []

        entities = []
        for row in rows:
            entities.append(TrackedEntity(
                entity_type=str(row["entity_type"]),
                value=str(row["entity_value"]),
                first_seen_turn=int(row.get("first_seen_turn") or 0),
                last_seen_turn=int(row.get("last_seen_turn") or 0),
                mention_count=int(row.get("mention_count") or 1),
                confidence=float(row.get("confidence") or 1.0),
            ))

        self._entity_cache[session_id] = entities
        return entities

    def _merge_entities(
        self,
        stored: list[TrackedEntity],
        new: list[TrackedEntity],
        current_turn: int,
    ) -> list[TrackedEntity]:
        """Merge stored and new entities, updating existing ones."""
        # Index stored entities by (type, value)
        entity_map: dict[tuple[str, str], TrackedEntity] = {
            (e.entity_type, e.value): e for e in stored
        }

        for ne in new:
            key = (ne.entity_type, ne.value)
            if key in entity_map:
                existing = entity_map[key]
                existing.last_seen_turn = current_turn
                existing.mention_count += 1
            else:
                entity_map[key] = ne

        return list(entity_map.values())

    def _get_intent_history(self, session_id: str) -> list[str]:
        """Get the history of intents for this session."""
        rows = self.db.execute(
            sql_text("""
                SELECT DISTINCT normalized_intent
                FROM agent_turns
                WHERE session_id = :session_id
                  AND role = 'assistant'
                  AND normalized_intent IS NOT NULL
                ORDER BY normalized_intent
            """),
            {"session_id": session_id},
        ).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def _get_session_summary(self, session_id: str) -> str | None:
        """Get session summary (for long-running conversations)."""
        try:
            row = self.db.execute(
                sql_text("""
                    SELECT metadata_json
                    FROM agent_sessions
                    WHERE session_id = :session_id
                """),
                {"session_id": session_id},
            ).mappings().first()

            if row and row.get("metadata_json"):
                meta = row["metadata_json"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                return meta.get("session_summary")
        except Exception:
            pass
        return None

    def persist_entities(
        self,
        session_id: str,
        entities: list[TrackedEntity],
    ) -> None:
        """
        Persist updated entities to DB.
        Uses UPSERT to handle duplicates.
        """
        for entity in entities:
            try:
                self.db.execute(
                    sql_text("""
                        INSERT INTO agent_entity_memory
                        (session_id, entity_type, entity_value, first_seen_turn,
                         last_seen_turn, mention_count, confidence)
                        VALUES (:session_id, :entity_type, :entity_value,
                                :first_seen_turn, :last_seen_turn, :mention_count, :confidence)
                        ON CONFLICT (session_id, entity_type, entity_value)
                        DO UPDATE SET
                            last_seen_turn = GREATEST(agent_entity_memory.last_seen_turn, EXCLUDED.last_seen_turn),
                            mention_count = agent_entity_memory.mention_count + 1,
                            confidence = GREATEST(agent_entity_memory.confidence, EXCLUDED.confidence)
                    """),
                    {
                        "session_id": session_id,
                        "entity_type": entity.entity_type,
                        "entity_value": entity.value,
                        "first_seen_turn": entity.first_seen_turn,
                        "last_seen_turn": entity.last_seen_turn,
                        "mention_count": entity.mention_count,
                        "confidence": entity.confidence,
                    },
                )
            except Exception as exc:
                # Graceful fallback if table doesn't exist
                logger.debug("[Memory] Entity persist skipped: %s", exc)
                break

        # Update cache
        self._entity_cache[session_id] = entities

    def format_context_for_prompt(self, context: ConversationContext) -> str:
        """
        Format conversation context into a text block for use in planning prompts.
        """
        parts = []

        # Session info
        parts.append(f"[Session: {context.session_id}]")

        # Active entities
        if context.active_tax_code:
            parts.append(f"[MST đang xử lý: {context.active_tax_code}]")
        if context.active_tax_period:
            parts.append(f"[Kỳ thuế: {context.active_tax_period}]")

        # Entity summary
        other_entities = [
            e for e in context.active_entities
            if e.entity_type not in ("tax_code", "tax_period")
        ]
        if other_entities:
            entity_strs = [f"{e.entity_type}={e.value}" for e in other_entities[:5]]
            parts.append(f"[Entities: {', '.join(entity_strs)}]")

        # Intent history
        if context.active_intent_history:
            parts.append(f"[Intents đã xử lý: {', '.join(context.active_intent_history[-5:])}]")

        # Session summary
        if context.session_summary:
            parts.append(f"[Tóm tắt session: {context.session_summary[:200]}]")

        if context.prior_answer_facts:
            parts.append("\n--- Facts tu cau tra loi truoc ---")
            for fact in context.prior_answer_facts[:6]:
                parts.append(f"- {str(fact.get('claim_text') or '')[:180]}")

        if context.active_artifact_refs:
            refs = [
                f"{ref.get('scope')}:{ref.get('filename') or ref.get('batch_id') or ref.get('summary', '')}"
                for ref in context.active_artifact_refs[:4]
            ]
            parts.append(f"[Artifact refs: {', '.join(refs)}]")

        # Recent conversation (abbreviated)
        if context.recent_turns:
            parts.append("\n--- Lịch sử gần đây ---")
            for turn in context.recent_turns[-4:]:
                role = "🧑 User" if turn["role"] == "user" else "🤖 Agent"
                msg = turn["message_text"][:150]
                parts.append(f"{role}: {msg}")

        return "\n".join(parts)
