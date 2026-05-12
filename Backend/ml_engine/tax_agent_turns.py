"""Durable turn allocation helpers for multi-agent chat sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllocatedTurnPair:
    user_turn_id: int
    assistant_turn_id: int
    user_turn_index: int
    assistant_turn_index: int


class AgentTurnRepository:
    """
    Allocate user/assistant turn pairs under a session-level lock.

    PostgreSQL uses `FOR UPDATE` on the owning session row. SQLite/dev and mock
    sessions fall back to best-effort max+insert with a short retry loop.
    """

    def __init__(self, db):
        self.db = db

    def allocate_turn_pair(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_placeholder: str = "",
        max_retries: int = 3,
    ) -> AllocatedTurnPair:
        last_exc: Exception | None = None
        for _attempt in range(max(1, max_retries)):
            try:
                self._lock_session(session_id)
                row = self.db.execute(
                    sql_text(
                        "SELECT COALESCE(MAX(turn_index), 0) FROM agent_turns "
                        "WHERE session_id = :session_id"
                    ),
                    {"session_id": session_id},
                ).fetchone()
                base = int(row[0] or 0) if row else 0
                user_index = base + 1
                assistant_index = base + 2
                user_id = self._insert_turn(
                    session_id=session_id,
                    turn_index=user_index,
                    role="user",
                    message_text=user_message,
                )
                assistant_id = self._insert_turn(
                    session_id=session_id,
                    turn_index=assistant_index,
                    role="assistant",
                    message_text=assistant_placeholder,
                )
                return AllocatedTurnPair(
                    user_turn_id=user_id,
                    assistant_turn_id=assistant_id,
                    user_turn_index=user_index,
                    assistant_turn_index=assistant_index,
                )
            except Exception as exc:
                last_exc = exc
                try:
                    self.db.rollback()
                except Exception:
                    pass
                logger.debug("[AgentTurnRepository] allocate retry after failure: %s", exc)
        raise RuntimeError(f"Could not allocate agent turn pair: {last_exc}") from last_exc

    def update_assistant_turn(
        self,
        *,
        turn_id: int,
        message_text: str,
        normalized_intent: str | None = None,
        confidence: float | None = None,
        citations_json: str = "[]",
    ) -> None:
        self.db.execute(
            sql_text(
                """
                UPDATE agent_turns
                SET message_text = :message_text,
                    normalized_intent = :normalized_intent,
                    confidence = :confidence,
                    citations_json = CAST(:citations_json AS jsonb)
                WHERE id = :turn_id
                """
            ),
            {
                "turn_id": turn_id,
                "message_text": message_text,
                "normalized_intent": normalized_intent,
                "confidence": confidence,
                "citations_json": citations_json,
            },
        )

    def _lock_session(self, session_id: str) -> None:
        try:
            bind = self.db.get_bind() if hasattr(self.db, "get_bind") else getattr(self.db, "bind", None)
            dialect = getattr(bind, "dialect", None)
            dialect_name = getattr(dialect, "name", "")
        except Exception:
            dialect_name = ""
        if dialect_name != "postgresql":
            return
        try:
            self.db.execute(
                sql_text(
                    "SELECT session_id FROM agent_sessions "
                    "WHERE session_id = :session_id FOR UPDATE"
                ),
                {"session_id": session_id},
            ).fetchone()
        except Exception:
            # Some test doubles or development DBs do not support row locks.
            try:
                self.db.rollback()
            except Exception:
                pass

    def _insert_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        role: str,
        message_text: str,
    ) -> int:
        params = {
            "session_id": session_id,
            "turn_index": turn_index,
            "role": role,
            "message_text": message_text,
        }
        try:
            bind = self.db.get_bind() if hasattr(self.db, "get_bind") else getattr(self.db, "bind", None)
            dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
        except Exception:
            dialect_name = ""
        if dialect_name == "postgresql":
            row = self.db.execute(
                sql_text(
                    """
                    INSERT INTO agent_turns (session_id, turn_index, role, message_text)
                    VALUES (:session_id, :turn_index, :role, :message_text)
                    RETURNING id
                    """
                ),
                params,
            ).fetchone()
            if row:
                return int(row[0])
        else:
            self.db.execute(
                sql_text(
                    """
                    INSERT INTO agent_turns (session_id, turn_index, role, message_text)
                    VALUES (:session_id, :turn_index, :role, :message_text)
                    """
                ),
                params,
            )
        fallback = self.db.execute(
            sql_text(
                """
                SELECT id FROM agent_turns
                WHERE session_id = :session_id AND turn_index = :turn_index AND role = :role
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"session_id": session_id, "turn_index": turn_index, "role": role},
        ).fetchone()
        if not fallback:
            raise RuntimeError("Could not retrieve inserted agent_turn id")
        return int(fallback[0])
