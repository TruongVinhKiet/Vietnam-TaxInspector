"""
tax_agent_telemetry.py – Telemetry & Monitoring Engine (Phase 5)
=================================================================
Real-time metrics collection, aggregation, and dashboard data provider
for the multi-agent tax intelligence system.

Capabilities:
    1. Per-request latency tracking (breakdown by component)
    2. Quality metrics aggregation (MRR, NDCG, confidence)
    3. Tool usage & error rate monitoring
    4. Intent distribution & drift detection
    5. Session analytics (turns per session, completion rates)
    6. System health monitoring (embedding tier, memory usage)
    7. Alerting thresholds for degradation detection
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single orchestrator request."""
    session_id: str
    timestamp: float
    intent: str
    complexity: str
    # Latency
    total_latency_ms: float
    latency_breakdown: dict[str, float]
    # Quality
    intent_confidence: float
    synthesis_confidence: float
    retrieval_hits: int
    evidence_count: int
    # Tools
    tools_invoked: int
    tools_succeeded: int
    tools_failed: int
    # Compliance
    abstained: bool
    escalated: bool
    compliance_decision: str
    warnings_count: int
    # Sub-agents
    sub_agents_used: list[str] = field(default_factory=list)
    # Classification
    intent_source: str = "unknown"


@dataclass
class AlertConfig:
    """Configuration for a monitoring alert."""
    name: str
    metric: str
    threshold: float
    direction: str  # "above" or "below"
    window_minutes: int = 5
    min_samples: int = 3


@dataclass
class Alert:
    """A triggered alert."""
    config: AlertConfig
    current_value: float
    triggered_at: float
    message: str


# ─── Default Alert Configurations ─────────────────────────────────────────────

DEFAULT_ALERTS = [
    AlertConfig("high_latency", "avg_latency_ms", 5000.0, "above", 5, 3),
    AlertConfig("low_confidence", "avg_confidence", 0.25, "below", 10, 5),
    AlertConfig("high_abstain_rate", "abstain_rate", 0.5, "above", 10, 5),
    AlertConfig("high_tool_error_rate", "tool_error_rate", 0.3, "above", 5, 3),
    AlertConfig("low_retrieval", "avg_retrieval_hits", 1.0, "below", 10, 5),
]


class TelemetryEngine:
    """
    Real-time telemetry engine for agent monitoring.

    Thread-safe, in-memory metrics with periodic DB flush.
    Provides aggregated dashboard data via get_dashboard().

    Usage:
        telemetry = TelemetryEngine()
        telemetry.record(metrics)
        dashboard = telemetry.get_dashboard()
    """

    def __init__(self, max_buffer: int = 10000, db=None):
        self._buffer: deque[RequestMetrics] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._db = db
        self._alerts: list[AlertConfig] = list(DEFAULT_ALERTS)
        self._active_alerts: list[Alert] = []
        # Counters for fast access
        self._total_requests = 0
        self._total_abstains = 0
        self._total_escalations = 0
        self._intent_counts: dict[str, int] = defaultdict(int)
        self._tool_calls: dict[str, int] = defaultdict(int)
        self._tool_errors: dict[str, int] = defaultdict(int)
        self._hourly_counts: dict[str, int] = defaultdict(int)

    def record(self, metrics: RequestMetrics) -> None:
        """Record metrics for a single request."""
        with self._lock:
            self._buffer.append(metrics)
            self._total_requests += 1
            if metrics.abstained:
                self._total_abstains += 1
            if metrics.escalated:
                self._total_escalations += 1
            self._intent_counts[metrics.intent] += 1

            # Track hourly distribution
            hour_key = datetime.fromtimestamp(metrics.timestamp).strftime("%Y-%m-%d-%H")
            self._hourly_counts[hour_key] += 1

        # Check alerts (non-blocking)
        self._check_alerts()

    def record_from_orchestrator(self, orch_response) -> None:
        """Convenience: record from OrchestratorResponse."""
        try:
            metrics = RequestMetrics(
                session_id=orch_response.session_id,
                timestamp=time.time(),
                intent=orch_response.intent,
                complexity=orch_response.complexity,
                total_latency_ms=orch_response.latency_ms,
                latency_breakdown=orch_response.latency_breakdown,
                intent_confidence=orch_response.intent_confidence,
                synthesis_confidence=orch_response.confidence,
                retrieval_hits=len(orch_response.citations),
                evidence_count=len(orch_response.citations),
                tools_invoked=len(orch_response.tools_used),
                tools_succeeded=sum(
                    1 for r in orch_response.tool_results.values()
                    if r.get("status") not in ("error",)
                ),
                tools_failed=sum(
                    1 for r in orch_response.tool_results.values()
                    if r.get("status") == "error"
                ),
                abstained=orch_response.abstained,
                escalated=orch_response.escalation_required,
                compliance_decision="allow" if not orch_response.abstained else "block",
                warnings_count=len(orch_response.compliance_warnings),
                intent_source=orch_response.synthesis_tier,
            )
            self.record(metrics)
        except Exception as exc:
            logger.debug("[Telemetry] record_from_orchestrator failed: %s", exc)

    def get_dashboard(self, window_minutes: int = 60, db=None) -> dict[str, Any]:
        """
        Build dashboard data for the monitoring UI.
        Returns aggregated metrics over the specified time window.
        """
        if db is not None:
            try:
                return self._get_dashboard_from_db(window_minutes=window_minutes, db=db)
            except Exception as exc:
                logger.debug("[Telemetry] DB dashboard fallback to memory: %s", exc)

        cutoff = 0 if window_minutes <= 0 else time.time() - (window_minutes * 60)
        with self._lock:
            recent = [m for m in self._buffer if m.timestamp >= cutoff]

        return self._build_dashboard_from_metrics(
            recent,
            window_minutes=window_minutes,
            source="memory",
            empty_reason="no_in_memory_telemetry",
        )

    def _get_dashboard_from_db(self, *, window_minutes: int, db) -> dict[str, Any]:
        from sqlalchemy import text as sql_text
        import datetime
        import json as _json

        all_time = window_minutes <= 0
        cutoff_ts = 0 if all_time else time.time() - (window_minutes * 60)
        cutoff_dt = datetime.datetime.fromtimestamp(cutoff_ts, tz=datetime.timezone.utc)
        where_clause = "" if all_time else "WHERE dt.created_at >= :cutoff"
        params = {} if all_time else {"cutoff": cutoff_dt}

        rows = db.execute(
            sql_text(f"""
                SELECT
                    dt.session_id,
                    dt.intent,
                    dt.selected_track,
                    dt.confidence,
                    dt.abstained,
                    dt.escalation_required,
                    dt.evidence_json,
                    dt.created_at,
                    COALESCE(ep.latency_ms, aq.total_latency_ms, 0) AS latency_ms,
                    COALESCE(ep.latency_breakdown, '{{}}'::jsonb) AS latency_breakdown,
                    COALESCE(aq.retrieval_hits, 0) AS retrieval_hits,
                    COALESCE(aq.evidence_count, 0) AS evidence_count,
                    COALESCE(aq.tools_invoked, 0) AS tools_invoked,
                    COALESCE(aq.tools_succeeded, 0) AS tools_succeeded,
                    COALESCE(aq.tools_failed, 0) AS tools_failed,
                    COALESCE(ar.dialogue_act, 'task') AS dialogue_act,
                    COALESCE(ar.answer_contract, 'risk_profile') AS answer_contract,
                    COALESCE(ar.focus_score, 1.0) AS focus_score,
                    COALESCE(ar.route_violation, FALSE) AS route_violation
                FROM agent_decision_traces dt
                LEFT JOIN agent_execution_plans ep ON ep.turn_id = dt.turn_id
                LEFT JOIN agent_quality_metrics aq
                  ON aq.session_id = dt.session_id
                 AND ABS(EXTRACT(EPOCH FROM (aq.created_at - dt.created_at))) <= 10
                LEFT JOIN agent_route_events ar ON ar.turn_id = dt.turn_id
                {where_clause}
                ORDER BY dt.created_at ASC
            """),
            params,
        ).mappings().all()

        metrics: list[RequestMetrics] = []
        route_rows: list[dict[str, Any]] = []
        react_retry_counts: dict[int, int] = defaultdict(int)
        debate_count = 0
        debate_escalations = 0
        debate_adjudicator_count = 0

        for row in rows:
            created_at = row["created_at"]
            ts = created_at.timestamp() if hasattr(created_at, "timestamp") else time.time()
            latency_breakdown = row.get("latency_breakdown") or {}
            if isinstance(latency_breakdown, str):
                try:
                    latency_breakdown = _json.loads(latency_breakdown)
                except Exception:
                    latency_breakdown = {}

            evidence_json = row.get("evidence_json") or {}
            if isinstance(evidence_json, str):
                try:
                    evidence_json = _json.loads(evidence_json)
                except Exception:
                    evidence_json = {}
            react_items = evidence_json.get("react") or []
            if react_items:
                bucket_key = int(ts // 60) * 60
                react_retry_counts[bucket_key] += sum(1 for item in react_items if item.get("should_retry"))
            debate = evidence_json.get("debate")
            if debate:
                debate_count += 1
                if float(debate.get("consensus_score", 1.0) or 1.0) < 0.58:
                    debate_escalations += 1
                if debate.get("adjudicator_verdict"):
                    debate_adjudicator_count += 1

            metrics.append(RequestMetrics(
                session_id=str(row["session_id"]),
                timestamp=ts,
                intent=str(row.get("intent") or "general_tax_query"),
                complexity=str(row.get("selected_track") or "unknown"),
                total_latency_ms=float(row.get("latency_ms") or 0.0),
                latency_breakdown=dict(latency_breakdown or {}),
                intent_confidence=float(row.get("confidence") or 0.0),
                synthesis_confidence=float(row.get("confidence") or 0.0),
                retrieval_hits=int(row.get("retrieval_hits") or 0),
                evidence_count=int(row.get("evidence_count") or 0),
                tools_invoked=int(row.get("tools_invoked") or 0),
                tools_succeeded=int(row.get("tools_succeeded") or 0),
                tools_failed=int(row.get("tools_failed") or 0),
                abstained=bool(row.get("abstained")),
                escalated=bool(row.get("escalation_required")),
                compliance_decision="block" if row.get("abstained") else "allow",
                warnings_count=0,
            ))
            route_rows.append({
                "timestamp": ts,
                "dialogue_act": str(row.get("dialogue_act") or "task"),
                "answer_contract": str(row.get("answer_contract") or "risk_profile"),
                "focus_score": float(row.get("focus_score") or 1.0),
                "route_violation": bool(row.get("route_violation")),
            })

        dashboard = self._build_dashboard_from_metrics(
            metrics,
            window_minutes=window_minutes,
            source="db",
            empty_reason="no_db_telemetry_in_window",
        )
        dashboard["all_time"] = all_time

        route_dist: dict[str, int] = defaultdict(int)
        dialogue_dist: dict[str, int] = defaultdict(int)
        violations = 0
        focus_scores = []
        for r in route_rows:
            route_dist[r["answer_contract"]] += 1
            dialogue_dist[r["dialogue_act"]] += 1
            violations += 1 if r["route_violation"] else 0
            focus_scores.append(r["focus_score"])

        dashboard["answer_contract_distribution"] = dict(route_dist)
        dashboard["dialogue_act_distribution"] = dict(dialogue_dist)
        dashboard["focus_violations"] = {
            "count": violations,
            "rate": round(violations / max(len(route_rows), 1), 4) if route_rows else 0,
        }
        dashboard["route_quality"] = {
            "avg_focus_score": round(sum(focus_scores) / len(focus_scores), 4) if focus_scores else 1.0,
            "violations": violations,
        }

        dashboard["tool_usage"] = self._query_tool_usage(db, cutoff_dt=cutoff_dt, all_time=all_time)
        dashboard["tools"] = self._legacy_tool_payload(dashboard["tool_usage"])
        dashboard["retrieval_quality"] = self._query_retrieval_quality(db, cutoff_dt=cutoff_dt, all_time=all_time)
        dashboard["legal_faithfulness"] = self._query_legal_faithfulness(db, cutoff_dt=cutoff_dt, all_time=all_time)
        dashboard["active_learning_summary"] = self._query_active_learning_summary(db, cutoff_dt=cutoff_dt, all_time=all_time)
        dashboard["debate_metrics"] = {
            "total": debate_count,
            "escalations": debate_escalations,
            "escalation_rate": round(debate_escalations / max(debate_count, 1), 4) if debate_count else 0,
            "adjudicator_triggered": debate_adjudicator_count,
            "adjudicator_rate": round(debate_adjudicator_count / max(debate_count, 1), 4) if debate_count else 0,
        }
        dashboard["react_retry_trend"] = [
            {"timestamp": ts, "retry_count": count}
            for ts, count in sorted(react_retry_counts.items())
        ]
        dashboard["dpo_status"] = self._get_dpo_status()
        return dashboard

    def _build_dashboard_from_metrics(
        self,
        metrics: list[RequestMetrics],
        *,
        window_minutes: int,
        source: str,
        empty_reason: str,
    ) -> dict[str, Any]:
        if not metrics:
            empty = self._empty_dashboard()
            empty.update({
                "window_minutes": window_minutes,
                "source": source,
                "empty_reason": empty_reason,
                "timeline": [],
                "intents": {},
                "intent_distribution": {},
                "tool_usage": [],
            })
            return empty

        n = len(metrics)
        latencies = [float(m.total_latency_ms or 0) for m in metrics]
        latency_sorted = sorted(latencies)
        intent_confs = [float(m.intent_confidence or 0) for m in metrics]
        synth_confs = [float(m.synthesis_confidence or 0) for m in metrics]
        retrieval_hits = [int(m.retrieval_hits or 0) for m in metrics]

        intent_dist: dict[str, int] = defaultdict(int)
        complexity_dist: dict[str, int] = defaultdict(int)
        component_latencies: dict[str, list[float]] = defaultdict(list)
        for m in metrics:
            intent_dist[m.intent] += 1
            if m.complexity != "unknown":
                complexity_dist[m.complexity] += 1
            for comp, lat in (m.latency_breakdown or {}).items():
                try:
                    component_latencies[comp].append(float(lat or 0))
                except Exception:
                    pass

        avg_component = {
            comp: round(sum(lats) / len(lats), 1)
            for comp, lats in component_latencies.items()
            if lats
        }

        timeline = self._build_timeline(metrics, window_minutes=window_minutes)
        confidence_hist = self._confidence_histogram(synth_confs)

        effective_minutes = max(window_minutes, 1)
        if window_minutes <= 0:
            span_seconds = max(m.timestamp for m in metrics) - min(m.timestamp for m in metrics)
            effective_minutes = max(1, int(span_seconds / 60))

        tools_invoked = sum(m.tools_invoked for m in metrics)
        tools_succeeded = sum(m.tools_succeeded for m in metrics)
        tools_failed = sum(m.tools_failed for m in metrics)

        return {
            "window_minutes": window_minutes,
            "source": source,
            "total_requests": n,
            "requests_per_minute": round(n / effective_minutes, 2),
            "latency": {
                "avg_ms": round(sum(latencies) / n, 1),
                "p50_ms": round(self._percentile(latency_sorted, 0.50), 1),
                "p95_ms": round(self._percentile(latency_sorted, 0.95), 1),
                "p99_ms": round(self._percentile(latency_sorted, 0.99), 1),
                "max_ms": round(max(latencies), 1),
                "min_ms": round(min(latencies), 1),
                "by_component": avg_component,
            },
            "latency_percentiles": {
                "p50_ms": round(self._percentile(latency_sorted, 0.50), 1),
                "p90_ms": round(self._percentile(latency_sorted, 0.90), 1),
                "p95_ms": round(self._percentile(latency_sorted, 0.95), 1),
                "p99_ms": round(self._percentile(latency_sorted, 0.99), 1),
            },
            "quality": {
                "avg_intent_confidence": round(sum(intent_confs) / n, 4),
                "avg_synthesis_confidence": round(sum(synth_confs) / n, 4),
                "avg_retrieval_hits": round(sum(retrieval_hits) / n, 2),
                "abstain_rate": round(sum(1 for m in metrics if m.abstained) / n, 4),
                "escalation_rate": round(sum(1 for m in metrics if m.escalated) / n, 4),
            },
            "confidence_histogram": confidence_hist,
            "intent_distribution": dict(intent_dist),
            "intents": dict(intent_dist),
            "complexity_distribution": dict(complexity_dist),
            "tools": {
                "total_invoked": tools_invoked,
                "total_succeeded": tools_succeeded,
                "total_failed": tools_failed,
            },
            "tool_usage": [],
            "timeline": timeline,
            "empty_reason": None,
            # G7: Debate & ReAct metrics (in-memory — limited data)
            "debate_metrics": {
                "total": 0,
                "escalations": 0,
                "escalation_rate": 0,
                "adjudicator_triggered": 0,
            },
            "react_retry_trend": [],
            # G8: DPO pipeline status
            "dpo_status": self._get_dpo_status(),
        }

    def _build_timeline(self, metrics: list[RequestMetrics], *, window_minutes: int) -> list[dict[str, Any]]:
        if not metrics:
            return []
        min_ts = min(m.timestamp for m in metrics)
        max_ts = max(m.timestamp for m in metrics)
        if window_minutes <= 15:
            bucket_seconds = 60
        elif window_minutes <= 60:
            bucket_seconds = 300
        elif window_minutes <= 1440:
            bucket_seconds = 1800
        else:
            bucket_seconds = 3600 if (max_ts - min_ts) <= 7 * 86400 else 86400

        start = int(min_ts // bucket_seconds) * bucket_seconds
        end = int(max_ts // bucket_seconds) * bucket_seconds
        buckets: dict[int, list[RequestMetrics]] = defaultdict(list)
        for m in metrics:
            buckets[int(m.timestamp // bucket_seconds) * bucket_seconds].append(m)

        timeline = []
        ts = start
        while ts <= end:
            bucket = buckets.get(ts, [])
            lats = sorted(float(m.total_latency_ms or 0) for m in bucket)
            timeline.append({
                "timestamp": ts,
                "count": len(bucket),
                "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
                "p95_latency_ms": round(self._percentile(lats, 0.95), 1) if lats else 0,
                "abstain_count": sum(1 for m in bucket if m.abstained),
                "escalation_count": sum(1 for m in bucket if m.escalated),
            })
            ts += bucket_seconds
        return timeline

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
        return float(values[idx])

    @staticmethod
    def _confidence_histogram(values: list[float]) -> list[dict[str, Any]]:
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        hist = []
        for label, (lo, hi) in zip(labels, bins):
            hist.append({"bucket": label, "count": sum(1 for v in values if lo <= float(v or 0) < hi)})
        return hist

    def _query_tool_usage(self, db, *, cutoff_dt, all_time: bool) -> list[dict[str, Any]]:
        from sqlalchemy import text as sql_text
        where = "" if all_time else "WHERE created_at >= :cutoff"
        rows = db.execute(sql_text(f"""
            SELECT tool_name,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status IN ('success', 'ok', 'found', 'analyzed') THEN 1 ELSE 0 END) AS success,
                   SUM(CASE WHEN status IN ('error', 'timeout') THEN 1 ELSE 0 END) AS failed,
                   AVG(COALESCE(latency_ms, 0)) AS avg_latency_ms
            FROM agent_tool_calls
            {where}
            GROUP BY tool_name
            ORDER BY total DESC
            LIMIT 20
        """), {} if all_time else {"cutoff": cutoff_dt}).mappings().all()
        return [
            {
                "tool": str(r["tool_name"]),
                "total": int(r["total"] or 0),
                "success": int(r["success"] or 0),
                "failed": int(r["failed"] or 0),
                "avg_latency_ms": round(float(r["avg_latency_ms"] or 0), 1),
            }
            for r in rows
        ]

    @staticmethod
    def _legacy_tool_payload(tool_usage: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "total_invoked": sum(int(t.get("total") or 0) for t in tool_usage),
            "total_succeeded": sum(int(t.get("success") or 0) for t in tool_usage),
            "total_failed": sum(int(t.get("failed") or 0) for t in tool_usage),
        }
        for item in tool_usage[:10]:
            payload[str(item.get("tool"))] = int(item.get("total") or 0)
        return payload

    def _query_retrieval_quality(self, db, *, cutoff_dt, all_time: bool) -> dict[str, Any]:
        from sqlalchemy import text as sql_text
        where = "" if all_time else "WHERE created_at >= :cutoff"
        row = db.execute(sql_text(f"""
            SELECT COUNT(*) AS total,
                   AVG(COALESCE(latency_ms, 0)) AS avg_latency_ms,
                   AVG(COALESCE(candidate_count, jsonb_array_length(COALESCE(retrieved_chunks, '[]'::jsonb)))) AS avg_candidates,
                   AVG(COALESCE(top_k, 0)) AS avg_top_k
            FROM retrieval_logs
            {where}
        """), {} if all_time else {"cutoff": cutoff_dt}).mappings().first()
        return {
            "total_queries": int(row["total"] or 0) if row else 0,
            "avg_latency_ms": round(float(row["avg_latency_ms"] or 0), 1) if row else 0,
            "avg_candidates": round(float(row["avg_candidates"] or 0), 2) if row else 0,
            "avg_top_k": round(float(row["avg_top_k"] or 0), 2) if row else 0,
        }

    def _query_legal_faithfulness(self, db, *, cutoff_dt, all_time: bool) -> dict[str, Any]:
        from sqlalchemy import text as sql_text
        where = "" if all_time else "WHERE created_at >= :cutoff"
        row = db.execute(sql_text(f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status = 'supported' OR support_score >= 0.65 THEN 1 ELSE 0 END) AS supported,
                   SUM(CASE WHEN status <> 'supported' AND support_score < 0.65 THEN 1 ELSE 0 END) AS unsupported,
                   AVG(COALESCE(support_score, 0)) AS avg_confidence
            FROM legal_claim_verifications
            {where}
        """), {} if all_time else {"cutoff": cutoff_dt}).mappings().first()
        total = int(row["total"] or 0) if row else 0
        supported = int(row["supported"] or 0) if row else 0
        unsupported = int(row["unsupported"] or 0) if row else 0
        return {
            "total_claims": total,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "unsupported_rate": round(unsupported / max(total, 1), 4) if total else 0,
            "avg_confidence": round(float(row["avg_confidence"] or 0), 4) if row else 0,
        }

    def _query_active_learning_summary(self, db, *, cutoff_dt, all_time: bool) -> dict[str, Any]:
        from sqlalchemy import text as sql_text
        where_turn = "" if all_time else "WHERE created_at >= :cutoff"
        where_feedback = "" if all_time else "WHERE created_at >= :cutoff"
        low_conf = db.execute(sql_text(f"""
            SELECT COUNT(*) AS total
            FROM agent_turns
            {where_turn}
              {"AND" if not all_time else "WHERE"} role = 'assistant'
              AND confidence IS NOT NULL
              AND confidence < 0.5
        """), {} if all_time else {"cutoff": cutoff_dt}).mappings().first()
        neg = db.execute(sql_text(f"""
            SELECT COUNT(*) AS total
            FROM agent_feedback_events
            {where_feedback}
              {"AND" if not all_time else "WHERE"} feedback_type IN ('negative', 'correction')
        """), {} if all_time else {"cutoff": cutoff_dt}).mappings().first()
        return {
            "low_confidence_turns": int((low_conf or {}).get("total") or 0),
            "negative_or_correction_feedback": int((neg or {}).get("total") or 0),
        }

    def get_intent_drift(self, window_hours: int = 24, db=None) -> dict[str, Any]:
        """Detect intent distribution drift over time."""
        now = time.time()
        half = window_hours * 3600 / 2

        if db is not None:
            try:
                from sqlalchemy import text as sql_text
                import datetime

                start_dt = datetime.datetime.fromtimestamp(now - window_hours * 3600, tz=datetime.timezone.utc)
                split_dt = datetime.datetime.fromtimestamp(now - half, tz=datetime.timezone.utc)
                rows = db.execute(sql_text("""
                    SELECT intent,
                           CASE WHEN created_at < :split THEN 'first' ELSE 'second' END AS half,
                           COUNT(*) AS total
                    FROM agent_decision_traces
                    WHERE created_at >= :start
                    GROUP BY intent, half
                """), {"start": start_dt, "split": split_dt}).mappings().all()
                dist_1: dict[str, float] = defaultdict(float)
                dist_2: dict[str, float] = defaultdict(float)
                for row in rows:
                    if row["half"] == "first":
                        dist_1[str(row["intent"] or "unknown")] += float(row["total"] or 0)
                    else:
                        dist_2[str(row["intent"] or "unknown")] += float(row["total"] or 0)
                return self._build_drift_payload(dist_1, dist_2)
            except Exception as exc:
                logger.debug("[Telemetry] DB drift fallback to memory: %s", exc)

        with self._lock:
            first_half = [m for m in self._buffer if now - m.timestamp >= half]
            second_half = [m for m in self._buffer if now - m.timestamp < half]

        if not first_half or not second_half:
            return {"drift_detected": False, "reason": "insufficient_data"}

        dist_1: dict[str, float] = defaultdict(float)
        dist_2: dict[str, float] = defaultdict(float)

        for m in first_half:
            dist_1[m.intent] += 1
        for m in second_half:
            dist_2[m.intent] += 1

        return self._build_drift_payload(dist_1, dist_2)

    def _build_drift_payload(self, dist_1: dict[str, float], dist_2: dict[str, float]) -> dict[str, Any]:
        if not dist_1 or not dist_2:
            return {"drift_detected": False, "reason": "insufficient_data"}

        # Normalize
        total_1 = max(sum(dist_1.values()), 1)
        total_2 = max(sum(dist_2.values()), 1)
        all_intents = set(dist_1.keys()) | set(dist_2.keys())

        # Jensen-Shannon-like divergence (simplified)
        divergence = 0.0
        for intent in all_intents:
            p = dist_1.get(intent, 0) / total_1
            q = dist_2.get(intent, 0) / total_2
            if p > 0 and q > 0:
                m_val = (p + q) / 2
                divergence += abs(p - q)

        drift_detected = divergence > 0.3

        return {
            "drift_detected": drift_detected,
            "divergence_score": round(divergence, 4),
            "first_half_distribution": {k: round(v / total_1, 4) for k, v in dist_1.items()},
            "second_half_distribution": {k: round(v / total_2, 4) for k, v in dist_2.items()},
        }

    def flush_to_db(self, db=None) -> int:
        """Flush buffered metrics to database."""
        from sqlalchemy import text as sql_text
        from app.database import SessionLocal

        with self._lock:
            to_flush = list(self._buffer)
            self._buffer.clear()

        flushed = 0
        if not to_flush:
            return 0

        session = db or SessionLocal()
        try:
            for m in to_flush:
                try:
                    session.execute(
                        sql_text("""
                            INSERT INTO agent_quality_metrics
                            (session_id, intent, retrieval_hits, synthesis_confidence,
                             evidence_count, citation_count, compliance_decision,
                             warnings_count, total_latency_ms, synthesis_tier,
                             tools_invoked, tools_succeeded, tools_failed, created_at)
                            VALUES
                            (:session_id, :intent, :retrieval_hits, :synthesis_confidence,
                             :evidence_count, :citation_count, :compliance_decision,
                             :warnings_count, :total_latency_ms, :synthesis_tier,
                             :tools_invoked, :tools_succeeded, :tools_failed,
                             to_timestamp(:ts))
                        """),
                        {
                            "session_id": m.session_id,
                            "intent": m.intent,
                            "retrieval_hits": m.retrieval_hits,
                            "synthesis_confidence": m.synthesis_confidence,
                            "evidence_count": m.evidence_count,
                            "citation_count": m.retrieval_hits,
                            "compliance_decision": m.compliance_decision,
                            "warnings_count": m.warnings_count,
                            "total_latency_ms": m.total_latency_ms,
                            "synthesis_tier": m.intent_source,
                            "tools_invoked": m.tools_invoked,
                            "tools_succeeded": m.tools_succeeded,
                            "tools_failed": m.tools_failed,
                            "ts": m.timestamp,
                        },
                    )
                    flushed += 1
                except Exception as e:
                    logger.error("[Telemetry] Insert failed: %s", e)

            if flushed:
                try:
                    session.commit()
                except Exception as e:
                    logger.error("[Telemetry] Commit failed: %s", e)
                    session.rollback()
        finally:
            if db is None:
                session.close()

        return flushed

    def _check_alerts(self) -> None:
        """Check alert thresholds."""
        try:
            dashboard = self.get_dashboard(window_minutes=10)
            new_alerts: list[Alert] = []

            for config in self._alerts:
                value = self._extract_metric(dashboard, config.metric)
                if value is None:
                    continue

                triggered = False
                if config.direction == "above" and value > config.threshold:
                    triggered = True
                elif config.direction == "below" and value < config.threshold:
                    triggered = True

                if triggered:
                    new_alerts.append(Alert(
                        config=config,
                        current_value=value,
                        triggered_at=time.time(),
                        message=(
                            f"{config.name}: {config.metric}={value:.2f} "
                            f"({config.direction} {config.threshold})"
                        ),
                    ))

            self._active_alerts = new_alerts

        except Exception:
            pass

    def _extract_metric(self, dashboard: dict, metric: str) -> Optional[float]:
        """Extract a metric value from dashboard data."""
        mappings = {
            "avg_latency_ms": ("latency", "avg_ms"),
            "avg_confidence": ("quality", "avg_synthesis_confidence"),
            "abstain_rate": ("quality", "abstain_rate"),
            "tool_error_rate": ("tools", "error_rate"),
            "avg_retrieval_hits": ("quality", "avg_retrieval_hits"),
        }
        path = mappings.get(metric)
        if not path:
            return None

        section = dashboard.get(path[0], {})
        return section.get(path[1])

    def _get_dpo_status(self) -> dict[str, Any]:
        """Get DPO training pipeline status for dashboard integration."""
        try:
            from ml_engine.rlhf_dpo_trainer import get_dpo_status_tracker
            return get_dpo_status_tracker().get_status()
        except Exception:
            return {
                "current_status": "unavailable",
                "has_deps": False,
                "total_runs": 0,
                "last_run": None,
                "history": [],
            }

    def _empty_dashboard(self) -> dict[str, Any]:
        return {
            "window_minutes": 0,
            "total_requests": 0,
            "requests_per_minute": 0,
            "latency": {},
            "quality": {},
            "intent_distribution": {},
            "tools": {},
            "active_alerts": [],
            "system": {"buffer_size": 0, "total_lifetime_requests": self._total_requests},
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

_telemetry: TelemetryEngine | None = None


def get_telemetry() -> TelemetryEngine:
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryEngine()
    return _telemetry
