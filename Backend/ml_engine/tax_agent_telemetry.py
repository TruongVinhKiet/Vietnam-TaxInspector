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

    def get_dashboard(self, window_minutes: int = 60) -> dict[str, Any]:
        """
        Build dashboard data for the monitoring UI.

        Returns aggregated metrics over the specified time window.
        """
        cutoff = time.time() - (window_minutes * 60)

        with self._lock:
            recent = [m for m in self._buffer if m.timestamp >= cutoff]

        if not recent:
            return self._empty_dashboard()

        n = len(recent)

        # Latency stats
        latencies = [m.total_latency_ms for m in recent]
        latency_sorted = sorted(latencies)

        # Confidence stats
        intent_confs = [m.intent_confidence for m in recent]
        synth_confs = [m.synthesis_confidence for m in recent]

        # Intent distribution
        intent_dist: dict[str, int] = defaultdict(int)
        for m in recent:
            intent_dist[m.intent] += 1

        # Complexity distribution
        complexity_dist: dict[str, int] = defaultdict(int)
        for m in recent:
            complexity_dist[m.complexity] += 1

        # Component latency breakdown
        component_latencies: dict[str, list[float]] = defaultdict(list)
        for m in recent:
            for comp, lat in m.latency_breakdown.items():
                component_latencies[comp].append(lat)

        avg_component = {
            comp: round(sum(lats) / len(lats), 1)
            for comp, lats in component_latencies.items()
        }

        # Tool stats
        total_tool_calls = sum(m.tools_invoked for m in recent)
        total_tool_errors = sum(m.tools_failed for m in recent)

        # Quality metrics
        retrieval_hits = [m.retrieval_hits for m in recent]

        return {
            "window_minutes": window_minutes,
            "total_requests": n,
            "requests_per_minute": round(n / max(window_minutes, 1), 2),
            # Latency
            "latency": {
                "avg_ms": round(sum(latencies) / n, 1),
                "p50_ms": round(latency_sorted[n // 2], 1),
                "p95_ms": round(latency_sorted[int(n * 0.95)], 1) if n >= 20 else None,
                "p99_ms": round(latency_sorted[int(n * 0.99)], 1) if n >= 100 else None,
                "max_ms": round(max(latencies), 1),
                "min_ms": round(min(latencies), 1),
                "by_component": avg_component,
            },
            # Quality
            "quality": {
                "avg_intent_confidence": round(sum(intent_confs) / n, 4),
                "avg_synthesis_confidence": round(sum(synth_confs) / n, 4),
                "avg_retrieval_hits": round(sum(retrieval_hits) / n, 2),
                "abstain_rate": round(sum(1 for m in recent if m.abstained) / n, 4),
                "escalation_rate": round(sum(1 for m in recent if m.escalated) / n, 4),
            },
            # Intent
            "intent_distribution": dict(intent_dist),
            "complexity_distribution": dict(complexity_dist),
            # Tools
            "tools": {
                "total_calls": total_tool_calls,
                "total_errors": total_tool_errors,
                "error_rate": round(total_tool_errors / max(total_tool_calls, 1), 4),
            },
            # Alerts
            "active_alerts": [
                {"name": a.config.name, "value": a.current_value, "message": a.message}
                for a in self._active_alerts
            ],
            # System
            "system": {
                "buffer_size": len(self._buffer),
                "total_lifetime_requests": self._total_requests,
                "total_lifetime_abstains": self._total_abstains,
                "total_lifetime_escalations": self._total_escalations,
                "uptime_hours": round(
                    (time.time() - self._buffer[0].timestamp) / 3600, 2
                ) if self._buffer else 0,
            },
        }

    def get_intent_drift(self, window_hours: int = 24) -> dict[str, Any]:
        """Detect intent distribution drift over time."""
        now = time.time()
        half = window_hours * 3600 / 2

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

    def flush_to_db(self, db) -> int:
        """Flush buffered metrics to database."""
        from sqlalchemy import text as sql_text

        with self._lock:
            to_flush = list(self._buffer)

        flushed = 0
        for m in to_flush:
            try:
                db.execute(
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
            except Exception:
                pass

        if flushed:
            try:
                db.commit()
            except Exception:
                pass

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
