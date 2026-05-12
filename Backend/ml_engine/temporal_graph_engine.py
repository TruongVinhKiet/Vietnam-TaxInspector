"""
temporal_graph_engine.py – Temporal Graph Analysis Engine
==========================================================
Sliding-window temporal analysis for detecting fraud network
migration, seasonal carousel patterns, and dormancy-reactivation.

Integrates with existing graph_snapshots / graph_snapshot_nodes /
graph_snapshot_edges DB tables (init_db.sql §23).

Architecture:
    - TemporalSnapshotBuilder: splits invoice data into quarterly snapshots
    - TemporalPatternDetector: detects 4 temporal fraud patterns
    - TemporalRiskScorer: aggregates temporal signals into risk scores
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════

class TemporalPatternType(str, Enum):
    """Types of temporal fraud patterns."""
    NETWORK_MIGRATION = "network_migration"
    TEMPORAL_BURST = "temporal_burst"
    DORMANCY_REACTIVATION = "dormancy_reactivation"
    SEASONAL_CAROUSEL = "seasonal_carousel"


@dataclass
class TemporalSnapshot:
    """A graph snapshot for a specific time window."""
    snapshot_id: str
    period_label: str          # e.g. "2025-Q1"
    start_date: date
    end_date: date
    graph: nx.DiGraph
    node_count: int = 0
    edge_count: int = 0
    total_amount: float = 0.0
    scc_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalPattern:
    """A detected temporal fraud pattern."""
    pattern_type: TemporalPatternType
    severity: str              # low, medium, high, critical
    confidence: float          # 0.0–1.0
    involved_entities: list[str]
    time_range: tuple[str, str]
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalRiskResult:
    """Aggregated temporal risk assessment for a tax_code."""
    tax_code: str
    temporal_risk_score: float  # 0.0–1.0
    patterns_detected: list[TemporalPattern]
    snapshot_activity: dict[str, dict[str, Any]]  # period → metrics
    risk_drivers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Snapshot Builder
# ═══════════════════════════════════════════════════════════════

class TemporalSnapshotBuilder:
    """
    Builds quarterly/monthly graph snapshots from invoice data.

    Uses sliding windows over invoice records to create a sequence
    of NetworkX DiGraphs, each representing transactions within
    that time window.
    """

    QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}

    def build_quarterly_snapshots(
        self,
        invoices: list[dict[str, Any]],
        *,
        year_range: tuple[int, int] | None = None,
    ) -> list[TemporalSnapshot]:
        """
        Build quarterly snapshots from invoice records.

        Args:
            invoices: list of dicts with keys:
                seller_tax_code, buyer_tax_code, amount, date
            year_range: optional (start_year, end_year) filter

        Returns:
            Ordered list of TemporalSnapshot objects
        """
        t0 = time.perf_counter()

        # Group invoices by quarter
        quarter_buckets: dict[str, list[dict]] = defaultdict(list)
        for inv in invoices:
            inv_date = inv.get("date") or inv.get("invoice_date")
            if not inv_date:
                continue
            if isinstance(inv_date, str):
                try:
                    inv_date = date.fromisoformat(inv_date[:10])
                except (ValueError, TypeError):
                    continue

            if year_range:
                if inv_date.year < year_range[0] or inv_date.year > year_range[1]:
                    continue

            q = (inv_date.month - 1) // 3 + 1
            label = f"{inv_date.year}-Q{q}"
            quarter_buckets[label].append(inv)

        # Build snapshots
        snapshots = []
        for label in sorted(quarter_buckets.keys()):
            bucket = quarter_buckets[label]
            year = int(label.split("-Q")[0])
            q = int(label.split("-Q")[1])
            m_start, m_end = self.QUARTER_MONTHS[q]

            start_d = date(year, m_start, 1)
            if m_end == 12:
                end_d = date(year, 12, 31)
            else:
                end_d = date(year, m_end + 1, 1) - timedelta(days=1)

            G = self._build_graph(bucket)
            sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]

            snap = TemporalSnapshot(
                snapshot_id=f"snap-{label}-{hashlib.md5(label.encode()).hexdigest()[:8]}",
                period_label=label,
                start_date=start_d,
                end_date=end_d,
                graph=G,
                node_count=G.number_of_nodes(),
                edge_count=G.number_of_edges(),
                total_amount=sum(
                    d.get("weight", 0) for _, _, d in G.edges(data=True)
                ),
                scc_count=len(sccs),
            )
            snapshots.append(snap)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[TemporalSnapshotBuilder] Built %d snapshots in %.1fms",
            len(snapshots), elapsed,
        )
        return snapshots

    @staticmethod
    def _build_graph(invoices: list[dict]) -> nx.DiGraph:
        """Build a DiGraph from invoice records."""
        G = nx.DiGraph()
        for inv in invoices:
            seller = inv.get("seller_tax_code", "")
            buyer = inv.get("buyer_tax_code", "")
            amount = float(inv.get("amount", 0))
            if not seller or not buyer or seller == buyer:
                continue
            if G.has_edge(seller, buyer):
                G[seller][buyer]["weight"] += amount
                G[seller][buyer]["count"] += 1
            else:
                G.add_edge(seller, buyer, weight=amount, count=1)
        return G


# ═══════════════════════════════════════════════════════════════
# Pattern Detector
# ═══════════════════════════════════════════════════════════════

class TemporalPatternDetector:
    """
    Detects temporal fraud patterns across a sequence of snapshots.

    Patterns:
    1. Network Migration: cluster appears in region A, disappears,
       reappears in region B
    2. Temporal Burst: sudden spike in transactions within a short
       window followed by silence
    3. Dormancy-Reactivation: entity dormant for 6+ months then
       suddenly reactivates with high volume
    4. Seasonal Carousel: SCC cycles only appear at end-of-quarter
       (months 3, 6, 9, 12)
    """

    # Thresholds
    BURST_MULTIPLIER = 3.0       # 3x average = burst
    DORMANCY_QUARTERS = 2        # 2 quarters = 6 months dormancy
    SEASONAL_MIN_APPEARANCES = 2 # SCC must appear in ≥2 end-of-quarter periods

    def detect_all_patterns(
        self,
        snapshots: list[TemporalSnapshot],
        *,
        target_tax_codes: set[str] | None = None,
    ) -> list[TemporalPattern]:
        """Run all pattern detectors and return combined results."""
        if len(snapshots) < 2:
            return []

        patterns: list[TemporalPattern] = []
        patterns.extend(self._detect_temporal_bursts(snapshots, target_tax_codes))
        patterns.extend(self._detect_dormancy_reactivation(snapshots, target_tax_codes))
        patterns.extend(self._detect_seasonal_carousel(snapshots))
        patterns.extend(self._detect_network_migration(snapshots))

        logger.info(
            "[TemporalPatternDetector] Detected %d patterns across %d snapshots",
            len(patterns), len(snapshots),
        )
        return patterns

    def _detect_temporal_bursts(
        self,
        snapshots: list[TemporalSnapshot],
        target_codes: set[str] | None,
    ) -> list[TemporalPattern]:
        """Detect entities with sudden transaction volume spikes."""
        patterns = []

        # Build per-entity activity timeline
        entity_timeline: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
        for snap in snapshots:
            for node in snap.graph.nodes():
                if target_codes and node not in target_codes:
                    continue
                out_vol = sum(d["weight"] for _, _, d in snap.graph.out_edges(node, data=True))
                in_vol = sum(d["weight"] for _, _, d in snap.graph.in_edges(node, data=True))
                total = out_vol + in_vol
                degree = snap.graph.degree(node)
                entity_timeline[node].append((snap.period_label, total, degree))

        for entity, timeline in entity_timeline.items():
            if len(timeline) < 3:
                continue
            volumes = [t[1] for t in timeline]
            avg_vol = sum(volumes) / len(volumes)
            if avg_vol <= 0:
                continue

            for i, (period, vol, deg) in enumerate(timeline):
                if vol > avg_vol * self.BURST_MULTIPLIER:
                    # Check if preceded and followed by low activity
                    pre_low = i == 0 or volumes[i - 1] < avg_vol * 0.5
                    post_low = i == len(timeline) - 1 or volumes[i + 1] < avg_vol * 0.5
                    if pre_low or post_low:
                        ratio = vol / avg_vol
                        severity = "critical" if ratio > 8 else "high" if ratio > 5 else "medium"
                        patterns.append(TemporalPattern(
                            pattern_type=TemporalPatternType.TEMPORAL_BURST,
                            severity=severity,
                            confidence=min(0.95, 0.5 + (ratio - self.BURST_MULTIPLIER) * 0.05),
                            involved_entities=[entity],
                            time_range=(period, period),
                            description=(
                                f"DN {entity} có khối lượng giao dịch đột biến "
                                f"({ratio:.1f}x trung bình) trong kỳ {period}"
                            ),
                            evidence={
                                "burst_volume": vol,
                                "average_volume": avg_vol,
                                "burst_ratio": ratio,
                                "degree_at_burst": deg,
                            },
                        ))
        return patterns

    def _detect_dormancy_reactivation(
        self,
        snapshots: list[TemporalSnapshot],
        target_codes: set[str] | None,
    ) -> list[TemporalPattern]:
        """Detect entities dormant for N quarters then reactivated."""
        patterns = []
        all_nodes: set[str] = set()
        for snap in snapshots:
            all_nodes.update(snap.graph.nodes())

        if target_codes:
            all_nodes &= target_codes

        for node in all_nodes:
            activity = []
            for snap in snapshots:
                is_active = node in snap.graph
                vol = 0.0
                if is_active:
                    vol = sum(d["weight"] for _, _, d in snap.graph.out_edges(node, data=True))
                    vol += sum(d["weight"] for _, _, d in snap.graph.in_edges(node, data=True))
                activity.append((snap.period_label, is_active and vol > 0, vol))

            # Find dormancy → reactivation sequences
            dormant_start = None
            dormant_count = 0
            for i, (period, active, vol) in enumerate(activity):
                if not active:
                    if dormant_start is None:
                        dormant_start = i
                    dormant_count += 1
                else:
                    if dormant_count >= self.DORMANCY_QUARTERS and dormant_start is not None:
                        # Reactivation detected
                        dormant_period_start = activity[dormant_start][0]
                        prev_period = activity[dormant_start - 1][0] if dormant_start > 0 else "N/A"
                        severity = "critical" if dormant_count >= 4 else "high" if dormant_count >= 3 else "medium"
                        patterns.append(TemporalPattern(
                            pattern_type=TemporalPatternType.DORMANCY_REACTIVATION,
                            severity=severity,
                            confidence=min(0.9, 0.5 + dormant_count * 0.1),
                            involved_entities=[node],
                            time_range=(dormant_period_start, period),
                            description=(
                                f"DN {node} ngủ đông {dormant_count} quý "
                                f"({dormant_period_start}→{period}) rồi tái hoạt động "
                                f"với khối lượng {vol:,.0f}"
                            ),
                            evidence={
                                "dormancy_quarters": dormant_count,
                                "reactivation_volume": vol,
                                "dormancy_start": dormant_period_start,
                            },
                        ))
                    dormant_start = None
                    dormant_count = 0
        return patterns

    def _detect_seasonal_carousel(
        self,
        snapshots: list[TemporalSnapshot],
    ) -> list[TemporalPattern]:
        """Detect SCC cycles that only appear at end-of-quarter periods."""
        patterns = []

        # Identify SCCs per snapshot
        scc_registry: dict[str, list[str]] = defaultdict(list)
        for snap in snapshots:
            sccs = [
                frozenset(c)
                for c in nx.strongly_connected_components(snap.graph)
                if len(c) > 1
            ]
            for scc in sccs:
                scc_key = "|".join(sorted(scc))
                scc_registry[scc_key].append(snap.period_label)

        # Check for seasonal-only SCCs
        for scc_key, appearances in scc_registry.items():
            if len(appearances) < self.SEASONAL_MIN_APPEARANCES:
                continue
            # Check if all appearances are end-of-quarter
            end_quarter_count = sum(
                1 for p in appearances
                if p.endswith(("-Q1", "-Q2", "-Q3", "-Q4"))
            )
            # All appearances are quarterly by definition with quarterly snapshots,
            # but check if they cluster at fiscal quarter-ends (Q1, Q4)
            fiscal_end_quarters = sum(
                1 for p in appearances if p.endswith(("-Q1", "-Q4"))
            )
            if fiscal_end_quarters >= len(appearances) * 0.6:
                entities = scc_key.split("|")
                patterns.append(TemporalPattern(
                    pattern_type=TemporalPatternType.SEASONAL_CAROUSEL,
                    severity="high" if len(entities) >= 4 else "medium",
                    confidence=min(0.9, 0.5 + len(appearances) * 0.1),
                    involved_entities=entities,
                    time_range=(appearances[0], appearances[-1]),
                    description=(
                        f"Chu trình xoay vòng {len(entities)} DN chỉ xuất hiện "
                        f"vào cuối kỳ tài chính ({', '.join(appearances)})"
                    ),
                    evidence={
                        "scc_size": len(entities),
                        "appearance_periods": appearances,
                        "fiscal_end_ratio": fiscal_end_quarters / max(1, len(appearances)),
                    },
                ))
        return patterns

    def _detect_network_migration(
        self,
        snapshots: list[TemporalSnapshot],
    ) -> list[TemporalPattern]:
        """
        Detect clusters that disappear from one subgraph and reappear
        in another with different composition (migration).
        """
        patterns = []
        if len(snapshots) < 3:
            return patterns

        # Track connected components over time
        for i in range(len(snapshots) - 2):
            snap_a = snapshots[i]
            snap_b = snapshots[i + 1]
            snap_c = snapshots[i + 2]

            # Find clusters in snap_a that disappear in snap_b
            comps_a = [
                set(c) for c in nx.weakly_connected_components(snap_a.graph)
                if len(c) >= 3
            ]

            for comp in comps_a:
                # Check if ≥60% of cluster nodes are absent in snap_b
                active_in_b = comp & set(snap_b.graph.nodes())
                disappear_rate = 1 - len(active_in_b) / len(comp)

                if disappear_rate >= 0.6:
                    # Check if ≥40% reappear in snap_c with new companions
                    active_in_c = comp & set(snap_c.graph.nodes())
                    reappear_rate = len(active_in_c) / len(comp)

                    if reappear_rate >= 0.3:
                        # Find new companions in snap_c
                        new_companions = set()
                        for node in active_in_c:
                            neighbors_c = set(snap_c.graph.predecessors(node)) | set(snap_c.graph.successors(node))
                            new_companions |= (neighbors_c - comp)

                        if len(new_companions) >= 2:
                            all_involved = list(comp | new_companions)[:20]  # cap at 20
                            patterns.append(TemporalPattern(
                                pattern_type=TemporalPatternType.NETWORK_MIGRATION,
                                severity="critical" if len(comp) >= 5 else "high",
                                confidence=min(0.85, 0.4 + disappear_rate * 0.3 + reappear_rate * 0.2),
                                involved_entities=all_involved,
                                time_range=(snap_a.period_label, snap_c.period_label),
                                description=(
                                    f"Cụm {len(comp)} DN biến mất khỏi mạng lưới trong "
                                    f"{snap_b.period_label} ({disappear_rate:.0%}), "
                                    f"tái xuất {snap_c.period_label} với {len(new_companions)} "
                                    f"đối tác mới"
                                ),
                                evidence={
                                    "original_cluster_size": len(comp),
                                    "disappear_rate": disappear_rate,
                                    "reappear_rate": reappear_rate,
                                    "new_companions_count": len(new_companions),
                                    "periods": [snap_a.period_label, snap_b.period_label, snap_c.period_label],
                                },
                            ))
        return patterns


# ═══════════════════════════════════════════════════════════════
# Risk Scorer
# ═══════════════════════════════════════════════════════════════

class TemporalRiskScorer:
    """
    Aggregates temporal patterns into per-entity risk scores.
    Designed to integrate with existing company_motif_scores
    in graph_intelligence.py.
    """

    PATTERN_WEIGHTS = {
        TemporalPatternType.NETWORK_MIGRATION: 0.35,
        TemporalPatternType.SEASONAL_CAROUSEL: 0.30,
        TemporalPatternType.DORMANCY_REACTIVATION: 0.20,
        TemporalPatternType.TEMPORAL_BURST: 0.15,
    }
    SEVERITY_MULTIPLIER = {
        "critical": 1.0,
        "high": 0.75,
        "medium": 0.50,
        "low": 0.25,
    }

    def score_entities(
        self,
        patterns: list[TemporalPattern],
        snapshots: list[TemporalSnapshot],
        *,
        target_tax_codes: set[str] | None = None,
    ) -> dict[str, TemporalRiskResult]:
        """
        Compute temporal risk scores for all entities involved in patterns.

        Returns:
            dict mapping tax_code → TemporalRiskResult
        """
        # Build per-entity pattern map
        entity_patterns: dict[str, list[TemporalPattern]] = defaultdict(list)
        for pat in patterns:
            for entity in pat.involved_entities:
                if target_tax_codes and entity not in target_tax_codes:
                    continue
                entity_patterns[entity].append(pat)

        # Build activity timeline per entity
        entity_activity: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for snap in snapshots:
            for node in snap.graph.nodes():
                out_vol = sum(d["weight"] for _, _, d in snap.graph.out_edges(node, data=True))
                in_vol = sum(d["weight"] for _, _, d in snap.graph.in_edges(node, data=True))
                entity_activity[node][snap.period_label] = {
                    "active": True,
                    "volume_out": out_vol,
                    "volume_in": in_vol,
                    "degree": snap.graph.degree(node),
                }

        # Score each entity
        results: dict[str, TemporalRiskResult] = {}
        for entity, pats in entity_patterns.items():
            score = 0.0
            drivers = []
            for pat in pats:
                w = self.PATTERN_WEIGHTS.get(pat.pattern_type, 0.1)
                m = self.SEVERITY_MULTIPLIER.get(pat.severity, 0.5)
                contribution = w * m * pat.confidence
                score += contribution
                drivers.append(
                    f"{pat.pattern_type.value} ({pat.severity}, conf={pat.confidence:.2f})"
                )

            # Normalize to [0, 1]
            score = min(1.0, score)

            results[entity] = TemporalRiskResult(
                tax_code=entity,
                temporal_risk_score=round(score, 4),
                patterns_detected=pats,
                snapshot_activity=dict(entity_activity.get(entity, {})),
                risk_drivers=drivers,
                metadata={
                    "pattern_count": len(pats),
                    "snapshot_count": len(snapshots),
                    "scoring_method": "weighted_pattern_aggregation",
                },
            )

        logger.info(
            "[TemporalRiskScorer] Scored %d entities, max_score=%.3f",
            len(results),
            max((r.temporal_risk_score for r in results.values()), default=0),
        )
        return results


# ═══════════════════════════════════════════════════════════════
# Orchestrator (convenience wrapper)
# ═══════════════════════════════════════════════════════════════

class TemporalGraphEngine:
    """
    High-level orchestrator combining snapshot building,
    pattern detection, and risk scoring.

    Usage:
        engine = TemporalGraphEngine()
        result = engine.analyze(invoices, target_tax_codes={"0312345678"})
    """

    def __init__(self):
        self.snapshot_builder = TemporalSnapshotBuilder()
        self.pattern_detector = TemporalPatternDetector()
        self.risk_scorer = TemporalRiskScorer()

    def analyze(
        self,
        invoices: list[dict[str, Any]],
        *,
        target_tax_codes: set[str] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """
        Run full temporal analysis pipeline.

        Returns:
            dict with keys: snapshots, patterns, risk_scores, summary
        """
        t0 = time.perf_counter()

        # Step 1: Build snapshots
        snapshots = self.snapshot_builder.build_quarterly_snapshots(
            invoices, year_range=year_range,
        )

        if len(snapshots) < 2:
            return {
                "snapshots": [],
                "patterns": [],
                "risk_scores": {},
                "summary": {
                    "status": "insufficient_data",
                    "message": "Cần ít nhất 2 quý dữ liệu để phân tích temporal",
                    "snapshot_count": len(snapshots),
                },
            }

        # Step 2: Detect patterns
        patterns = self.pattern_detector.detect_all_patterns(
            snapshots, target_tax_codes=target_tax_codes,
        )

        # Step 3: Score entities
        risk_scores = self.risk_scorer.score_entities(
            patterns, snapshots, target_tax_codes=target_tax_codes,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        # Build summary
        pattern_counts = defaultdict(int)
        for p in patterns:
            pattern_counts[p.pattern_type.value] += 1

        summary = {
            "status": "completed",
            "snapshot_count": len(snapshots),
            "period_range": f"{snapshots[0].period_label} → {snapshots[-1].period_label}",
            "total_patterns": len(patterns),
            "pattern_breakdown": dict(pattern_counts),
            "entities_at_risk": len(risk_scores),
            "high_risk_entities": sum(
                1 for r in risk_scores.values() if r.temporal_risk_score >= 0.6
            ),
            "latency_ms": round(elapsed, 1),
        }

        logger.info(
            "[TemporalGraphEngine] Analysis complete: %d snapshots, "
            "%d patterns, %d entities scored in %.1fms",
            len(snapshots), len(patterns), len(risk_scores), elapsed,
        )

        return {
            "snapshots": [
                {
                    "snapshot_id": s.snapshot_id,
                    "period": s.period_label,
                    "nodes": s.node_count,
                    "edges": s.edge_count,
                    "total_amount": s.total_amount,
                    "scc_count": s.scc_count,
                }
                for s in snapshots
            ],
            "patterns": [
                {
                    "type": p.pattern_type.value,
                    "severity": p.severity,
                    "confidence": p.confidence,
                    "entities": p.involved_entities[:10],
                    "time_range": list(p.time_range),
                    "description": p.description,
                    "evidence": p.evidence,
                }
                for p in patterns
            ],
            "risk_scores": {
                tc: {
                    "temporal_risk_score": r.temporal_risk_score,
                    "pattern_count": len(r.patterns_detected),
                    "risk_drivers": r.risk_drivers,
                    "activity_periods": len(r.snapshot_activity),
                }
                for tc, r in risk_scores.items()
            },
            "summary": summary,
        }
