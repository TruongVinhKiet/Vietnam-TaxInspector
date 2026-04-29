"""
tax_agent_investigation_agent.py – Investigation Sub-Agent (Phase 3)
=====================================================================
Specialized agent for deep-dive fraud investigation using graph intelligence.

Capabilities:
    1. Network topology analysis (transaction graph structure)
    2. Suspicious pattern identification (carousel, shell, layering)
    3. UBO (Ultimate Beneficial Owner) chain tracing
    4. Cross-entity relationship mapping
    5. Risk propagation through ownership networks
    6. Evidence package assembly for investigation reports

Architecture:
    Input: Sub-task from Orchestrator (with tax_code + graph data)
    Processing: Graph tools → pattern analysis → evidence synthesis
    Output: Investigation report with network visualizations data
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SuspicionLevel(str, Enum):
    CONFIRMED = "confirmed"         # Clear evidence of fraud/violation
    HIGHLY_SUSPICIOUS = "highly_suspicious"  # Strong indicators
    SUSPICIOUS = "suspicious"       # Moderate indicators
    WATCH = "watch"                 # Worth monitoring
    CLEAR = "clear"                 # No indicators found


class PatternType(str, Enum):
    CAROUSEL = "carousel"           # Circular VAT fraud
    SHELL_COMPANY = "shell_company" # Fake/dormant companies
    LAYERING = "layering"           # Transaction layering
    PHOENIX = "phoenix"             # Company dissolution & rebirth
    MISSING_TRADER = "missing_trader" # Missing trader intra-community fraud
    BACK_TO_BACK = "back_to_back"   # Back-to-back trading
    CONTROL_CHAIN = "control_chain" # Hidden control via nested ownership


@dataclass
class SuspiciousPattern:
    """A detected suspicious pattern."""
    pattern_type: PatternType
    description: str
    entities_involved: list[str]    # tax codes
    evidence: str
    suspicion_score: float          # 0.0 - 1.0
    recommended_action: str


@dataclass
class OwnershipChain:
    """An ownership chain from target to UBO."""
    chain: list[dict[str, Any]]     # [{entity, ownership_pct, type}]
    total_depth: int
    ubo_identified: bool
    ubo_entity: str | None = None
    shell_indicators: int = 0       # Number of shell company indicators


@dataclass
class NetworkAnalysis:
    """Network topology analysis results."""
    total_nodes: int                 # Companies in network
    total_edges: int                 # Transactions/relationships
    density: float                   # Graph density
    clustering_coefficient: float    # How clustered the network is
    betweenness_centrality: float    # Target's centrality
    communities: int                 # Number of communities
    isolated_nodes: int              # Disconnected entities
    hub_score: float                 # How much of a hub the target is


@dataclass
class InvestigationReport:
    """Full investigation report."""
    tax_code: str
    company_name: str
    # Suspicion assessment
    suspicion_level: SuspicionLevel
    overall_score: float
    confidence: float
    # Patterns
    suspicious_patterns: list[SuspiciousPattern]
    # Ownership
    ownership_chains: list[OwnershipChain]
    ubo_status: str                 # "identified", "complex", "unknown"
    # Network
    network_analysis: NetworkAnalysis | None
    # Investigation narrative
    executive_summary: str
    detailed_findings: str
    evidence_package: list[dict[str, Any]]
    # Recommendations
    recommended_actions: list[str]
    escalation_level: str           # "routine", "priority", "urgent", "critical"
    # Metadata
    tools_used: list[str]
    latency_ms: float = 0.0


class InvestigationAgent:
    """
    Specialized agent for deep-dive fraud investigation.

    Investigation Flow:
    1. Analyze ownership structure → identify UBO chains
    2. Analyze transaction graph → compute network metrics
    3. Detect suspicious patterns → carousel, shell, layering
    4. Assess risk propagation → who else is affected?
    5. Assemble evidence package → structured for investigation
    6. Generate investigation report → actionable findings

    Usage:
        agent = InvestigationAgent()
        report = agent.investigate(
            tax_code="0312345678",
            tool_results={...},
        )
    """

    # Suspicion scoring thresholds
    SUSPICION_THRESHOLDS = {
        SuspicionLevel.CONFIRMED: 0.90,
        SuspicionLevel.HIGHLY_SUSPICIOUS: 0.70,
        SuspicionLevel.SUSPICIOUS: 0.45,
        SuspicionLevel.WATCH: 0.20,
    }

    def investigate(
        self,
        tax_code: str,
        tool_results: dict[str, dict[str, Any]],
        *,
        company_name: str = "",
        intent: str = "osint_ownership",
    ) -> InvestigationReport:
        """
        Conduct deep-dive investigation.

        Args:
            tax_code: Target company
            tool_results: Results from tool executor
            company_name: Company name
            intent: Investigation intent

        Returns:
            InvestigationReport with detailed findings
        """
        t0 = time.perf_counter()

        if not company_name:
            cr = tool_results.get("company_risk_lookup", {})
            company_name = str(cr.get("company_name", f"DN {tax_code}"))

        # Step 1: Analyze ownership chains
        ownership_chains = self._analyze_ownership(tool_results, tax_code)
        ubo_status = self._assess_ubo_status(ownership_chains)

        # Step 2: Network topology analysis
        network = self._analyze_network(tool_results, tax_code)

        # Step 3: Detect suspicious patterns
        patterns = self._detect_patterns(tool_results, tax_code, ownership_chains, network)

        # Step 4: Compute suspicion score
        overall_score = self._compute_suspicion_score(patterns, ownership_chains, network, tool_results)
        suspicion_level = self._classify_suspicion(overall_score)

        # Step 5: Compute confidence
        confidence = self._compute_confidence(tool_results, patterns)

        # Step 6: Assemble evidence package
        evidence_package = self._assemble_evidence(tool_results, patterns, ownership_chains)

        # Step 7: Generate report narrative
        summary = self._generate_summary(
            company_name, tax_code, suspicion_level, overall_score,
            patterns, ownership_chains,
        )
        findings = self._generate_findings(
            company_name, tax_code, patterns, ownership_chains,
            network, tool_results,
        )

        # Step 8: Generate recommendations
        actions = self._generate_actions(suspicion_level, patterns, ubo_status)
        escalation = self._determine_escalation(suspicion_level, patterns)

        tools_used = [k for k in tool_results.keys()]
        latency = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "[InvestigationAgent] Report for %s: suspicion=%s(%.2f), "
            "patterns=%d, ownership_chains=%d in %.0fms",
            tax_code, suspicion_level.value, overall_score,
            len(patterns), len(ownership_chains), latency,
        )

        return InvestigationReport(
            tax_code=tax_code,
            company_name=company_name,
            suspicion_level=suspicion_level,
            overall_score=round(overall_score, 4),
            confidence=round(confidence, 4),
            suspicious_patterns=patterns,
            ownership_chains=ownership_chains,
            ubo_status=ubo_status,
            network_analysis=network,
            executive_summary=summary,
            detailed_findings=findings,
            evidence_package=evidence_package,
            recommended_actions=actions,
            escalation_level=escalation,
            tools_used=tools_used,
            latency_ms=latency,
        )

    def _analyze_ownership(
        self,
        tool_results: dict[str, dict[str, Any]],
        tax_code: str,
    ) -> list[OwnershipChain]:
        """Analyze ownership structure from tool results."""
        own = tool_results.get("ownership_analysis", {})
        chains: list[OwnershipChain] = []

        if not own or own.get("status") in ("error", "no_data"):
            return chains

        clusters = own.get("clusters", [])
        common_controllers = own.get("common_controllers", [])

        # Build chains from clusters
        for cluster in clusters:
            entities = cluster if isinstance(cluster, list) else [cluster]
            shell_indicators = 0

            chain_data = []
            for entity in entities:
                if isinstance(entity, dict):
                    chain_data.append(entity)
                    # Check for shell company indicators
                    if entity.get("ownership_percent", 0) > 95:
                        shell_indicators += 1
                else:
                    chain_data.append({"entity": str(entity), "type": "unknown"})

            ubo_entity = None
            if common_controllers:
                ubo_entity = str(common_controllers[0]) if common_controllers else None

            chains.append(OwnershipChain(
                chain=chain_data,
                total_depth=len(chain_data),
                ubo_identified=bool(ubo_entity),
                ubo_entity=ubo_entity,
                shell_indicators=shell_indicators,
            ))

        return chains

    def _analyze_network(
        self,
        tool_results: dict[str, dict[str, Any]],
        tax_code: str,
    ) -> NetworkAnalysis | None:
        """Extract network metrics from motif/graph tool results."""
        motif = tool_results.get("motif_detection", {})
        gnn = tool_results.get("gnn_analysis", {})

        if motif.get("status") not in ("analyzed",) and gnn.get("status") not in ("found",):
            return None

        summary = motif.get("summary", {})

        # Estimate network metrics from available data
        total_patterns = sum(
            int(v) for v in summary.values() if isinstance(v, (int, float))
        )

        # GNN outputs may have graph metrics
        gnn_outputs = gnn.get("gnn_outputs", {}) if gnn.get("status") == "found" else {}

        return NetworkAnalysis(
            total_nodes=int(gnn_outputs.get("graph_nodes", summary.get("total_nodes", 0))),
            total_edges=int(gnn_outputs.get("graph_edges", summary.get("total_edges", 0))),
            density=float(gnn_outputs.get("density", 0)),
            clustering_coefficient=float(gnn_outputs.get("clustering", 0)),
            betweenness_centrality=float(gnn_outputs.get("betweenness", 0)),
            communities=int(gnn_outputs.get("communities", 1)),
            isolated_nodes=int(gnn_outputs.get("isolated", 0)),
            hub_score=float(gnn_outputs.get("hub_score", 0)),
        )

    def _detect_patterns(
        self,
        tool_results: dict[str, dict[str, Any]],
        tax_code: str,
        ownership_chains: list[OwnershipChain],
        network: NetworkAnalysis | None,
    ) -> list[SuspiciousPattern]:
        """Detect suspicious patterns from all available data."""
        patterns: list[SuspiciousPattern] = []

        motif = tool_results.get("motif_detection", {})
        summary = motif.get("summary", {})

        # Pattern 1: Carousel (circular transactions)
        triangles = int(summary.get("total_triangles", 0))
        if triangles > 0:
            patterns.append(SuspiciousPattern(
                pattern_type=PatternType.CAROUSEL,
                description=f"Phát hiện {triangles} mẫu giao dịch vòng tròn (A→B→C→A)",
                entities_involved=[tax_code],
                evidence=f"Motif detection: {triangles} triangles",
                suspicion_score=min(1.0, triangles * 0.3),
                recommended_action="Điều tra chi tiết chuỗi hóa đơn VAT vòng tròn",
            ))

        # Pattern 2: Shell company indicators (fan-out/fan-in)
        fan_out = int(summary.get("total_fan_out", 0))
        fan_in = int(summary.get("total_fan_in", 0))
        if fan_out > 3 or fan_in > 3:
            patterns.append(SuspiciousPattern(
                pattern_type=PatternType.SHELL_COMPANY,
                description=f"Fan-out: {fan_out}, Fan-in: {fan_in} — mẫu công ty vỏ bọc",
                entities_involved=[tax_code],
                evidence=f"Asymmetric transaction flow: out={fan_out}, in={fan_in}",
                suspicion_score=min(1.0, (fan_out + fan_in) * 0.1),
                recommended_action="Kiểm tra tính thực chất của hoạt động kinh doanh",
            ))

        # Pattern 3: Layering (long chains)
        chains = int(summary.get("total_chains", 0))
        if chains > 2:
            patterns.append(SuspiciousPattern(
                pattern_type=PatternType.LAYERING,
                description=f"Phát hiện {chains} chuỗi giao dịch dài (layering)",
                entities_involved=[tax_code],
                evidence=f"Chain detection: {chains} chains",
                suspicion_score=min(1.0, chains * 0.2),
                recommended_action="Truy vết chuỗi giao dịch, xác định điểm cuối",
            ))

        # Pattern 4: Control chain (from ownership analysis)
        for chain in ownership_chains:
            if chain.shell_indicators > 0:
                patterns.append(SuspiciousPattern(
                    pattern_type=PatternType.CONTROL_CHAIN,
                    description=f"Chuỗi sở hữu phức tạp (depth={chain.total_depth}, shells={chain.shell_indicators})",
                    entities_involved=[tax_code] + [
                        str(e.get("entity", "")) for e in chain.chain[:3]
                    ],
                    evidence=f"Ownership depth={chain.total_depth}, UBO={'identified' if chain.ubo_identified else 'hidden'}",
                    suspicion_score=min(1.0, chain.total_depth * 0.15 + chain.shell_indicators * 0.2),
                    recommended_action="Truy vết UBO thực sự qua các lớp sở hữu",
                ))

        # Sort by suspicion score
        patterns.sort(key=lambda p: p.suspicion_score, reverse=True)
        return patterns

    def _compute_suspicion_score(
        self,
        patterns: list[SuspiciousPattern],
        chains: list[OwnershipChain],
        network: NetworkAnalysis | None,
        tool_results: dict[str, dict[str, Any]],
    ) -> float:
        """Compute overall suspicion score."""
        if not patterns:
            return 0.05

        # Pattern-based score (max of top 3 patterns)
        top_pattern_scores = sorted(
            [p.suspicion_score for p in patterns], reverse=True
        )[:3]
        pattern_score = sum(top_pattern_scores) / len(top_pattern_scores) * 0.5

        # Ownership complexity score
        ownership_score = 0.0
        if chains:
            max_depth = max(c.total_depth for c in chains)
            total_shells = sum(c.shell_indicators for c in chains)
            ownership_score = min(1.0, max_depth * 0.1 + total_shells * 0.15) * 0.25

        # Company risk multiplier
        cr = tool_results.get("company_risk_lookup", {})
        risk_factor = float(cr.get("risk_score", 50)) / 100.0 * 0.25

        return min(1.0, pattern_score + ownership_score + risk_factor)

    def _classify_suspicion(self, score: float) -> SuspicionLevel:
        """Classify suspicion level from score."""
        for level, threshold in self.SUSPICION_THRESHOLDS.items():
            if score >= threshold:
                return level
        return SuspicionLevel.CLEAR

    def _compute_confidence(
        self,
        tool_results: dict[str, dict[str, Any]],
        patterns: list[SuspiciousPattern],
    ) -> float:
        """Compute investigation confidence."""
        total_tools = len(tool_results)
        successful = sum(
            1 for r in tool_results.values()
            if isinstance(r, dict) and r.get("status") not in ("error", "no_data", "not_found")
        )

        tool_coverage = successful / max(total_tools, 1) * 0.5
        pattern_confidence = min(1.0, len(patterns) / 3.0) * 0.3
        data_quality = min(1.0, successful / 3.0) * 0.2

        return min(1.0, tool_coverage + pattern_confidence + data_quality)

    def _assess_ubo_status(self, chains: list[OwnershipChain]) -> str:
        """Assess UBO identification status."""
        if not chains:
            return "unknown"
        if any(c.ubo_identified for c in chains):
            return "identified"
        if any(c.total_depth > 2 for c in chains):
            return "complex"
        return "unknown"

    def _assemble_evidence(
        self,
        tool_results: dict[str, dict[str, Any]],
        patterns: list[SuspiciousPattern],
        chains: list[OwnershipChain],
    ) -> list[dict[str, Any]]:
        """Assemble evidence package for investigation."""
        evidence = []

        for i, pattern in enumerate(patterns):
            evidence.append({
                "evidence_id": f"PAT-{i+1}",
                "type": "pattern",
                "pattern_type": pattern.pattern_type.value,
                "description": pattern.description,
                "score": pattern.suspicion_score,
                "entities": pattern.entities_involved,
                "source": pattern.evidence,
            })

        for i, chain in enumerate(chains):
            evidence.append({
                "evidence_id": f"OWN-{i+1}",
                "type": "ownership",
                "chain_depth": chain.total_depth,
                "ubo": chain.ubo_entity,
                "shell_indicators": chain.shell_indicators,
            })

        return evidence

    def _generate_summary(
        self,
        company_name: str,
        tax_code: str,
        level: SuspicionLevel,
        score: float,
        patterns: list[SuspiciousPattern],
        chains: list[OwnershipChain],
    ) -> str:
        """Generate executive summary."""
        level_vi = {
            SuspicionLevel.CONFIRMED: "XÁC NHẬN VI PHẠM",
            SuspicionLevel.HIGHLY_SUSPICIOUS: "RẤT ĐÁNG NGỜ",
            SuspicionLevel.SUSPICIOUS: "ĐÁNG NGỜ",
            SuspicionLevel.WATCH: "CẦN GIÁM SÁT",
            SuspicionLevel.CLEAR: "BÌNH THƯỜNG",
        }

        pattern_str = ""
        if patterns:
            top_patterns = ", ".join(p.pattern_type.value for p in patterns[:3])
            pattern_str = f" Phát hiện {len(patterns)} mẫu đáng ngờ ({top_patterns})."

        ownership_str = ""
        if chains:
            max_depth = max(c.total_depth for c in chains)
            ownership_str = f" Cấu trúc sở hữu {max_depth} tầng."

        return (
            f"**{company_name}** (MST: {tax_code}) — "
            f"Mức nghi vấn: **{level_vi.get(level, level.value)}** "
            f"(score: {score:.0%})."
            f"{pattern_str}{ownership_str}"
        )

    def _generate_findings(
        self,
        company_name: str,
        tax_code: str,
        patterns: list[SuspiciousPattern],
        chains: list[OwnershipChain],
        network: NetworkAnalysis | None,
        tool_results: dict[str, dict[str, Any]],
    ) -> str:
        """Generate detailed investigation findings."""
        parts = []
        parts.append(f"### Kết quả điều tra — {company_name}\n")

        # Patterns
        if patterns:
            parts.append("**Mẫu giao dịch đáng ngờ:**")
            for i, p in enumerate(patterns, 1):
                icon = "🔴" if p.suspicion_score > 0.7 else "🟡" if p.suspicion_score > 0.4 else "🟢"
                parts.append(f"{i}. {icon} **{p.pattern_type.value}** ({p.suspicion_score:.0%})")
                parts.append(f"   {p.description}")
                parts.append(f"   → {p.recommended_action}")

        # Ownership
        if chains:
            parts.append("\n**Cấu trúc sở hữu:**")
            for i, chain in enumerate(chains, 1):
                ubo_str = f"UBO: {chain.ubo_entity}" if chain.ubo_identified else "UBO: chưa xác định"
                parts.append(
                    f"{i}. Depth={chain.total_depth}, Shells={chain.shell_indicators}, {ubo_str}"
                )

        # Network
        if network and network.total_nodes > 0:
            parts.append(f"\n**Phân tích mạng lưới:**")
            parts.append(f"- Nodes: {network.total_nodes}, Edges: {network.total_edges}")
            parts.append(f"- Density: {network.density:.4f}")
            parts.append(f"- Communities: {network.communities}")
            parts.append(f"- Hub score: {network.hub_score:.4f}")

        return "\n".join(parts)

    def _generate_actions(
        self,
        level: SuspicionLevel,
        patterns: list[SuspiciousPattern],
        ubo_status: str,
    ) -> list[str]:
        """Generate recommended investigation actions."""
        actions = []

        if level in (SuspicionLevel.CONFIRMED, SuspicionLevel.HIGHLY_SUSPICIOUS):
            actions.append("🚨 Chuyển hồ sơ cho Đội điều tra/kiểm tra ngay.")
            actions.append("📎 Thu thập bằng chứng bổ sung từ ngân hàng, hải quan.")

        for pattern in patterns[:3]:
            actions.append(f"🔍 {pattern.recommended_action}")

        if ubo_status == "complex":
            actions.append("👤 Truy vết UBO qua OSINT và đăng ký kinh doanh.")
        elif ubo_status == "unknown":
            actions.append("👤 Yêu cầu doanh nghiệp công bố chủ sở hữu hưởng lợi.")

        if not actions:
            actions.append("✅ Tiếp tục giám sát theo chu kỳ.")

        return actions

    def _determine_escalation(
        self,
        level: SuspicionLevel,
        patterns: list[SuspiciousPattern],
    ) -> str:
        """Determine escalation level."""
        if level == SuspicionLevel.CONFIRMED:
            return "critical"
        if level == SuspicionLevel.HIGHLY_SUSPICIOUS:
            return "urgent"
        if level == SuspicionLevel.SUSPICIOUS:
            return "priority"
        return "routine"
