"""
tax_agent_analytics_agent.py – Analytics Sub-Agent (Phase 3)
=============================================================
Specialized agent for quantitative risk analysis and data-driven insights.

Capabilities:
    1. Multi-model risk aggregation (GNN + XGBoost + Delinquency + Invoice)
    2. Risk factor decomposition with SHAP-like explanations
    3. Temporal trend analysis (payment history, filing patterns)
    4. Peer comparison (industry benchmarks, regional statistics)
    5. Composite risk scoring with confidence intervals

Architecture:
    Input: Sub-task from Orchestrator (with tax_code)
    Processing: Parallel tool calls → signal aggregation → scoring
    Output: Structured risk report with explanations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    CRITICAL = "critical"       # ≥ 0.85
    HIGH = "high"               # ≥ 0.65
    MODERATE = "moderate"       # ≥ 0.40
    LOW = "low"                 # ≥ 0.20
    MINIMAL = "minimal"         # < 0.20

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if score >= 0.85:
            return cls.CRITICAL
        elif score >= 0.65:
            return cls.HIGH
        elif score >= 0.40:
            return cls.MODERATE
        elif score >= 0.20:
            return cls.LOW
        return cls.MINIMAL


@dataclass
class RiskSignal:
    """A single risk signal from a model/tool."""
    source: str                # Tool/model name
    signal_name: str           # Human-readable signal name
    score: float               # 0.0 - 1.0
    weight: float              # Weight in composite score
    details: str               # Human-readable explanation
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class RiskFactor:
    """A risk factor with explanation and contribution."""
    factor_name: str
    contribution: float        # How much this factor contributes to total risk
    direction: str             # "increases" or "decreases"
    evidence: str
    source_tool: str


@dataclass
class AnalyticsReport:
    """Structured analytics report."""
    tax_code: str
    company_name: str
    # Composite scores
    composite_risk_score: float
    risk_level: RiskLevel
    confidence: float
    # Individual signals
    risk_signals: list[RiskSignal]
    # Risk factor decomposition
    top_risk_factors: list[RiskFactor]
    # Analysis
    summary: str
    detailed_analysis: str
    recommendations: list[str]
    # Peer comparison
    industry_percentile: float | None = None
    industry_benchmark: str | None = None
    # Temporal
    risk_trend: str = "stable"     # "increasing", "decreasing", "stable", "volatile"
    # Metadata
    tools_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class AnalyticsAgent:
    """
    Specialized analytics agent for quantitative risk analysis.

    Multi-signal aggregation:
    1. Company risk profile → baseline risk
    2. Delinquency prediction → payment behavior risk
    3. Invoice risk scan → transaction integrity risk
    4. GNN analysis → graph/network risk
    5. Motif detection → pattern risk

    Each signal gets a weight, scores are aggregated into composite risk.

    Usage:
        agent = AnalyticsAgent()
        report = agent.analyze(
            tax_code="0312345678",
            tool_results={...},
        )
    """

    # Signal weights for composite scoring (9 signals: 5 ML + 4 DL)
    SIGNAL_WEIGHTS = {
        # Classic ML signals
        "company_risk_lookup": 0.12,
        "delinquency_check": 0.10,
        "invoice_risk_scan": 0.10,
        "gnn_analysis": 0.10,
        "motif_detection": 0.08,
        # Deep Learning signals
        "temporal_delinquency_deep": 0.18,
        "hetero_gnn_risk": 0.14,
        "vae_anomaly_scan": 0.12,
        "causal_uplift_recommend": 0.06,
    }

    def analyze(
        self,
        tax_code: str,
        tool_results: dict[str, dict[str, Any]],
        *,
        intent: str = "general_tax_query",
    ) -> AnalyticsReport:
        """
        Generate comprehensive analytics report from tool results.

        Args:
            tax_code: Target company tax code
            tool_results: Results from tool executor
            intent: Classified intent (for weighting)

        Returns:
            AnalyticsReport with composite scoring
        """
        t0 = time.perf_counter()

        # Extract company name
        company_result = tool_results.get("company_risk_lookup", {})
        company_name = str(company_result.get("company_name", f"DN {tax_code}"))

        # Step 1: Extract risk signals from each tool result
        signals = self._extract_signals(tool_results, intent)

        # Step 2: Compute composite risk score
        composite_score = self._compute_composite_score(signals)
        risk_level = RiskLevel.from_score(composite_score)

        # Step 3: Decompose into risk factors
        risk_factors = self._decompose_risk_factors(tool_results, signals)

        # Step 4: Compute confidence
        confidence = self._compute_confidence(signals, tool_results)

        # Step 5: Analyze risk trend
        risk_trend = self._analyze_trend(tool_results)

        # Step 6: Peer comparison
        industry = str(company_result.get("industry", ""))
        industry_percentile = self._estimate_percentile(composite_score, industry)

        # Step 7: Generate summary
        summary = self._generate_summary(
            company_name, tax_code, composite_score, risk_level,
            risk_factors, confidence,
        )

        # Step 8: Generate detailed analysis
        detailed = self._generate_detailed_analysis(
            company_name, tax_code, signals, risk_factors,
            tool_results, intent,
        )

        # Step 9: Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level, risk_factors, intent,
        )

        tools_used = [k for k in tool_results.keys() if tool_results[k].get("status") not in ("error", "not_found")]
        latency = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "[AnalyticsAgent] Report for %s: score=%.2f (%s), "
            "signals=%d, conf=%.2f in %.0fms",
            tax_code, composite_score, risk_level.value,
            len(signals), confidence, latency,
        )

        return AnalyticsReport(
            tax_code=tax_code,
            company_name=company_name,
            composite_risk_score=round(composite_score, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4),
            risk_signals=signals,
            top_risk_factors=risk_factors[:5],
            summary=summary,
            detailed_analysis=detailed,
            recommendations=recommendations,
            industry_percentile=industry_percentile,
            industry_benchmark=f"Ngành {industry}" if industry else None,
            risk_trend=risk_trend,
            tools_used=tools_used,
            latency_ms=latency,
        )

    def _extract_signals(
        self,
        tool_results: dict[str, dict[str, Any]],
        intent: str,
    ) -> list[RiskSignal]:
        """Extract risk signals from tool results."""
        signals: list[RiskSignal] = []

        # Company risk signal
        cr = tool_results.get("company_risk_lookup", {})
        if cr.get("status") == "found":
            score = float(cr.get("risk_score", 0)) / 100.0
            signals.append(RiskSignal(
                source="company_risk_lookup",
                signal_name="Điểm rủi ro tổng thể",
                score=min(1.0, score),
                weight=self.SIGNAL_WEIGHTS["company_risk_lookup"],
                details=f"Mức {cr.get('risk_level', 'N/A')} ({cr.get('risk_score', 0)}/100)",
                raw_data=cr,
            ))

        # Delinquency signal
        dq = tool_results.get("delinquency_check", {})
        if dq.get("status") == "analyzed":
            # Use max probability across time horizons
            prob_30 = float(dq.get("prob_30d", 0))
            prob_60 = float(dq.get("prob_60d", 0))
            prob_90 = float(dq.get("prob_90d", 0))
            max_prob = max(prob_30, prob_60, prob_90)
            signals.append(RiskSignal(
                source="delinquency_check",
                signal_name="Rủi ro nợ đọng",
                score=min(1.0, max_prob),
                weight=self.SIGNAL_WEIGHTS["delinquency_check"],
                details=f"P(30d)={prob_30:.0%}, P(60d)={prob_60:.0%}, P(90d)={prob_90:.0%}",
                raw_data=dq,
            ))

        # Invoice risk signal
        ir = tool_results.get("invoice_risk_scan", {})
        if ir.get("status") == "analyzed":
            risk_ratio = float(ir.get("risk_ratio", 0))
            signals.append(RiskSignal(
                source="invoice_risk_scan",
                signal_name="Rủi ro hóa đơn",
                score=min(1.0, risk_ratio * 2.0),  # Scale up: 50% risky = 100% score
                weight=self.SIGNAL_WEIGHTS["invoice_risk_scan"],
                details=(
                    f"{ir.get('risky_invoices', 0)}/{ir.get('total_invoices', 0)} "
                    f"hóa đơn rủi ro ({risk_ratio:.0%})"
                ),
                raw_data=ir,
            ))

        # GNN signal
        gnn = tool_results.get("gnn_analysis", {})
        if gnn.get("status") == "found":
            gnn_outputs = gnn.get("gnn_outputs", {})
            gnn_prob = float(gnn_outputs.get("risk_probability", gnn_outputs.get("fraud_probability", 0)))
            signals.append(RiskSignal(
                source="gnn_analysis",
                signal_name="Rủi ro đồ thị giao dịch",
                score=min(1.0, gnn_prob),
                weight=self.SIGNAL_WEIGHTS["gnn_analysis"],
                details=f"GNN score: {gnn_prob:.2f}",
                raw_data=gnn,
            ))

        # Motif signal
        motif = tool_results.get("motif_detection", {})
        if motif.get("status") == "analyzed":
            summary = motif.get("summary", {})
            total_patterns = sum(
                int(v) for v in summary.values()
                if isinstance(v, (int, float))
            )
            motif_score = min(1.0, total_patterns / 10.0)
            signals.append(RiskSignal(
                source="motif_detection",
                signal_name="Mẫu giao dịch đáng ngờ",
                score=motif_score,
                weight=self.SIGNAL_WEIGHTS["motif_detection"],
                details=f"{total_patterns} patterns detected",
                raw_data=motif,
            ))

        # ═══ NEW DEEP LEARNING SIGNALS ═══

        # Temporal Transformer signal
        temporal = tool_results.get("temporal_delinquency_deep", {})
        if temporal.get("status") == "analyzed":
            prob_30 = float(temporal.get("prob_30d", 0))
            prob_60 = float(temporal.get("prob_60d", 0))
            prob_90 = float(temporal.get("prob_90d", 0))
            max_prob = max(prob_30, prob_60, prob_90)
            signals.append(RiskSignal(
                source="temporal_delinquency_deep",
                signal_name="[DL] Dự báo nợ đọng (Transformer)",
                score=min(1.0, max_prob),
                weight=self.SIGNAL_WEIGHTS["temporal_delinquency_deep"],
                details=f"P(30d)={prob_30:.0%}, P(60d)={prob_60:.0%}, P(90d)={prob_90:.0%} | {temporal.get('architecture', '')}",
                raw_data=temporal,
            ))

        # HeteroGNN signal
        hgnn = tool_results.get("hetero_gnn_risk", {})
        if hgnn.get("status") == "analyzed":
            fraud_prob = float(hgnn.get("fraud_probability", 0))
            signals.append(RiskSignal(
                source="hetero_gnn_risk",
                signal_name="[DL] Rủi ro đồ thị dị thể (HGT)",
                score=min(1.0, fraud_prob),
                weight=self.SIGNAL_WEIGHTS["hetero_gnn_risk"],
                details=f"HGT fraud={fraud_prob:.2f} | Neighbors={hgnn.get('total_neighbors', 0)} | {hgnn.get('architecture', '')}",
                raw_data=hgnn,
            ))

        # VAE Anomaly signal
        vae = tool_results.get("vae_anomaly_scan", {})
        if vae.get("status") == "analyzed":
            anomaly_ratio = float(vae.get("anomaly_ratio", 0))
            signals.append(RiskSignal(
                source="vae_anomaly_scan",
                signal_name="[DL] Bất thường hóa đơn (VAE)",
                score=min(1.0, anomaly_ratio * 5.0),  # Scale: 20% anomaly = 100%
                weight=self.SIGNAL_WEIGHTS["vae_anomaly_scan"],
                details=f"{vae.get('anomaly_count', 0)}/{vae.get('total_invoices', 0)} anomalies ({anomaly_ratio:.1%}) | {vae.get('architecture', '')}",
                raw_data=vae,
            ))

        # Causal Uplift signal
        uplift = tool_results.get("causal_uplift_recommend", {})
        if uplift.get("status") == "analyzed":
            cate = float(uplift.get("cate_score", 0))
            signals.append(RiskSignal(
                source="causal_uplift_recommend",
                signal_name="[CI] Hiệu quả hành động (CATE)",
                score=min(1.0, abs(cate)),
                weight=self.SIGNAL_WEIGHTS["causal_uplift_recommend"],
                details=f"CATE={cate:.4f} | Action: {uplift.get('recommended_action', 'N/A')}",
                raw_data=uplift,
            ))

        return signals

    def _compute_composite_score(self, signals: list[RiskSignal]) -> float:
        """Compute weighted composite risk score."""
        if not signals:
            return 0.0

        weighted_sum = sum(s.weighted_score for s in signals)
        total_weight = sum(s.weight for s in signals)

        if total_weight < 1e-9:
            return 0.0

        return min(1.0, weighted_sum / total_weight)

    def _decompose_risk_factors(
        self,
        tool_results: dict[str, dict[str, Any]],
        signals: list[RiskSignal],
    ) -> list[RiskFactor]:
        """Decompose risk into contributing factors."""
        factors: list[RiskFactor] = []
        total_score = sum(s.weighted_score for s in signals)

        for signal in signals:
            if signal.score < 0.1:
                direction = "decreases"
            else:
                direction = "increases"

            contribution = signal.weighted_score / max(total_score, 1e-9)
            factors.append(RiskFactor(
                factor_name=signal.signal_name,
                contribution=round(contribution, 4),
                direction=direction,
                evidence=signal.details,
                source_tool=signal.source,
            ))

        # Sort by contribution (highest first)
        factors.sort(key=lambda f: abs(f.contribution), reverse=True)
        return factors

    def _compute_confidence(
        self,
        signals: list[RiskSignal],
        tool_results: dict[str, dict[str, Any]],
    ) -> float:
        """Compute confidence in the composite score."""
        if not signals:
            return 0.0

        # Factor 1: Number of signals
        signal_coverage = min(1.0, len(signals) / 4.0) * 0.4

        # Factor 2: Signal agreement (low variance = high confidence)
        scores = [s.score for s in signals]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        agreement = max(0.0, 1.0 - variance * 4.0) * 0.3

        # Factor 3: Tool success rate
        total_tools = len(tool_results)
        successful = sum(
            1 for r in tool_results.values()
            if isinstance(r, dict) and r.get("status") not in ("error", "no_data", "not_found")
        )
        success_rate = (successful / max(total_tools, 1)) * 0.3

        return min(1.0, signal_coverage + agreement + success_rate)

    def _analyze_trend(self, tool_results: dict[str, dict[str, Any]]) -> str:
        """Analyze risk trend from delinquency data."""
        dq = tool_results.get("delinquency_check", {})
        if dq.get("status") != "analyzed":
            return "unknown"

        prob_30 = float(dq.get("prob_30d", 0))
        prob_60 = float(dq.get("prob_60d", 0))
        prob_90 = float(dq.get("prob_90d", 0))

        if prob_90 > prob_60 > prob_30:
            return "increasing"
        elif prob_30 > prob_60 > prob_90:
            return "decreasing"
        elif abs(prob_90 - prob_30) < 0.1:
            return "stable"
        return "volatile"

    def _estimate_percentile(self, score: float, industry: str) -> float | None:
        """Estimate industry percentile (simplified)."""
        # Simplified: assume normal distribution around 0.5
        # In production, this would use actual industry data
        if score >= 0.8:
            return 95.0
        elif score >= 0.6:
            return 80.0
        elif score >= 0.4:
            return 60.0
        elif score >= 0.2:
            return 30.0
        return 10.0

    def _generate_summary(
        self,
        company_name: str,
        tax_code: str,
        score: float,
        level: RiskLevel,
        factors: list[RiskFactor],
        confidence: float,
    ) -> str:
        """Generate executive summary."""
        level_vi = {
            RiskLevel.CRITICAL: "NGHIÊM TRỌNG",
            RiskLevel.HIGH: "CAO",
            RiskLevel.MODERATE: "TRUNG BÌNH",
            RiskLevel.LOW: "THẤP",
            RiskLevel.MINIMAL: "RẤT THẤP",
        }

        top_factor = factors[0].factor_name if factors else "chưa xác định"

        return (
            f"Doanh nghiệp **{company_name}** (MST: {tax_code}) — "
            f"Mức rủi ro: **{level_vi.get(level, level.value)}** "
            f"(điểm: {score:.0%}, độ tin cậy: {confidence:.0%}). "
            f"Yếu tố rủi ro chính: {top_factor}."
        )

    def _generate_detailed_analysis(
        self,
        company_name: str,
        tax_code: str,
        signals: list[RiskSignal],
        factors: list[RiskFactor],
        tool_results: dict[str, dict[str, Any]],
        intent: str,
    ) -> str:
        """Generate detailed analysis."""
        parts = []
        parts.append(f"### Phân tích rủi ro chi tiết — {company_name}\n")

        # Signal breakdown
        parts.append("| Chỉ số | Điểm | Trọng số | Chi tiết |")
        parts.append("|--------|------|----------|---------|")
        for s in signals:
            bar = "█" * int(s.score * 10)
            parts.append(
                f"| {s.signal_name} | {s.score:.0%} {bar} | {s.weight:.0%} | {s.details} |"
            )

        # Factor decomposition
        if factors:
            parts.append("\n**Phân tích yếu tố rủi ro:**")
            for i, f in enumerate(factors[:5], 1):
                arrow = "↑" if f.direction == "increases" else "↓"
                parts.append(
                    f"{i}. {arrow} **{f.factor_name}** — đóng góp {f.contribution:.0%}: {f.evidence}"
                )

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        factors: list[RiskFactor],
        intent: str,
    ) -> list[str]:
        """Generate targeted recommendations."""
        recs = []

        if risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            recs.append("🔴 Ưu tiên cao: đưa vào danh sách thanh tra/kiểm tra ngay.")

        # Factor-specific recommendations
        for f in factors[:5]:
            if f.source_tool == "delinquency_check" and f.direction == "increases":
                recs.append("📋 Theo dõi sát tình hình nộp thuế, xem xét cưỡng chế nếu tiếp tục vi phạm.")
            elif f.source_tool == "invoice_risk_scan" and f.direction == "increases":
                recs.append("🔍 Rà soát hóa đơn đầu vào, đối chiếu chéo với bên bán.")
            elif f.source_tool == "gnn_analysis" and f.direction == "increases":
                recs.append("🕸️ Mở rộng điều tra mạng lưới giao dịch liên quan.")
            elif f.source_tool == "motif_detection" and f.direction == "increases":
                recs.append("⭕ Điều tra chi tiết mẫu giao dịch vòng tròn / carousel.")
            # DL model recommendations
            elif f.source_tool == "temporal_delinquency_deep" and f.direction == "increases":
                recs.append("🧠 [DL] Temporal Transformer cho thấy xu hướng nợ đọng gia tăng — cần hành động sớm.")
            elif f.source_tool == "hetero_gnn_risk" and f.direction == "increases":
                recs.append("🔗 [DL] HGT phát hiện rủi ro lan truyền từ các thực thể liên quan trong đồ thị.")
            elif f.source_tool == "vae_anomaly_scan" and f.direction == "increases":
                recs.append("🔬 [DL] VAE phát hiện các hóa đơn có mẫu bất thường — cần kiểm tra chi tiết.")
            elif f.source_tool == "causal_uplift_recommend":
                recs.append("📊 [CI] Causal Inference đề xuất hành động thu nợ có hiệu quả cao nhất.")

        if risk_level in (RiskLevel.LOW, RiskLevel.MINIMAL):
            recs.append("✅ Hồ sơ rủi ro thấp. Tiếp tục giám sát theo chu kỳ thường niên.")

        return recs
