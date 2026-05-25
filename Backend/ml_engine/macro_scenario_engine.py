"""
macro_scenario_engine.py – Province-Level Economic Scenario Engine
===================================================================
Deterministic + AI-augmented scenario simulation for 63 Vietnamese provinces.
Computes projected tax revenue under different economic event scenarios,
then generates Vietnamese-language narrative reports via LLM.

Architecture:
    1. Load province economic baseline from JSON/DB
    2. Apply event impact coefficients (COVID, trade war, natural disaster, etc.)
    3. Apply user slider adjustments (GDP, tax rate, compliance)
    4. Compute projected metrics
    5. Generate narrative via Groq/Gemini LLM

Novel contribution: First integration of province-level economic digital twin
with event-driven scenario simulation for tax administration research.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────────────────────────────────────────
#  Data Loading
# ────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"
_PROVINCES_PATH = _DATA_DIR / "vietnam_provinces.json"
_PROVINCES_34_PATH = _DATA_DIR / "vietnam_provinces_34_2025.json"
_EVENTS_PATH = _DATA_DIR / "historical_economic_events.json"

_provinces_cache: Optional[List[Dict]] = None
_provinces_34_cache: Optional[List[Dict]] = None
_events_cache: Optional[List[Dict]] = None


def load_provinces(boundary_version: Optional[str] = None) -> List[Dict]:
    if boundary_version == "vn_34_2025":
        return load_provinces_34()
    global _provinces_cache
    if _provinces_cache is not None:
        return _provinces_cache
    if _PROVINCES_PATH.exists():
        with open(_PROVINCES_PATH, "r", encoding="utf-8") as f:
            _provinces_cache = json.load(f)
    else:
        _provinces_cache = []
    return _provinces_cache


def load_provinces_34() -> List[Dict]:
    global _provinces_34_cache
    if _provinces_34_cache is not None:
        return _provinces_34_cache
    if _PROVINCES_34_PATH.exists():
        with open(_PROVINCES_34_PATH, "r", encoding="utf-8") as f:
            _provinces_34_cache = json.load(f)
    else:
        _provinces_34_cache = []
    return _provinces_34_cache


def load_events() -> List[Dict]:
    global _events_cache
    if _events_cache is not None:
        return _events_cache
    if _EVENTS_PATH.exists():
        with open(_EVENTS_PATH, "r", encoding="utf-8") as f:
            _events_cache = json.load(f)
    else:
        _events_cache = []
    return _events_cache


def get_province_by_code(code: str) -> Optional[Dict]:
    for p in load_provinces():
        if p.get("province_code") == code:
            return p
    for p in load_provinces_34():
        if p.get("province_code") == code:
            return p
    return None


def get_event_by_key(key: str) -> Optional[Dict]:
    for e in load_events():
        if e.get("event_key") == key:
            return e
    return None


def _parse_date_key(value: Any) -> str:
    raw = str(value or "")
    return raw if re.match(r"^\d{4}-\d{2}-\d{2}$", raw) else "1900-01-01"


def _severity_weight(severity: str) -> float:
    return {
        "low": 0.05,
        "medium": 0.10,
        "high": 0.16,
        "extreme": 0.24,
    }.get(str(severity or "").lower(), 0.10)


def event_relevance_score(event: Dict[str, Any], province: Dict[str, Any]) -> float:
    """Score how relevant a macro event is to a province."""
    province_code = str(province.get("province_code") or "")
    member_codes = {str(x) for x in (province.get("member_codes") or [])}
    affected = [str(x) for x in (event.get("affected_provinces") or [])]
    dominant = {str(x).lower() for x in (province.get("dominant_sectors") or [])}
    sectors = {str(x).lower() for x in (event.get("affected_sectors") or [])}

    score = 0.0
    if not affected:
        score += 1.0
    elif province_code in affected or member_codes.intersection(affected):
        score += 2.4

    if dominant and sectors:
        score += 0.45 * len(dominant & sectors)

    score += _severity_weight(str(event.get("severity") or "medium"))
    impact = abs(float(event.get("impact_gdp_pct") or 0.0)) + abs(float(event.get("impact_tax_revenue_pct") or 0.0))
    score += min(0.9, impact / 20.0)

    # Newer events are more useful for narrative context.
    try:
        year = int(str(event.get("start_date") or "1900")[:4])
        score += max(0.0, min(0.35, (year - 1990) / 100.0))
    except Exception:
        pass
    return round(float(score), 6)


def get_events_for_province(
    province_code: str,
    *,
    event_type: Optional[str] = None,
    limit: int = 60,
) -> List[Dict[str, Any]]:
    """Return events relevant to a province, including global events."""
    province = get_province_by_code(province_code)
    if not province:
        return []

    rows: List[Dict[str, Any]] = []
    province_codes = {str(province_code)}
    province_codes.update(str(x) for x in (province.get("member_codes") or []))
    for event in load_events():
        if event_type and str(event.get("event_type")) != str(event_type):
            continue
        affected = [str(x) for x in (event.get("affected_provinces") or [])]
        if affected and not province_codes.intersection(affected):
            continue
        score = event_relevance_score(event, province)
        item = dict(event)
        item["relevance_score"] = score
        item["applies_to_selected_province"] = (not affected) or (province_code in affected)
        rows.append(item)

    rows.sort(key=lambda e: (float(e.get("relevance_score") or 0.0), _parse_date_key(e.get("start_date"))), reverse=True)
    return rows[: max(1, min(int(limit), 250))]


def build_vietnam_geojson(boundary_version: Optional[str] = None) -> Dict[str, Any]:
    """Load versioned Vietnam boundaries or an explicitly marked fallback."""
    from ml_engine.admin_boundary_manager import load_boundary_geojson

    return load_boundary_geojson(boundary_version=boundary_version)


# ────────────────────────────────────────────────────────────
#  Scenario Computation
# ────────────────────────────────────────────────────────────

@dataclass
class ScenarioParams:
    """User-adjustable scenario parameters."""
    gdp_delta_pct: float = 0.0        # % change to province GDP
    tax_rate_delta: float = 0.0       # absolute change to tax rate
    compliance_delta: float = 0.0     # absolute change to compliance rate
    unemployment_delta: float = 0.0   # absolute change to unemployment
    fdi_delta_pct: float = 0.0        # % change to FDI
    event_key: Optional[str] = None   # historical event to apply


@dataclass
class ScenarioResult:
    """Output of scenario computation."""
    province_code: str
    province_name: str
    region: str
    scenario_title: str
    # Baseline
    baseline_gdp: float
    baseline_revenue: float
    baseline_risk: str
    baseline_tax_rate: float
    # Projected
    projected_gdp: float
    projected_revenue: float
    projected_risk: str
    projected_tax_rate: float
    # Deltas
    delta_revenue_pct: float
    delta_gdp_pct: float
    # Metrics
    projected_compliance: float
    projected_unemployment: float
    projected_fdi: float
    # Event
    event_applied: Optional[str] = None
    event_impact_description: str = ""
    related_events: List[Dict[str, Any]] = field(default_factory=list)
    impact_drivers: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    uncertainty_band_revenue: Dict[str, float] = field(default_factory=dict)
    # Narrative
    narrative_text: str = ""
    # Metadata
    generated_at: str = ""
    model_version: str = "macro_scenario_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_scenario(
    province_code: str,
    params: ScenarioParams,
) -> ScenarioResult:
    """
    Compute a province-level economic scenario.

    Revenue = GDP × effective_tax_rate × compliance_rate × event_multiplier
    Risk = f(unemployment, compliance, sector_vulnerability)
    """
    # Clone params to avoid mutating the caller's object (critical for batch loops)
    params = ScenarioParams(
        gdp_delta_pct=params.gdp_delta_pct,
        tax_rate_delta=params.tax_rate_delta,
        compliance_delta=params.compliance_delta,
        unemployment_delta=params.unemployment_delta,
        fdi_delta_pct=params.fdi_delta_pct,
        event_key=params.event_key,
    )

    province = get_province_by_code(province_code)
    if not province:
        raise ValueError(f"Province not found: {province_code}")

    # Extract baseline
    base_gdp = float(province.get("gdp_billion_vnd", 0))
    base_revenue = float(province.get("tax_revenue_billion_vnd", 0))
    base_compliance = float(province.get("compliance_rate", 0.85))
    base_unemployment = float(province.get("unemployment_rate", 2.5))
    base_fdi = float(province.get("fdi_billion_usd", 0))
    base_risk = str(province.get("risk_level", "medium"))
    # Calibrate the effective tax take to each province baseline. A fixed 10%
    # heuristic made no-change scenarios drift far away from observed revenue.
    if base_gdp > 0 and base_compliance > 0:
        base_tax_rate = base_revenue / max(base_gdp * base_compliance, 0.01)
        base_tax_rate = max(0.02, base_tax_rate)
    else:
        base_tax_rate = 0.10

    # Apply event impacts if specified
    event_multiplier = 1.0
    event_description = ""
    event = None
    if params.event_key:
        event = get_event_by_key(params.event_key)
        if event:
            # Check if this province is affected
            affected = event.get("affected_provinces", [])
            is_affected = (not affected) or (province_code in affected)

            if is_affected:
                gdp_impact = float(event.get("impact_gdp_pct", 0)) / 100.0
                tax_impact = float(event.get("impact_tax_revenue_pct", 0)) / 100.0
                unemployment_impact = float(event.get("impact_unemployment_pct", 0))

                event_multiplier = 1.0 + tax_impact
                params.gdp_delta_pct += gdp_impact * 100
                params.unemployment_delta += unemployment_impact

                event_description = event.get("description_vi", event.get("event_name_vi", ""))

    # Compute projected values
    projected_gdp = base_gdp * (1.0 + params.gdp_delta_pct / 100.0)
    effective_tax_rate = max(0.0, base_tax_rate + params.tax_rate_delta)
    effective_compliance = min(1.0, max(0.3, base_compliance + params.compliance_delta))
    projected_unemployment = max(0.0, base_unemployment + params.unemployment_delta)
    projected_fdi = base_fdi * (1.0 + params.fdi_delta_pct / 100.0)

    # Revenue projection
    # Revenue = GDP × tax_rate × compliance × event_multiplier × (1 - unemployment_penalty)
    unemployment_penalty = max(-0.08, min(0.25, (projected_unemployment - base_unemployment) * 0.02))
    projected_revenue = (
        projected_gdp
        * effective_tax_rate
        * effective_compliance
        * event_multiplier
        * (1.0 - unemployment_penalty)
    )

    # Risk level computation
    risk_score = 0.0
    if projected_unemployment > 5.0:
        risk_score += 0.3
    if effective_compliance < 0.75:
        risk_score += 0.3
    if params.gdp_delta_pct < -5.0:
        risk_score += 0.2
    if event and float(event.get("impact_gdp_pct", 0)) < -3.0:
        risk_score += 0.2

    if risk_score >= 0.5:
        projected_risk = "high"
    elif risk_score >= 0.2:
        projected_risk = "medium"
    else:
        projected_risk = "low"

    # Delta calculations
    delta_revenue = projected_revenue - base_revenue
    delta_revenue_pct = (delta_revenue / max(base_revenue, 0.01)) * 100
    delta_gdp_pct = ((projected_gdp - base_gdp) / max(base_gdp, 0.01)) * 100

    # Build scenario title
    if event:
        title = f"Kịch bản: {event.get('event_name_vi', 'Sự kiện')} tại {province.get('province_name', '')}"
    else:
        title = f"Kịch bản giả định tại {province.get('province_name', '')}"

    adjustment_pressure = (
        abs(params.gdp_delta_pct) / 50.0
        + abs(params.tax_rate_delta) / 0.1
        + abs(params.compliance_delta) / 0.5
        + abs(params.unemployment_delta) / 10.0
        + abs(params.fdi_delta_pct) / 100.0
    )
    event_uncertainty = _severity_weight(str(event.get("severity") if event else "medium")) if event else 0.05
    uncertainty_pct = min(0.35, 0.045 + event_uncertainty + 0.035 * adjustment_pressure)
    confidence_score = max(0.45, min(0.92, 0.88 - uncertainty_pct * 0.9))
    related_events = get_events_for_province(province_code, limit=6)
    if params.event_key:
        related_events = [e for e in related_events if e.get("event_key") != params.event_key][:5]

    impact_drivers = [
        {
            "factor": str(event.get("event_name_vi") or event.get("event_name") or "Historical event") if event else "No selected historical event",
            "delta_pct": round(float(event.get("impact_tax_revenue_pct") or 0.0), 3) if event else 0.0,
            "direction": "positive" if (float(event.get("impact_tax_revenue_pct") or 0.0) if event else 0.0) >= 0 else "negative",
        },
        {"factor": "GDP/GRDP", "delta_pct": round(float(params.gdp_delta_pct), 3), "direction": "positive" if params.gdp_delta_pct >= 0 else "negative"},
        {"factor": "Effective tax rate", "delta_pct": round(float(params.tax_rate_delta * 100.0), 3), "direction": "positive" if params.tax_rate_delta >= 0 else "negative"},
        {"factor": "Tax compliance", "delta_pct": round(float(params.compliance_delta * 100.0), 3), "direction": "positive" if params.compliance_delta >= 0 else "negative"},
        {"factor": "Unemployment pressure", "delta_pct": round(float(params.unemployment_delta), 3), "direction": "negative" if params.unemployment_delta > 0 else "positive"},
    ]

    return ScenarioResult(
        province_code=province_code,
        province_name=province.get("province_name", ""),
        region=province.get("region", ""),
        scenario_title=title,
        baseline_gdp=round(base_gdp, 2),
        baseline_revenue=round(base_revenue, 2),
        baseline_risk=base_risk,
        baseline_tax_rate=round(base_tax_rate, 4),
        projected_gdp=round(projected_gdp, 2),
        projected_revenue=round(projected_revenue, 2),
        projected_risk=projected_risk,
        projected_tax_rate=round(effective_tax_rate, 4),
        delta_revenue_pct=round(delta_revenue_pct, 2),
        delta_gdp_pct=round(delta_gdp_pct, 2),
        projected_compliance=round(effective_compliance, 4),
        projected_unemployment=round(projected_unemployment, 2),
        projected_fdi=round(projected_fdi, 2),
        event_applied=params.event_key,
        event_impact_description=event_description,
        related_events=related_events,
        impact_drivers=impact_drivers,
        confidence_score=round(float(confidence_score), 4),
        uncertainty_band_revenue={
            "low": round(projected_revenue * (1.0 - uncertainty_pct), 2),
            "base": round(projected_revenue, 2),
            "high": round(projected_revenue * (1.0 + uncertainty_pct), 2),
            "uncertainty_pct": round(float(uncertainty_pct * 100.0), 2),
        },
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ────────────────────────────────────────────────────────────
#  Monte Carlo Simulation (Stochastic Policy Analysis)
# ────────────────────────────────────────────────────────────

def run_monte_carlo(
    province_code: str,
    params: ScenarioParams,
    *,
    n_simulations: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run N stochastic perturbations of the scenario to produce a
    probability distribution of projected revenue outcomes.

    Each simulation jitters every input parameter by a Gaussian noise
    proportional to its magnitude, then runs compute_scenario().

    Returns percentiles (P5, P10, P25, P50, P75, P90, P95),
    Value-at-Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall).

    Reference:
        Glasserman, "Monte Carlo Methods in Financial Engineering", Springer 2003
    """
    import random as _random
    rng = _random.Random(seed)

    revenues: List[float] = []
    gdps: List[float] = []

    for _ in range(n_simulations):
        jittered = ScenarioParams(
            gdp_delta_pct=params.gdp_delta_pct + rng.gauss(0, max(0.5, abs(params.gdp_delta_pct) * 0.15)),
            tax_rate_delta=params.tax_rate_delta + rng.gauss(0, max(0.002, abs(params.tax_rate_delta) * 0.12)),
            compliance_delta=params.compliance_delta + rng.gauss(0, max(0.005, abs(params.compliance_delta) * 0.12)),
            unemployment_delta=params.unemployment_delta + rng.gauss(0, max(0.15, abs(params.unemployment_delta) * 0.15)),
            fdi_delta_pct=params.fdi_delta_pct + rng.gauss(0, max(1.0, abs(params.fdi_delta_pct) * 0.18)),
            event_key=params.event_key,
        )
        try:
            result = compute_scenario(province_code, jittered)
            revenues.append(result.projected_revenue)
            gdps.append(result.projected_gdp)
        except Exception:
            continue

    if len(revenues) < 10:
        return {"error": "insufficient_simulations", "completed": len(revenues)}

    revenues.sort()
    n = len(revenues)

    def percentile(arr: List[float], p: float) -> float:
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = min(f + 1, n - 1)
        return arr[f] + (arr[c] - arr[f]) * (k - f)

    p5 = percentile(revenues, 5)
    p10 = percentile(revenues, 10)
    p25 = percentile(revenues, 25)
    p50 = percentile(revenues, 50)
    p75 = percentile(revenues, 75)
    p90 = percentile(revenues, 90)
    p95 = percentile(revenues, 95)
    mean_rev = sum(revenues) / n
    std_rev = math.sqrt(sum((r - mean_rev) ** 2 for r in revenues) / n)

    # VaR at 5% (worst 5% of outcomes)
    var_5 = p5
    # CVaR / Expected Shortfall (average of worst 5%)
    tail_count = max(1, int(n * 0.05))
    cvar_5 = sum(revenues[:tail_count]) / tail_count

    # Build histogram bins for frontend visualization
    bin_count = 20
    min_rev = revenues[0]
    max_rev = revenues[-1]
    bin_width = max(0.01, (max_rev - min_rev) / bin_count)
    histogram = []
    for i in range(bin_count):
        lo = min_rev + i * bin_width
        hi = lo + bin_width
        count = sum(1 for r in revenues if lo <= r < hi) if i < bin_count - 1 else sum(1 for r in revenues if lo <= r <= hi)
        histogram.append({
            "bin_start": round(lo, 2),
            "bin_end": round(hi, 2),
            "count": count,
            "density": round(count / n, 4),
        })

    return {
        "n_simulations": n,
        "percentiles": {
            "p5": round(p5, 2), "p10": round(p10, 2), "p25": round(p25, 2),
            "p50": round(p50, 2), "p75": round(p75, 2), "p90": round(p90, 2),
            "p95": round(p95, 2),
        },
        "mean": round(mean_rev, 2),
        "std": round(std_rev, 2),
        "var_5pct": round(var_5, 2),
        "cvar_5pct": round(cvar_5, 2),
        "histogram": histogram,
        "coefficient_of_variation": round(std_rev / max(mean_rev, 0.01), 4),
    }


# ────────────────────────────────────────────────────────────
#  Sensitivity / Tornado Analysis
# ────────────────────────────────────────────────────────────

def run_sensitivity_analysis(
    province_code: str,
    params: ScenarioParams,
    *,
    sweep_pct: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Tornado-style one-at-a-time (OAT) sensitivity analysis.

    For each input parameter, sweep it by ±sweep_pct while holding
    all other parameters at their base values. Measure the resulting
    change in projected_revenue to determine which parameter has the
    largest marginal impact.

    Reference:
        Saltelli et al., "Sensitivity Analysis in Practice", Wiley 2004
    """
    base_result = compute_scenario(province_code, params)
    base_revenue = base_result.projected_revenue

    factors = [
        ("GDP Delta (%)", "gdp_delta_pct", params.gdp_delta_pct, 1.0),
        ("Thuế suất hiệu dụng", "tax_rate_delta", params.tax_rate_delta, 0.01),
        ("Tuân thủ thuế", "compliance_delta", params.compliance_delta, 0.01),
        ("Thất nghiệp", "unemployment_delta", params.unemployment_delta, 0.3),
        ("FDI Delta (%)", "fdi_delta_pct", params.fdi_delta_pct, 1.0),
    ]

    results = []
    for label, attr, base_val, min_step in factors:
        step = max(min_step, abs(base_val) * sweep_pct / 100.0)

        low_params = ScenarioParams(
            gdp_delta_pct=params.gdp_delta_pct,
            tax_rate_delta=params.tax_rate_delta,
            compliance_delta=params.compliance_delta,
            unemployment_delta=params.unemployment_delta,
            fdi_delta_pct=params.fdi_delta_pct,
            event_key=params.event_key,
        )
        high_params = ScenarioParams(
            gdp_delta_pct=params.gdp_delta_pct,
            tax_rate_delta=params.tax_rate_delta,
            compliance_delta=params.compliance_delta,
            unemployment_delta=params.unemployment_delta,
            fdi_delta_pct=params.fdi_delta_pct,
            event_key=params.event_key,
        )
        setattr(low_params, attr, base_val - step)
        setattr(high_params, attr, base_val + step)

        try:
            low_result = compute_scenario(province_code, low_params)
            high_result = compute_scenario(province_code, high_params)
            low_rev = low_result.projected_revenue
            high_rev = high_result.projected_revenue
        except Exception:
            low_rev = base_revenue
            high_rev = base_revenue

        spread = high_rev - low_rev
        results.append({
            "factor": label,
            "param": attr,
            "base_value": round(base_val, 4),
            "low_value": round(base_val - step, 4),
            "high_value": round(base_val + step, 4),
            "revenue_low": round(low_rev, 2),
            "revenue_high": round(high_rev, 2),
            "revenue_base": round(base_revenue, 2),
            "spread": round(abs(spread), 2),
            "direction": "positive" if spread >= 0 else "negative",
        })

    # Sort by absolute spread descending (most impactful first)
    results.sort(key=lambda x: x["spread"], reverse=True)
    return results


# ────────────────────────────────────────────────────────────
#  Narrative Generation
# ────────────────────────────────────────────────────────────

def _build_template_narrative(result: ScenarioResult) -> str:
    """Fallback template-based narrative when LLM is unavailable."""
    direction = "tăng trưởng" if result.delta_revenue_pct > 0 else "suy giảm"
    abs_delta = abs(result.delta_revenue_pct)

    narrative = f"""📊 **{result.scenario_title}**

**Tổng quan:** {result.province_name} ({result.region}) hiện có GDP {result.baseline_gdp:,.0f} tỷ VND với thu thuế {result.baseline_revenue:,.0f} tỷ VND.

**Kịch bản dự báo:** Trong kịch bản này, doanh thu thuế dự kiến {direction} {abs_delta:.1f}%, đạt mức {result.projected_revenue:,.0f} tỷ VND. GDP tỉnh dự kiến thay đổi {result.delta_gdp_pct:+.1f}%.

**Các chỉ số quan trọng:**
- Tỷ lệ tuân thủ thuế: {result.projected_compliance*100:.1f}%
- Tỷ lệ thất nghiệp: {result.projected_unemployment:.1f}%
- FDI: {result.projected_fdi:.2f} tỷ USD
- Mức rủi ro: {"🔴 Cao" if result.projected_risk == "high" else "🟡 Trung bình" if result.projected_risk == "medium" else "🟢 Thấp"}
"""

    if result.event_applied and result.event_impact_description:
        narrative += f"\n**Tác động sự kiện:** {result.event_impact_description}\n"

    if result.projected_risk == "high":
        narrative += "\n⚠️ **Khuyến nghị:** Cần tăng cường giám sát thu thuế, mở rộng diện kiểm tra, hỗ trợ doanh nghiệp duy trì tuân thủ."
    elif result.delta_revenue_pct > 5:
        narrative += "\n✅ **Khuyến nghị:** Cơ hội mở rộng cơ sở thuế. Có thể đầu tư vào số hóa quy trình thu thuế."
    else:
        narrative += "\n📋 **Khuyến nghị:** Duy trì giám sát ổn định, theo dõi các chỉ số kinh tế vĩ mô hàng quý."

    return narrative


async def generate_narrative_llm(result: ScenarioResult) -> str:
    """
    Generate a Vietnamese-language economic narrative using Gemini API.
    Falls back to template if API fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _build_template_narrative(result)

    prompt = f"""Bạn là chuyên gia kinh tế vĩ mô Việt Nam. Hãy viết một bài phân tích ngắn (200-300 từ) bằng tiếng Việt về kịch bản kinh tế sau:

Tỉnh/Thành: {result.province_name} ({result.region})
GDP hiện tại: {result.baseline_gdp:,.0f} tỷ VND
Thu thuế hiện tại: {result.baseline_revenue:,.0f} tỷ VND
Thu thuế dự kiến: {result.projected_revenue:,.0f} tỷ VND (thay đổi {result.delta_revenue_pct:+.1f}%)
GDP dự kiến thay đổi: {result.delta_gdp_pct:+.1f}%
Tỷ lệ tuân thủ: {result.projected_compliance*100:.1f}%
Thất nghiệp: {result.projected_unemployment:.1f}%
FDI: {result.projected_fdi:.2f} tỷ USD
Mức rủi ro: {result.projected_risk}
{"Sự kiện áp dụng: " + result.event_impact_description if result.event_impact_description else "Không có sự kiện đặc biệt"}

Yêu cầu:
- Viết theo phong cách bài báo kinh tế chuyên nghiệp
- Phân tích nguyên nhân và hệ quả
- Đưa ra 2-3 khuyến nghị chính sách cụ thể
- Sử dụng số liệu minh họa"""

    try:
        text = await asyncio.to_thread(_generate_gemini_text, api_key, prompt)
        if text:
            return text
    except Exception as e:
        print(f"[MacroScenario] Gemini API error: {e}")

    # Fallback to template
    return _build_template_narrative(result)


def generate_narrative_sync(result: ScenarioResult) -> str:
    """Synchronous version of narrative generation (template-only, no API call)."""
    return _build_template_narrative(result)


def _generate_gemini_text(api_key: str, prompt: str) -> Optional[str]:
    """Blocking LLM waterfall call isolated in a worker thread by generate_narrative_llm."""
    import urllib.request
    import json
    
    # Helper to send POST request
    def http_post(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        try:
            payload = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", **headers},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"[LLM Narrative REST Debug] POST to {url} failed: {e}")
            return None

    # Step 1: Try Gemini SDK
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text and len(str(text).strip()) > 30:
            return str(text).strip()
    except Exception as sdk_err:
        print(f"[LLM Narrative SDK Failed] {sdk_err}, trying REST endpoints...")

    # Step 2: Try Gemini REST (v1)
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.25}}
        resp = http_post(url, body, {})
        if resp:
            parts = ((((resp.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or [])
            text = "\n".join(str(p.get("text") or "") for p in parts).strip()
            if text and len(text) > 30:
                return text
    except Exception as e:
        print(f"[LLM Narrative Gemini REST Failed] {e}")

    # Step 3: Try OpenRouter
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            body = {
                "model": "openrouter/auto",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25
            }
            resp = http_post(url, body, {"Authorization": f"Bearer {openrouter_key}"})
            if resp:
                text = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                if text and len(text) > 30:
                    return text
        except Exception as e:
            print(f"[LLM Narrative OpenRouter Failed] {e}")

    # Step 4: Try GitHub PAT
    github_key = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    if github_key:
        try:
            url = "https://models.inference.ai.azure.com/chat/completions"
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25
            }
            resp = http_post(url, body, {"Authorization": f"Bearer {github_key}"})
            if resp:
                text = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                if text and len(text) > 30:
                    return text
        except Exception as e:
            print(f"[LLM Narrative GitHub Failed] {e}")

    # Step 5: Try Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            body = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25
            }
            resp = http_post(url, body, {"Authorization": f"Bearer {groq_key}"})
            if resp:
                text = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                if text and len(text) > 30:
                    return text
        except Exception as e:
            print(f"[LLM Narrative Groq Failed] {e}")

    # Step 6: Try Cohere
    cohere_key = os.environ.get("COHERE_API_KEY")
    if cohere_key:
        try:
            url = "https://api.cohere.com/v2/chat"
            body = {
                "model": "command-r7b-12-2024",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25
            }
            resp = http_post(url, body, {"Authorization": f"Bearer {cohere_key}"})
            if resp:
                content = resp.get("message", {}).get("content", [])
                text = "\n".join(str(part.get("text") or "") for part in content).strip()
                if text and len(text) > 30:
                    return text
        except Exception as e:
            print(f"[LLM Narrative Cohere Failed] {e}")

    return None


# ────────────────────────────────────────────────────────────
#  Aggregate Analysis
# ────────────────────────────────────────────────────────────

def compute_national_scenario(
    params: ScenarioParams,
    provinces: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Run scenario across all 63 provinces and aggregate national metrics.
    """
    if provinces is None:
        provinces = load_provinces()

    results = []
    for p in provinces:
        try:
            r = compute_scenario(p["province_code"], params)
            results.append(r)
        except Exception:
            continue

    if not results:
        return {"error": "no_results"}

    total_baseline_gdp = sum(r.baseline_gdp for r in results)
    total_projected_gdp = sum(r.projected_gdp for r in results)
    total_baseline_revenue = sum(r.baseline_revenue for r in results)
    total_projected_revenue = sum(r.projected_revenue for r in results)

    risk_distribution = {"low": 0, "medium": 0, "high": 0}
    for r in results:
        risk_distribution[r.projected_risk] = risk_distribution.get(r.projected_risk, 0) + 1

    return {
        "total_provinces": len(results),
        "total_baseline_gdp": round(total_baseline_gdp, 2),
        "total_projected_gdp": round(total_projected_gdp, 2),
        "delta_gdp_pct": round((total_projected_gdp - total_baseline_gdp) / max(total_baseline_gdp, 0.01) * 100, 2),
        "total_baseline_revenue": round(total_baseline_revenue, 2),
        "total_projected_revenue": round(total_projected_revenue, 2),
        "delta_revenue_pct": round((total_projected_revenue - total_baseline_revenue) / max(total_baseline_revenue, 0.01) * 100, 2),
        "risk_distribution": risk_distribution,
        "event_applied": params.event_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
