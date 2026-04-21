"""
simulation.py – Digital Twin & Scenario Simulation Engine (DB-backed)
======================================================================
Queries real data from PostgreSQL (companies, tax_payments, ai_risk_assessments,
delinquency_predictions) to compute baseline metrics, then applies elasticity-based
heuristic simulation to project impacts of macro-economic policy changes.

Endpoints:
    POST /api/simulation/run-scenario     – Run a single scenario
    POST /api/simulation/compare          – Compare multiple scenarios side-by-side
    GET  /api/simulation/presets          – List available preset scenarios
    GET  /api/simulation/baseline        – Get current baseline metrics from real DB
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
import joblib
import numpy as np
from pathlib import Path

from ..database import get_db

router = APIRouter(prefix="/api/simulation", tags=["Digital Twin Simulation"])

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "simulation_lgbm.joblib"
_simulation_model = None

def get_simulation_model():
    global _simulation_model
    if _simulation_model is None and MODEL_PATH.exists():
        try:
            _simulation_model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[Simulation] Error loading model: {e}")
    return _simulation_model


# ────────────────────────────────────────────────────────────
#  Schemas
# ────────────────────────────────────────────────────────────

class ScenarioInput(BaseModel):
    vat_rate: float = Field(default=10.0, ge=0.0, le=25.0)
    cit_rate: float = Field(default=20.0, ge=0.0, le=40.0)
    audit_coverage_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    penalty_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)
    interest_rate: float = Field(default=6.0, ge=0.0, le=30.0)
    economic_growth_pct: float = Field(default=6.5, ge=-10.0, le=20.0)
    industry_filter: Optional[str] = None
    province_filter: Optional[str] = None


class IndustryImpact(BaseModel):
    industry: str
    baseline_delinquency_rate: float
    simulated_delinquency_rate: float
    delta_pct: float
    company_count: int
    estimated_revenue_change: float


class TimeSeriesPoint(BaseModel):
    quarter: str
    baseline_value: float
    simulated_value: float


class ScenarioResult(BaseModel):
    scenario_name: str
    parameters: Dict[str, Any]
    baseline_total_companies: int
    baseline_high_risk_count: int
    simulated_high_risk_count: int
    delta_high_risk: int
    delta_high_risk_pct: float
    baseline_delinquency_rate: float
    simulated_delinquency_rate: float
    baseline_estimated_loss: float
    simulated_estimated_loss: float
    delta_estimated_loss: float
    baseline_total_revenue: float
    simulated_total_revenue: float
    delta_revenue: float
    delta_revenue_pct: float
    industry_impacts: List[IndustryImpact]
    quarterly_projection: List[TimeSeriesPoint]
    risk_distribution: Dict[str, int]
    generated_at: str
    data_source: str = "postgresql"


class CompareRequest(BaseModel):
    scenarios: List[ScenarioInput] = Field(..., min_length=1, max_length=5)
    scenario_names: Optional[List[str]] = None


class CompareResponse(BaseModel):
    baseline: ScenarioResult
    scenarios: List[ScenarioResult]
    best_scenario: Optional[str] = None
    worst_scenario: Optional[str] = None


class PresetScenario(BaseModel):
    id: str
    name: str
    description: str
    parameters: ScenarioInput


# ────────────────────────────────────────────────────────────
#  Real DB baseline query
# ────────────────────────────────────────────────────────────

INDUSTRY_MARGINS = {
    "Xây dựng": 0.06, "Bất động sản": 0.12, "Thương mại XNK": 0.04,
    "Sản xuất công nghiệp": 0.08, "Nông nghiệp": 0.05, "Vận tải & Logistics": 0.07,
    "Công nghệ thông tin": 0.15, "Dịch vụ tài chính": 0.18, "Y tế & Dược phẩm": 0.14,
    "Giáo dục & Đào tạo": 0.10, "Thực phẩm & Đồ uống": 0.09, "May mặc & Giầy da": 0.06,
    "Khoáng sản & Năng lượng": 0.11, "Du lịch & Khách sạn": 0.08, "Viễn thông": 0.13,
}

BASELINE_VAT_RATE = 10.0
BASELINE_CIT_RATE = 20.0
BASELINE_AUDIT_COVERAGE = 5.0
BASELINE_PENALTY_MULTIPLIER = 1.0
BASELINE_INTEREST_RATE = 6.0
BASELINE_GROWTH_RATE = 6.5


def _query_industry_baselines(db: Session, industry_filter: Optional[str] = None, province_filter: Optional[str] = None) -> Dict[str, Dict]:
    """Query real aggregated data per industry from companies + tax_payments + delinquency_predictions."""
    
    where_clauses = ["c.industry IS NOT NULL", "c.industry != ''", "c.industry != 'Offshore Entity'"]
    params: Dict[str, Any] = {}
    if industry_filter:
        where_clauses.append("c.industry = :industry")
        params["industry"] = industry_filter
    if province_filter:
        where_clauses.append("c.province = :province")
        params["province"] = province_filter

    where_sql = " AND ".join(where_clauses)

    # Count companies per industry and avg revenue from tax_returns
    rows = db.execute(text(f"""
        SELECT 
            c.industry,
            COUNT(DISTINCT c.tax_code) as company_count,
            COALESCE(AVG(tr.revenue), 0) as avg_revenue,
            COALESCE(SUM(tp.penalty_amount), 0) as total_penalties
        FROM companies c
        LEFT JOIN tax_returns tr ON tr.tax_code = c.tax_code
        LEFT JOIN tax_payments tp ON tp.tax_code = c.tax_code AND tp.status = 'overdue'
        WHERE {where_sql}
        GROUP BY c.industry
        ORDER BY company_count DESC
    """), params).fetchall()

    # Get delinquency rates per industry
    delinq_rows = db.execute(text(f"""
        SELECT
            c.industry,
            COUNT(DISTINCT dp.tax_code) as delinquent_count,
            COUNT(DISTINCT c.tax_code) as total_count
        FROM companies c
        LEFT JOIN delinquency_predictions dp ON dp.tax_code = c.tax_code AND dp.prob_90d >= 0.5
        WHERE {where_sql}
        GROUP BY c.industry
    """), params).fetchall()

    delinq_map = {r[0]: {"delinq_count": r[1], "total": r[2]} for r in delinq_rows}

    result = {}
    for row in rows:
        industry = row[0]
        count = row[1]
        avg_rev = float(row[2])
        total_pen = float(row[3])

        d = delinq_map.get(industry, {"delinq_count": 0, "total": count})
        delinq_rate = d["delinq_count"] / max(1, d["total"])

        # If no real delinquency data, estimate from payment history
        if delinq_rate == 0:
            overdue_count = db.execute(text("""
                SELECT COUNT(DISTINCT tp.tax_code)
                FROM tax_payments tp
                JOIN companies c ON c.tax_code = tp.tax_code
                WHERE c.industry = :ind AND tp.status IN ('overdue', 'partial')
            """), {"ind": industry}).scalar() or 0
            delinq_rate = overdue_count / max(1, count)

        # Fallback margin
        margin = INDUSTRY_MARGINS.get(industry, 0.08)

        result[industry] = {
            "company_count": count,
            "avg_revenue": avg_rev if avg_rev > 0 else 5e9,
            "avg_margin": margin,
            "delinq_rate": max(0.02, min(0.95, delinq_rate)) if delinq_rate > 0 else max(0.05, margin * 1.5),
            "total_penalties": total_pen,
        }

    return result


def _compute_scenario(params: ScenarioInput, db: Session, name: str = "Custom") -> ScenarioResult:
    """Compute scenario using real DB data + elasticity-based heuristic simulation."""

    industry_baselines = _query_industry_baselines(db, params.industry_filter, params.province_filter)

    if not industry_baselines:
        raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu doanh nghiệp trong CSDL. Hãy chạy seed_db.py trước.")

    # Elasticity coefficients
    vat_elasticity = -0.08
    cit_elasticity = -0.05
    audit_elasticity = -0.015
    penalty_elasticity = -0.04
    interest_elasticity = 0.03
    growth_elasticity = -0.02

    d_vat = params.vat_rate - BASELINE_VAT_RATE
    d_cit = params.cit_rate - BASELINE_CIT_RATE
    d_audit = params.audit_coverage_pct - BASELINE_AUDIT_COVERAGE
    d_penalty = params.penalty_multiplier - BASELINE_PENALTY_MULTIPLIER
    d_interest = params.interest_rate - BASELINE_INTEREST_RATE
    d_growth = params.economic_growth_pct - BASELINE_GROWTH_RATE

    delinq_shift = (
        d_vat * vat_elasticity + d_cit * cit_elasticity
        + d_audit * audit_elasticity + d_penalty * penalty_elasticity
        + d_interest * interest_elasticity + d_growth * growth_elasticity
    )

    revenue_shift_pct = (
        (d_vat / BASELINE_VAT_RATE * 0.3)
        + (d_cit / BASELINE_CIT_RATE * 0.25)
        + (d_growth / 100.0 * 0.8)
    )

    industry_impacts = []
    total_baseline_companies = 0
    total_baseline_high_risk = 0
    total_simulated_high_risk = 0
    total_baseline_loss = 0.0
    total_simulated_loss = 0.0
    total_baseline_revenue = 0.0
    total_simulated_revenue = 0.0

    model = get_simulation_model()

    for ind, stats in industry_baselines.items():
        count = stats["company_count"]
        base_rate = stats["delinq_rate"]
        avg_rev = stats["avg_revenue"]
        margin = stats["avg_margin"]

        if model is not None:
            # Macro features + Industry features
            features = np.array([[
                params.vat_rate, params.cit_rate, params.audit_coverage_pct,
                params.penalty_multiplier, params.interest_rate, params.economic_growth_pct,
                base_rate, margin, count
            ]])
            sim_rate = float(model.predict(features)[0])
            sim_rate = max(0.01, min(0.95, sim_rate))
        else:
            margin_sensitivity = max(0.5, 1.0 - margin * 3)
            sim_rate = max(0.01, min(0.95, base_rate + delinq_shift * margin_sensitivity))

        base_high_risk = int(count * base_rate)
        sim_high_risk = int(count * sim_rate)

        ind_base_revenue = count * avg_rev
        ind_sim_revenue = ind_base_revenue * (1.0 + revenue_shift_pct)

        base_loss = base_high_risk * avg_rev * 0.02 * BASELINE_PENALTY_MULTIPLIER
        sim_loss = sim_high_risk * avg_rev * 0.02 * params.penalty_multiplier

        industry_impacts.append(IndustryImpact(
            industry=ind,
            baseline_delinquency_rate=round(base_rate * 100, 2),
            simulated_delinquency_rate=round(sim_rate * 100, 2),
            delta_pct=round((sim_rate - base_rate) * 100, 2),
            company_count=count,
            estimated_revenue_change=round(ind_sim_revenue - ind_base_revenue, 0),
        ))

        total_baseline_companies += count
        total_baseline_high_risk += base_high_risk
        total_simulated_high_risk += sim_high_risk
        total_baseline_loss += base_loss
        total_simulated_loss += sim_loss
        total_baseline_revenue += ind_base_revenue
        total_simulated_revenue += ind_sim_revenue

    # Quarterly projection (8 quarters forward)
    quarters = []
    for q in range(1, 9):
        label = f"Q{((q - 1) % 4) + 1}/{2025 + (q - 1) // 4}"
        growth_factor = 1.0 + (params.economic_growth_pct / 100.0) * (q / 8.0)
        base_val = total_baseline_revenue / 4.0 * (1.0 + BASELINE_GROWTH_RATE / 100.0 * (q / 8.0))
        sim_val = total_simulated_revenue / 4.0 * growth_factor
        quarters.append(TimeSeriesPoint(
            quarter=label,
            baseline_value=round(base_val / 1e9, 2),
            simulated_value=round(sim_val / 1e9, 2),
        ))

    base_delinq_rate = total_baseline_high_risk / max(1, total_baseline_companies)
    sim_delinq_rate = total_simulated_high_risk / max(1, total_baseline_companies)

    risk_dist = {
        "low": int(total_baseline_companies * (1 - sim_delinq_rate) * 0.6),
        "medium": int(total_baseline_companies * (1 - sim_delinq_rate) * 0.25),
        "high": int(total_baseline_companies * sim_delinq_rate * 0.6),
        "critical": int(total_baseline_companies * sim_delinq_rate * 0.4),
    }

    return ScenarioResult(
        scenario_name=name,
        parameters=params.model_dump(),
        baseline_total_companies=total_baseline_companies,
        baseline_high_risk_count=total_baseline_high_risk,
        simulated_high_risk_count=total_simulated_high_risk,
        delta_high_risk=total_simulated_high_risk - total_baseline_high_risk,
        delta_high_risk_pct=round((total_simulated_high_risk - total_baseline_high_risk) / max(1, total_baseline_high_risk) * 100, 2),
        baseline_delinquency_rate=round(base_delinq_rate * 100, 2),
        simulated_delinquency_rate=round(sim_delinq_rate * 100, 2),
        baseline_estimated_loss=round(total_baseline_loss, 0),
        simulated_estimated_loss=round(total_simulated_loss, 0),
        delta_estimated_loss=round(total_simulated_loss - total_baseline_loss, 0),
        baseline_total_revenue=round(total_baseline_revenue, 0),
        simulated_total_revenue=round(total_simulated_revenue, 0),
        delta_revenue=round(total_simulated_revenue - total_baseline_revenue, 0),
        delta_revenue_pct=round((total_simulated_revenue - total_baseline_revenue) / max(1, total_baseline_revenue) * 100, 2),
        industry_impacts=sorted(industry_impacts, key=lambda x: x.delta_pct),
        quarterly_projection=quarters,
        risk_distribution=risk_dist,
        generated_at=datetime.utcnow().isoformat() + "Z",
        data_source="postgresql",
    )


# ────────────────────────────────────────────────────────────
#  Preset Scenarios
# ────────────────────────────────────────────────────────────

PRESETS: List[PresetScenario] = [
    PresetScenario(id="vat_reduction", name="Giảm VAT xuống 8%",
        description="Mô phỏng tác động khi Quốc hội giảm thuế GTGT từ 10% xuống 8% để kích thích kinh tế.",
        parameters=ScenarioInput(vat_rate=8.0)),
    PresetScenario(id="aggressive_audit", name="Tăng cường thanh tra (15%)",
        description="Tăng diện thanh tra từ 5% lên 15% doanh nghiệp, đánh giá khả năng răn đe.",
        parameters=ScenarioInput(audit_coverage_pct=15.0)),
    PresetScenario(id="economic_downturn", name="Suy thoái kinh tế",
        description="GDP tăng trưởng chỉ 2%, lãi suất tăng lên 12% – đánh giá rủi ro nợ đọng.",
        parameters=ScenarioInput(economic_growth_pct=2.0, interest_rate=12.0)),
    PresetScenario(id="strict_enforcement", name="Siết chặt xử phạt",
        description="Tăng gấp 3 mức phạt + tăng thanh tra 10% – đánh giá hiệu quả cưỡng chế.",
        parameters=ScenarioInput(penalty_multiplier=3.0, audit_coverage_pct=10.0)),
    PresetScenario(id="optimistic", name="Kịch bản lạc quan",
        description="GDP 8%, lãi suất thấp 4%, giảm VAT 8% – kịch bản tăng trưởng tốt nhất.",
        parameters=ScenarioInput(vat_rate=8.0, economic_growth_pct=8.0, interest_rate=4.0)),
]


# ────────────────────────────────────────────────────────────
#  API Endpoints
# ────────────────────────────────────────────────────────────

@router.get("/presets", response_model=List[PresetScenario])
def list_presets():
    return PRESETS


@router.get("/baseline", response_model=ScenarioResult)
def get_baseline(db: Session = Depends(get_db)):
    return _compute_scenario(ScenarioInput(), db, name="Baseline (hiện tại)")


@router.post("/run-scenario", response_model=ScenarioResult)
def run_scenario(params: ScenarioInput, name: str = "Custom Scenario", db: Session = Depends(get_db)):
    return _compute_scenario(params, db, name=name)


@router.post("/compare", response_model=CompareResponse)
def compare_scenarios(req: CompareRequest, db: Session = Depends(get_db)):
    baseline = _compute_scenario(ScenarioInput(), db, name="Baseline")
    results = []
    names = req.scenario_names or [f"Kịch bản {i+1}" for i in range(len(req.scenarios))]
    for i, s in enumerate(req.scenarios):
        label = names[i] if i < len(names) else f"Kịch bản {i+1}"
        results.append(_compute_scenario(s, db, name=label))
    best = min(results, key=lambda r: r.simulated_estimated_loss) if results else None
    worst = max(results, key=lambda r: r.simulated_estimated_loss) if results else None
    return CompareResponse(
        baseline=baseline, scenarios=results,
        best_scenario=best.scenario_name if best else None,
        worst_scenario=worst.scenario_name if worst else None,
    )
