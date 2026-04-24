"""
scoring.py – Fraud Risk Scoring Router (REAL – Phase 0.1)
===========================================================
Replaces the old mock endpoint (random.uniform) with real AI pipeline
inference. Connects to:
    - TaxFraudPipeline (Isolation Forest + XGBoost + SHAP)
    - ai_risk_assessments table (cached results)
    - companies + tax_returns tables (financial data source)

Endpoints:
    POST /api/scoring/{tax_code}   – Score a company's fraud risk (real ML)
"""

import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional
import uuid

from ..database import get_db
from .. import models, schemas

# Ensure ml_engine is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

router = APIRouter(prefix="/api", tags=["Fraud Risk Scoring"])

# ---- Singleton pipeline (lazy-loaded) ----
_pipeline = None


def _get_pipeline():
    """Lazy-load the ML pipeline to avoid startup failure when models aren't trained yet."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from ml_engine.pipeline import TaxFraudPipeline
        _pipeline = TaxFraudPipeline()
        _pipeline.load_models()
        return _pipeline
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"[WARN] Failed to load fraud pipeline for scoring: {exc}")
        return None


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _classify_risk_level(score: float) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _build_red_flag_strings(red_flags_json) -> list[str]:
    """Convert structured red_flags JSON to flat string list for backward compatibility."""
    if not red_flags_json:
        return []
    if isinstance(red_flags_json, list):
        result = []
        for flag in red_flags_json:
            if isinstance(flag, str):
                result.append(flag)
            elif isinstance(flag, dict):
                label = flag.get("label") or flag.get("flag") or flag.get("description", "")
                if label:
                    result.append(str(label))
        return result
    return []


def _build_top_features(assessment) -> list[dict]:
    """Extract top contributing features from a cached assessment."""
    features = []
    feature_map = {
        "f1_divergence": "Lệch pha Tăng trưởng (F1)",
        "f2_ratio_limit": "Đội Chi phí/Doanh thu (F2)",
        "f3_vat_structure": "VAT Vòng lặp (F3)",
        "f4_peer_comparison": "Thấp hơn Ngành (F4)",
    }
    for attr, label in feature_map.items():
        val = getattr(assessment, attr, None)
        if val is not None:
            features.append({
                "feature": attr,
                "label": label,
                "value": round(float(val), 4),
            })

    features.sort(key=lambda x: abs(x["value"]), reverse=True)
    return features


@router.post("/scoring/{tax_code}", response_model=schemas.FraudRiskPredictionEnhanced)
def score_company_risk(tax_code: str, db: Session = Depends(get_db)):
    """
    Tab 1: Chấm điểm rủi ro gian lận cho một doanh nghiệp theo MST.
    Uses real AI pipeline (XGBoost + Isolation Forest + SHAP).
    Falls back to cached assessment if pipeline is not available.
    """
    # Validate company exists
    company = db.query(models.Company).filter(models.Company.tax_code == tax_code).first()
    if not company:
        raise HTTPException(status_code=404, detail="Không tìm thấy doanh nghiệp với MST này.")

    # Strategy 1: Try real-time ML pipeline inference
    pipeline = _get_pipeline()
    if pipeline is not None:
        from ml_engine.model_registry import ModelRegistryService, AuditContext, InferenceTimer
        timer = InferenceTimer()
        request_id = uuid.uuid4().hex
        tax_returns = (
            db.query(models.TaxReturn)
            .filter(models.TaxReturn.tax_code == tax_code)
            .order_by(models.TaxReturn.filing_date.desc())
            .limit(30)
            .all()
        )

        if tax_returns:
            # Build input rows from tax returns
            yearly_data: dict[int, dict] = {}
            for tr in tax_returns:
                year = tr.filing_date.year if tr.filing_date else None
                if year is None:
                    continue
                bucket = yearly_data.setdefault(year, {
                    "tax_code": tax_code,
                    "company_name": company.name,
                    "industry": company.industry or "Unknown",
                    "year": year,
                    "revenue": 0.0,
                    "total_expenses": 0.0,
                    "net_profit": 0.0,
                    "vat_input": 0.0,
                    "vat_output": 0.0,
                    "industry_avg_profit_margin": 0.08,
                })
                bucket["revenue"] += _to_float(tr.revenue)
                bucket["total_expenses"] += _to_float(tr.expenses)

            if yearly_data:
                for year_bucket in yearly_data.values():
                    rev = year_bucket["revenue"]
                    exp = year_bucket["total_expenses"]
                    year_bucket["net_profit"] = rev - exp
                    year_bucket["vat_output"] = rev * 0.10
                    year_bucket["vat_input"] = exp * 0.75 * 0.10

                company_rows = sorted(yearly_data.values(), key=lambda x: x["year"])

                try:
                    result = pipeline.predict_single(company_rows)
                    risk_score = _to_float(result.get("risk_score"), 0.0)
                    risk_level = _classify_risk_level(risk_score)

                    response_payload = schemas.FraudRiskPredictionEnhanced(
                        tax_code=tax_code,
                        company_name=company.name,
                        risk_score=risk_score,
                        risk_level=risk_level,
                        red_flags=_build_red_flag_strings(result.get("red_flags")),
                        model_confidence=result.get("model_confidence"),
                        model_version=result.get("model_version"),
                        f1_divergence=result.get("f1_divergence"),
                        f2_ratio_limit=result.get("f2_ratio_limit"),
                        f3_vat_structure=result.get("f3_vat_structure"),
                        f4_peer_comparison=result.get("f4_peer_comparison"),
                        anomaly_score=result.get("anomaly_score"),
                        top_features=[
                            {"feature": f, "label": f, "value": round(float(result.get(f, 0)), 4)}
                            for f in ["f1_divergence", "f2_ratio_limit", "f3_vat_structure", "f4_peer_comparison"]
                            if result.get(f) is not None
                        ],
                    )
                    try:
                        registry = ModelRegistryService(db)
                        registry.log_inference(
                            model_name="fraud_scoring",
                            model_version=str(result.get("model_version") or "unknown"),
                            entity_type="company",
                            entity_id=tax_code,
                            input_features={"tax_code": tax_code, "rows": len(company_rows)},
                            outputs={"risk_score": risk_score, "risk_level": risk_level},
                            latency_ms=timer.elapsed_ms(),
                            ctx=AuditContext(request_id=request_id),
                        )
                    except Exception:
                        db.rollback()

                    return response_payload
                except Exception as exc:
                    print(f"[WARN] Pipeline inference failed for {tax_code}, trying cached: {exc}")

    # Strategy 2: Use cached assessment from ai_risk_assessments
    cached = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.tax_code == tax_code)
        .order_by(desc(models.AIRiskAssessment.created_at))
        .first()
    )

    if cached:
        risk_score = _to_float(cached.risk_score, 0.0)
        return schemas.FraudRiskPredictionEnhanced(
            tax_code=tax_code,
            company_name=company.name,
            risk_score=risk_score,
            risk_level=cached.risk_level or _classify_risk_level(risk_score),
            red_flags=_build_red_flag_strings(cached.red_flags),
            model_confidence=float(cached.model_confidence) if cached.model_confidence else None,
            model_version=cached.model_version,
            f1_divergence=float(cached.f1_divergence) if cached.f1_divergence else None,
            f2_ratio_limit=float(cached.f2_ratio_limit) if cached.f2_ratio_limit else None,
            f3_vat_structure=float(cached.f3_vat_structure) if cached.f3_vat_structure else None,
            f4_peer_comparison=float(cached.f4_peer_comparison) if cached.f4_peer_comparison else None,
            anomaly_score=float(cached.anomaly_score) if cached.anomaly_score else None,
            top_features=_build_top_features(cached),
        )

    # Strategy 3: No data at all – return zeros with guidance
    return schemas.FraudRiskPredictionEnhanced(
        tax_code=tax_code,
        company_name=company.name,
        risk_score=0.0,
        risk_level="unknown",
        red_flags=["Chưa có dữ liệu tài chính. Vui lòng upload CSV hoặc import tờ khai thuế."],
        model_confidence=None,
        model_version="no-data",
        top_features=[],
    )
