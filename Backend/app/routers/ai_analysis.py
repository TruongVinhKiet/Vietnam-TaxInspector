"""
ai_analysis.py – API Router cho Hệ thống AI Chấm điểm Rủi ro
================================================================
4 Endpoints:
    1. POST /api/ai/single-query/{tax_code}  – Tra cứu đơn lẻ real-time
    2. POST /api/ai/batch-upload              – Upload CSV & bắt đầu phân tích lô
    3. GET  /api/ai/batch-status/{batch_id}   – Kiểm tra tiến độ batch
    4. GET  /api/ai/batch-results/{batch_id}  – Lấy kết quả đầy đủ batch
    5. POST /api/ai/what-if/{tax_code}        – Mô phỏng What-If

Enhancements:
    - Single query tự lưu kết quả vào cache để What-If hoạt động ngay
    - Stale-cache detection: tái phân tích khi dữ liệu tài chính mới hơn cache
    - Hỗ trợ tra cứu bằng Tên doanh nghiệp ngoài Mã số thuế
"""

import os
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session

from ..database import get_db
from .. import models

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml_engine.pipeline import TaxFraudPipeline
from ml_engine.feature_engineering import TaxFeatureEngineer

router = APIRouter(prefix="/api/ai", tags=["AI Risk Analysis"])

# ---- Upload directory ----
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---- Singleton pipeline ----
_pipeline = None


def get_pipeline() -> TaxFraudPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TaxFraudPipeline()
        _pipeline.load_models()
    return _pipeline


# ==================================================================
# 1. SINGLE QUERY (Real-time)
# ==================================================================
@router.post("/single-query/{tax_code}")
def single_query(tax_code: str, db: Session = Depends(get_db)):
    """
    Chế độ 1: Tra cứu đơn lẻ.
    Tìm dữ liệu 3 năm gần nhất trong DB, chạy AI pipeline real-time.
    Nếu không có trong DB, trả về lỗi kèm gợi ý upload CSV.
    Hỗ trợ tra cứu bằng Tên DN: nếu tax_code không phải là số thuần,
    hệ thống sẽ tìm DN theo tên (ILIKE) rồi lấy MST.
    """
    resolved_tax_code = tax_code
    company = None

    # --- Hỗ trợ tra cứu bằng tên doanh nghiệp ---
    if not tax_code.replace("-", "").isdigit():
        # Treat input as company name search
        company = db.query(models.Company).filter(
            models.Company.name.ilike(f"%{tax_code}%")
        ).first()
        if company:
            resolved_tax_code = company.tax_code
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy doanh nghiệp tên '{tax_code}'. "
                       "Hãy nhập MST chính xác hoặc upload CSV."
            )
    else:
        company = db.query(models.Company).filter(
            models.Company.tax_code == resolved_tax_code
        ).first()

    # Check if we have tax returns for this company
    tax_returns = (
        db.query(models.TaxReturn)
        .filter(models.TaxReturn.tax_code == resolved_tax_code)
        .order_by(models.TaxReturn.filing_date.desc())
        .limit(12)  # up to 3 years * 4 quarters
        .all()
    )

    # Check if we have a cached risk assessment
    cached_assessment = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.tax_code == resolved_tax_code)
        .order_by(models.AIRiskAssessment.created_at.desc())
        .first()
    )

    # --- Stale-cache detection ---
    # Two conditions invalidate cache:
    # 1) Newer financial data exists than cached assessment
    # 2) Cache is older than 30 days (ensures periodic model re-evaluation)
    cache_is_stale = False
    if cached_assessment:
        # Condition 1: newer filing data
        if tax_returns:
            newest_filing = max(
                (tr.filing_date for tr in tax_returns if tr.filing_date),
                default=None,
            )
            if newest_filing and cached_assessment.created_at:
                cache_created = cached_assessment.created_at
                from datetime import date as date_type
                filing_dt = datetime.combine(newest_filing, datetime.min.time()) \
                    if isinstance(newest_filing, date_type) else newest_filing
                cache_dt = cache_created.replace(tzinfo=None) \
                    if hasattr(cache_created, 'tzinfo') and cache_created.tzinfo else cache_created
                if filing_dt > cache_dt:
                    cache_is_stale = True

        # Condition 2: cache age > 30 days
        if not cache_is_stale and cached_assessment.created_at:
            from datetime import timedelta
            cache_dt = cached_assessment.created_at.replace(tzinfo=None) \
                if hasattr(cached_assessment.created_at, 'tzinfo') and cached_assessment.created_at.tzinfo \
                else cached_assessment.created_at
            if (datetime.utcnow() - cache_dt) > timedelta(days=30):
                cache_is_stale = True

    if cached_assessment and not cache_is_stale:
        # Yearly history: prefer DB column (populated by batch), fallback to tax_returns
        yearly_history = cached_assessment.yearly_history or []
        if not yearly_history and tax_returns:
            yearly_agg = {}
            for tr in tax_returns:
                y = tr.filing_date.year if tr.filing_date else 2024
                if y not in yearly_agg:
                    yearly_agg[y] = {"year": y, "revenue": 0, "total_expenses": 0}
                yearly_agg[y]["revenue"] += float(tr.revenue or 0)
                yearly_agg[y]["total_expenses"] += float(tr.expenses or 0)
            yearly_history = sorted(yearly_agg.values(), key=lambda x: x["year"])

        # Model confidence: read real value from DB column if available,
        # fallback to approximation for legacy records without the column
        if cached_assessment.model_confidence is not None:
            cached_confidence = cached_assessment.model_confidence
        else:
            cached_risk = cached_assessment.risk_score or 0
            fraud_prob_approx = cached_risk / 100.0
            cached_confidence = round(max(fraud_prob_approx, 1 - fraud_prob_approx) * 100, 1)

        # Return cached result
        return {
            "tax_code": cached_assessment.tax_code,
            "company_name": cached_assessment.company_name or (company.name if company else ""),
            "industry": cached_assessment.industry or (company.industry if company else ""),
            "year": cached_assessment.year,
            "revenue": float(cached_assessment.revenue or 0),
            "total_expenses": float(cached_assessment.total_expenses or 0),
            "f1_divergence": cached_assessment.f1_divergence,
            "f2_ratio_limit": cached_assessment.f2_ratio_limit,
            "f3_vat_structure": cached_assessment.f3_vat_structure,
            "f4_peer_comparison": cached_assessment.f4_peer_comparison,
            "anomaly_score": cached_assessment.anomaly_score,
            "model_confidence": cached_confidence,
            "risk_score": cached_assessment.risk_score,
            "risk_level": cached_assessment.risk_level,
            "red_flags": cached_assessment.red_flags or [],
            "shap_explanation": cached_assessment.shap_explanation or [],
            "yearly_history": yearly_history,
            "source": "cached",
        }

    # If we have tax_returns data, build financial data for pipeline
    if tax_returns and len(tax_returns) >= 1:
        import pandas as pd

        # Aggregate quarterly data into yearly
        yearly_data = {}
        for tr in tax_returns:
            year = tr.filing_date.year if tr.filing_date else 2024
            if year not in yearly_data:
                yearly_data[year] = {
                    "tax_code": resolved_tax_code,
                    "company_name": company.name if company else "",
                    "industry": company.industry if company else "",
                    "year": year,
                    "revenue": 0,
                    "cost_of_goods": 0,
                    "operating_expenses": 0,
                    "total_expenses": 0,
                    "net_profit": 0,
                    "vat_input": 0,
                    "vat_output": 0,
                    "industry_avg_profit_margin": 0.08,
                }
            yd = yearly_data[year]
            yd["revenue"] += float(tr.revenue or 0)
            yd["total_expenses"] += float(tr.expenses or 0)
            yd["net_profit"] = yd["revenue"] - yd["total_expenses"]
            yd["cost_of_goods"] = yd["total_expenses"] * 0.75
            yd["operating_expenses"] = yd["total_expenses"] * 0.25
            yd["vat_output"] = yd["revenue"] * 0.10
            yd["vat_input"] = yd["cost_of_goods"] * 0.10

        company_data = list(yearly_data.values())
        pipeline = get_pipeline()
        result = pipeline.predict_single(company_data)
        result["source"] = "realtime"

        # Build yearly history for trend chart + cache persistence
        yearly_history_for_cache = sorted([
            {"year": d["year"], "revenue": d["revenue"], "total_expenses": d["total_expenses"]}
            for d in company_data
        ], key=lambda x: x["year"])

        # --- Save result into cache (AIRiskAssessment) so What-If works immediately ---
        try:
            new_assessment = models.AIRiskAssessment(
                batch_id=None,
                tax_code=result.get("tax_code", resolved_tax_code),
                company_name=result.get("company_name"),
                industry=result.get("industry"),
                year=result.get("year"),
                revenue=result.get("revenue"),
                total_expenses=result.get("total_expenses"),
                f1_divergence=result.get("f1_divergence"),
                f2_ratio_limit=result.get("f2_ratio_limit"),
                f3_vat_structure=result.get("f3_vat_structure"),
                f4_peer_comparison=result.get("f4_peer_comparison"),
                anomaly_score=result.get("anomaly_score"),
                model_confidence=result.get("model_confidence"),
                risk_score=result.get("risk_score", 0),
                risk_level=result.get("risk_level", "low"),
                red_flags=result.get("red_flags"),
                shap_explanation=result.get("shap_explanation"),
                yearly_history=yearly_history_for_cache,
            )
            db.add(new_assessment)
            db.commit()
        except Exception as cache_err:
            db.rollback()
            print(f"[WARN] Failed to cache realtime assessment: {cache_err}")

        result["yearly_history"] = yearly_history_for_cache

        return result

    # No data at all - return informative error
    if not company:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy doanh nghiệp MST {resolved_tax_code}. "
                   "Hãy upload file CSV để nhập dữ liệu tài chính."
        )

    # Company exists but no financial data
    raise HTTPException(
        status_code=404,
        detail=f"DN {company.name} (MST: {resolved_tax_code}) chưa có dữ liệu tài chính. "
               "Hãy upload file CSV chứa báo cáo tài chính."
    )


# ==================================================================
# 2. BATCH UPLOAD
# ==================================================================
@router.post("/batch-upload")
async def batch_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Chế độ 2: Upload file CSV để phân tích lô.
    - Lưu file vào disk
    - Tạo batch record
    - Khởi động background task (Celery hoặc thread đồng bộ)
    - Trả về batch_id cho Frontend polling
    """
    # Validate file
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file CSV (.csv)")

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    save_filename = f"{file_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_filename

    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > 200:  # 200MB limit
        raise HTTPException(status_code=400, detail="File quá lớn (tối đa 200MB)")

    with open(save_path, "wb") as f:
        f.write(content)

    # Create batch record in DB
    batch = models.AIAnalysisBatch(
        filename=file.filename,
        file_path=str(save_path),
        status="pending",
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    # Try Celery first, fallback to synchronous thread
    try:
        from app.worker import CELERY_AVAILABLE
        from app.tasks import analyze_batch_csv_task, run_batch_analysis

        if CELERY_AVAILABLE:
            # Dispatch to Celery worker
            analyze_batch_csv_task.delay(str(save_path), batch.id)
        else:
            # Fallback: run in background thread
            thread = threading.Thread(
                target=run_batch_analysis,
                args=(str(save_path), batch.id),
                daemon=True,
            )
            thread.start()
    except Exception:
        # Ultimate fallback: background thread
        from app.tasks import run_batch_analysis
        thread = threading.Thread(
            target=run_batch_analysis,
            args=(str(save_path), batch.id),
            daemon=True,
        )
        thread.start()

    return {
        "batch_id": batch.id,
        "filename": file.filename,
        "file_size_mb": round(file_size_mb, 2),
        "status": "pending",
        "message": "File đã được nhận. Hệ thống AI đang xử lý...",
    }


# ==================================================================
# 3. BATCH STATUS (for polling)
# ==================================================================
@router.get("/batch-status/{batch_id}")
def batch_status(batch_id: int, db: Session = Depends(get_db)):
    """
    Kiểm tra tiến độ xử lý batch.
    Frontend dùng setInterval() để gọi endpoint này mỗi 1-2 giây.
    """
    batch = db.query(models.AIAnalysisBatch).filter(
        models.AIAnalysisBatch.id == batch_id
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch không tồn tại")

    progress = 0
    if batch.total_rows and batch.total_rows > 0:
        progress = round((batch.processed_rows or 0) / batch.total_rows * 100, 1)

    return {
        "batch_id": batch.id,
        "filename": batch.filename,
        "status": batch.status,
        "total_rows": batch.total_rows,
        "processed_rows": batch.processed_rows,
        "progress_percent": progress,
        "error_message": batch.error_message,
        "created_at": str(batch.created_at) if batch.created_at else None,
        "started_at": str(batch.started_at) if batch.started_at else None,
        "completed_at": str(batch.completed_at) if batch.completed_at else None,
    }


# ==================================================================
# 4. BATCH RESULTS (full dashboard data)
# ==================================================================
@router.get("/batch-results/{batch_id}")
def batch_results(batch_id: int, db: Session = Depends(get_db)):
    """
    Lấy kết quả đầy đủ batch: thống kê, top 50, scatter data...
    Chỉ trả về khi status == 'done'.
    """
    batch = db.query(models.AIAnalysisBatch).filter(
        models.AIAnalysisBatch.id == batch_id
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch không tồn tại")

    if batch.status != "done":
        raise HTTPException(
            status_code=202,
            detail=f"Batch đang ở trạng thái: {batch.status}. Vui lòng đợi xử lý hoàn tất."
        )

    # Get individual assessments for the data table
    assessments = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.batch_id == batch_id)
        .order_by(models.AIRiskAssessment.risk_score.desc())
        .all()
    )

    assessment_list = []
    for a in assessments:
        assessment_list.append({
            "tax_code": a.tax_code,
            "company_name": a.company_name,
            "industry": a.industry,
            "year": a.year,
            "revenue": float(a.revenue or 0),
            "total_expenses": float(a.total_expenses or 0),
            "f1_divergence": a.f1_divergence,
            "f2_ratio_limit": a.f2_ratio_limit,
            "f3_vat_structure": a.f3_vat_structure,
            "f4_peer_comparison": a.f4_peer_comparison,
            "anomaly_score": a.anomaly_score,
            "risk_score": a.risk_score,
            "risk_level": a.risk_level,
            "red_flags": a.red_flags or [],
        })

    return {
        "batch_id": batch.id,
        "filename": batch.filename,
        "status": batch.status,
        "total_records": batch.total_rows or 0,
        "total_companies": len(assessment_list),
        "statistics": batch.result_summary or {},
        "assessments": assessment_list,
    }


# ==================================================================
# 5. WHAT-IF SIMULATION (Scenario Analysis)
# ==================================================================
@router.post("/what-if/{tax_code}")
def what_if_simulation(tax_code: str, adjustments: dict, db: Session = Depends(get_db)):
    """
    Chế độ 5: Mô phỏng Tình huống (What-If Analysis).
    Nhận điều chỉnh % cho các chỉ số tài chính, chạy lại AI pipeline
    và trả về điểm rủi ro mới để so sánh.
    
    Body JSON example:
    {
        "revenue": -20,          // Giảm doanh thu 20%
        "total_expenses": 30     // Tăng chi phí 30%
    }
    """
    # Find cached assessment
    cached = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.tax_code == tax_code)
        .order_by(models.AIRiskAssessment.created_at.desc())
        .first()
    )

    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"Chưa có dữ liệu phân tích cho MST {tax_code}. Hãy phân tích AI trước."
        )

    # Build base data from cached assessment
    base_data = {
        "tax_code": cached.tax_code,
        "company_name": cached.company_name or "",
        "industry": cached.industry or "",
        "year": cached.year,
        "revenue": float(cached.revenue or 0),
        "total_expenses": float(cached.total_expenses or 0),
        "net_profit": float(cached.revenue or 0) - float(cached.total_expenses or 0),
        "cost_of_goods": float(cached.total_expenses or 0) * 0.75,
        "operating_expenses": float(cached.total_expenses or 0) * 0.25,
        "vat_output": float(cached.revenue or 0) * 0.10,
        "vat_input": float(cached.total_expenses or 0) * 0.75 * 0.10,
        "industry_avg_profit_margin": 0.08,
    }

    pipeline = get_pipeline()
    result = pipeline.predict_whatif(base_data, adjustments)
    result["original_risk_score"] = cached.risk_score
    result["original_risk_level"] = cached.risk_level
    result["delta_risk"] = round(result["simulated_risk_score"] - cached.risk_score, 2)

    return result
