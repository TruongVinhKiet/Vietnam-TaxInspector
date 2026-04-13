"""
ai_analysis.py – API Router cho Hệ thống AI Chấm điểm Rủi ro
================================================================
4 Endpoints:
    1. POST /api/ai/single-query/{tax_code}  – Tra cứu đơn lẻ real-time
    2. POST /api/ai/batch-upload              – Upload CSV & bắt đầu phân tích lô
    3. GET  /api/ai/batch-status/{batch_id}   – Kiểm tra tiến độ batch
    4. GET  /api/ai/batch-results/{batch_id}  – Lấy kết quả đầy đủ batch
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
    """
    # Lookup company data
    company = db.query(models.Company).filter(
        models.Company.tax_code == tax_code
    ).first()

    # Check if we have tax returns for this company
    tax_returns = (
        db.query(models.TaxReturn)
        .filter(models.TaxReturn.tax_code == tax_code)
        .order_by(models.TaxReturn.filing_date.desc())
        .limit(12)  # up to 3 years * 4 quarters
        .all()
    )

    # Check if we have a cached risk assessment
    cached_assessment = (
        db.query(models.AIRiskAssessment)
        .filter(models.AIRiskAssessment.tax_code == tax_code)
        .order_by(models.AIRiskAssessment.created_at.desc())
        .first()
    )

    if cached_assessment:
        # Build yearly history from tax_returns for trend chart
        yearly_history = []
        if tax_returns:
            yearly_agg = {}
            for tr in tax_returns:
                y = tr.filing_date.year if tr.filing_date else 2024
                if y not in yearly_agg:
                    yearly_agg[y] = {"year": y, "revenue": 0, "total_expenses": 0}
                yearly_agg[y]["revenue"] += float(tr.revenue or 0)
                yearly_agg[y]["total_expenses"] += float(tr.expenses or 0)
            yearly_history = sorted(yearly_agg.values(), key=lambda x: x["year"])

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
            "model_confidence": round(max(cached_assessment.risk_score, 100 - cached_assessment.risk_score), 1),
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
                    "tax_code": tax_code,
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

        # Attach yearly history for trend chart
        yearly_history = sorted([
            {"year": d["year"], "revenue": d["revenue"], "total_expenses": d["total_expenses"]}
            for d in company_data
        ], key=lambda x: x["year"])
        result["yearly_history"] = yearly_history

        return result

    # No data at all - return informative error
    if not company:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy doanh nghiệp MST {tax_code}. "
                   "Hãy upload file CSV để nhập dữ liệu tài chính."
        )

    # Company exists but no financial data
    raise HTTPException(
        status_code=404,
        detail=f"DN {company.name} (MST: {tax_code}) chưa có dữ liệu tài chính. "
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
