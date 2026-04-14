"""
tasks.py – Background Tasks (Celery / Synchronous Fallback)
=============================================================
Task chính: analyze_batch_csv_task
    1. Đọc CSV bằng Pandas
    2. Chạy Feature Engineering (F1, F2, F3, F4)
    3. Đẩy qua Isolation Forest + XGBoost
    4. Lưu kết quả AI risk assessments vào PostgreSQL
    5. Cập nhật trạng thái batch (processing → done / failed)

Nếu Celery/Redis không khả dụng, task chạy đồng bộ (synchronous).
"""

import os
import json
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine.pipeline import TaxFraudPipeline
from app.database import SessionLocal

# Try to import Celery app
try:
    from app.worker import celery_app, CELERY_AVAILABLE
except Exception:
    celery_app = None
    CELERY_AVAILABLE = False


# Singleton pipeline instance (loaded once)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = TaxFraudPipeline()
        _pipeline.load_models()
    return _pipeline


def _update_batch_status(batch_id: int, **kwargs):
    """Update batch record in DB."""
    from app.models import AIAnalysisBatch
    db = SessionLocal()
    try:
        batch = db.query(AIAnalysisBatch).filter(AIAnalysisBatch.id == batch_id).first()
        if batch:
            for key, value in kwargs.items():
                setattr(batch, key, value)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"[ERROR] _update_batch_status: {e}")
    finally:
        db.close()


def _save_assessments(assessments: list):
    """Save individual risk assessments to DB."""
    from app.models import AIRiskAssessment
    db = SessionLocal()
    try:
        for a in assessments:
            record = AIRiskAssessment(
                batch_id=a.get("batch_id"),
                tax_code=a["tax_code"],
                company_name=a.get("company_name"),
                industry=a.get("industry"),
                year=a.get("year"),
                revenue=a.get("revenue"),
                total_expenses=a.get("total_expenses"),
                f1_divergence=a.get("f1_divergence"),
                f2_ratio_limit=a.get("f2_ratio_limit"),
                f3_vat_structure=a.get("f3_vat_structure"),
                f4_peer_comparison=a.get("f4_peer_comparison"),
                anomaly_score=a.get("anomaly_score"),
                model_confidence=a.get("model_confidence"),
                risk_score=a.get("risk_score", 0),
                risk_level=a.get("risk_level", "low"),
                red_flags=a.get("red_flags"),
                shap_explanation=a.get("shap_explanation"),
                yearly_history=a.get("yearly_history"),
            )
            db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[ERROR] _save_assessments: {e}")
        raise
    finally:
        db.close()


def run_batch_analysis(file_path: str, batch_id: int) -> dict:
    """
    Core batch analysis logic.
    Can be called by Celery task or directly (synchronous).
    """
    try:
        # Mark as processing
        _update_batch_status(
            batch_id,
            status="processing",
            started_at=datetime.utcnow(),
        )

        # Read CSV
        df = pd.read_csv(file_path)
        # Use number of unique companies as total (not CSV row count)
        # because progress_callback in pipeline counts per-company, not per-row
        total_companies = df["tax_code"].nunique() if "tax_code" in df.columns else len(df)

        _update_batch_status(batch_id, total_rows=total_companies)

        # Load and run pipeline
        pipeline = get_pipeline()

        def progress_cb(processed, total):
            _update_batch_status(batch_id, processed_rows=processed)

        result = pipeline.predict_batch(df, batch_id=batch_id,
                                         progress_callback=progress_cb)

        # Save individual assessments to DB
        _save_assessments(result["assessments"])

        # Save summary statistics to batch record
        _update_batch_status(
            batch_id,
            status="done",
            processed_rows=result["total_companies"],
            result_summary=result["statistics"],
            completed_at=datetime.utcnow(),
        )

        return {
            "batch_id": batch_id,
            "status": "done",
            "total_companies": result["total_companies"],
            "statistics": result["statistics"],
        }

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        _update_batch_status(
            batch_id,
            status="failed",
            error_message=error_msg,
            completed_at=datetime.utcnow(),
        )
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}


# ---- Celery Task (if available) ----
if CELERY_AVAILABLE and celery_app:
    @celery_app.task(bind=True, name="tasks.analyze_batch_csv")
    def analyze_batch_csv_task(self, file_path: str, batch_id: int):
        """
        Celery background task for batch CSV analysis.
        Updates task state so Frontend can poll progress.
        """
        return run_batch_analysis(file_path, batch_id)


# ---- Cache Cleanup: Remove stale single-query assessments > 30 days ----
def cleanup_stale_assessments(max_age_days: int = 30) -> int:
    """
    Remove old AIRiskAssessment records from single-query cache (batch_id IS NULL)
    that are older than max_age_days. Batch assessments are kept intact.

    Returns:
        Number of deleted records.
    """
    from app.models import AIRiskAssessment
    from datetime import timedelta

    db = SessionLocal()
    deleted_count = 0
    try:
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        stale = db.query(AIRiskAssessment).filter(
            AIRiskAssessment.batch_id.is_(None),
            AIRiskAssessment.created_at < cutoff,
        )
        deleted_count = stale.count()
        stale.delete(synchronize_session=False)
        db.commit()
        print(f"[CLEANUP] Deleted {deleted_count} stale single-query assessments older than {max_age_days} days")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] cleanup_stale_assessments: {e}")
    finally:
        db.close()
    return deleted_count
