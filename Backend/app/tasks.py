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


def _save_assessments(assessments: list, chunk_size: int = 2000):
    """Save individual risk assessments to DB using chunked bulk inserts."""
    from app.models import AIRiskAssessment

    if not assessments:
        return

    safe_chunk_size = max(200, int(chunk_size or 2000))

    def _to_record(a: dict) -> dict:
        return {
            "batch_id": a.get("batch_id"),
            "tax_code": str(a.get("tax_code", "")),
            "company_name": a.get("company_name"),
            "industry": a.get("industry"),
            "year": int(a.get("year") or 0),
            "revenue": float(a.get("revenue") or 0.0),
            "total_expenses": float(a.get("total_expenses") or 0.0),
            "f1_divergence": float(a.get("f1_divergence") or 0.0),
            "f2_ratio_limit": float(a.get("f2_ratio_limit") or 0.0),
            "f3_vat_structure": float(a.get("f3_vat_structure") or 0.0),
            "f4_peer_comparison": float(a.get("f4_peer_comparison") or 0.0),
            "anomaly_score": float(a.get("anomaly_score") or 0.0),
            "model_confidence": float(a.get("model_confidence") or 0.0),
            "model_version": a.get("model_version"),
            "risk_score": float(a.get("risk_score") or 0.0),
            "risk_level": a.get("risk_level", "low"),
            "red_flags": a.get("red_flags") or [],
            "shap_explanation": a.get("shap_explanation"),
            "yearly_history": a.get("yearly_history"),
        }

    db = SessionLocal()
    try:
        total = len(assessments)
        for start in range(0, total, safe_chunk_size):
            chunk = assessments[start:start + safe_chunk_size]
            rows = [_to_record(a) for a in chunk]
            db.bulk_insert_mappings(AIRiskAssessment, rows)
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

        # Read CSV with tax_code as string to preserve leading-zero MST values.
        df = pd.read_csv(file_path, dtype={"tax_code": "string"}, low_memory=False)
        if "tax_code" in df.columns:
            df["tax_code"] = (
                df["tax_code"]
                .astype("string")
                .str.strip()
            )
        # Use number of unique companies as total (not CSV row count)
        # because progress_callback in pipeline counts per-company, not per-row
        total_companies = df["tax_code"].nunique() if "tax_code" in df.columns else len(df)

        _update_batch_status(batch_id, total_rows=total_companies)

        # Load and run pipeline
        pipeline = get_pipeline()

        last_reported = 0

        def progress_cb(processed, total):
            nonlocal last_reported
            # Throttle DB progress writes to reduce lock/contention for very large batches.
            if processed == total or (processed - last_reported) >= 1000:
                _update_batch_status(batch_id, processed_rows=processed)
                last_reported = processed

        result = pipeline.predict_batch(df, batch_id=batch_id,
                                         progress_callback=progress_cb)

        # Persisting many rows can take noticeable time; expose a finalizing status.
        _update_batch_status(
            batch_id,
            status="finalizing",
            processed_rows=result["total_companies"],
        )

        # Save individual assessments to DB
        _save_assessments(result["assessments"], chunk_size=2000)

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
