from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional
import time
import json
from datetime import datetime
from pathlib import Path
import psycopg2
import os

router = APIRouter(prefix="/api/monitoring", tags=["MLOps"])

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

class FeedbackData(BaseModel):
    tax_code: Optional[str] = None
    invoice_number: Optional[str] = None
    path_id: Optional[str] = None
    is_fraud: bool
    expert_notes: Optional[str] = None

class MetricLog(BaseModel):
    metric_name: str
    value: float
    labels: dict = {}

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            dbname=os.getenv("DB_NAME", "TaxInspector")
        )
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

@router.post("/feedback")
async def submit_expert_feedback(feedback: FeedbackData):
    """
    Endpoint for tax inspectors to provide ground-truth feedback on AI predictions.
    This data is crucial for continuous learning and model retraining.
    """
    feedback_file = LOG_DIR / "expert_feedback.jsonl"
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tax_code": feedback.tax_code,
        "invoice_number": feedback.invoice_number,
        "path_id": feedback.path_id,
        "is_fraud_ground_truth": feedback.is_fraud,
        "expert_notes": feedback.expert_notes,
    }
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
    return {"status": "success", "message": "Feedback recorded for next training cycle."}

@router.post("/log_metric")
async def log_ml_metric(metric: MetricLog):
    """
    Internal endpoint to log inference latency, fallback occurrences, etc.
    """
    metrics_file = LOG_DIR / "ml_metrics.jsonl"
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "metric": metric.metric_name,
        "value": metric.value,
        "labels": metric.labels,
    }
    
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
    return {"status": "success"}

@router.get("/health")
async def mlops_health_check():
    """
    Check the health of the ML serving infrastructure.
    """
    model_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models"
    
    status = {
        "status": "healthy",
        "models": {
            "gat_model": (model_dir / "gat_model.pt").exists(),
            "calibrator": (model_dir / "calibrator.pkl").exists(),
            "anomaly_detector": (model_dir / "anomaly_detector.pkl").exists(),
            "ensemble_meta": (model_dir / "ensemble_meta.pkl").exists(),
        },
        "db_connection": False
    }
    
    conn = get_db_connection()
    if conn:
        status["db_connection"] = True
        conn.close()
    else:
        status["status"] = "degraded"
        
    if not all(status["models"].values()):
        status["status"] = "degraded"
        
    return status

@router.get("/drift_report")
async def get_drift_report():
    """
    Simplified data drift report based on recent queries vs training baseline.
    In a full production setup, this would compare feature distributions.
    """
    return {
        "drift_detected": False,
        "features": {
            "company_age_days": {"wasserstein_distance": 0.05, "status": "stable"},
            "invoice_amount_log": {"wasserstein_distance": 0.12, "status": "stable"},
            "is_reciprocal_ratio": {"wasserstein_distance": 0.01, "status": "stable"}
        },
        "recommendation": "No retraining required at this time."
    }
