from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import random

from ..database import get_db
from .. import models, schemas

router = APIRouter(prefix="/api", tags=["Fraud Risk Scoring"])


@router.post("/scoring/{tax_code}", response_model=schemas.FraudRiskPrediction)
def score_company_risk(tax_code: str, db: Session = Depends(get_db)):
    """
    Tab 1: Chấm điểm rủi ro gian lận cho một doanh nghiệp theo MST.
    TODO: Thay thế mock bằng XGBoost / Isolation Forest thật.
    """
    company = db.query(models.Company).filter(models.Company.tax_code == tax_code).first()
    if not company:
        raise HTTPException(status_code=404, detail="Không tìm thấy doanh nghiệp với MST này.")

    # --- Mock Prediction Logic (sẽ thay bằng model thật) ---
    mock_score = round(random.uniform(10.0, 95.0), 2)
    mock_red_flags = []

    if mock_score > 70:
        mock_red_flags.append("Doanh thu tăng 200% nhưng nộp thuế giảm 50%")
    if mock_score > 85:
        mock_red_flags.append("Chi phí đầu vào tăng đột biến trong Quý gần nhất")
    if mock_score > 90:
        mock_red_flags.append("Tỷ suất lợi nhuận thấp bất thường so với trung bình ngành")

    return schemas.FraudRiskPrediction(
        tax_code=tax_code,
        risk_score=mock_score,
        red_flags=mock_red_flags,
    )
