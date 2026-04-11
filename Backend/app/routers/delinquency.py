from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from .. import schemas

router = APIRouter(prefix="/api", tags=["Delinquency Prediction"])


@router.get("/delinquency", response_model=List[schemas.DelinquencyPrediction])
def get_delinquency_forecast(db: Session = Depends(get_db)):
    """
    Tab 3: Dự báo nguy cơ trễ hạn nộp thuế.
    TODO: Thay thế bằng DBSCAN/KMeans + RandomForest pipeline thật.
    """
    predictions = [
        {"tax_code": "0100112233", "company_name": "CTY TNHH Xây Dựng Số 1", "probability": 0.89, "cluster": "Nhóm rủi ro cao"},
        {"tax_code": "0311554422", "company_name": "CTY CP Bán Lẻ Nhanh", "probability": 0.72, "cluster": "Nhóm rủi ro cao"},
        {"tax_code": "0400557766", "company_name": "Dịch vụ Vận tải Hải Vân", "probability": 0.65, "cluster": "Vấn đề Dòng tiền"},
        {"tax_code": "0500889900", "company_name": "CTY TNHH Sản Xuất Thép", "probability": 0.58, "cluster": "Suy giảm Theo mùa"},
        {"tax_code": "0600112244", "company_name": "Tập đoàn Viễn thông Omega", "probability": 0.52, "cluster": "Vấn đề Dòng tiền"},
    ]

    return predictions
