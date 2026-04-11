from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from .. import schemas

router = APIRouter(prefix="/api", tags=["VAT Invoice Graph"])


@router.get("/graph", response_model=schemas.GraphResponse)
def get_vat_invoice_graph(db: Session = Depends(get_db)):
    """
    Tab 2: Phân tích đồ thị mạng lưới mua bán hóa đơn.
    TODO: Fetch invoices từ DB -> NetworkX tìm vòng tròn giao dịch.
    """
    nodes = [
        {"id": "C01", "label": "CTY Alpha", "group": "suspicious"},
        {"id": "C02", "label": "CTY Beta", "group": "suspicious"},
        {"id": "C03", "label": "CTY Gamma", "group": "suspicious"},
        {"id": "C04", "label": "CTY Delta", "group": "normal"},
    ]
    edges = [
        {"from_node": "C01", "to_node": "C02", "value": 500_000_000},
        {"from_node": "C02", "to_node": "C03", "value": 450_000_000},
        {"from_node": "C03", "to_node": "C01", "value": 420_000_000},  # Circular!
        {"from_node": "C04", "to_node": "C01", "value": 100_000_000},
    ]

    return schemas.GraphResponse(
        nodes=nodes,
        edges=edges,
        suspicious_clusters=[["C01", "C02", "C03"]],
    )
