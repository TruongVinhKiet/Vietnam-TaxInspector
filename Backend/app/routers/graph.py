"""
graph.py – VAT Network Graph API (GNN-powered)
================================================
Endpoints:
    GET  /api/graph?tax_code=...  → Full subgraph analysis with GNN inference
    GET  /api/graph/search?q=...  → Search companies for graph exploration
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional
import traceback

from ..database import get_db

router = APIRouter(prefix="/api", tags=["VAT Invoice Graph (GNN)"])

# Lazy-loaded GNN inference engine
_gnn_engine = None


def _get_gnn_engine():
    """Lazy-load the GNN inference engine (singleton)."""
    global _gnn_engine
    if _gnn_engine is None:
        try:
            from ml_engine.gnn_model import GNNInference
            _gnn_engine = GNNInference()
            _gnn_engine.load()
        except Exception as e:
            print(f"[WARN] GNN engine failed to load: {e}")
            from ml_engine.gnn_model import GNNInference
            _gnn_engine = GNNInference()
            _gnn_engine._loaded = False
    return _gnn_engine


@router.get("/graph")
def get_vat_invoice_graph(
    tax_code: Optional[str] = Query(None, description="Tax code tâm điểm để dựng subgraph"),
    depth: int = Query(2, ge=1, le=4, description="Độ sâu truy vết (1-4 bước)"),
    db: Session = Depends(get_db),
):
    """
    Phân tích đồ thị mạng lưới mua bán hóa đơn với GNN.
    
    - Nếu tax_code được cung cấp: trích xuất subgraph xung quanh công ty đó
    - Nếu không: trả về toàn bộ mạng lưới (giới hạn 200 nodes)
    """
    try:
        if tax_code:
            companies, invoices = _extract_subgraph(db, tax_code, depth)
        else:
            companies, invoices = _extract_full_graph(db, limit=200)

        if not companies:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu cho mã số thuế này.")

        # Run GNN inference
        engine = _get_gnn_engine()
        result = engine.predict(companies, invoices)

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích đồ thị: {str(e)}")


@router.get("/graph/search")
def search_companies_for_graph(
    q: str = Query(..., min_length=2, description="Từ khoá tìm kiếm (mã thuế hoặc tên)"),
    db: Session = Depends(get_db),
):
    """Tìm kiếm công ty để dựng đồ thị mạng lưới."""
    try:
        result = db.execute(text("""
            SELECT tax_code, name, industry, risk_score
            FROM companies
            WHERE tax_code ILIKE :q OR name ILIKE :q
            ORDER BY risk_score DESC
            LIMIT 10
        """), {"q": f"%{q}%"})
        
        rows = result.fetchall()
        return {
            "results": [
                {
                    "tax_code": r[0],
                    "name": r[1],
                    "industry": r[2],
                    "risk_score": float(r[3]) if r[3] else 0.0,
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/companies")
def get_all_companies(db: Session = Depends(get_db)):
    """Lấy danh sách các doanh nghiệp cho bảng tổng hợp (limit 500)."""
    try:
        result = db.execute(text("""
            SELECT tax_code, name, industry, is_active, risk_score
            FROM companies
            ORDER BY risk_score DESC
            LIMIT 500
        """))
        
        rows = result.fetchall()
        return {
            "results": [
                {
                    "tax_code": r[0],
                    "name": r[1],
                    "industry": r[2],
                    "is_active": bool(r[3]),
                    "risk_score": float(r[4]) if r[4] else 0.0,
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _extract_subgraph(db: Session, center_tax_code: str, depth: int, max_nodes: int = 150) -> tuple:
    """
    Extract a subgraph centered on a company, expanding outward by `depth` hops.
    Returns (companies, invoices) lists.
    """
    # BFS-expand from center
    visited_codes = {center_tax_code}
    frontier = {center_tax_code}

    for current_depth in range(depth):
        if not frontier or len(visited_codes) >= max_nodes:
            break
        placeholders = ",".join([f":tc{i}" for i in range(len(frontier))])
        params = {f"tc{i}": tc for i, tc in enumerate(frontier)}

        # Lấy các giao dịch mà trong đó source/target nằm trong tập frontier
        result = db.execute(text(f"""
            SELECT DISTINCT seller_tax_code, buyer_tax_code 
            FROM invoices
            WHERE seller_tax_code IN ({placeholders}) 
               OR buyer_tax_code IN ({placeholders})
        """), params)

        new_frontier = set()
        for row in result.fetchall():
            # Mở rộng nếu chưa chạm max_nodes
            for tc in [row[0], row[1]]:
                if tc not in visited_codes:
                    if len(visited_codes) < max_nodes:
                        new_frontier.add(tc)
                        visited_codes.add(tc)
                    else:
                        break
            if len(visited_codes) >= max_nodes:
                break
        frontier = new_frontier

    if not visited_codes:
        return [], []

    # Lấy thông tin các doanh nghiệp (Company)
    placeholders = ",".join([f":tc{i}" for i in range(len(visited_codes))])
    params = {f"tc{i}": tc for i, tc in enumerate(visited_codes)}

    has_geom = True
    try:
        db.execute(text("SELECT geom FROM companies LIMIT 1"))
    except Exception:
        has_geom = False
        db.rollback()

    if has_geom:
        comp_result = db.execute(text(f"""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   ST_Y(geom) as lat, ST_X(geom) as lng
            FROM companies
            WHERE tax_code IN ({placeholders})
        """), params)
    else:
        comp_result = db.execute(text(f"""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   0.0 as lat, 0.0 as lng
            FROM companies
            WHERE tax_code IN ({placeholders})
        """), params)

    columns = [desc[0] for desc in comp_result.cursor.description] if hasattr(comp_result, 'cursor') else \
              [col for col in comp_result.keys()]
    companies = [dict(zip(columns, row)) for row in comp_result.fetchall()]

    # Lấy các hóa đơn giữa những doanh nghiệp nằm trong sub-graph
    # Giới hạn số lượng invoice để tránh crash DOM frontend (max 500 đường nối)
    inv_result = db.execute(text(f"""
        SELECT seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number
        FROM invoices
        WHERE seller_tax_code IN ({placeholders})
          AND buyer_tax_code IN ({placeholders})
        ORDER BY amount DESC
        LIMIT 500
    """), params)

    inv_columns = [col for col in inv_result.keys()]
    invoices = [dict(zip(inv_columns, row)) for row in inv_result.fetchall()]

    return companies, invoices


def _extract_full_graph(db: Session, limit: int = 200) -> tuple:
    """Extract the full graph (limited to top N companies by risk & activity)."""
    has_geom = True
    try:
        db.execute(text("SELECT geom FROM companies LIMIT 1"))
    except Exception:
        has_geom = False
        db.rollback()

    if has_geom:
        comp_result = db.execute(text("""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   ST_Y(geom) as lat, ST_X(geom) as lng
            FROM companies
            ORDER BY risk_score DESC
            LIMIT :limit
        """), {"limit": limit})
    else:
        comp_result = db.execute(text("""
            SELECT tax_code, name, industry, registration_date, risk_score, is_active,
                   0.0 as lat, 0.0 as lng
            FROM companies
            ORDER BY risk_score DESC
            LIMIT :limit
        """), {"limit": limit})

    columns = [col for col in comp_result.keys()]
    companies = [dict(zip(columns, row)) for row in comp_result.fetchall()]

    if not companies:
        return [], []

    tax_codes = [c["tax_code"] for c in companies]
    placeholders = ",".join([f":tc{i}" for i in range(len(tax_codes))])
    params = {f"tc{i}": tc for i, tc in enumerate(tax_codes)}

    inv_result = db.execute(text(f"""
        SELECT seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number
        FROM invoices
        WHERE seller_tax_code IN ({placeholders})
          AND buyer_tax_code IN ({placeholders})
        ORDER BY date
    """), params)

    inv_columns = [col for col in inv_result.keys()]
    invoices = [dict(zip(inv_columns, row)) for row in inv_result.fetchall()]

    return companies, invoices
