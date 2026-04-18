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
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Any

from ..database import get_db
from ..observability import get_structured_logger, log_event
from . import monitoring as monitoring_router

router = APIRouter(prefix="/api", tags=["VAT Invoice Graph (GNN)"])
logger = get_structured_logger("taxinspector.graph")

# Lazy-loaded GNN inference engine
_gnn_engine = None


def _get_gnn_engine():
    """Lazy-load the GNN inference engine (singleton)."""
    global _gnn_engine
    if _gnn_engine is None:
        from ml_engine.gnn_model import GNNInference
        _gnn_engine = GNNInference()
        try:
            _gnn_engine.load()
            if _gnn_engine._loaded:
                _gnn_engine._load_error = None
                log_event(logger, "info", "graph_gnn_engine_ready", model_loaded=True)
            else:
                _gnn_engine._load_error = "model_artifacts_missing"
                log_event(
                    logger,
                    "warning",
                    "graph_gnn_engine_fallback",
                    reason="model_artifacts_missing",
                )
        except (FileNotFoundError, OSError) as e:
            _gnn_engine._loaded = False
            _gnn_engine._load_error = "artifact_io_error"
            log_event(
                logger,
                "error",
                "graph_gnn_engine_load_failed",
                error_type=type(e).__name__,
                reason="artifact_io_error",
                error=str(e),
            )
        except RuntimeError as e:
            _gnn_engine._loaded = False
            _gnn_engine._load_error = "runtime_error"
            log_event(
                logger,
                "error",
                "graph_gnn_engine_load_failed",
                error_type=type(e).__name__,
                reason="runtime_error",
                error=str(e),
            )
        except Exception as e:
            _gnn_engine._loaded = False
            _gnn_engine._load_error = "initialization_error"
            log_event(
                logger,
                "error",
                "graph_gnn_engine_load_failed",
                error_type=type(e).__name__,
                reason="initialization_error",
                error=str(e),
            )
    return _gnn_engine


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sanitize_policy(policy: Any) -> dict:
    if not isinstance(policy, dict):
        return {}
    return {
        "cold_start_degree_threshold": int(_to_float(policy.get("cold_start_degree_threshold", 0), 0)),
        "cold_start_threshold_delta": round(_to_float(policy.get("cold_start_threshold_delta", 0.0), 0.0), 4),
        "node_blend_alpha_gnn": round(_to_float(policy.get("node_blend_alpha_gnn", 0.0), 0.0), 4),
    }


def _enrich_graph_result(result: dict, engine: Any, tax_code: Optional[str], depth: int) -> dict:
    model_loaded = bool(result.get("model_loaded", getattr(engine, "_loaded", False)))
    ensemble_active = bool(result.get("ensemble_active", False))
    fallback_active = bool(result.get("fallback_active", not model_loaded))
    if fallback_active and not result.get("fallback_reason"):
        result["fallback_reason"] = getattr(engine, "_load_error", "model_unavailable")

    thresholds_raw = result.get("decision_thresholds")
    if isinstance(thresholds_raw, dict):
        node_threshold = _to_float(thresholds_raw.get("node", 0.5), 0.5)
        edge_threshold = _to_float(thresholds_raw.get("edge", 0.5), 0.5)
        policy = _sanitize_policy(thresholds_raw.get("policy"))
    else:
        node_threshold = 0.5
        edge_threshold = 0.5
        policy = {}

    decision_thresholds = {
        "node": round(node_threshold, 4),
        "edge": round(edge_threshold, 4),
        "policy": policy,
    }
    result["decision_thresholds"] = decision_thresholds

    nodes = result.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            shell_probability = _to_float(
                node.get("shell_probability"),
                _to_float(node.get("risk_score", 0.0), 0.0) / 100.0,
            )
            node_threshold_local = _to_float(node.get("decision_threshold", node_threshold), node_threshold)
            node["shell_probability"] = round(shell_probability, 4)
            node["decision_threshold"] = round(node_threshold_local, 4)
            node["threshold_margin"] = round(shell_probability - node_threshold_local, 4)

    edges = result.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            circular_probability = _to_float(edge.get("circular_probability"), 0.0)
            edge["circular_probability"] = round(circular_probability, 4)
            edge["decision_threshold"] = round(edge_threshold, 4)
            edge["threshold_margin"] = round(circular_probability - edge_threshold, 4)

    attention_weights = result.get("attention_weights")
    attention_top = []
    if isinstance(attention_weights, list):
        normalized_attention = []
        for item in attention_weights:
            if not isinstance(item, dict):
                continue
            normalized_attention.append(
                {
                    "from": str(item.get("from", "")),
                    "to": str(item.get("to", "")),
                    "weight": round(_to_float(item.get("weight", 0.0), 0.0), 4),
                }
            )
        attention_top = sorted(normalized_attention, key=lambda row: row["weight"], reverse=True)[:5]
        attention_count = len(normalized_attention)
    else:
        attention_count = 0

    result["attention_summary"] = {
        "count": attention_count,
        "top_edges": attention_top,
    }

    result["query_context"] = {
        "tax_code": tax_code,
        "depth": int(depth),
    }

    result["contract_version"] = "graph-intelligence-v1"
    result["model_info"] = {
        "contract_version": "graph-intelligence-v1",
        "inference_mode": "gnn_ensemble" if model_loaded else "heuristic_fallback",
        "model_loaded": model_loaded,
        "ensemble_active": ensemble_active,
        "fallback_active": fallback_active,
        "fallback_reason": result.get("fallback_reason"),
        "decision_thresholds": decision_thresholds,
    }

    return result


def _build_split_trigger_status_context(snapshot_source: str = "graph_main") -> dict[str, Any]:
    payload = monitoring_router.get_split_trigger_status_snapshot(
        persist_snapshot=False,
        snapshot_source=snapshot_source,
    )
    if isinstance(payload, dict):
        return payload
    return {
        "ready": False,
        "schema_ready": False,
        "readiness_score": 0,
        "reason": "Không thể tải split-trigger status.",
        "track_status": {},
        "totals": {"enabled_rules": 0, "passed_rules": 0},
        "generated_at": "",
    }


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
    except SQLAlchemyError as e:
        log_event(
            logger,
            "error",
            "graph_data_extraction_failed",
            error_type=type(e).__name__,
            tax_code=tax_code,
            depth=depth,
        )
        raise HTTPException(status_code=500, detail="Lỗi truy vấn dữ liệu đồ thị.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_data_extraction_unexpected_error",
            error_type=type(e).__name__,
            tax_code=tax_code,
            depth=depth,
        )
        raise HTTPException(status_code=500, detail=f"Lỗi dựng subgraph: {str(e)}")

    if not companies:
        raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu cho mã số thuế này.")

    # Run GNN inference
    try:
        engine = _get_gnn_engine()
    except (ImportError, ModuleNotFoundError) as e:
        log_event(
            logger,
            "error",
            "graph_gnn_dependency_error",
            error_type=type(e).__name__,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Không thể khởi tạo GNN engine do lỗi phụ thuộc.")

    try:
        result = engine.predict(companies, invoices)
    except RuntimeError as e:
        log_event(
            logger,
            "error",
            "graph_gnn_inference_runtime_error",
            error_type=type(e).__name__,
            error=str(e),
            tax_code=tax_code,
        )
        raise HTTPException(status_code=500, detail="Lỗi runtime khi suy luận GNN.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_gnn_inference_error",
            error_type=type(e).__name__,
            error=str(e),
            tax_code=tax_code,
        )
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích đồ thị: {str(e)}")

    if isinstance(result, dict):
        fallback_active = not bool(getattr(engine, "_loaded", False))
        result.setdefault("fallback_active", fallback_active)
        if fallback_active:
            result.setdefault("fallback_reason", getattr(engine, "_load_error", "model_unavailable"))
            log_event(
                logger,
                "warning",
                "graph_response_fallback_active",
                tax_code=tax_code,
                reason=result.get("fallback_reason"),
            )

        result = _enrich_graph_result(result, engine=engine, tax_code=tax_code, depth=depth)
        result["split_trigger_status"] = _build_split_trigger_status_context(
            snapshot_source="graph_main",
        )

    return result


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
    except SQLAlchemyError as e:
        log_event(
            logger,
            "error",
            "graph_company_search_failed",
            error_type=type(e).__name__,
            query=q,
        )
        raise HTTPException(status_code=500, detail="Lỗi truy vấn tìm kiếm doanh nghiệp.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_company_search_unexpected_error",
            error_type=type(e).__name__,
            query=q,
        )
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
    except SQLAlchemyError as e:
        log_event(
            logger,
            "error",
            "graph_companies_list_failed",
            error_type=type(e).__name__,
        )
        raise HTTPException(status_code=500, detail="Lỗi truy vấn danh sách doanh nghiệp.")
    except Exception as e:
        log_event(
            logger,
            "error",
            "graph_companies_list_unexpected_error",
            error_type=type(e).__name__,
        )
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


# ════════════════════════════════════════════════════════════════
#  Graph Intelligence 2.0 (Program B) – Flagship Endpoints
# ════════════════════════════════════════════════════════════════

@router.get("/graph/motifs")
def detect_graph_motifs(
    tax_code: Optional[str] = Query(None, description="Tax code tâm điểm"),
    depth: int = Query(2, ge=1, le=4),
    db: Session = Depends(get_db),
):
    """
    Motif Detection: Identify suspicious transaction patterns in the VAT invoice graph.
    Detects: triangles (carousel fraud), stars (shell hubs), chains (layering),
    fan-out/fan-in patterns.
    """
    try:
        if tax_code:
            companies, invoices = _extract_subgraph(db, tax_code, depth)
        else:
            companies, invoices = _extract_full_graph(db, limit=200)
    except Exception as e:
        log_event(logger, "error", "graph_motif_data_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu: {str(e)}")

    if not invoices:
        return {
            "status": "no_data",
            "message": "Không có dữ liệu giao dịch để phân tích motif.",
            "motifs": {},
            "summary": {},
        }

    try:
        from ml_engine.graph_intelligence import MotifDetector
        detector = MotifDetector()
        result = detector.detect_all(companies, invoices)

        log_event(
            logger, "info", "graph_motif_detection_complete",
            triangles=result["summary"]["total_triangles"],
            stars=result["summary"]["total_stars"],
            chains=result["summary"]["total_chains"],
        )
        return result

    except Exception as e:
        log_event(logger, "error", "graph_motif_detection_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích motif: {str(e)}")


@router.get("/graph/link-prediction")
def predict_graph_links(
    tax_code: Optional[str] = Query(None),
    depth: int = Query(2, ge=1, le=4),
    top_k: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Link Prediction: Predict likely future fraudulent connections between companies.
    Uses Jaccard coefficient + Adamic-Adar index + topology risk.
    """
    try:
        if tax_code:
            companies, invoices = _extract_subgraph(db, tax_code, depth)
        else:
            companies, invoices = _extract_full_graph(db, limit=200)
    except Exception as e:
        log_event(logger, "error", "graph_link_pred_data_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu: {str(e)}")

    if not invoices:
        return {"predictions": [], "total": 0, "message": "Không đủ dữ liệu giao dịch."}

    try:
        from ml_engine.graph_intelligence import LinkPredictor
        predictor = LinkPredictor()
        predictions = predictor.predict_new_links(companies, invoices, top_k=top_k)

        log_event(
            logger, "info", "graph_link_prediction_complete",
            predicted_links=len(predictions),
        )
        return {
            "predictions": predictions,
            "total": len(predictions),
            "query_context": {"tax_code": tax_code, "depth": depth, "top_k": top_k},
        }

    except Exception as e:
        log_event(logger, "error", "graph_link_prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán liên kết: {str(e)}")


@router.get("/graph/ownership")
def analyze_ownership_network(
    tax_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Ownership Graph Analysis: Detect shell company networks through ownership relationships.
    Identifies common controllers, ownership chains, and cross-ownership trades.
    """
    try:
        # Load ownership links
        if tax_code:
            ownership_result = db.execute(text("""
                SELECT ol.id, ol.parent_tax_code, ol.child_tax_code, ol.ownership_percent,
                       ol.relationship_type, ol.person_name, ol.person_id,
                       ol.effective_date, ol.end_date, ol.verified
                FROM ownership_links ol
                WHERE ol.parent_tax_code = :tc OR ol.child_tax_code = :tc
                   OR ol.parent_tax_code IN (
                       SELECT child_tax_code FROM ownership_links WHERE parent_tax_code = :tc
                   )
                   OR ol.child_tax_code IN (
                       SELECT parent_tax_code FROM ownership_links WHERE child_tax_code = :tc
                   )
                LIMIT 500
            """), {"tc": tax_code})
        else:
            ownership_result = db.execute(text("""
                SELECT id, parent_tax_code, child_tax_code, ownership_percent,
                       relationship_type, person_name, person_id,
                       effective_date, end_date, verified
                FROM ownership_links
                ORDER BY ownership_percent DESC
                LIMIT 500
            """))

        ownership_cols = [col for col in ownership_result.keys()]
        ownership_links = [dict(zip(ownership_cols, row)) for row in ownership_result.fetchall()]

        # Load relevant invoices for cross-ownership trade detection
        if tax_code:
            _, invoices = _extract_subgraph(db, tax_code, depth=2)
        else:
            _, invoices = _extract_full_graph(db, limit=200)

    except Exception as e:
        log_event(logger, "error", "graph_ownership_data_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu sở hữu: {str(e)}")

    try:
        from ml_engine.graph_intelligence import OwnershipGraphAnalyzer
        analyzer = OwnershipGraphAnalyzer()
        result = analyzer.analyze(ownership_links, invoices)

        log_event(
            logger, "info", "graph_ownership_analysis_complete",
            clusters=result.get("summary", {}).get("total_clusters", 0),
            cross_trades=result.get("summary", {}).get("total_cross_trades", 0),
        )
        return result

    except Exception as e:
        log_event(logger, "error", "graph_ownership_analysis_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích sở hữu: {str(e)}")


@router.get("/graph/ring-scoring")
def score_transaction_rings(
    tax_code: Optional[str] = Query(None),
    depth: int = Query(2, ge=1, le=4),
    db: Session = Depends(get_db),
):
    """
    Ring Scoring: Score circular transaction rings by severity.
    Multi-factor analysis: amount, speed, and complexity.
    """
    try:
        if tax_code:
            companies, invoices = _extract_subgraph(db, tax_code, depth)
        else:
            companies, invoices = _extract_full_graph(db, limit=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn dữ liệu: {str(e)}")

    if not invoices:
        return {"rings": [], "total": 0}

    # Use existing GNN engine to get cycles
    try:
        engine = _get_gnn_engine()
        result = engine.predict(companies, invoices)
        cycles = result.get("cycles", [])
    except Exception:
        cycles = []

    try:
        from ml_engine.graph_intelligence import RingScorer
        scorer = RingScorer()
        scored_rings = scorer.score_rings(cycles, invoices)

        log_event(
            logger, "info", "graph_ring_scoring_complete",
            rings_scored=len(scored_rings),
        )
        return {
            "rings": scored_rings,
            "total": len(scored_rings),
            "critical_count": sum(1 for r in scored_rings if r["risk_level"] == "critical"),
            "query_context": {"tax_code": tax_code, "depth": depth},
        }

    except Exception as e:
        log_event(logger, "error", "graph_ring_scoring_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi chấm điểm vòng lặp: {str(e)}")

