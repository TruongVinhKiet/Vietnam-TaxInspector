"""
tax_agent_nl_query.py – Natural Language Query Executor
========================================================
Handles non-MST natural language queries:
  1. top_n_query: "Top 10 DN rủi ro cao nhất" → DB query
  2. company_name_lookup: "Phân tích công ty ABC" → fuzzy name match
  3. batch_analysis: CSV file → inline batch processing

Designed to plug into the orchestrator's NL fast-path (Step 2.5).
"""

from __future__ import annotations

import csv
import io
import logging
import time
from typing import Any

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


class NLQueryExecutor:
    """
    Execute natural language queries that don't require a specific MST.

    Usage:
        executor = NLQueryExecutor()
        result = executor.execute_top_n(db, n=10, sort_by="risk_score", mode="fraud")
    """

    # ─── Top-N Risky Companies ────────────────────────────────────────────

    def execute_top_n(
        self,
        db,
        *,
        n: int = 10,
        sort_by: str = "risk_score",
        mode: str = "full",
    ) -> dict[str, Any]:
        """
        Query top N companies by risk score from ai_risk_assessments.

        Args:
            db: SQLAlchemy session
            n: Number of companies to return (1-50)
            sort_by: Column to sort by (risk_score, anomaly_score)
            mode: Model mode for context

        Returns:
            {
                "companies": [{tax_code, company_name, industry, risk_score, ...}],
                "total": int,
                "status": "success"
            }
        """
        t0 = time.perf_counter()
        n = min(50, max(1, n))

        sort_column = "risk_score"
        if sort_by == "anomaly_score":
            sort_column = "anomaly_score"

        try:
            # Query from ai_risk_assessments (populated by batch analysis)
            rows = db.execute(
                sql_text(f"""
                    SELECT DISTINCT ON (a.tax_code)
                        a.tax_code,
                        a.company_name,
                        a.industry,
                        a.risk_score,
                        a.risk_level,
                        a.anomaly_score,
                        a.f1_divergence,
                        a.f2_ratio_limit,
                        a.f3_vat_structure,
                        a.f4_peer_comparison,
                        a.model_version,
                        a.year
                    FROM ai_risk_assessments a
                    WHERE a.risk_score IS NOT NULL
                    ORDER BY a.tax_code, a.{sort_column} DESC
                """),
            ).fetchall()

            # Sort all results by risk_score descending, take top N
            all_companies = []
            for row in rows:
                all_companies.append({
                    "tax_code": str(row[0] or ""),
                    "company_name": str(row[1] or ""),
                    "industry": str(row[2] or ""),
                    "risk_score": round(float(row[3] or 0), 2),
                    "risk_level": str(row[4] or ""),
                    "anomaly_score": round(float(row[5] or 0), 4),
                    "f1_divergence": round(float(row[6] or 0), 4),
                    "f2_ratio_limit": round(float(row[7] or 0), 4),
                    "f3_vat_structure": round(float(row[8] or 0), 4),
                    "f4_peer_comparison": round(float(row[9] or 0), 4),
                    "model_version": str(row[10] or ""),
                    "year": int(row[11] or 0),
                })

            # Sort by chosen column descending
            all_companies.sort(
                key=lambda c: c.get(sort_column, 0), reverse=True,
            )
            top_companies = all_companies[:n]

            # Add rank
            for i, c in enumerate(top_companies, 1):
                c["stt"] = i

            latency = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "[NLQuery] top_n: n=%d sort=%s found=%d latency=%.0fms",
                n, sort_by, len(top_companies), latency,
            )

            return {
                "companies": top_companies,
                "total": len(all_companies),
                "query_n": n,
                "sort_by": sort_by,
                "status": "success",
            }

        except Exception as exc:
            logger.warning("[NLQuery] top_n query failed: %s", exc)
            # Fallback: try companies table directly
            try:
                rows = db.execute(
                    sql_text("""
                        SELECT tax_code, company_name, industry
                        FROM companies
                        ORDER BY tax_code
                        LIMIT :n
                    """),
                    {"n": n},
                ).fetchall()

                companies = [
                    {
                        "stt": i + 1,
                        "tax_code": str(row[0] or ""),
                        "company_name": str(row[1] or ""),
                        "industry": str(row[2] or ""),
                        "risk_score": 0,
                        "risk_level": "unknown",
                    }
                    for i, row in enumerate(rows)
                ]
                return {"companies": companies, "total": len(companies), "status": "partial"}
            except Exception:
                return {"companies": [], "total": 0, "status": "error", "error": str(exc)}

    # ─── Company Name Fuzzy Search ────────────────────────────────────────

    def execute_company_name_search(
        self,
        db,
        *,
        name: str,
    ) -> dict[str, Any]:
        """
        Search for companies by name (case-insensitive LIKE match).

        Args:
            db: SQLAlchemy session
            name: Company name to search for

        Returns:
            {
                "matches": [{tax_code, company_name, industry, similarity}],
                "query": str,
                "status": "success"
            }
        """
        t0 = time.perf_counter()
        if not name or len(name.strip()) < 2:
            return {"matches": [], "query": name, "status": "empty_query"}

        search_term = name.strip()

        try:
            # Try exact-ish match first (ILIKE)
            rows = db.execute(
                sql_text("""
                    SELECT c.tax_code, c.company_name, c.industry
                    FROM companies c
                    WHERE c.company_name ILIKE :pattern
                    ORDER BY
                        CASE WHEN c.company_name ILIKE :exact THEN 0 ELSE 1 END,
                        c.company_name
                    LIMIT 20
                """),
                {
                    "pattern": f"%{search_term}%",
                    "exact": search_term,
                },
            ).fetchall()

            matches = []
            for row in rows:
                company_name = str(row[1] or "")
                # Simple similarity score
                similarity = _compute_similarity(search_term.lower(), company_name.lower())
                matches.append({
                    "tax_code": str(row[0] or ""),
                    "company_name": company_name,
                    "industry": str(row[2] or ""),
                    "similarity": round(similarity, 3),
                })

            # Sort by similarity descending
            matches.sort(key=lambda m: m["similarity"], reverse=True)

            latency = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "[NLQuery] name_search: query='%s' found=%d latency=%.0fms",
                search_term[:30], len(matches), latency,
            )

            return {
                "matches": matches[:10],
                "query": search_term,
                "total": len(matches),
                "status": "success",
            }

        except Exception as exc:
            logger.warning("[NLQuery] name_search failed: %s", exc)
            return {"matches": [], "query": search_term, "status": "error", "error": str(exc)}

    # ─── Inline Batch CSV Analysis ────────────────────────────────────────

    def execute_batch_inline(
        self,
        db,
        *,
        csv_content: bytes,
        filename: str,
    ) -> dict[str, Any]:
        """
        Perform inline batch analysis on CSV content.
        Reuses the batch scoring logic from ai_analysis.

        Args:
            db: SQLAlchemy session
            csv_content: Raw CSV bytes
            filename: Original filename

        Returns:
            {
                "total": int,
                "by_level": {critical, high, medium, low},
                "top_5": [{tax_code, company_name, risk_score, ...}],
                "assessments": [...],
                "filename": str,
                "status": "success"
            }
        """
        t0 = time.perf_counter()

        try:
            # Parse CSV
            text_content = csv_content.decode("utf-8-sig", errors="replace")
            reader = csv.DictReader(io.StringIO(text_content))
            rows = list(reader)

            if not rows:
                return {"total": 0, "status": "empty_file", "filename": filename}

            # Score each row using simplified risk heuristics
            assessments = []
            for row in rows:
                tax_code = str(row.get("tax_code", row.get("mst", row.get("MST", "")))).strip()
                company_name = str(row.get("company_name", row.get("ten_dn", row.get("TEN_DN", "")))).strip()
                industry = str(row.get("industry", row.get("nganh", ""))).strip()

                revenue = _safe_float(row.get("revenue", row.get("doanh_thu", 0)))
                expenses = _safe_float(row.get("total_expenses", row.get("chi_phi", 0)))
                tax_paid = _safe_float(row.get("tax_paid", row.get("thue_nop", 0)))

                # Compute risk signals
                risk_score, risk_level, signals = _compute_risk_signals(
                    revenue=revenue, expenses=expenses, tax_paid=tax_paid,
                )

                assessments.append({
                    "tax_code": tax_code,
                    "company_name": company_name,
                    "industry": industry,
                    "revenue": revenue,
                    "total_expenses": expenses,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "signals": signals,
                })

            # Sort by risk_score descending
            assessments.sort(key=lambda a: a["risk_score"], reverse=True)

            # Compute by_level
            by_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for a in assessments:
                level = a["risk_level"]
                if level in by_level:
                    by_level[level] += 1

            latency = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "[NLQuery] batch_inline: file=%s rows=%d latency=%.0fms",
                filename, len(assessments), latency,
            )

            return {
                "total": len(assessments),
                "by_level": by_level,
                "top_5": assessments[:5],
                "assessments": assessments[:50],
                "filename": filename,
                "status": "success",
            }

        except Exception as exc:
            logger.warning("[NLQuery] batch_inline failed: %s", exc)
            return {"total": 0, "status": "error", "error": str(exc), "filename": filename}


# ─── Utility Functions ────────────────────────────────────────────────────────

def _compute_similarity(query: str, target: str) -> float:
    """Simple substring-based similarity score."""
    if query == target:
        return 1.0
    if query in target:
        return 0.8 + 0.2 * (len(query) / max(len(target), 1))
    # Character overlap ratio
    query_set = set(query)
    target_set = set(target)
    if not query_set:
        return 0.0
    overlap = len(query_set & target_set) / len(query_set | target_set)
    return round(overlap * 0.6, 3)


def _safe_float(val) -> float:
    """Safely convert a value to float."""
    if val is None or val == "":
        return 0.0
    try:
        # Handle Vietnamese number formatting (1.234.567,89)
        s = str(val).strip().replace(".", "").replace(",", ".")
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _compute_risk_signals(
    *,
    revenue: float,
    expenses: float,
    tax_paid: float,
) -> tuple[float, str, list[str]]:
    """
    Compute simplified risk score from financial signals.
    Returns: (risk_score 0-100, risk_level, list of signal descriptions)
    """
    signals = []
    score = 25.0  # base score

    if revenue > 0:
        # Signal 1: expense/revenue ratio
        expense_ratio = expenses / revenue
        if expense_ratio > 0.95:
            score += 25
            signals.append(f"Tỷ lệ chi phí/doanh thu rất cao: {expense_ratio:.1%}")
        elif expense_ratio > 0.85:
            score += 15
            signals.append(f"Tỷ lệ chi phí/doanh thu cao: {expense_ratio:.1%}")

        # Signal 2: effective tax rate
        effective_tax = tax_paid / revenue if revenue > 0 else 0
        if effective_tax < 0.01 and revenue > 1_000_000_000:
            score += 20
            signals.append(f"Thuế suất hiệu dụng rất thấp: {effective_tax:.2%}")
        elif effective_tax < 0.03:
            score += 10
            signals.append(f"Thuế suất hiệu dụng thấp: {effective_tax:.2%}")

    # Signal 3: suspicious zero revenue
    if revenue == 0 and expenses > 0:
        score += 15
        signals.append("Doanh thu bằng 0 nhưng có chi phí")

    # Signal 4: negative profit
    if revenue > 0 and expenses > revenue:
        score += 10
        signals.append("Lỗ kinh doanh (chi phí > doanh thu)")

    # Clamp score
    score = min(100.0, max(0.0, score))

    # Determine level
    if score >= 75:
        level = "critical"
    elif score >= 55:
        level = "high"
    elif score >= 35:
        level = "medium"
    else:
        level = "low"

    return round(score, 1), level, signals
