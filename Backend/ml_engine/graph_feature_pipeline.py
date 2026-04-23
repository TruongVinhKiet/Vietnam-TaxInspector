from __future__ import annotations

from collections import defaultdict
from datetime import datetime, date
from statistics import mean, pstdev
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


class GraphFeaturePipeline:
    """Compute forensic signals outside of GNN core inference."""

    CONTRACT_VERSION = "graph-intelligence-v2.1"

    def run(
        self,
        db: Session,
        companies: list[dict],
        invoices: list[dict],
        nodes: list[dict],
        edges: list[dict],
        ownership_links: list[dict] | None = None,
    ) -> dict[str, Any]:
        if not invoices:
            return {
                "contract_version": self.CONTRACT_VERSION,
                "nodes": nodes,
                "edges": edges,
                "integrity_signals": {"available": False, "reason": "no_invoice_context"},
                "pricing_signals": {"available": False, "reason": "no_invoice_context"},
                "phoenix_signals": {"available": False, "reason": "no_invoice_context"},
                "forensic_logs": [],
                "evidence_paths": [],
            }

        invoice_numbers = {str(inv.get("invoice_number", "")).strip() for inv in invoices}
        scoped_invoices = [inv for inv in invoices if str(inv.get("invoice_number", "")).strip() in invoice_numbers]
        inv_map = {str(inv.get("invoice_number", "")).strip(): inv for inv in scoped_invoices}
        inv_id_map = self._fetch_invoice_ids(db, invoice_numbers)

        line_items = self._fetch_line_items(db, list(inv_id_map.values()))
        lifecycle_events = self._fetch_lifecycle_events(db, list(inv_id_map.values()))
        payments = self._fetch_invoice_payments(db, list(inv_id_map.values()))

        edge_by_number = {str(e.get("invoice_number", "")).strip(): e for e in edges}
        node_by_tax_code = {str(n.get("tax_code", n.get("id", ""))).strip(): n for n in nodes}
        company_by_tax_code = {str(c.get("tax_code", "")).strip(): c for c in companies}

        id_to_number = {v: k for k, v in inv_id_map.items()}
        integrity = self._compute_integrity(scoped_invoices, lifecycle_events, payments, id_to_number)
        washout = self._compute_washout(scoped_invoices)
        pricing = self._compute_pricing(scoped_invoices, line_items, company_by_tax_code)
        phoenix = self._compute_phoenix(companies, scoped_invoices, ownership_links or [])
        payment_consistency = self._compute_payment_consistency(scoped_invoices, payments, inv_id_map)

        for tc, stats in washout["node_scores"].items():
            node = node_by_tax_code.get(tc)
            if node:
                node["vat_washout_score"] = round(stats["vat_washout_score"], 4)

        for tc, exposure in pricing["node_exposure"].items():
            node = node_by_tax_code.get(tc)
            if node:
                node["industry_mismatch_exposure"] = round(exposure, 4)

        for tc, score in phoenix["node_scores"].items():
            node = node_by_tax_code.get(tc)
            if node:
                node["phoenix_candidate_score"] = round(score, 4)

        for inv_no, edge in edge_by_number.items():
            edge["lifecycle_state"] = integrity["invoice_state"].get(inv_no, "issued")
            edge["effective"] = edge["lifecycle_state"] not in {"canceled"}
            edge["industry_goods_mismatch_score"] = round(pricing["edge_mismatch"].get(inv_no, 0.0), 4)
            edge["price_deviation_score"] = round(pricing["edge_price_deviation"].get(inv_no, 0.0), 4)
            edge["invoice_payment_match_score"] = round(payment_consistency["edge_match_score"].get(inv_no, 0.0), 4)

        forensic_logs = []
        forensic_logs.extend(integrity.get("logs", []))
        forensic_logs.extend(washout.get("logs", []))
        forensic_logs.extend(pricing.get("logs", []))
        forensic_logs.extend(payment_consistency.get("logs", []))
        forensic_logs.extend(phoenix.get("logs", []))

        evidence_paths = []
        evidence_paths.extend(phoenix.get("succession_links", []))
        evidence_paths.extend(integrity.get("incidents_paths", []))
        evidence_paths.extend(payment_consistency.get("mismatch_paths", []))

        return {
            "contract_version": self.CONTRACT_VERSION,
            "nodes": nodes,
            "edges": edges,
            "integrity_signals": integrity["payload"],
            "pricing_signals": pricing["payload"],
            "phoenix_signals": phoenix["payload"],
            "forensic_logs": forensic_logs[:20],
            "evidence_paths": evidence_paths[:20],
            "forensic_metrics_patch": {
                "vat_washout_node_count": washout["high_risk_count"],
                "mismatch_edge_count": pricing["mismatch_count"],
                "phoenix_link_count": len(phoenix.get("succession_links", [])),
                "payment_mismatch_count": payment_consistency["mismatch_count"],
            },
        }

    def _fetch_invoice_ids(self, db: Session, invoice_numbers: set[str]) -> dict[str, int]:
        if not invoice_numbers:
            return {}
        try:
            rows = db.execute(
                text(
                    """
                    SELECT id, invoice_number
                    FROM invoices
                    WHERE invoice_number = ANY(:invoice_numbers)
                    """
                ),
                {"invoice_numbers": list(invoice_numbers)},
            ).fetchall()
        except Exception:
            db.rollback()
            return {}
        return {str(row[1]): int(row[0]) for row in rows}

    def _fetch_line_items(self, db: Session, invoice_ids: list[int]) -> list[dict]:
        if not invoice_ids:
            return []
        try:
            rows = db.execute(
                text(
                    """
                    SELECT invoice_id, item_code, quantity, unit_price, line_amount, unit
                    FROM invoice_line_items
                    WHERE invoice_id = ANY(:invoice_ids)
                    """
                ),
                {"invoice_ids": invoice_ids},
            ).fetchall()
        except Exception:
            db.rollback()
            return []
        return [
            {
                "invoice_id": int(r[0]),
                "item_code": str(r[1] or ""),
                "quantity": _to_float(r[2]),
                "unit_price": _to_float(r[3]),
                "line_amount": _to_float(r[4]),
                "unit": str(r[5] or ""),
            }
            for r in rows
        ]

    def _fetch_lifecycle_events(self, db: Session, invoice_ids: list[int]) -> list[dict]:
        if not invoice_ids:
            return []
        try:
            rows = db.execute(
                text(
                    """
                    SELECT invoice_id, event_type, event_time, replacement_invoice_id, source
                    FROM invoice_lifecycle_events
                    WHERE invoice_id = ANY(:invoice_ids)
                    """
                ),
                {"invoice_ids": invoice_ids},
            ).fetchall()
        except Exception:
            db.rollback()
            return []
        return [
            {
                "invoice_id": int(r[0]),
                "event_type": str(r[1] or "issued"),
                "event_time": r[2],
                "replacement_invoice_id": r[3],
                "source": str(r[4] or ""),
            }
            for r in rows
        ]

    def _fetch_invoice_payments(self, db: Session, invoice_ids: list[int]) -> list[dict]:
        if not invoice_ids:
            return []
        try:
            rows = db.execute(
                text(
                    """
                    SELECT invoice_id, payer_tax_code, payee_tax_code, paid_amount, paid_at, reference_no
                    FROM invoice_payments
                    WHERE invoice_id = ANY(:invoice_ids)
                    """
                ),
                {"invoice_ids": invoice_ids},
            ).fetchall()
        except Exception:
            db.rollback()
            return []
        return [
            {
                "invoice_id": int(r[0]),
                "payer_tax_code": str(r[1] or ""),
                "payee_tax_code": str(r[2] or ""),
                "paid_amount": _to_float(r[3]),
                "paid_at": r[4],
                "reference_no": str(r[5] or ""),
            }
            for r in rows
        ]

    def _compute_integrity(
        self,
        invoices: list[dict],
        events: list[dict],
        payments: list[dict],
        id_to_number: dict[int, str],
    ) -> dict[str, Any]:
        if not events:
            return {
                "payload": {"available": False, "reason": "missing_invoice_lifecycle_events"},
                "invoice_state": {},
                "logs": [],
                "incidents_paths": [],
            }
        by_invoice_id = defaultdict(list)
        for ev in events:
            by_invoice_id[int(ev["invoice_id"])].append(ev)
        for inv_events in by_invoice_id.values():
            inv_events.sort(key=lambda x: str(x.get("event_time", "")))

        total = max(1, len(by_invoice_id))
        canceled = 0
        replace_latencies = []
        quarter_end_cancel = 0
        invoice_state = {}
        top_incidents = []

        payment_by_invoice = defaultdict(float)
        for p in payments:
            payment_by_invoice[int(p["invoice_id"])] += _to_float(p.get("paid_amount"))

        for invoice_id, inv_events in by_invoice_id.items():
            state = inv_events[-1]["event_type"] if inv_events else "issued"
            invoice_no = id_to_number.get(invoice_id, str(invoice_id))
            invoice_state[invoice_no] = state
            if state == "canceled":
                canceled += 1
            issued_time = next((ev["event_time"] for ev in inv_events if ev["event_type"] == "issued"), None)
            replaced_time = next((ev["event_time"] for ev in inv_events if ev["event_type"] == "replaced"), None)
            canceled_time = next((ev["event_time"] for ev in inv_events if ev["event_type"] == "canceled"), None)
            if issued_time and replaced_time:
                try:
                    replace_latencies.append(max(0.0, (replaced_time - issued_time).total_seconds() / 3600.0))
                except Exception:
                    pass
            if issued_time and canceled_time:
                if issued_time.month in {3, 6, 9, 12} and issued_time.day >= 25:
                    quarter_end_cancel += 1
            if state in {"canceled", "replaced"} and payment_by_invoice.get(invoice_id, 0.0) > 0:
                top_incidents.append(
                    {
                        "invoice_id": invoice_id,
                        "state": state,
                        "paid_amount": round(payment_by_invoice.get(invoice_id, 0.0), 2),
                    }
                )

        cancel_rate = canceled / total
        replace_latency = mean(replace_latencies) if replace_latencies else 0.0
        cross_party_mismatch_count = len(top_incidents)
        integrity_risk_score = min(
            1.0,
            0.45 * cancel_rate
            + 0.25 * min(1.0, replace_latency / 72.0)
            + 0.15 * min(1.0, quarter_end_cancel / max(1, total // 20))
            + 0.15 * min(1.0, cross_party_mismatch_count / max(1, total // 15)),
        )
        incidents_paths = [
            {
                "path_id": f"INTEGRITY-{idx+1}",
                "summary": f"Hoa don #{item['invoice_id']} co trang thai {item['state']} nhung da co thanh toan.",
                "risk_level": "high",
                "companies": [],
                "hops": [],
            }
            for idx, item in enumerate(top_incidents[:5])
        ]
        return {
            "payload": {
                "available": True,
                "cancel_rate": round(cancel_rate, 4),
                "replace_latency_hours": round(replace_latency, 2),
                "quarter_end_issue_then_cancel": int(quarter_end_cancel),
                "cross_party_mismatch_count": int(cross_party_mismatch_count),
                "integrity_risk_score": round(integrity_risk_score, 4),
                "top_incidents": top_incidents[:10],
            },
            "invoice_state": invoice_state,
            "logs": [
                {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": "high" if integrity_risk_score >= 0.6 else "medium",
                    "title": "Invoice Lifecycle Integrity",
                    "description": f"cancel_rate={cancel_rate:.1%}, quarter_end_cancel={quarter_end_cancel}, mismatch={cross_party_mismatch_count}.",
                }
            ],
            "incidents_paths": incidents_paths,
        }

    def _compute_washout(self, invoices: list[dict]) -> dict[str, Any]:
        per_node_in = defaultdict(float)
        per_node_out = defaultdict(float)
        for inv in invoices:
            s = str(inv.get("seller_tax_code", ""))
            b = str(inv.get("buyer_tax_code", ""))
            amount = _to_float(inv.get("amount"), 0.0)
            per_node_out[s] += amount
            per_node_in[b] += amount
        node_scores = {}
        high_risk = 0
        for tc in set(per_node_in) | set(per_node_out):
            in_total = per_node_in.get(tc, 0.0)
            out_total = per_node_out.get(tc, 0.0)
            ratio = min(in_total, out_total) / max(in_total, out_total) if max(in_total, out_total) > 0 else 0.0
            vat_washout_score = min(1.0, ratio)
            if vat_washout_score >= 0.92 and max(in_total, out_total) > 1_000_000_000:
                high_risk += 1
            node_scores[tc] = {
                "in_out_balance_ratio": round(ratio, 4),
                "balance_stability_k": round(abs(in_total - out_total) / max(1.0, in_total + out_total), 4),
                "vat_washout_score": round(vat_washout_score, 4),
                "layer_chain_depth": 2 if ratio > 0.9 else 1,
            }
        return {
            "node_scores": node_scores,
            "high_risk_count": high_risk,
            "logs": [
                {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": "medium" if high_risk > 0 else "low",
                    "title": "VAT Washout Detection",
                    "description": f"Phat hien {high_risk} node co in_out_balance_ratio cao bat thuong.",
                }
            ],
        }

    def _compute_pricing(self, invoices: list[dict], line_items: list[dict], company_map: dict[str, dict]) -> dict[str, Any]:
        if not line_items:
            return {
                "payload": {"available": False, "reason": "missing_invoice_line_items"},
                "edge_mismatch": {},
                "edge_price_deviation": {},
                "node_exposure": {},
                "mismatch_count": 0,
                "logs": [],
            }
        inv_by_id: dict[int, dict] = {}
        for inv in invoices:
            inv_id = inv.get("id")
            if inv_id is not None:
                inv_by_id[int(inv_id)] = inv
        baseline = defaultdict(list)
        for li in line_items:
            invoice = inv_by_id.get(int(li["invoice_id"]))
            if not invoice:
                continue
            seller_tc = str(invoice.get("seller_tax_code", ""))
            seller = company_map.get(seller_tc, {})
            industry = str(seller.get("industry") or "")
            date_obj = _to_date(invoice.get("date"))
            month_bucket = date_obj.strftime("%Y-%m") if date_obj else "unknown"
            key = (industry, str(invoice.get("goods_category") or ""), month_bucket, str(li.get("unit") or ""))
            baseline[key].append(_to_float(li.get("unit_price"), 0.0))

        edge_mismatch = defaultdict(float)
        edge_price = defaultdict(float)
        node_exp = defaultdict(list)
        mismatch_count = 0
        for li in line_items:
            invoice = inv_by_id.get(int(li["invoice_id"]))
            if not invoice:
                continue
            invoice_no = str(invoice.get("invoice_number", ""))
            seller_tc = str(invoice.get("seller_tax_code", ""))
            seller = company_map.get(seller_tc, {})
            industry = str(seller.get("industry") or "")
            goods = str(invoice.get("goods_category") or "")
            mismatch = 0.0
            if "Công nghệ thông tin" in industry and goods in {"Xỉ than", "Sắt thép xây dựng", "Thủy hải sản tươi sống"}:
                mismatch = 1.0
            elif goods and industry and goods not in industry:
                mismatch = 0.45
            edge_mismatch[invoice_no] = max(edge_mismatch[invoice_no], mismatch)

            date_obj = _to_date(invoice.get("date"))
            month_bucket = date_obj.strftime("%Y-%m") if date_obj else "unknown"
            key = (industry, goods, month_bucket, str(li.get("unit") or ""))
            values = baseline.get(key, [])
            unit_price = _to_float(li.get("unit_price"), 0.0)
            if len(values) >= 5:
                mu = mean(values)
                sigma = pstdev(values) or 1.0
                z = abs((unit_price - mu) / sigma)
                deviation = min(1.0, z / 4.0)
            else:
                deviation = 0.0
            edge_price[invoice_no] = max(edge_price[invoice_no], deviation)
            node_exp[seller_tc].append(max(mismatch, deviation))
            if mismatch >= 0.6:
                mismatch_count += 1

        node_exposure = {tc: (sum(vals) / max(1, len(vals))) for tc, vals in node_exp.items()}
        return {
            "payload": {
                "available": True,
                "mismatch_edge_count": mismatch_count,
                "top_price_outliers": sorted(
                    [{"invoice_number": k, "price_deviation_score": round(v, 4)} for k, v in edge_price.items()],
                    key=lambda x: x["price_deviation_score"],
                    reverse=True,
                )[:10],
            },
            "edge_mismatch": edge_mismatch,
            "edge_price_deviation": edge_price,
            "node_exposure": node_exposure,
            "mismatch_count": mismatch_count,
            "logs": [
                {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": "high" if mismatch_count > 0 else "low",
                    "title": "Industry-Goods + Pricing Anomaly",
                    "description": f"Phat hien {mismatch_count} giao dich mismatch nganh-hang va da tinh price deviation baseline.",
                }
            ],
        }

    def _compute_phoenix(self, companies: list[dict], invoices: list[dict], ownership_links: list[dict]) -> dict[str, Any]:
        if not companies:
            return {"payload": {"available": False, "reason": "no_company_context"}, "node_scores": {}, "succession_links": [], "logs": []}
        by_tax = {str(c.get("tax_code")): c for c in companies}
        buyers = defaultdict(set)
        sellers = defaultdict(set)
        for inv in invoices:
            s = str(inv.get("seller_tax_code", ""))
            b = str(inv.get("buyer_tax_code", ""))
            buyers[s].add(b)
            sellers[b].add(s)
        own_pairs = {(str(o.get("parent_tax_code", "")), str(o.get("child_tax_code", ""))) for o in ownership_links}
        node_scores = defaultdict(float)
        links = []
        company_items = list(by_tax.items())
        for old_tc, old in company_items:
            old_reg = _to_date(old.get("registration_date"))
            if not old_reg:
                continue
            for new_tc, new in company_items:
                if old_tc == new_tc:
                    continue
                new_reg = _to_date(new.get("registration_date"))
                if not new_reg or new_reg <= old_reg:
                    continue
                days = (new_reg - old_reg).days
                if days > 400:
                    continue
                overlap = len((buyers[old_tc] | sellers[old_tc]) & (buyers[new_tc] | sellers[new_tc]))
                denom = max(1, len(buyers[old_tc] | sellers[old_tc] | buyers[new_tc] | sellers[new_tc]))
                similarity = overlap / denom
                ownership_bridge = (old_tc, new_tc) in own_pairs or (new_tc, old_tc) in own_pairs
                score = min(1.0, similarity * 0.7 + (0.3 if ownership_bridge else 0.0))
                if score < 0.5:
                    continue
                node_scores[new_tc] = max(node_scores[new_tc], score)
                links.append(
                    {
                        "from": old_tc,
                        "to": new_tc,
                        "successor_similarity": round(similarity, 4),
                        "time_to_reincorporate_days": int(days),
                        "risk_inheritance_score": round(score, 4),
                    }
                )
        succession_paths = [
            {
                "path_id": f"PHX-{idx+1}",
                "summary": f"Chuoi ke thua rui ro {lnk['from']} -> {lnk['to']} (similarity={lnk['successor_similarity']:.2f}).",
                "risk_level": "high" if lnk["risk_inheritance_score"] >= 0.7 else "medium",
                "companies": [lnk["from"], lnk["to"]],
                "hops": [{"from": lnk["from"], "to": lnk["to"], "fraud_probability": lnk["risk_inheritance_score"], "date": str(lnk["time_to_reincorporate_days"])}],
            }
            for idx, lnk in enumerate(sorted(links, key=lambda x: x["risk_inheritance_score"], reverse=True)[:10])
        ]
        available = True if companies else False
        return {
            "payload": {
                "available": available,
                "successions": links[:15],
            },
            "node_scores": node_scores,
            "succession_links": succession_paths,
            "logs": [
                {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": "medium" if links else "low",
                    "title": "Phoenix Sequencing",
                    "description": f"Phat hien {len(links)} chuoi ke thua rui ro tiem nang.",
                }
            ],
        }

    def _compute_payment_consistency(
        self,
        invoices: list[dict],
        payments: list[dict],
        inv_id_map: dict[str, int],
    ) -> dict[str, Any]:
        if not payments:
            return {
                "edge_match_score": {},
                "payload": {"available": False, "reason": "missing_invoice_payments"},
                "mismatch_count": 0,
                "logs": [],
                "mismatch_paths": [],
            }
        payments_by_id = defaultdict(list)
        for p in payments:
            payments_by_id[int(p["invoice_id"])].append(p)
        edge_match_score = {}
        mismatch_count = 0
        mismatch_paths = []
        for inv in invoices:
            inv_no = str(inv.get("invoice_number", ""))
            inv_id = inv_id_map.get(inv_no)
            if not inv_id:
                edge_match_score[inv_no] = 0.0
                continue
            linked = payments_by_id.get(inv_id, [])
            if not linked:
                edge_match_score[inv_no] = 0.0
                mismatch_count += 1
                continue
            invoice_amount = _to_float(inv.get("amount"), 0.0)
            paid_amount = sum(_to_float(p.get("paid_amount"), 0.0) for p in linked)
            ratio = min(1.0, paid_amount / max(1.0, invoice_amount))
            third_party = any(
                str(p.get("payer_tax_code", "")) != str(inv.get("buyer_tax_code", ""))
                or str(p.get("payee_tax_code", "")) != str(inv.get("seller_tax_code", ""))
                for p in linked
            )
            score = max(0.0, ratio - (0.35 if third_party else 0.0))
            edge_match_score[inv_no] = min(1.0, score)
            if score < 0.55:
                mismatch_count += 1
                mismatch_paths.append(
                    {
                        "path_id": f"PAY-{len(mismatch_paths)+1}",
                        "summary": f"Payment mismatch tai hoa don {inv_no}: match_score={score:.2f}.",
                        "risk_level": "high" if score < 0.3 else "medium",
                        "companies": [str(inv.get("seller_tax_code", "")), str(inv.get("buyer_tax_code", ""))],
                        "hops": [],
                    }
                )
        match_rate = (
            sum(1 for score in edge_match_score.values() if score >= 0.8) / max(1, len(edge_match_score))
        )
        return {
            "edge_match_score": edge_match_score,
            "payload": {
                "available": True,
                "invoice_payment_match_rate": round(match_rate, 4),
                "payment_lag_anomaly": round(1.0 - match_rate, 4),
                "third_party_settlement_flag": mismatch_count > 0,
                "mismatch_incidents": mismatch_paths[:10],
            },
            "mismatch_count": mismatch_count,
            "logs": [
                {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": "high" if mismatch_count > 0 else "low",
                    "title": "Invoice-Payment Consistency",
                    "description": f"invoice_payment_match_rate={match_rate:.1%}, mismatch={mismatch_count}.",
                }
            ],
            "mismatch_paths": mismatch_paths[:10],
        }
