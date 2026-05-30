# -*- coding: utf-8 -*-
"""Rule-first intelligence helpers for the taxpayer portal.

The service is intentionally lightweight. It produces useful ML-style scores,
forecasts, anomaly flags, and recommendations from internal taxpayer data, then
leaves room for model_registry/model_serving to replace each heuristic later.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from hashlib import sha256
import json
import math
import re
from typing import Any
from xml.etree import ElementTree

from .taxpayer_rules import (
    BASELINE_SOURCES,
    CASH_PAYMENT_LIMIT,
    PASSPORT_BAN_DAY_THRESHOLD,
    PASSPORT_BAN_DEBT_THRESHOLD,
    VND_1B,
    VND_3B,
    VND_500M,
    build_deadlines,
    calculate_tax_by_industry,
    classify_household_group,
    debt_days_overdue,
    e_invoice_requirement,
    evaluate_expense,
    passport_ban_risk,
    revenue_threshold_summary,
)


MODEL_NAME = "taxpayer_intelligence_baseline"
MODEL_VERSION = "quickwins-2026.05"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    if isinstance(value, dict):
        return value
    return {}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 2) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)


class TaxpayerIntelligenceService:
    """Advisory intelligence that can be replaced by trained models per method."""

    model_name = MODEL_NAME
    model_version = MODEL_VERSION

    def input_hash(self, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return sha256(encoded).hexdigest()

    def model_meta(self, payload: dict[str, Any], confidence: str = "medium", score: float | None = None) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "confidence": confidence,
            "confidence_score": score if score is not None else {"low": 0.35, "medium": 0.68, "high": 0.86}.get(confidence, 0.5),
            "input_hash": self.input_hash(payload),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def build_snapshot(self, dataset: dict[str, Any]) -> dict[str, Any]:
        today = _date(dataset.get("today")) or date.today()
        year = _int(dataset.get("year"), today.year)
        profile = dataset.get("profile") or {}
        annual_plan = _float(profile.get("annual_revenue"), 0.0)
        industry = str(profile.get("industry") or "commerce")
        household_group = _int(profile.get("household_group"), classify_household_group(annual_plan)["group"])

        revenue_entries = dataset.get("revenue_entries") or []
        expense_entries = dataset.get("expense_entries") or []
        invoices = dataset.get("invoices") or []
        filings = dataset.get("filings") or []
        payments = dataset.get("payments") or []
        debts = dataset.get("debts") or []
        documents = dataset.get("documents") or []
        claims = dataset.get("claims") or []
        deadlines = dataset.get("deadlines") or build_deadlines(year, household_group, today=today)

        monthly: dict[int, dict[str, float]] = {idx: {"revenue": 0.0, "expense": 0.0, "profit": 0.0} for idx in range(1, 13)}
        channel_revenue: dict[str, float] = defaultdict(float)
        revenue_total = 0.0
        for item in revenue_entries:
            amount = _float(item.get("amount"))
            entry_date = _date(item.get("entry_date") or item.get("date"))
            if entry_date and entry_date.year == year:
                revenue_total += amount
                monthly[entry_date.month]["revenue"] += amount
                channel_revenue[str(item.get("channel") or "direct")] += amount

        expense_total = 0.0
        deductible_expense = 0.0
        evidence_gap_count = 0
        cash_payment_violations = 0
        expense_by_category: dict[str, float] = defaultdict(float)
        for item in expense_entries:
            amount = _float(item.get("amount"))
            expense_date = _date(item.get("expense_date") or item.get("date"))
            if expense_date and expense_date.year == year:
                expense_total += amount
                monthly[expense_date.month]["expense"] += amount
                category = str(item.get("category") or "other")
                expense_by_category[category] += amount
                status = str(item.get("deductible_status") or "").lower()
                if status in {"deductible", "needs_evidence"}:
                    deductible_expense += amount
                if status in {"needs_evidence", "needs_invoice"}:
                    evidence_gap_count += 1
                if _float(item.get("amount")) >= CASH_PAYMENT_LIMIT and str(item.get("payment_method") or "").lower() in {"cash", "tien_mat"}:
                    cash_payment_violations += 1

        for month in monthly.values():
            month["profit"] = month["revenue"] - month["expense"]

        profit = revenue_total - expense_total
        profit_margin = profit / revenue_total if revenue_total else 0.0
        expense_ratio = expense_total / revenue_total if revenue_total else 0.0
        active_months = len([item for item in monthly.values() if item["revenue"] > 0 or item["expense"] > 0])
        avg_monthly_revenue = revenue_total / max(1, active_months)
        remaining_months = max(0, 12 - today.month)
        projected_year_end = max(annual_plan, revenue_total + avg_monthly_revenue * remaining_months)

        deadline_overdue = len([item for item in deadlines if item.get("status") == "overdue"])
        deadline_soon = len([item for item in deadlines if item.get("status") in {"soon", "due_today"}])
        debt_total = 0.0
        max_days_overdue = 0
        for item in debts:
            outstanding = max(0.0, _float(item.get("amount_due")) - _float(item.get("amount_paid")))
            debt_total += outstanding
            due_date = item.get("due_date")
            if due_date:
                max_days_overdue = max(max_days_overdue, debt_days_overdue(due_date, today=today))

        payment_pending = sum(max(0.0, _float(item.get("amount_due")) - _float(item.get("amount_paid"))) for item in payments if str(item.get("status") or "").lower() != "paid")
        risky_invoice_count = 0
        invoice_amount_total = 0.0
        invoice_numbers: dict[str, int] = defaultdict(int)
        for item in invoices:
            invoice_amount_total += _float(item.get("total_amount") or item.get("amount"))
            number = str(item.get("invoice_number") or "").strip()
            if number:
                invoice_numbers[number] += 1
            risk = _json(item.get("risk_json"))
            status = str(item.get("status") or "").lower()
            if status in {"invalid", "risky", "cancelled", "missing"} or risk.get("risk_flags"):
                risky_invoice_count += 1
        duplicate_invoice_count = len([num for num, count in invoice_numbers.items() if count > 1])

        data_fields = [
            profile.get("tax_code"),
            profile.get("full_name"),
            profile.get("business_name"),
            profile.get("industry"),
            profile.get("annual_revenue"),
            profile.get("email"),
            profile.get("phone"),
            profile.get("address"),
        ]
        data_quality_score = 100.0 * sum(1 for value in data_fields if value not in (None, "", 0)) / len(data_fields)
        if documents:
            data_quality_score = min(100.0, data_quality_score + 6.0)

        threshold = revenue_threshold_summary(revenue_total, annual_plan)
        snapshot = {
            "year": year,
            "profile": {
                "tax_code": profile.get("tax_code"),
                "business_name": profile.get("business_name") or profile.get("full_name"),
                "industry": industry,
                "household_group": household_group,
                "annual_revenue_plan": annual_plan,
            },
            "revenue": {
                "total": _round(revenue_total),
                "annual_plan": _round(annual_plan),
                "projected_year_end": _round(projected_year_end),
                "avg_monthly": _round(avg_monthly_revenue),
                "channels": dict(channel_revenue),
                "threshold": threshold,
            },
            "expenses": {
                "total": _round(expense_total),
                "deductible_total": _round(deductible_expense),
                "ratio": _round(expense_ratio, 4),
                "by_category": dict(expense_by_category),
                "evidence_gap_count": evidence_gap_count,
                "cash_payment_violations": cash_payment_violations,
            },
            "profit": {
                "amount": _round(profit),
                "margin": _round(profit_margin, 4),
            },
            "compliance": {
                "deadline_overdue": deadline_overdue,
                "deadline_soon": deadline_soon,
                "filing_count": len(filings),
                "pending_payment": _round(payment_pending),
                "debt_total": _round(debt_total),
                "max_days_overdue": max_days_overdue,
                "passport_ban": passport_ban_risk(debt_total, max_days_overdue),
            },
            "invoices": {
                "count": len(invoices),
                "amount_total": _round(invoice_amount_total),
                "risky_count": risky_invoice_count,
                "duplicate_count": duplicate_invoice_count,
            },
            "documents": {
                "count": len(documents),
                "claims_count": len(claims),
            },
            "monthly": [{"month": month, **values} for month, values in monthly.items()],
            "data_quality_score": _round(data_quality_score),
            "sample_size": {
                "revenue_entries": len(revenue_entries),
                "expense_entries": len(expense_entries),
                "invoices": len(invoices),
                "documents": len(documents),
            },
        }
        snapshot["input_hash"] = self.input_hash(snapshot)
        return snapshot

    def overview(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        revenue = snapshot["revenue"]["total"]
        profit_margin = snapshot["profit"]["margin"]
        debt_total = snapshot["compliance"]["debt_total"]
        avg_monthly_revenue = snapshot["revenue"]["avg_monthly"]
        deadline_overdue = snapshot["compliance"]["deadline_overdue"]
        risky_invoice_count = snapshot["invoices"]["risky_count"]
        evidence_gap_count = snapshot["expenses"]["evidence_gap_count"]

        financial_score = 58.0 + profit_margin * 120.0 - min(28.0, debt_total / max(revenue, 1.0) * 60.0)
        if revenue == 0 and snapshot["profile"]["annual_revenue_plan"] > 0:
            financial_score = 55.0
        compliance_score = 92.0 - deadline_overdue * 18.0 - snapshot["compliance"]["deadline_soon"] * 5.0 - risky_invoice_count * 7.0 - evidence_gap_count * 4.0
        cashflow_score = 62.0 + profit_margin * 90.0 - min(35.0, debt_total / max(avg_monthly_revenue, 1.0) * 18.0)
        data_quality_score = snapshot["data_quality_score"]

        scores = {
            "financial_health": _round(_clamp(financial_score)),
            "compliance": _round(_clamp(compliance_score)),
            "cashflow": _round(_clamp(cashflow_score)),
            "data_quality": _round(_clamp(data_quality_score)),
        }
        alerts = self._alerts(snapshot, scores)
        recommendations = self.recommendations(dataset, snapshot=snapshot)["recommendations"]
        confidence = "medium" if sum(snapshot["sample_size"].values()) >= 5 else "low"
        result = {
            "scores": scores,
            "alerts": alerts,
            "top_recommendations": recommendations[:5],
            "snapshot": snapshot,
            "model": self.model_meta(snapshot, confidence=confidence),
            "disclaimer": "AI insights chi mang tinh ho tro, can doi chieu quy dinh va chung tu thuc te.",
        }
        return result

    def forecast(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        today = _date(dataset.get("today")) or date.today()
        monthly = snapshot["monthly"]
        current_month = today.month
        historical = [item["revenue"] for item in monthly[:current_month] if item["revenue"] > 0]
        if historical:
            last3 = historical[-3:]
            prev3 = historical[-6:-3]
            base = sum(last3) / len(last3)
            prev = sum(prev3) / len(prev3) if prev3 else base
        else:
            base = snapshot["profile"]["annual_revenue_plan"] / 12.0 if snapshot["profile"]["annual_revenue_plan"] else 0.0
            prev = base
        trend = 0.0 if prev <= 0 else _clamp((base - prev) / prev, -0.35, 0.35)
        months = []
        for idx in range(1, 7):
            month_no = ((current_month + idx - 1) % 12) + 1
            year = today.year + ((current_month + idx - 1) // 12)
            revenue = max(0.0, base * (1.0 + trend * idx / 3.0))
            expense_ratio = snapshot["expenses"]["ratio"] if snapshot["expenses"]["ratio"] > 0 else 0.72
            expense = revenue * expense_ratio
            months.append(
                {
                    "period": f"{year}-{month_no:02d}",
                    "revenue": _round(revenue),
                    "expense": _round(expense),
                    "profit": _round(revenue - expense),
                }
            )
        projected_year_end = snapshot["revenue"]["total"] + sum(item["revenue"] for item in months[: max(0, 12 - current_month)])
        projected_year_end = max(projected_year_end, snapshot["revenue"]["projected_year_end"])

        def probability(threshold: float) -> float:
            if snapshot["revenue"]["total"] >= threshold:
                return 1.0
            band = max(threshold * 0.18, 1.0)
            return _round(_clamp(0.5 + (projected_year_end - threshold) / band, 0.0, 1.0), 4)

        confidence = "medium" if len(historical) >= 3 else "low"
        result = {
            "forecast_months": months,
            "projected_year_end_revenue": _round(projected_year_end),
            "threshold_probabilities": {
                "taxable_500m": probability(VND_500M),
                "einvoice_1b": probability(VND_1B),
                "group3_3b": probability(VND_3B),
            },
            "cashflow_30_60_90": {
                "days_30": _round((months[0]["profit"] if months else 0.0) - snapshot["compliance"]["pending_payment"]),
                "days_60": _round(sum(item["profit"] for item in months[:2]) - snapshot["compliance"]["pending_payment"]),
                "days_90": _round(sum(item["profit"] for item in months[:3]) - snapshot["compliance"]["pending_payment"]),
            },
            "trend_pct": _round(trend * 100.0, 2),
            "model": self.model_meta({"snapshot": snapshot, "months": months}, confidence=confidence),
        }
        return result

    def what_if(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        profile = (dataset.get("profile") or {})
        revenue = max(0.0, _float(payload.get("revenue"), _float(profile.get("annual_revenue"))))
        expenses = max(0.0, _float(payload.get("expenses"), revenue * 0.7))
        industry = str(payload.get("industry") or profile.get("industry") or "commerce")
        months_active = max(1, min(12, _int(payload.get("months_active"), 12)))
        annualized_revenue = revenue * 12.0 / months_active
        taxes = calculate_tax_by_industry(revenue, industry)
        profit_before_tax = revenue - expenses
        profit_after_tax = profit_before_tax - taxes["total_tax"]
        scenarios = []
        for pct in (-0.15, 0.0, 0.15):
            scenario_revenue = revenue * (1.0 + pct)
            scenario_tax = calculate_tax_by_industry(scenario_revenue, industry)
            scenarios.append(
                {
                    "change_pct": int(pct * 100),
                    "revenue": _round(scenario_revenue),
                    "tax": scenario_tax["total_tax"],
                    "profit_after_tax": _round(scenario_revenue - expenses - scenario_tax["total_tax"]),
                }
            )
        result = {
            "input": {"revenue": revenue, "expenses": expenses, "industry": industry, "months_active": months_active},
            "annualized_revenue": _round(annualized_revenue),
            "household_group": classify_household_group(annualized_revenue),
            "einvoice": e_invoice_requirement(annualized_revenue),
            "taxes": taxes,
            "profit_before_tax": _round(profit_before_tax),
            "profit_after_tax": _round(profit_after_tax),
            "profit_margin_after_tax": _round(profit_after_tax / revenue if revenue else 0.0, 4),
            "scenarios": scenarios,
            "model": self.model_meta(payload, confidence="medium"),
        }
        return result

    def classify_expense(self, payload: dict[str, Any]) -> dict[str, Any]:
        evaluation = evaluate_expense(payload)
        amount = _float(payload.get("amount"))
        status = evaluation["status"]
        risk_score = 15.0
        if status == "non_deductible":
            risk_score = 88.0
        elif status in {"needs_invoice", "needs_evidence"}:
            risk_score = 58.0
        if amount >= CASH_PAYMENT_LIMIT and str(payload.get("payment_method") or "").lower() in {"cash", "tien_mat"}:
            risk_score = max(risk_score, 92.0)
        if "luong chu" in str(payload.get("description") or "").lower():
            risk_score = max(risk_score, 90.0)
        result = {
            "label": status,
            "risk_score": _round(risk_score),
            "evaluation": evaluation,
            "recommended_action": self._expense_action(evaluation),
            "model": self.model_meta(payload, confidence="medium", score=_round(1.0 - risk_score / 100.0, 4)),
        }
        return result

    def invoice_risk(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        invoices = dataset.get("invoices") or []
        seller = str(payload.get("seller_tax_code") or payload.get("tax_code") or "").strip()
        buyer = str(payload.get("buyer_tax_code") or "").strip()
        number = str(payload.get("invoice_number") or "").strip()
        amount = _float(payload.get("amount") or payload.get("total_amount"))
        flags: list[str] = []
        score = 12.0
        if not seller or len(re.sub(r"\D", "", seller)) < 10:
            flags.append("missing_or_invalid_seller_tax_code")
            score += 28
        if buyer and seller and buyer == seller:
            flags.append("seller_buyer_same_tax_code")
            score += 18
        if not number:
            flags.append("missing_invoice_number")
            score += 14
        if amount <= 0:
            flags.append("missing_amount")
            score += 12
        elif amount >= 200_000_000:
            flags.append("large_invoice_amount")
            score += 12
        duplicate = any(str(item.get("invoice_number") or "").strip() == number for item in invoices if number)
        if duplicate:
            flags.append("duplicate_invoice_number_in_taxpayer_log")
            score += 30
        if seller.startswith(("000", "999")):
            flags.append("suspicious_tax_code_pattern")
            score += 18
        score = _clamp(score, 0.0, 100.0)
        if score >= 75:
            level = "high"
            action = "Tam dung ghi nhan chi phi, yeu cau doi tac xac minh hoa don va MST."
        elif score >= 45:
            level = "medium"
            action = "Can bo sung chung tu, doi chieu trang thai hoa don truoc khi ke khai."
        else:
            level = "low"
            action = "Co the ghi nhan tam thoi, tiep tuc luu bang chung thanh toan."
        return {
            "risk_level": level,
            "risk_score": _round(score),
            "risk_flags": flags,
            "recommended_action": action,
            "model": self.model_meta({"payload": payload, "invoice_count": len(invoices)}, confidence="medium"),
        }

    def extract_document(self, content: bytes | None, filename: str | None, doc_type: str = "evidence") -> dict[str, Any]:
        text = ""
        if content:
            try:
                text = content[:100_000].decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        fields: dict[str, Any] = {}
        confidence = "low"
        if text.strip().startswith("<"):
            try:
                root = ElementTree.fromstring(text.encode("utf-8"))
                for elem in root.iter():
                    local = elem.tag.split("}")[-1].lower()
                    if local in {"mst", "nbmst", "nmmst", "seller tax code", "sellertaxcode"} and elem.text:
                        fields.setdefault("tax_code", elem.text.strip())
                    if local in {"shdon", "invoice number", "invoicenumber", "sohoadon"} and elem.text:
                        fields.setdefault("invoice_number", elem.text.strip())
                    if local in {"tgtcthue", "totalamount", "amount"} and elem.text:
                        fields.setdefault("amount", _float(elem.text))
                confidence = "medium" if fields else "low"
            except Exception:
                pass
        if not fields and text:
            tax_code = re.search(r"\b\d{10}(?:\d{3})?\b", text)
            amount = re.search(r"(\d[\d\.,]{5,})", text)
            if tax_code:
                fields["tax_code"] = tax_code.group(0)
            if amount:
                fields["amount_text"] = amount.group(1)
            confidence = "low" if fields else "low"
        result = {
            "doc_type": doc_type,
            "filename": filename,
            "extracted_fields": fields,
            "quality_flags": [] if fields else ["no_structured_fields_detected"],
            "suggested_category": self._suggest_document_category(filename or "", text, doc_type),
            "model": self.model_meta({"filename": filename, "doc_type": doc_type, "fields": fields}, confidence=confidence),
        }
        return result

    def peer_benchmark(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        industry = snapshot["profile"]["industry"]
        references = {
            "commerce": {"margin_low": 0.08, "margin_high": 0.24, "expense_ratio_high": 0.92},
            "service": {"margin_low": 0.18, "margin_high": 0.45, "expense_ratio_high": 0.78},
            "manufacture": {"margin_low": 0.10, "margin_high": 0.30, "expense_ratio_high": 0.88},
            "rental": {"margin_low": 0.35, "margin_high": 0.70, "expense_ratio_high": 0.62},
            "other": {"margin_low": 0.10, "margin_high": 0.35, "expense_ratio_high": 0.86},
        }
        ref = references.get(industry, references["other"])
        margin = snapshot["profit"]["margin"]
        expense_ratio = snapshot["expenses"]["ratio"]
        if margin < ref["margin_low"]:
            margin_position = "below_peer_range"
        elif margin > ref["margin_high"]:
            margin_position = "above_peer_range"
        else:
            margin_position = "within_peer_range"
        result = {
            "industry": industry,
            "taxpayer_metrics": {
                "profit_margin": margin,
                "expense_ratio": expense_ratio,
                "revenue": snapshot["revenue"]["total"],
            },
            "peer_range": ref,
            "signals": {
                "margin_position": margin_position,
                "expense_ratio_flag": expense_ratio > ref["expense_ratio_high"],
                "evidence_gap_count": snapshot["expenses"]["evidence_gap_count"],
            },
            "model": self.model_meta(snapshot, confidence="low" if snapshot["revenue"]["total"] == 0 else "medium"),
        }
        return result

    def chart_analytics(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        monthly = snapshot["monthly"]
        cumulative = 0.0
        monthly_series = []
        for item in monthly:
            cumulative += _float(item.get("revenue"))
            monthly_series.append(
                {
                    "period": f"{snapshot['year']}-{int(item['month']):02d}",
                    "revenue": _round(item.get("revenue")),
                    "expense": _round(item.get("expense")),
                    "profit": _round(item.get("profit")),
                    "cumulative_revenue": _round(cumulative),
                }
            )
        channel_total = max(1.0, sum(snapshot["revenue"]["channels"].values()))
        category_total = max(1.0, sum(snapshot["expenses"]["by_category"].values()))
        channel_breakdown = [
            {"label": key, "value": _round(value), "share": _round(value / channel_total, 4)}
            for key, value in sorted(snapshot["revenue"]["channels"].items(), key=lambda item: item[1], reverse=True)
        ]
        expense_breakdown = [
            {"label": key, "value": _round(value), "share": _round(value / category_total, 4)}
            for key, value in sorted(snapshot["expenses"]["by_category"].items(), key=lambda item: item[1], reverse=True)
        ]
        kpi_cards = [
            {
                "key": "revenue",
                "label": "Doanh thu luy ke",
                "value": snapshot["revenue"]["total"],
                "trend": self._series_trend([item["revenue"] for item in monthly_series]),
            },
            {
                "key": "profit_margin",
                "label": "Bien loi nhuan",
                "value": snapshot["profit"]["margin"],
                "trend": self._series_trend([item["profit"] for item in monthly_series]),
            },
            {
                "key": "expense_ratio",
                "label": "Ty le chi phi",
                "value": snapshot["expenses"]["ratio"],
                "trend": self._series_trend([item["expense"] for item in monthly_series]),
            },
            {
                "key": "compliance",
                "label": "Canh bao tuan thu",
                "value": snapshot["compliance"]["deadline_soon"] + snapshot["compliance"]["deadline_overdue"],
                "trend": "watch",
            },
        ]
        return {
            "kpi_cards": kpi_cards,
            "monthly_series": monthly_series,
            "channel_breakdown": channel_breakdown,
            "expense_breakdown": expense_breakdown,
            "threshold_markers": [
                {"label": "500M", "value": VND_500M, "meaning": "Bat dau phat sinh nghia vu thue/ke khai theo nguong moi"},
                {"label": "1B", "value": VND_1B, "meaning": "Can chu y hoa don dien tu"},
                {"label": "3B", "value": VND_3B, "meaning": "Nhom 3, so sach day du hon"},
            ],
            "model": self.model_meta(snapshot, confidence="medium" if snapshot["sample_size"]["revenue_entries"] else "low"),
        }

    def anomaly_insights(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        anomalies: list[dict[str, Any]] = []
        revenue_values = [item["revenue"] for item in snapshot["monthly"]]
        expense_values = [item["expense"] for item in snapshot["monthly"]]
        anomalies.extend(self._spike_anomalies("revenue", revenue_values, snapshot["year"]))
        anomalies.extend(self._spike_anomalies("expense", expense_values, snapshot["year"]))
        if snapshot["expenses"]["cash_payment_violations"]:
            anomalies.append(
                {
                    "type": "cash_payment_violation",
                    "severity": "high",
                    "title": "Chi phi tien mat tu 5 trieu",
                    "description": "Co giao dich tien mat vuot nguong, de bi loai khoi chi phi duoc tru.",
                    "recommended_action": "Doi chieu chung tu va uu tien thanh toan chuyen khoan cho giao dich lon.",
                }
            )
        if snapshot["invoices"]["duplicate_count"]:
            anomalies.append(
                {
                    "type": "duplicate_invoice",
                    "severity": "high",
                    "title": "Hoa don trung so",
                    "description": "Nhat ky co so hoa don lap lai, can xac minh truoc khi ke khai.",
                    "recommended_action": "Mo trang hoa don de ra soat va danh dau hoa don hop le.",
                }
            )
        if snapshot["profit"]["margin"] < -0.05 and snapshot["revenue"]["total"] > 0:
            anomalies.append(
                {
                    "type": "negative_margin",
                    "severity": "medium",
                    "title": "Bien loi nhuan am",
                    "description": "Chi phi dang vuot doanh thu, can ra soat phan loai chi phi va dong tien.",
                    "recommended_action": "Dung benchmark va bao cao so sach de tim nhom chi phi bat thuong.",
                }
            )
        if not anomalies:
            anomalies.append(
                {
                    "type": "normal",
                    "severity": "low",
                    "title": "Chua phat hien bat thuong lon",
                    "description": "Du lieu hien tai khong co spike ro rang. Tiep tuc ghi nhan doanh thu/chi phi hang ngay.",
                    "recommended_action": "Duy tri cap nhat so lieu va tai chung tu theo thang.",
                }
            )
        return {
            "anomalies": anomalies,
            "summary": {
                "count": len(anomalies),
                "high": len([item for item in anomalies if item["severity"] == "high"]),
                "medium": len([item for item in anomalies if item["severity"] == "medium"]),
            },
            "model": self.model_meta(snapshot, confidence="medium" if snapshot["sample_size"]["revenue_entries"] >= 3 else "low"),
        }

    def optimize_tax(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        profile = snapshot["profile"]
        revenue = _float(payload.get("revenue"), snapshot["revenue"]["projected_year_end"] or profile["annual_revenue_plan"])
        expenses = _float(payload.get("expenses"), snapshot["expenses"]["total"])
        industry = str(payload.get("industry") or profile["industry"] or "commerce")
        method_revenue = calculate_tax_by_industry(revenue, industry)
        profit = max(0.0, revenue - expenses)
        progressive_like_tax = self._simple_progressive_profit_tax(profit)
        saving = method_revenue["total_tax"] - progressive_like_tax["tax"]
        if saving > 0:
            recommendation = "Phuong phap tinh tren loi nhuan co the co loi hon neu co du chung tu chi phi."
            preferred = "profit_based"
        else:
            recommendation = "Phuong phap ty le tren doanh thu dang don gian va co the thap hon voi bien loi nhuan hien tai."
            preferred = "revenue_percentage"
        checklist = [
            "Doi chieu doanh thu san TMDT va doanh thu tu khai de tranh trung.",
            "Bo sung hoa don/bang ke cho cac khoan chi lon.",
            "Kiem tra giao dich tu 5 trieu da thanh toan chuyen khoan.",
            "Cap nhat nhom HKD neu du bao vuot nguong 500M/1B/3B.",
        ]
        if snapshot["expenses"]["evidence_gap_count"] > 0:
            checklist.insert(0, f"Xu ly {snapshot['expenses']['evidence_gap_count']} khoan chi thieu chung tu.")
        return {
            "input": {"revenue": revenue, "expenses": expenses, "industry": industry},
            "methods": [
                {
                    "key": "revenue_percentage",
                    "label": "Ty le tren doanh thu",
                    "estimated_tax": method_revenue["total_tax"],
                    "details": method_revenue,
                },
                {
                    "key": "profit_based",
                    "label": "Loi nhuan sau chi phi",
                    "estimated_tax": progressive_like_tax["tax"],
                    "details": progressive_like_tax,
                },
            ],
            "preferred_method": preferred,
            "estimated_saving": _round(abs(saving)),
            "recommendation": recommendation,
            "checklist": checklist,
            "model": self.model_meta({"payload": payload, "snapshot": snapshot}, confidence="low"),
        }

    def claim_assist(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        description = str(payload.get("description") or payload.get("appeal_reason") or "").strip()
        decision_no = str(payload.get("decision_no") or "").strip()
        evidence_items = payload.get("evidence_items") or []
        if isinstance(evidence_items, str):
            evidence_items = [item.strip() for item in evidence_items.split(",") if item.strip()]
        score = 35.0
        gaps = []
        if decision_no:
            score += 18
        else:
            gaps.append("Can bo sung so quyet dinh/bien ban bi khieu nai.")
        if len(description) >= 80:
            score += 20
        else:
            gaps.append("Mo ta ly do khieu nai con ngan, nen neu ro so tien/ky thue/can cu.")
        if len(evidence_items) >= 2:
            score += 18
        else:
            gaps.append("Can toi thieu 2 nhom bang chung: to khai, bien lai, hoa don, hop dong, sao ke.")
        if any(token in description.lower() for token in ["qua han", "90 ngay", "thoi han"]):
            score += 7
        readiness = "ready_to_submit" if score >= 75 else "needs_more_evidence" if score >= 50 else "weak_file"
        outline = [
            "Thong tin nguoi nop thue va MST/CCCD.",
            "Quyet dinh hoac bien ban dang khieu nai.",
            "Noi dung khieu nai: phan khong dong y, so tien/ky thue, ly do.",
            "Can cu phap ly va chung tu kem theo.",
            "Yeu cau xu ly: dieu chinh, huy, hoan/bu tru hoac tam dung tinh cham nop phan tranh chap.",
        ]
        return {
            "readiness": readiness,
            "readiness_score": _round(_clamp(score)),
            "evidence_gaps": gaps,
            "draft_outline": outline,
            "strategy": "Nen nop phan khong tranh chap de giam tien cham nop, dong thoi khieu nai phan co can cu.",
            "model": self.model_meta({"payload": payload, "dataset_size": len(dataset.get("claims") or [])}, confidence="low"),
        }

    def model_catalog(self) -> dict[str, Any]:
        items = [
            {
                "key": "document_ocr",
                "name": "OCR chung tu/hoa don",
                "taxinspector_module": "document_ocr_engine.py",
                "taxpayer_use": "Tu dong doc hoa don, hop dong, bang ke va gan vao so chi phi.",
                "status": "baseline_wired",
            },
            {
                "key": "invoice_risk",
                "name": "Invoice risk model",
                "taxinspector_module": "invoice_risk_model.py",
                "taxpayer_use": "Phat hien hoa don trung, MST bat thuong, nha cung cap rui ro.",
                "status": "baseline_wired",
            },
            {
                "key": "delinquency",
                "name": "Delinquency / no thue",
                "taxinspector_module": "delinquency_model.py",
                "taxpayer_use": "Du bao nguy co cham nop, de xuat phan ky va canh bao cuong che.",
                "status": "adapter_ready",
            },
            {
                "key": "forecast",
                "name": "Revenue forecast",
                "taxinspector_module": "revenue_forecast_model.py",
                "taxpayer_use": "Du bao doanh thu, dong tien, thoi diem vuot nguong 500M/1B/3B.",
                "status": "baseline_wired",
            },
            {
                "key": "anomaly",
                "name": "VAE/Isolation anomaly",
                "taxinspector_module": "vae_anomaly.py",
                "taxpayer_use": "Phat hien spike doanh thu/chi phi, bien loi nhuan bat thuong.",
                "status": "baseline_wired",
            },
            {
                "key": "graph_risk",
                "name": "Graph/GNN supplier risk",
                "taxinspector_module": "graph_intelligence.py, gnn_model.py",
                "taxpayer_use": "Danh gia mang luoi nha cung cap/doi tac co dau hieu rui ro.",
                "status": "adapter_ready",
            },
            {
                "key": "legal_rag",
                "name": "Legal GraphRAG",
                "taxinspector_module": "tax_agent_graphrag.py, tax_agent_legal_intelligence.py",
                "taxpayer_use": "Tra loi cau hoi phap ly co citation va canh bao van ban moi.",
                "status": "baseline_wired",
            },
            {
                "key": "uplift_recommendation",
                "name": "Next-best-action / uplift",
                "taxinspector_module": "causal_uplift_model.py",
                "taxpayer_use": "Chon hanh dong co tac dong cao nhat: bo sung chung tu, nop to khai, xu ly no.",
                "status": "roadmap",
            },
        ]
        return {
            "items": items,
            "default_strategy": "rule_first_then_model_adapter",
            "governance": [
                "Moi model can co confidence, explanation, model_version va feedback loop.",
                "Khong dung du lieu can bo/officer de hien thi cho taxpayer neu chua duoc phep.",
                "Ket qua AI la tu van ho tro, khong phai ket luan phap ly.",
            ],
            "model": self.model_meta({"catalog_size": len(items)}, confidence="high"),
        }

    def scenario_dashboard(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        forecast = self.forecast(dataset)
        benchmark = self.peer_benchmark(dataset)
        recommendations = self.recommendations(dataset, snapshot=snapshot)["recommendations"]
        projected = _float(forecast.get("projected_year_end_revenue"))
        group = classify_household_group(max(projected, snapshot["revenue"]["total"], snapshot["profile"]["annual_revenue_plan"]))
        channels = snapshot["revenue"]["channels"]
        segments = []
        if channels.get("ecommerce", 0) > 0 or channels.get("marketplace", 0) > 0:
            segments.append("hkd_tmdt")
        if channels.get("direct", 0) > 0:
            segments.append("storefront")
        if projected >= VND_3B:
            segments.append("enterprise_transition_candidate")
        elif projected >= VND_1B:
            segments.append("einvoice_maturity")
        elif projected >= VND_500M:
            segments.append("growth_taxpayer")
        else:
            segments.append("small_taxpayer")

        risk_heatmap = [
            self._heat("deadline", "Han ke khai", snapshot["compliance"]["deadline_overdue"] * 40 + snapshot["compliance"]["deadline_soon"] * 18),
            self._heat("cashflow", "Dong tien nop thue", 100 - _float(self.overview(dataset)["scores"]["cashflow"])),
            self._heat("invoice", "Hoa don/doi tac", snapshot["invoices"]["risky_count"] * 35 + snapshot["invoices"]["duplicate_count"] * 45),
            self._heat("evidence", "Chung tu chi phi", snapshot["expenses"]["evidence_gap_count"] * 18 + snapshot["expenses"]["cash_payment_violations"] * 40),
            self._heat("debt", "No thue/cuong che", snapshot["compliance"]["debt_total"] / max(1.0, PASSPORT_BAN_DEBT_THRESHOLD) * 100),
        ]
        strategy_cards = [
            {
                "key": "operate",
                "title": "Van hanh tuan thu",
                "status": "needs_attention" if any(item["severity"] == "high" for item in risk_heatmap) else "stable",
                "metric": len([item for item in risk_heatmap if item["severity"] != "low"]),
                "action": "Xu ly deadline, hoa don va chung tu truoc ky nop thue tiep theo.",
            },
            {
                "key": "cash",
                "title": "Dong tien va thue",
                "status": "watch" if forecast["cashflow_30_60_90"]["days_90"] < 0 else "healthy",
                "metric": forecast["cashflow_30_60_90"]["days_90"],
                "action": "Duy tri quy du phong thue theo doanh thu du bao.",
            },
            {
                "key": "digital",
                "title": "So hoa so sach",
                "status": "improve" if snapshot["data_quality_score"] < 85 else "ready",
                "metric": snapshot["data_quality_score"],
                "action": "Chuan hoa ho so, chung tu va tai khoan kinh doanh.",
            },
            {
                "key": "growth",
                "title": "Tang truong/chuyen doi",
                "status": "prepare" if projected >= VND_1B else "monitor",
                "metric": projected,
                "action": "Theo doi nguong hoa don dien tu va san sang chuyen mo hinh neu vuot 3 ty.",
            },
        ]
        return {
            "persona": {
                "household_group": group,
                "segments": list(dict.fromkeys(segments)),
                "projected_year_end_revenue": _round(projected),
            },
            "strategy_cards": strategy_cards,
            "risk_heatmap": risk_heatmap,
            "benchmark": benchmark,
            "next_best_actions": recommendations[:6],
            "model": self.model_meta({"snapshot": snapshot, "forecast": forecast}, confidence="medium"),
        }

    def cashflow_risk(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        forecast = self.forecast(dataset)
        pending_tax = snapshot["compliance"]["pending_payment"] + snapshot["compliance"]["debt_total"]
        projected_tax = calculate_tax_by_industry(max(snapshot["revenue"]["projected_year_end"], snapshot["revenue"]["total"]), snapshot["profile"]["industry"])
        reserve_needed = max(0.0, projected_tax["total_tax"] - pending_tax)
        cashflow = forecast["cashflow_30_60_90"]
        pressure = 0.0
        if cashflow["days_30"] < 0:
            pressure += 35
        if cashflow["days_60"] < 0:
            pressure += 25
        if cashflow["days_90"] < 0:
            pressure += 20
        pressure += min(20.0, pending_tax / max(snapshot["revenue"]["avg_monthly"], 1.0) * 8.0)
        risk_score = _clamp(pressure)
        plan = []
        if pending_tax > 0:
            plan.append({"action": "Uu tien thanh toan no/thue den han", "amount": _round(pending_tax), "priority": "high"})
        if reserve_needed > 0:
            plan.append({"action": "Trich lap quy du phong thue", "amount": _round(reserve_needed), "priority": "medium"})
        if snapshot["profit"]["margin"] < 0.08 and snapshot["revenue"]["total"] > 0:
            plan.append({"action": "Ra soat chi phi bien loi nhuan thap", "amount": snapshot["expenses"]["total"], "priority": "medium"})
        if not plan:
            plan.append({"action": "Duy tri ghi nhan dong tien hang thang", "amount": 0, "priority": "low"})
        return {
            "risk_score": _round(risk_score),
            "risk_level": self._risk_level(risk_score),
            "cashflow_30_60_90": cashflow,
            "pending_tax_and_debt": _round(pending_tax),
            "projected_tax": projected_tax,
            "reserve_needed": _round(reserve_needed),
            "payment_plan": plan,
            "model": self.model_meta({"snapshot": snapshot, "forecast": forecast}, confidence="medium" if snapshot["sample_size"]["revenue_entries"] >= 3 else "low"),
        }

    def supplier_risk_graph(self, dataset: dict[str, Any], tax_code: str | None = None) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        invoices = dataset.get("invoices") or []
        taxpayer_code = str(snapshot["profile"].get("tax_code") or "taxpayer")
        nodes: dict[str, dict[str, Any]] = {
            taxpayer_code: {"id": taxpayer_code, "label": snapshot["profile"].get("business_name") or "Taxpayer", "type": "taxpayer", "risk_score": 0.0}
        }
        edges = []
        supplier_stats: dict[str, dict[str, Any]] = {}
        for item in invoices:
            partner = str(item.get("seller_tax_code") or item.get("buyer_tax_code") or item.get("partner_name") or "unknown").strip() or "unknown"
            if tax_code and partner != tax_code:
                continue
            amount = _float(item.get("total_amount") or item.get("amount"))
            risk_json = _json(item.get("risk_json"))
            status = str(item.get("status") or "").lower()
            flags = list(risk_json.get("risk_flags") or [])
            score = 12.0 + min(25.0, amount / 200_000_000 * 10.0)
            if status in {"invalid", "risky", "cancelled", "missing"}:
                score += 35
            if flags:
                score += min(35, len(flags) * 14)
            if partner.startswith(("000", "999")):
                score += 22
            stats = supplier_stats.setdefault(partner, {"amount": 0.0, "count": 0, "max_score": 0.0, "flags": set(), "name": item.get("partner_name")})
            stats["amount"] += amount
            stats["count"] += 1
            stats["max_score"] = max(stats["max_score"], score)
            stats["flags"].update(flags)
            nodes[partner] = {"id": partner, "label": item.get("partner_name") or partner, "type": "supplier", "risk_score": _round(_clamp(stats["max_score"]))}
            edges.append({"source": taxpayer_code, "target": partner, "amount": _round(amount), "invoice_number": item.get("invoice_number"), "risk_score": _round(_clamp(score))})
        top_risks = [
            {
                "tax_code": key,
                "partner_name": value.get("name") or key,
                "invoice_count": value["count"],
                "amount": _round(value["amount"]),
                "risk_score": _round(_clamp(value["max_score"])),
                "risk_level": self._risk_level(value["max_score"]),
                "risk_flags": sorted(value["flags"]),
            }
            for key, value in supplier_stats.items()
        ]
        top_risks.sort(key=lambda item: item["risk_score"], reverse=True)
        return {
            "graph": {"nodes": list(nodes.values()), "edges": edges[:120]},
            "top_risks": top_risks[:10],
            "summary": {"supplier_count": max(0, len(nodes) - 1), "edge_count": len(edges), "high_risk_count": len([item for item in top_risks if item["risk_score"] >= 70])},
            "model": self.model_meta({"tax_code": tax_code, "snapshot": snapshot, "invoice_count": len(invoices)}, confidence="medium" if invoices else "low"),
        }

    def ocr_reconcile(self, content: bytes | None, filename: str | None, doc_type: str, dataset: dict[str, Any]) -> dict[str, Any]:
        extraction = self.extract_document(content, filename, doc_type)
        fields = extraction.get("extracted_fields") or {}
        amount = self._amount_from_fields(fields)
        tax_code = str(fields.get("tax_code") or "").strip()
        invoice_number = str(fields.get("invoice_number") or "").strip()
        matches = []
        for item in dataset.get("invoices") or []:
            reasons = []
            if invoice_number and invoice_number == str(item.get("invoice_number") or ""):
                reasons.append("invoice_number")
            if tax_code and tax_code in {str(item.get("seller_tax_code") or ""), str(item.get("buyer_tax_code") or "")}:
                reasons.append("tax_code")
            item_amount = _float(item.get("total_amount") or item.get("amount"))
            if amount and abs(item_amount - amount) <= max(10_000, amount * 0.02):
                reasons.append("amount")
            if reasons:
                matches.append({"type": "invoice", "id": item.get("id"), "invoice_number": item.get("invoice_number"), "match_reasons": reasons, "amount": item_amount})
        direction = "expense" if doc_type in {"invoice_in", "expense", "evidence"} else "revenue"
        suggestion = self._bookkeeping_entry_from_fields(fields, filename or "", direction)
        return {
            "extraction": extraction,
            "reconciliation_matches": matches[:8],
            "suggested_book_entries": [suggestion],
            "reconciliation_status": "matched" if matches else "needs_review",
            "model": self.model_meta({"fields": fields, "matches": matches, "doc_type": doc_type}, confidence="medium" if matches else extraction["model"]["confidence"]),
        }

    def auto_bookkeeping(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raw_items = [payload]
        proposed = []
        controls = []
        for idx, item in enumerate(raw_items, start=1):
            description = str(item.get("description") or item.get("memo") or item.get("filename") or "")
            amount = _float(item.get("amount") or item.get("total_amount"))
            direction = str(item.get("direction") or self._infer_direction(description, amount)).lower()
            category = str(item.get("category") or self._category_from_text(description, direction))
            if direction == "revenue":
                proposed.append(
                    {
                        "row_id": idx,
                        "book_code": "S1a-HKD",
                        "entry_type": "revenue",
                        "entry_date": item.get("entry_date") or item.get("date"),
                        "channel": item.get("channel") or "direct",
                        "amount": _round(amount),
                        "description": description,
                        "confidence": "medium" if amount > 0 else "low",
                    }
                )
            else:
                evaluation = evaluate_expense({**item, "amount": amount, "description": description, "category": category})
                proposed.append(
                    {
                        "row_id": idx,
                        "book_code": "S2a-HKD",
                        "entry_type": "expense",
                        "expense_date": item.get("expense_date") or item.get("date"),
                        "category": category,
                        "amount": _round(amount),
                        "description": description,
                        "deductible_status": evaluation["status"],
                        "required_evidence": evaluation["required_evidence"],
                        "confidence": "medium" if amount > 0 else "low",
                    }
                )
                if evaluation["status"] != "deductible":
                    controls.append({"row_id": idx, "severity": "medium", "message": "; ".join(evaluation["reasons"])})
        return {
            "proposed_entries": proposed,
            "control_warnings": controls,
            "posting_mode": payload.get("posting_mode") or "draft_only",
            "model": self.model_meta({"items": raw_items, "proposed_count": len(proposed)}, confidence="medium" if proposed else "low"),
        }

    def tax_return_precheck(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        declared_revenue = _float(payload.get("revenue") or payload.get("declared_revenue"), snapshot["revenue"]["total"])
        declared_expenses = _float(payload.get("expenses") or payload.get("declared_expenses"), snapshot["expenses"]["total"])
        issues = []
        book_revenue = snapshot["revenue"]["total"]
        invoice_total = snapshot["invoices"]["amount_total"]
        if book_revenue and abs(declared_revenue - book_revenue) / max(book_revenue, 1.0) > 0.1:
            issues.append({"severity": "high", "type": "revenue_mismatch", "message": "Doanh thu tren to khai lech tren 10% so voi so doanh thu."})
        if invoice_total and declared_revenue < invoice_total * 0.85:
            issues.append({"severity": "high", "type": "invoice_revenue_gap", "message": "Tong gia tri hoa don cao hon doanh thu khai bao dang ke."})
        if snapshot["expenses"]["cash_payment_violations"]:
            issues.append({"severity": "high", "type": "cash_payment", "message": "Co chi phi tien mat tu 5 trieu can loai/bo sung xu ly."})
        if snapshot["expenses"]["evidence_gap_count"]:
            issues.append({"severity": "medium", "type": "evidence_gap", "message": "Mot so chi phi can hoa don/bang ke/chung tu thanh toan."})
        projected_group = classify_household_group(max(declared_revenue, snapshot["revenue"]["projected_year_end"]))
        current_group = _int(snapshot["profile"].get("household_group"), projected_group["group"])
        if projected_group["group"] != current_group:
            issues.append({"severity": "medium", "type": "group_change", "message": "Doanh thu du bao khac nhom HKD hien tai, can cap nhat lich va so sach."})
        penalty = sum(28 if item["severity"] == "high" else 12 for item in issues)
        readiness_score = _round(_clamp(100 - penalty))
        taxes = calculate_tax_by_industry(declared_revenue, payload.get("industry") or snapshot["profile"]["industry"])
        return {
            "readiness": "ready" if readiness_score >= 80 else "needs_review" if readiness_score >= 55 else "blocked",
            "readiness_score": readiness_score,
            "issues": issues,
            "computed_tax": taxes,
            "threshold": revenue_threshold_summary(declared_revenue, snapshot["profile"]["annual_revenue_plan"]),
            "recommended_fixes": self._precheck_fixes(issues),
            "model": self.model_meta({"payload": payload, "snapshot": snapshot}, confidence="medium"),
        }

    def policy_impact(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        revenue = _float(payload.get("revenue"), max(snapshot["revenue"]["projected_year_end"], snapshot["revenue"]["total"]))
        industry = str(payload.get("industry") or snapshot["profile"]["industry"] or "commerce")
        channels = payload.get("channels") or list(snapshot["revenue"]["channels"].keys())
        if isinstance(channels, str):
            channels = [channels]
        impacts = []
        impacts.append(self._impact("tt18_2026", "Ke khai/thong bao doanh thu", "high", "Can dung dung mau ho so, thoi han va kenh nop theo nhom HKD."))
        impacts.append(self._impact("tt152_2025", "So sach ke toan HKD", "high" if revenue >= VND_500M else "medium", "Can lap so doanh thu, chi phi va luu tru chung tu theo nhom."))
        if revenue >= VND_1B:
            impacts.append(self._impact("nd70_2025", "Hoa don dien tu bat buoc", "high", "Doanh thu tu 1 ty can san sang hoa don dien tu/may tinh tien theo quy dinh."))
        elif revenue >= VND_500M:
            impacts.append(self._impact("nd70_2025", "Hoa don dien tu khuyen khich", "medium", "Nen chu dong dung hoa don dien tu neu giao dich voi doanh nghiep/to chuc."))
        if any(str(channel).lower() in {"ecommerce", "marketplace", "tmdt", "shopee", "tiktok"} for channel in channels):
            impacts.append(self._impact("nd68_2026", "Doanh thu TMDT", "high", "Can tach doanh thu san da khau tru va doanh thu tu khai de tranh trung."))
        if revenue >= VND_3B:
            impacts.append(self._impact("nd68_2026", "Chuyen nhom doanh thu cao", "high", "Can chuan bi so sach day du, hang ton kho va quy trinh ke khai chat che hon."))
        if industry in {"rental", "service"}:
            impacts.append(self._impact("pit_109_2025", "Thu nhap ca nhan/nganh dich vu", "medium", "Can doi chieu ty le thue va giam tru neu co thu nhap nhieu nguon."))
        return {
            "context": {"industry": industry, "revenue": revenue, "channels": channels},
            "impacts": impacts,
            "citations": [item["citation"] for item in impacts],
            "model": self.model_meta({"payload": payload, "impact_count": len(impacts)}, confidence="medium"),
        }

    def business_upgrade_readiness(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        revenue = max(snapshot["revenue"]["projected_year_end"], snapshot["revenue"]["total"])
        components = {
            "scale": _clamp(revenue / VND_3B * 100),
            "data_quality": snapshot["data_quality_score"],
            "bookkeeping": _clamp(100 - snapshot["expenses"]["evidence_gap_count"] * 12 - snapshot["expenses"]["cash_payment_violations"] * 22),
            "einvoice": 90 if revenue >= VND_1B and snapshot["invoices"]["count"] > 0 else 55 if revenue >= VND_1B else 70,
            "cashflow": _float(self.overview(dataset)["scores"]["cashflow"]),
            "compliance": _float(self.overview(dataset)["scores"]["compliance"]),
        }
        score = _round(sum(components.values()) / len(components))
        missing = []
        if components["bookkeeping"] < 75:
            missing.append("Hoan thien chung tu va phan loai chi phi.")
        if components["einvoice"] < 75:
            missing.append("Chuan bi nha cung cap hoa don dien tu va quy trinh xuat hoa don.")
        if components["data_quality"] < 85:
            missing.append("Cap nhat ho so, nganh nghe, tai khoan ngan hang va thong tin lien he.")
        if components["cashflow"] < 60:
            missing.append("Lap ke hoach dong tien truoc khi chuyen doi.")
        timeline = [
            {"phase": "30_ngay", "action": "Lam sach so lieu doanh thu, chi phi, hoa don va tai khoan kinh doanh."},
            {"phase": "60_ngay", "action": "Chuan hoa so sach, nha cung cap HDDT, chu ky so va quy trinh ke khai."},
            {"phase": "90_ngay", "action": "Mo phong thue HKD vs TNHH, chot phuong an va ho so chuyen doi."},
        ]
        return {
            "readiness_score": score,
            "readiness_level": "ready" if score >= 78 else "prepare" if score >= 55 else "not_ready",
            "components": {key: _round(value) for key, value in components.items()},
            "missing_capabilities": missing,
            "transition_timeline": timeline,
            "model": self.model_meta({"snapshot": snapshot, "components": components}, confidence="medium"),
        }

    def copilot(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        page = str(payload.get("page") or "business_dashboard.html")
        text = question.lower()
        citations = []
        actions = []
        if any(token in text for token in ["hoa don", "invoice", "nha cung cap", "doi tac"]):
            supplier = self.supplier_risk_graph(dataset)
            answer = f"Co {supplier['summary']['supplier_count']} doi tac trong graph hoa don; {supplier['summary']['high_risk_count']} doi tac dang co diem rui ro cao."
            actions.append({"label": "Mo hoa don", "target_page": "business_invoices.html"})
            cards = supplier["top_risks"][:3]
        elif any(token in text for token in ["dong tien", "cash", "no thue", "nop thue"]):
            cash = self.cashflow_risk(dataset)
            answer = f"Rui ro dong tien hien o muc {cash['risk_level']} voi diem {cash['risk_score']}/100. Du phong thue can them khoang {cash['reserve_needed']:,.0f} VND."
            actions.append({"label": "Mo no thue", "target_page": "business_debts.html"})
            cards = cash["payment_plan"]
        elif any(token in text for token in ["luat", "chinh sach", "nghi dinh", "thong tu"]):
            impact = self.policy_impact(payload, dataset)
            answer = f"Co {len(impact['impacts'])} tac dong chinh sach lien quan toi nganh/doanh thu hien tai."
            citations = impact["citations"]
            actions.append({"label": "Mo tra cuu", "target_page": "business_legal.html"})
            cards = impact["impacts"][:3]
        elif any(token in text for token in ["chuyen doi", "cong ty", "tnhh", "doanh nghiep"]):
            readiness = self.business_upgrade_readiness(dataset)
            answer = f"Diem san sang chuyen doi hien la {readiness['readiness_score']}/100, trang thai {readiness['readiness_level']}."
            actions.append({"label": "Mo thay doi mo hinh", "target_page": "business_growth.html"})
            cards = readiness["transition_timeline"]
        else:
            scenario = self.scenario_dashboard(dataset)
            answer = "Uu tien hien tai la xu ly cac canh bao co muc cao, giu dong tien nop thue va hoan thien chung tu."
            actions.extend({"label": item.get("action_label"), "target_page": item.get("target_page")} for item in scenario["next_best_actions"][:2])
            cards = scenario["next_best_actions"][:3]
        return {
            "answer": answer,
            "page": page,
            "cards": cards,
            "actions": actions,
            "citations": citations,
            "model": self.model_meta({"question": question, "page": page}, confidence="medium" if question else "low"),
        }

    def advanced_dashboard(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        overview = self.overview(dataset)
        forecast = self.probabilistic_forecast(dataset)
        graph = self.graph_risk(dataset)
        cash = self.cashflow_delinquency(dataset)
        nba = self.next_best_action(dataset)
        anomaly = self.anomaly_insights(dataset)
        evidence_score = _clamp(100 - snapshot["expenses"]["evidence_gap_count"] * 12 - snapshot["expenses"]["cash_payment_violations"] * 24)
        readiness = self.business_upgrade_readiness(dataset)
        command_center = {
            "financial_health": overview["scores"]["financial_health"],
            "compliance": overview["scores"]["compliance"],
            "cashflow": overview["scores"]["cashflow"],
            "data_quality": overview["scores"]["data_quality"],
            "evidence": _round(evidence_score),
            "graph_risk": graph["summary"]["graph_risk_score"],
            "delinquency_hazard": cash["hazard_90d"],
            "upgrade_readiness": readiness["readiness_score"],
        }
        risk_heatmap = [
            {"axis": "forecast_threshold", "score": _round(100 * max(forecast["threshold_probabilities"].values() or [0])), "label": "Vuot nguong"},
            {"axis": "invoice_graph", "score": graph["summary"]["graph_risk_score"], "label": "Graph hoa don"},
            {"axis": "cashflow", "score": cash["risk_score"], "label": "Dong tien"},
            {"axis": "evidence", "score": _round(100 - evidence_score), "label": "Chung tu"},
            {"axis": "anomaly", "score": _round(min(100, anomaly["summary"]["high"] * 35 + anomaly["summary"]["medium"] * 16)), "label": "Bat thuong"},
        ]
        return {
            "command_center": command_center,
            "risk_heatmap": risk_heatmap,
            "probabilistic_forecast": forecast,
            "graph_summary": graph["summary"],
            "top_actions": nba["actions"][:5],
            "explanation": {
                "reason_codes": self._reason_codes(snapshot),
                "counterfactual": {
                    "cash_payment_fix": "Neu loai/doi chung tu cac giao dich tien mat >=5 trieu, diem evidence co the tang.",
                    "evidence_upload": "Tai them chung tu cho chi phi thieu bang chung se giam rui ro precheck.",
                },
                "research_basis": [
                    "OECD: AI thue can giai thich duoc, quan tri rui ro va co human oversight.",
                    "World Bank: cong thong tin/e-payment giam chi phi tuan thu cho nguoi nop thue.",
                    "Time-series + conformal intervals phu hop SME du lieu it vi tra ve khoang bat dinh.",
                ],
            },
            "model": self.model_meta({"snapshot": snapshot, "command_center": command_center}, confidence="medium"),
        }

    def document_ai_extract(self, content: bytes | None, filename: str | None, doc_type: str, dataset: dict[str, Any]) -> dict[str, Any]:
        extraction = self.extract_document(content, filename, doc_type)
        fields = extraction.get("extracted_fields") or {}
        field_confidence = {}
        for key, value in fields.items():
            base = 0.78 if value not in (None, "", 0) else 0.25
            if key in {"tax_code", "invoice_number"}:
                base += 0.08
            field_confidence[key] = _round(min(0.94, base), 4)
        quality_flags = list(extraction.get("quality_flags") or [])
        if not fields.get("tax_code") and doc_type in {"invoice", "invoice_in", "invoice_out"}:
            quality_flags.append("missing_tax_code")
        if not fields.get("amount") and not fields.get("amount_text"):
            quality_flags.append("missing_amount")
        layout_blocks = [
            {"block_type": "header", "confidence": 0.62, "fields": [key for key in fields.keys() if key in {"tax_code", "invoice_number"}]},
            {"block_type": "table_or_total", "confidence": 0.58, "fields": [key for key in fields.keys() if key in {"amount", "amount_text"}]},
        ]
        active_learning = {
            "needs_human_review": bool(quality_flags) or extraction["model"]["confidence"] == "low",
            "review_fields": sorted(set(flag.replace("missing_", "") for flag in quality_flags if flag.startswith("missing_"))),
            "labeling_hint": "Xac nhan MST, so hoa don, tong tien va ngay chung tu de cai thien OCR.",
        }
        result = {
            **extraction,
            "layout_blocks": layout_blocks,
            "field_confidence": field_confidence,
            "quality_flags": quality_flags,
            "table_extraction": {
                "detected": bool(fields.get("amount") or fields.get("amount_text")),
                "rows": [],
                "method": "layoutlm_donut_table_transformer_baseline",
            },
            "active_learning": active_learning,
            "explanation": {
                "reason_codes": quality_flags or ["structured_fields_detected"],
                "research_basis": ["Multimodal document AI ket hop OCR, layout va table parsing; baseline dang chay rule-first."],
            },
            "model": self.model_meta({"filename": filename, "doc_type": doc_type, "fields": fields}, confidence=extraction["model"]["confidence"]),
        }
        return result

    def document_ai_reconcile(self, content: bytes | None, filename: str | None, doc_type: str, dataset: dict[str, Any]) -> dict[str, Any]:
        base = self.ocr_reconcile(content, filename, doc_type, dataset)
        extraction = base.get("extraction") or {}
        fields = extraction.get("extracted_fields") or {}
        matches = base.get("reconciliation_matches") or []
        bank_matches = []
        amount = self._amount_from_fields(fields)
        tax_code = str(fields.get("tax_code") or "")
        for item in dataset.get("bank_transactions") or []:
            reasons = []
            if tax_code and tax_code == str(item.get("counterparty_tax_code") or ""):
                reasons.append("counterparty_tax_code")
            tx_amount = abs(_float(item.get("amount")))
            if amount and abs(tx_amount - amount) <= max(10_000, amount * 0.02):
                reasons.append("amount")
            if reasons:
                bank_matches.append({"type": "bank_transaction", "id": item.get("id"), "match_reasons": reasons, "amount": tx_amount})
        controls = []
        if not matches and not bank_matches:
            controls.append({"severity": "medium", "message": "Chua tim thay hoa don/sao ke khop voi chung tu OCR."})
        if amount >= CASH_PAYMENT_LIMIT and not bank_matches and doc_type in {"invoice_in", "expense", "evidence"}:
            controls.append({"severity": "high", "message": "Chi phi tu 5 trieu nen co chung tu thanh toan khong dung tien mat."})
        return {
            **base,
            "bank_matches": bank_matches[:8],
            "control_warnings": controls,
            "reconciliation_score": _round(_clamp(len(matches) * 42 + len(bank_matches) * 35 - len(controls) * 18)),
            "explanation": {
                "reason_codes": ["invoice_match" if matches else "invoice_unmatched", "bank_match" if bank_matches else "bank_unmatched"],
                "counterfactual": {"upload_bank_statement": "Them sao ke ngan hang giup doi soat giao dich tu 5 trieu."},
            },
            "model": self.model_meta({"fields": fields, "matches": matches, "bank_matches": bank_matches}, confidence="medium" if matches or bank_matches else "low"),
        }

    def probabilistic_forecast(self, dataset: dict[str, Any]) -> dict[str, Any]:
        point = self.forecast(dataset)
        snapshot = self.build_snapshot(dataset)
        monthly = snapshot["monthly"]
        values = [_float(item.get("revenue")) for item in monthly if _float(item.get("revenue")) > 0]
        mean = sum(values) / len(values) if values else max(snapshot["profile"]["annual_revenue_plan"] / 12.0, 0.0)
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        sigma = math.sqrt(variance) if variance > 0 else max(mean * 0.18, 2_000_000)
        intervals = []
        cumulative_p50 = snapshot["revenue"]["total"]
        for idx, item in enumerate(point["forecast_months"], start=1):
            p50 = _float(item.get("revenue"))
            width = sigma * math.sqrt(idx) * 1.2816
            intervals.append(
                {
                    "period": item["period"],
                    "p10": _round(max(0.0, p50 - width)),
                    "p50": _round(p50),
                    "p90": _round(p50 + width),
                    "expense_p50": item.get("expense"),
                    "profit_p50": item.get("profit"),
                }
            )
            cumulative_p50 += p50
        threshold_probs = {}
        for threshold_key, threshold_value in {"taxable_500m": VND_500M, "einvoice_1b": VND_1B, "group3_3b": VND_3B}.items():
            projected_p90 = snapshot["revenue"]["total"] + sum(item["p90"] for item in intervals[: max(0, 12 - (_date(dataset.get("today")) or date.today()).month)])
            projected_p10 = snapshot["revenue"]["total"] + sum(item["p10"] for item in intervals[: max(0, 12 - (_date(dataset.get("today")) or date.today()).month)])
            if projected_p10 >= threshold_value:
                prob = 0.92
            elif projected_p90 < threshold_value:
                prob = 0.12
            else:
                prob = _clamp((cumulative_p50 - threshold_value) / max(projected_p90 - projected_p10, 1.0) + 0.5, 0.15, 0.88)
            threshold_probs[threshold_key] = _round(prob, 4)
        return {
            "intervals": intervals,
            "projected_year_end": point["projected_year_end_revenue"],
            "threshold_probabilities": threshold_probs,
            "method_stack": ["seasonal_baseline", "tft_ready", "nbeats_ready", "deepar_ready", "conformal_prediction_interval"],
            "data_sufficiency": {
                "observations": len(values),
                "status": "model_ready" if len(values) >= 12 else "few_shot_baseline",
            },
            "explanation": {
                "reason_codes": ["few_observations" if len(values) < 12 else "enough_history", "conformal_interval"],
                "counterfactual": {"add_daily_revenue": "Ghi doanh thu hang ngay lam khoang P10-P90 hep hon."},
            },
            "model": self.model_meta({"snapshot": snapshot, "intervals": intervals}, confidence="medium" if len(values) >= 4 else "low"),
        }

    def digital_twin_simulate(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        base_revenue = max(0.0, _float(payload.get("revenue"), snapshot["revenue"]["projected_year_end"] or snapshot["profile"]["annual_revenue_plan"]))
        base_expenses = max(0.0, _float(payload.get("expenses"), snapshot["expenses"]["total"] or base_revenue * 0.72))
        growth_rate = _float(payload.get("growth_rate_pct"), 0.0) / 100.0
        cost_change = _float(payload.get("cost_change_pct"), 0.0) / 100.0
        months_active = max(1, min(12, _int(payload.get("months_active"), 12)))
        industry = str(payload.get("industry") or snapshot["profile"]["industry"] or "commerce")
        simulated_revenue = base_revenue * (1.0 + growth_rate) * months_active / 12.0
        simulated_expense = base_expenses * (1.0 + cost_change) * months_active / 12.0
        hkd_revenue_tax = calculate_tax_by_industry(simulated_revenue, industry)
        hkd_profit_tax = self._simple_progressive_profit_tax(max(0.0, simulated_revenue - simulated_expense))
        llc_tax = max(0.0, simulated_revenue - simulated_expense) * 0.2
        variants = [
            {
                "key": "hkd_revenue_percentage",
                "label": "HKD - ty le doanh thu",
                "revenue": _round(simulated_revenue),
                "expenses": _round(simulated_expense),
                "tax": hkd_revenue_tax["total_tax"],
                "profit_after_tax": _round(simulated_revenue - simulated_expense - hkd_revenue_tax["total_tax"]),
            },
            {
                "key": "hkd_profit_based",
                "label": "HKD - doanh thu tru chi phi",
                "revenue": _round(simulated_revenue),
                "expenses": _round(simulated_expense),
                "tax": hkd_profit_tax["tax"],
                "profit_after_tax": _round(simulated_revenue - simulated_expense - hkd_profit_tax["tax"]),
            },
            {
                "key": "llc_simplified",
                "label": "Cong ty TNHH - mo phong don gian",
                "revenue": _round(simulated_revenue),
                "expenses": _round(simulated_expense),
                "tax": _round(llc_tax),
                "profit_after_tax": _round(simulated_revenue - simulated_expense - llc_tax),
                "needs_accountant_review": True,
            },
        ]
        variants.sort(key=lambda item: item["profit_after_tax"], reverse=True)
        return {
            "input": {"revenue": base_revenue, "expenses": base_expenses, "growth_rate_pct": growth_rate * 100, "months_active": months_active, "industry": industry},
            "variants": variants,
            "threshold": revenue_threshold_summary(simulated_revenue, snapshot["profile"]["annual_revenue_plan"]),
            "recommended_variant": variants[0]["key"],
            "transition_triggers": [
                {"threshold": "500M", "active": simulated_revenue >= VND_500M, "action": "Chuan bi ke khai/thong bao doanh thu."},
                {"threshold": "1B", "active": simulated_revenue >= VND_1B, "action": "Chuan bi hoa don dien tu va quy trinh xuat hoa don."},
                {"threshold": "3B", "active": simulated_revenue >= VND_3B, "action": "Danh gia chuyen mo hinh doanh nghiep/so sach day du."},
            ],
            "explanation": {
                "reason_codes": ["digital_twin_scenario", "threshold_simulation", "tax_method_comparison"],
                "counterfactual": {"lower_cost_ratio": "Giam ty le chi phi hoac co du chung tu co the doi phuong an toi uu."},
            },
            "model": self.model_meta({"payload": payload, "variants": variants}, confidence="medium"),
        }

    def graph_risk(self, dataset: dict[str, Any]) -> dict[str, Any]:
        supplier = self.supplier_risk_graph(dataset)
        explicit_nodes = dataset.get("graph_nodes") or []
        explicit_edges = dataset.get("graph_edges") or []
        nodes = supplier["graph"]["nodes"] + [
            {"id": item.get("node_key"), "label": item.get("label") or item.get("node_key"), "type": item.get("node_type"), "risk_score": _round(item.get("risk_score"))}
            for item in explicit_nodes
        ]
        edges = supplier["graph"]["edges"] + [
            {"source": item.get("source_key"), "target": item.get("target_key"), "edge_type": item.get("edge_type"), "amount": _round(item.get("amount")), "risk_score": _round(item.get("risk_score"))}
            for item in explicit_edges
        ]
        node_count = len({str(item.get("id")) for item in nodes if item.get("id")})
        edge_count = len(edges)
        max_risk = max([_float(item.get("risk_score")) for item in nodes + edges] or [0.0])
        density = edge_count / max(1, node_count * (node_count - 1))
        centrality = []
        degree: dict[str, int] = defaultdict(int)
        for edge in edges:
            degree[str(edge.get("source"))] += 1
            degree[str(edge.get("target"))] += 1
        for key, value in sorted(degree.items(), key=lambda item: item[1], reverse=True)[:8]:
            centrality.append({"node_key": key, "degree": value, "risk_score": next((_float(node.get("risk_score")) for node in nodes if str(node.get("id")) == key), 0.0)})
        communities = [
            {"community_id": "invoice_network", "node_count": node_count, "edge_count": edge_count, "risk_score": _round(max_risk), "explanation": "Cong dong tao tu hoa don, doi tac, sao ke va node graph noi bo."}
        ]
        return {
            "graph": {"nodes": nodes[:300], "edges": edges[:500]},
            "summary": {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": _round(density, 4),
                "graph_risk_score": _round(_clamp(max_risk * 0.72 + density * 100 * 0.28)),
                "high_risk_nodes": len([item for item in nodes if _float(item.get("risk_score")) >= 70]),
            },
            "centrality": centrality,
            "communities": communities,
            "gnn_signals": {
                "method_stack": ["heterogeneous_graph", "temporal_edges", "hgt_rgcn_ready", "metapath_supplier_invoice_bank"],
                "artifact_status": "adapter_ready",
            },
            "explanation": {
                "reason_codes": ["supplier_invoice_edges", "centrality_degree", "risk_propagation_baseline"],
                "counterfactual": {"verify_supplier": "Xac minh nha cung cap rui ro cao se giam diem graph risk."},
            },
            "model": self.model_meta({"node_count": node_count, "edge_count": edge_count}, confidence="medium" if edge_count else "low"),
        }

    def ledger_autopost(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        base = self.auto_bookkeeping(payload, dataset)
        entries = base.get("proposed_entries") or []
        ledger_entries = []
        missing_evidence = []
        for item in entries:
            entry_type = item.get("entry_type")
            amount = _float(item.get("amount"))
            ledger_entries.append(
                {
                    "book_code": item.get("book_code"),
                    "entry_type": entry_type,
                    "account_code": "511" if entry_type == "revenue" else "642",
                    "amount": _round(amount),
                    "description": item.get("description"),
                    "confidence_score": 0.72 if item.get("confidence") == "medium" else 0.42,
                    "posting_status": "draft_ai_suggested",
                }
            )
            if item.get("deductible_status") in {"needs_invoice", "needs_evidence", "non_deductible"}:
                missing_evidence.append({"description": item.get("description"), "status": item.get("deductible_status"), "required_evidence": item.get("required_evidence")})
        return {
            **base,
            "ledger_entries": ledger_entries,
            "missing_evidence": missing_evidence,
            "book_coverage": sorted(set(item.get("book_code") for item in ledger_entries if item.get("book_code"))),
            "explanation": {
                "reason_codes": ["auto_category", "deductibility_rule", "ledger_mapping"],
                "counterfactual": {"attach_invoice": "Gan hoa don hop le co the tang confidence va doi status sang deductible."},
            },
            "model": self.model_meta({"payload": payload, "entry_count": len(ledger_entries)}, confidence="medium" if ledger_entries else "low"),
        }

    def filing_precheck_advanced(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        base = self.tax_return_precheck(payload, dataset)
        forecast = self.probabilistic_forecast(dataset)
        anomalies = self.anomaly_insights(dataset)
        graph = self.graph_risk(dataset)
        issues = list(base.get("issues") or [])
        if max(forecast["threshold_probabilities"].values() or [0]) >= 0.75:
            issues.append({"severity": "medium", "type": "probabilistic_threshold", "message": "Forecast P10/P50/P90 cho thay kha nang vuot nguong doanh thu dang cao."})
        if anomalies["summary"]["high"]:
            issues.append({"severity": "high", "type": "anomaly_high", "message": "Co bat thuong muc cao trong doanh thu/chi phi/hoa don can giai trinh."})
        if graph["summary"]["graph_risk_score"] >= 70:
            issues.append({"severity": "high", "type": "supplier_graph_risk", "message": "Graph nha cung cap/hoa don co diem rui ro cao."})
        penalty = sum(24 if item["severity"] == "high" else 10 for item in issues)
        advanced_score = _round(_clamp(100 - penalty))
        return {
            **base,
            "advanced_readiness_score": advanced_score,
            "advanced_readiness": "ready" if advanced_score >= 82 else "needs_review" if advanced_score >= 55 else "blocked",
            "issues": issues,
            "forecast_intervals": forecast["intervals"][:3],
            "anomaly_summary": anomalies["summary"],
            "graph_summary": graph["summary"],
            "explainable_delta": {
                "base_score": base.get("readiness_score"),
                "advanced_score": advanced_score,
                "main_drivers": [item["type"] for item in issues[:5]],
            },
            "explanation": {
                "reason_codes": ["rule_precheck", "forecast_interval", "anomaly_scan", "graph_risk"],
                "counterfactual": {"fix_high_issues": "Xu ly cac loi severity high truoc khi ky/nop se tang diem san sang."},
            },
            "model": self.model_meta({"payload": payload, "issue_count": len(issues)}, confidence="medium"),
        }

    def cashflow_delinquency(self, dataset: dict[str, Any]) -> dict[str, Any]:
        base = self.cashflow_risk(dataset)
        snapshot = self.build_snapshot(dataset)
        debt_ratio = snapshot["compliance"]["debt_total"] / max(snapshot["revenue"]["avg_monthly"], 1.0)
        hazard_30 = _clamp(base["risk_score"] * 0.004 + min(0.25, debt_ratio * 0.04), 0.02, 0.92)
        hazard_60 = _clamp(hazard_30 + 0.12 + (0.08 if base["cashflow_30_60_90"]["days_60"] < 0 else 0), 0.03, 0.96)
        hazard_90 = _clamp(hazard_60 + 0.10 + (0.08 if base["cashflow_30_60_90"]["days_90"] < 0 else 0), 0.05, 0.98)
        survival_curve = [
            {"horizon": "30d", "survival_probability": _round(1 - hazard_30, 4), "hazard": _round(hazard_30, 4)},
            {"horizon": "60d", "survival_probability": _round(1 - hazard_60, 4), "hazard": _round(hazard_60, 4)},
            {"horizon": "90d", "survival_probability": _round(1 - hazard_90, 4), "hazard": _round(hazard_90, 4)},
        ]
        optimizer = []
        total_due = base["pending_tax_and_debt"]
        if total_due > 0:
            optimizer = [
                {"period": "now", "amount": _round(total_due * 0.4), "goal": "Giam tien cham nop va rui ro cuong che"},
                {"period": "30d", "amount": _round(total_due * 0.35), "goal": "Duy tri dong tien toi thieu"},
                {"period": "60d", "amount": _round(total_due * 0.25), "goal": "Tat toan phan con lai neu duoc chap nhan phan ky"},
            ]
        return {
            **base,
            "hazard_30d": _round(hazard_30 * 100),
            "hazard_60d": _round(hazard_60 * 100),
            "hazard_90d": _round(hazard_90 * 100),
            "survival_curve": survival_curve,
            "payment_optimizer": optimizer,
            "reserve_policy": {
                "recommended_tax_reserve_rate": _round(min(0.3, max(0.08, base["projected_tax"]["total_tax"] / max(snapshot["revenue"]["projected_year_end"], 1.0))), 4),
                "cash_buffer_months": 2 if base["risk_level"] == "high" else 1,
            },
            "explanation": {
                "reason_codes": ["debt_ratio", "cashflow_forecast", "deadline_pressure"],
                "counterfactual": {"partial_payment": "Nop truoc mot phan no thue co the giam hazard 30-90 ngay."},
            },
            "model": self.model_meta({"base": base, "snapshot": snapshot}, confidence="medium" if snapshot["sample_size"]["revenue_entries"] >= 3 else "low"),
        }

    def legal_graphrag(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        snapshot = self.build_snapshot(dataset)
        lower = question.lower()
        selected = []
        for source in BASELINE_SOURCES:
            haystack = f"{source.get('key')} {source.get('title')} {source.get('category')} {source.get('article_ref')}".lower()
            if any(token in haystack for token in re.findall(r"[a-zA-Z0-9_]+", lower)) or not selected:
                selected.append(source)
            if len(selected) >= 4:
                break
        impacts = self.policy_impact({"industry": snapshot["profile"]["industry"], "revenue": snapshot["revenue"]["projected_year_end"], "channels": list(snapshot["revenue"]["channels"].keys())}, dataset)
        citations = [
            {
                "title": item.get("title"),
                "source_url": item.get("source_url"),
                "article_ref": item.get("article_ref"),
                "effective_from": item.get("effective_from"),
            }
            for item in selected[:4]
        ]
        needs_verification = not bool(question) or not citations
        answer = (
            "Theo du lieu ho so hien tai, can doi chieu nghia vu theo nhom doanh thu, hoa don dien tu, so sach va chung tu chi phi. "
            "Ket qua nay la goi y co citation, nen kiem tra lai van ban goc truoc khi nop ho so."
        )
        if "hoa don" in lower:
            answer = "Trong tinh huong hoa don, uu tien kiem tra MST doi tac, trang thai hoa don, phuong thuc thanh toan va nguong doanh thu de quyet dinh cach ghi nhan."
        elif "chi phi" in lower:
            answer = "Voi chi phi duoc tru, can co lien quan kinh doanh, du chung tu va giao dich tu 5 trieu nen thanh toan khong dung tien mat."
        elif "chuyen" in lower or "tnhh" in lower:
            answer = "Khi doanh thu tang nhanh, nen mo phong HKD vs TNHH, kha nang so sach, hoa don dien tu, dong tien va trach nhiem phap ly."
        return {
            "answer": answer,
            "needs_verification": needs_verification,
            "citations": citations,
            "policy_impacts": impacts["impacts"][:4],
            "citation_verifier": {
                "citation_count": len(citations),
                "unsupported_claims": [] if citations else ["no_citation_available"],
                "quote_policy": "Chi trich dan ngan, uu tien tom tat va link van ban goc.",
            },
            "explanation": {
                "reason_codes": ["graph_retrieval_baseline", "policy_impact_context", "citation_guard"],
                "counterfactual": {"add_question_detail": "Nhap nganh, doanh thu, kenh ban va ky thue giup truy xuat citation sat hon."},
            },
            "model": self.model_meta({"question": question, "citations": citations}, confidence="medium" if citations and question else "low"),
        }

    def next_best_action(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        base = self.recommendations(dataset, snapshot=snapshot)["recommendations"]
        cash = self.cashflow_risk(dataset)
        actions = []
        priority_weight = {"high": 42, "medium": 24, "low": 10}
        for idx, item in enumerate(base, start=1):
            uplift = priority_weight.get(item.get("priority"), 18) + max(0, 16 - idx * 2)
            if item.get("key") == "debt_repayment_plan" and cash["risk_level"] == "high":
                uplift += 16
            if item.get("key") == "evidence_gap" and snapshot["expenses"]["evidence_gap_count"] > 2:
                uplift += 10
            actions.append(
                {
                    **item,
                    "uplift_score": _round(_clamp(uplift)),
                    "expected_impact": self._nba_impact(item.get("key")),
                    "policy": "contextual_bandit_baseline",
                    "treatment": item.get("key"),
                }
            )
        actions.sort(key=lambda item: item["uplift_score"], reverse=True)
        return {
            "actions": actions,
            "ranking_policy": {
                "method_stack": ["doubly_robust_ready", "causal_forest_ready", "contextual_bandit_baseline"],
                "exploration_rate": 0.05,
                "fairness_guard": "Khong dung thuoc tinh nhay cam de giam quyen tiep canh bao.",
            },
            "explanation": {
                "reason_codes": ["priority", "cashflow_context", "evidence_context", "deadline_context"],
                "counterfactual": {"feedback": "Phan hoi huu ich/chua dung se doi trong so next-best-action lan sau."},
            },
            "model": self.model_meta({"snapshot": snapshot, "actions": actions}, confidence="medium"),
        }

    def model_governance(self, dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        feedback = dataset.get("ai_feedback") or []
        consents = {item.get("consent_key"): item.get("status") for item in dataset.get("privacy_consents") or []}
        sample_count = sum(snapshot["sample_size"].values())
        drift_score = _round(_clamp((snapshot["expenses"]["cash_payment_violations"] * 10) + (snapshot["invoices"]["risky_count"] * 8) + (0 if sample_count >= 8 else 22)))
        calibration_bins = [
            {"bin": "low", "count": max(1, len([item for item in feedback if item.get("signal") == "not_relevant"])), "estimated_accuracy": 0.55},
            {"bin": "medium", "count": max(1, len([item for item in feedback if item.get("signal") == "helpful"])), "estimated_accuracy": 0.72},
            {"bin": "high", "count": max(1, sample_count // 4), "estimated_accuracy": 0.86},
        ]
        model_cards = [
            {"model": "DocumentAIEngine", "status": "baseline_adapter", "risk": "medium", "human_review_required": True},
            {"model": "ProbabilisticForecastEngine", "status": "baseline_adapter", "risk": "medium", "human_review_required": False},
            {"model": "LedgerGraphEngine", "status": "baseline_adapter", "risk": "high", "human_review_required": True},
            {"model": "LegalGraphRAGEngine", "status": "citation_guard", "risk": "high", "human_review_required": True},
            {"model": "CausalNBAEngine", "status": "baseline_adapter", "risk": "medium", "human_review_required": False},
        ]
        return {
            "model_cards": model_cards,
            "drift": {"score": drift_score, "level": self._risk_level(drift_score), "method": "psi_adwin_page_hinkley_ready"},
            "calibration": {"bins": calibration_bins, "status": "needs_more_feedback" if len(feedback) < 20 else "monitoring"},
            "fairness": {"status": "guarded", "notes": "Taxpayer advisory khong xep hang theo thuoc tinh nhay cam; can audit khi co du lieu lon."},
            "privacy": {
                "bank_training_consent": consents.get("bank_training", "not_granted"),
                "ocr_training_consent": consents.get("ocr_training", "not_granted"),
                "federated_learning_ready": True,
                "differential_privacy_export_ready": True,
            },
            "feedback_quality": {
                "feedback_count": len(feedback),
                "helpful_count": len([item for item in feedback if item.get("signal") == "helpful"]),
                "not_relevant_count": len([item for item in feedback if item.get("signal") == "not_relevant"]),
            },
            "explanation": {
                "reason_codes": ["model_card", "drift_monitor", "calibration_bins", "privacy_consent"],
                "counterfactual": {"more_feedback": "Them feedback va du lieu giao dich se cai thien calibration."},
            },
            "model": self.model_meta({"sample_count": sample_count, "feedback_count": len(feedback)}, confidence="medium"),
        }

    def reconcile_4way(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        bank_transactions = dataset.get("bank_transactions") or []
        ledger_entries = dataset.get("ledger_entries") or []
        filings = dataset.get("filings") or []
        invoices = dataset.get("invoices") or []
        platform_orders = dataset.get("platform_orders") or []

        bank_in = sum(_float(item.get("amount")) for item in bank_transactions if str(item.get("direction") or "in").lower() == "in")
        bank_out = sum(_float(item.get("amount")) for item in bank_transactions if str(item.get("direction") or "").lower() == "out")
        invoice_out = sum(_float(item.get("total_amount") or item.get("amount")) for item in invoices if str(item.get("direction") or "out").lower() == "out")
        invoice_in = sum(_float(item.get("total_amount") or item.get("amount")) for item in invoices if str(item.get("direction") or "").lower() == "in")
        ledger_revenue = sum(_float(item.get("amount")) for item in ledger_entries if str(item.get("entry_type") or "").lower() == "revenue")
        ledger_expense = sum(_float(item.get("amount")) for item in ledger_entries if str(item.get("entry_type") or "").lower() == "expense")
        platform_gross = sum(_float(item.get("gross_amount")) for item in platform_orders)
        latest_filing = filings[0] if filings else {}
        declared_revenue = _float(payload.get("declared_revenue") or latest_filing.get("revenue"), snapshot["revenue"]["total"])
        declared_expense = _float(payload.get("declared_expenses") or latest_filing.get("expenses"), snapshot["expenses"]["total"])

        totals = {
            "bank_in": _round(bank_in),
            "bank_out": _round(bank_out),
            "invoice_out": _round(invoice_out),
            "invoice_in": _round(invoice_in),
            "ledger_revenue": _round(ledger_revenue),
            "ledger_expense": _round(ledger_expense),
            "book_revenue": snapshot["revenue"]["total"],
            "book_expense": snapshot["expenses"]["total"],
            "declared_revenue": _round(declared_revenue),
            "declared_expense": _round(declared_expense),
            "platform_gross": _round(platform_gross),
        }

        def delta_case(key: str, lhs: float, rhs: float, title: str, description: str, high_threshold: float = 0.18) -> dict[str, Any] | None:
            base = max(abs(lhs), abs(rhs), 1.0)
            ratio = abs(lhs - rhs) / base
            if ratio < 0.08 and abs(lhs - rhs) < 5_000_000:
                return None
            severity = "high" if ratio >= high_threshold or abs(lhs - rhs) >= 50_000_000 else "medium"
            return {
                "case_key": key,
                "case_type": "4way_reconciliation",
                "severity": severity,
                "status": "open",
                "title": title,
                "description": description,
                "score": _round(_clamp(ratio * 100)),
                "entity_refs": [{"metric": "lhs", "amount": _round(lhs)}, {"metric": "rhs", "amount": _round(rhs)}],
                "suggested_actions": [
                    "Doi chieu giao dich/hoa don theo ngay va so tien.",
                    "Tao but toan dieu chinh o trang so sach neu la ghi thieu.",
                    "Lap to khai bo sung neu doanh thu/chi phi da nop bi lech.",
                ],
            }

        cases = [
            delta_case("bank_vs_book_revenue", bank_in, snapshot["revenue"]["total"], "Doanh thu ngan hang lech so doanh thu", "Tong tien vao ngan hang khac voi doanh thu ghi so."),
            delta_case("invoice_vs_book_revenue", invoice_out, snapshot["revenue"]["total"], "Hoa don dau ra lech doanh thu", "Tong hoa don dau ra khong khop doanh thu ghi nhan."),
            delta_case("ledger_vs_book_revenue", ledger_revenue, snapshot["revenue"]["total"], "But toan doanh thu lech so S1a", "Ledger revenue khac voi doanh thu business_revenue_entries."),
            delta_case("filing_vs_book_revenue", declared_revenue, snapshot["revenue"]["total"], "To khai lech doanh thu ghi so", "Doanh thu tren to khai/draft khac voi so doanh thu."),
            delta_case("bank_out_vs_expense", bank_out, snapshot["expenses"]["total"], "Chi ngan hang lech chi phi", "Tong tien ra ngan hang khac voi chi phi ghi so."),
            delta_case("invoice_in_vs_expense", invoice_in, snapshot["expenses"]["total"], "Hoa don dau vao lech chi phi", "Tong hoa don dau vao khac chi phi duoc ghi nhan."),
            delta_case("filing_vs_book_expense", declared_expense, snapshot["expenses"]["total"], "To khai lech chi phi", "Chi phi tren to khai/draft khac voi so chi phi."),
        ]
        cases = [item for item in cases if item]
        if platform_gross and snapshot["revenue"]["channels"].get("ecommerce") and platform_gross + snapshot["revenue"]["channels"].get("ecommerce", 0) > bank_in * 1.25:
            cases.append(
                {
                    "case_key": "ecommerce_duplicate_risk",
                    "case_type": "channel_attribution",
                    "severity": "high",
                    "status": "open",
                    "title": "Nguy co tinh trung doanh thu san TMĐT",
                    "description": "Doanh thu san va doanh thu kenh ecommerce trong so ghi co the dang bi ghi lap.",
                    "score": 82,
                    "entity_refs": [{"metric": "platform_gross", "amount": _round(platform_gross)}],
                    "suggested_actions": ["Danh dau doanh thu san da khau tru/nop thay.", "Chi tu khai phan doanh thu chua duoc san nop thay."],
                }
            )

        completeness = _clamp(100 - len(cases) * 14 - snapshot["expenses"]["evidence_gap_count"] * 5)
        auto_drafts = []
        for case in cases[:6]:
            if "doanh thu" in str(case.get("title", "")).lower():
                auto_drafts.append({"draft_type": "ledger_adjustment", "case_key": case["case_key"], "book_code": "S1a-HKD", "status": "draft_ai_suggested"})
            elif "chi" in str(case.get("title", "")).lower() or "hoa don dau vao" in str(case.get("title", "")).lower():
                auto_drafts.append({"draft_type": "ledger_adjustment", "case_key": case["case_key"], "book_code": "S2a-HKD", "status": "draft_ai_suggested"})
        result = {
            "summary": {
                "reconciliation_score": _round(completeness),
                "open_case_count": len(cases),
                "high_case_count": len([item for item in cases if item["severity"] == "high"]),
                "data_sources": {
                    "bank_transactions": len(bank_transactions),
                    "invoices": len(invoices),
                    "ledger_entries": len(ledger_entries),
                    "filings": len(filings),
                    "platform_orders": len(platform_orders),
                },
            },
            "totals": totals,
            "cases": cases,
            "auto_drafts": auto_drafts,
            "human_review_queue": [item["case_key"] for item in cases if item["severity"] == "high"],
            "explanation": {
                "reason_codes": ["bank_invoice_ledger_filing_delta", "materiality_threshold", "idempotent_exception_queue"],
                "counterfactual": {"import_missing_sources": "Nhap du sao ke, HĐĐT va so cai se giam false positive doi soat."},
            },
        }
        return self._production_contract(result, {"payload": payload, "totals": totals, "case_count": len(cases)}, "medium" if bank_transactions or invoices else "low", result["explanation"]["reason_codes"], bool(cases))

    def reconciliation_cases(self, dataset: dict[str, Any]) -> dict[str, Any]:
        cases = dataset.get("reconciliation_cases") or []
        open_cases = [item for item in cases if str(item.get("status") or "open").lower() in {"open", "needs_review"}]
        result = {
            "cases": cases,
            "summary": {
                "case_count": len(cases),
                "open_case_count": len(open_cases),
                "high_case_count": len([item for item in open_cases if str(item.get("severity")) == "high"]),
            },
            "next_action": "Xu ly case high truoc khi ky/nop to khai." if open_cases else "Chua co case doi soat dang mo.",
            "explanation": {"reason_codes": ["exception_queue", "human_review_workflow"]},
        }
        return self._production_contract(result, {"case_count": len(cases)}, "medium" if cases else "low", result["explanation"]["reason_codes"], bool(open_cases))

    def channel_attribution(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        platform_orders = dataset.get("platform_orders") or []
        bank_transactions = dataset.get("bank_transactions") or []
        channels = defaultdict(float)
        for channel, amount in (snapshot["revenue"]["channels"] or {}).items():
            channels[str(channel or "direct")] += _float(amount)
        for item in platform_orders:
            channels[str(item.get("platform") or "marketplace").lower()] += _float(item.get("gross_amount"))
        bank_in = 0.0
        for item in bank_transactions:
            if str(item.get("direction") or "in").lower() == "in":
                bank_in += _float(item.get("amount"))
                channel = str(item.get("channel") or "bank_transfer").lower()
                if channel not in channels:
                    channels[channel] += 0.0
        recognized = sum(channels.values())
        duplicate_risks = []
        if channels.get("ecommerce", 0) and sum(_float(item.get("gross_amount")) for item in platform_orders) > 0:
            duplicate_risks.append({"type": "marketplace_book_overlap", "severity": "high", "message": "Doanh thu ecommerce ghi so va order san co the bi cong trung."})
        if bank_in and recognized > bank_in * 1.25:
            duplicate_risks.append({"type": "recognized_exceeds_bank_in", "severity": "medium", "message": "Tong doanh thu theo kenh cao hon tien vao ngan hang dang ke."})
        missing = max(0.0, bank_in - recognized)
        attribution = [
            {
                "channel": key,
                "amount": _round(value),
                "share": _round(value / max(recognized, 1.0), 4),
                "confidence": "medium" if value else "low",
            }
            for key, value in sorted(channels.items(), key=lambda item: item[1], reverse=True)
        ]
        result = {
            "attribution": attribution,
            "bank_in": _round(bank_in),
            "recognized_revenue": _round(recognized),
            "missing_unattributed_revenue": _round(missing),
            "duplicate_risks": duplicate_risks,
            "method_stack": ["weak_supervision", "transaction_text_embedding_ready", "hierarchical_channel_reconciliation"],
            "explanation": {
                "reason_codes": ["platform_order_matching", "bank_text_channel", "duplicate_revenue_detection"],
                "counterfactual": {"map_order_ids": "Gan ma don hang vao sao ke/COD giup phan bo kenh chinh xac hon."},
            },
        }
        confidence = "medium" if platform_orders or bank_transactions else "low"
        return self._production_contract(result, {"payload": payload, "channels": attribution}, confidence, result["explanation"]["reason_codes"], bool(duplicate_risks))

    def tax_reserve_optimize(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.build_snapshot(dataset)
        cash = self.cashflow_delinquency(dataset)
        probabilistic = self.probabilistic_forecast(dataset)
        current_cash = _float(payload.get("current_cash"))
        if current_cash <= 0:
            bank_in = sum(_float(item.get("amount")) for item in dataset.get("bank_transactions") or [] if str(item.get("direction") or "in").lower() == "in")
            bank_out = sum(_float(item.get("amount")) for item in dataset.get("bank_transactions") or [] if str(item.get("direction") or "").lower() == "out")
            current_cash = max(0.0, bank_in - bank_out + snapshot["profit"]["total"])
        projected_tax = cash["projected_tax"]["total_tax"]
        pending = cash["pending_tax_and_debt"]
        penalty_rate_daily = 0.0003
        horizons = [30, 60, 90, 180]
        fan_chart = []
        schedules = []
        for days in horizons:
            interval_idx = min(max(days // 30 - 1, 0), max(0, len(probabilistic["intervals"]) - 1))
            interval = probabilistic["intervals"][interval_idx] if probabilistic["intervals"] else {"p10": 0, "p50": 0, "p90": 0}
            expected_cash = current_cash + _float(interval.get("profit_p50")) - pending * min(1.0, days / 90.0)
            fan_chart.append({"horizon_days": days, "cash_p10": _round(expected_cash - _float(interval.get("p90")) * 0.18), "cash_p50": _round(expected_cash), "cash_p90": _round(expected_cash + _float(interval.get("p90")) * 0.12)})
        reserve_rate = _clamp((projected_tax + pending) / max(snapshot["revenue"]["projected_year_end"], 1.0), 0.05, 0.35)
        immediate = min(current_cash * 0.45, pending)
        remaining = max(0.0, pending - immediate)
        schedules.extend(
            [
                {"date_offset_days": 0, "amount": _round(immediate), "objective": "Giam tien cham nop va hazard 30 ngay"},
                {"date_offset_days": 30, "amount": _round(remaining * 0.55), "objective": "Duy tri buffer dong tien"},
                {"date_offset_days": 60, "amount": _round(remaining * 0.45), "objective": "Tat toan no con lai neu duoc chap nhan phan ky"},
            ]
        )
        penalty_avoided = remaining * penalty_rate_daily * 60
        result = {
            "current_cash": _round(current_cash),
            "recommended_reserve_rate": _round(reserve_rate, 4),
            "monthly_reserve_amount": _round(snapshot["revenue"]["avg_monthly"] * reserve_rate),
            "optimized_payment_schedule": schedules,
            "cash_fan_chart": fan_chart,
            "expected_penalty_avoided": _round(penalty_avoided),
            "risk_after_plan": "low" if pending and immediate >= pending * 0.4 else cash["risk_level"],
            "method_stack": ["stochastic_cashflow_simulation", "conformal_forecast", "penalty_aware_payment_optimization"],
            "explanation": {
                "reason_codes": ["tax_due", "cash_buffer", "late_payment_penalty", "probabilistic_cashflow"],
                "counterfactual": {"increase_reserve": "Tang ty le du phong hang thang giup giam xac suat thieu tien khi den han."},
            },
        }
        return self._production_contract(result, {"payload": payload, "snapshot": snapshot}, cash["model"]["confidence"], result["explanation"]["reason_codes"], bool(pending))

    def supplier_account_risk(self, dataset: dict[str, Any]) -> dict[str, Any]:
        graph = self.graph_risk(dataset)
        bank_transactions = dataset.get("bank_transactions") or []
        supplier_accounts: dict[str, dict[str, Any]] = {}
        for tx in bank_transactions:
            if str(tx.get("direction") or "").lower() != "out":
                continue
            key = str(tx.get("counterparty_tax_code") or tx.get("counterparty_name") or "unknown")
            meta = _json(tx.get("metadata_json"))
            account = str(meta.get("counterparty_account") or meta.get("beneficiary_account") or tx.get("bank_account") or "unknown")
            stats = supplier_accounts.setdefault(key, {"accounts": set(), "amount": 0.0, "count": 0, "name": tx.get("counterparty_name")})
            stats["accounts"].add(account)
            stats["amount"] += _float(tx.get("amount"))
            stats["count"] += 1
        alerts = []
        for key, stats in supplier_accounts.items():
            account_count = len(stats["accounts"])
            score = min(95.0, account_count * 24 + min(30.0, stats["amount"] / 200_000_000 * 12))
            if account_count >= 2 or score >= 55:
                alerts.append(
                    {
                        "supplier_key": key,
                        "partner_name": stats.get("name") or key,
                        "account_count": account_count,
                        "payment_amount": _round(stats["amount"]),
                        "risk_score": _round(score),
                        "severity": self._risk_level(score),
                        "reason": "Nha cung cap co nhieu tai khoan nhan tien hoac dong tien ra lon.",
                    }
                )
        alerts.sort(key=lambda item: item["risk_score"], reverse=True)
        result = {
            "account_change_alerts": alerts[:12],
            "graph_summary": graph["summary"],
            "centrality": graph["centrality"],
            "recommended_controls": [
                "Xac minh van ban thay doi tai khoan nhan tien cua nha cung cap.",
                "Doi chieu MST, ten don vi, hoa don va sao ke truoc khi tinh chi phi duoc tru.",
                "Danh dau giao dich can human-confirm neu risk_score >= 70.",
            ],
            "explanation": {
                "reason_codes": ["beneficiary_account_change", "outgoing_payment_graph", "supplier_centrality"],
                "counterfactual": {"supplier_confirmation": "Tai len xac nhan tai khoan cua nha cung cap de giam rui ro."},
            },
        }
        return self._production_contract(result, {"alerts": alerts, "graph": graph["summary"]}, "medium" if bank_transactions else "low", result["explanation"]["reason_codes"], bool(alerts))

    def inventory_analyze(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        movements = list(dataset.get("inventory_movements") or [])
        for item in payload.get("movements") or []:
            if isinstance(item, dict):
                movements.append(item)
        line_items = dataset.get("einvoice_line_items") or []
        expenses = dataset.get("expense_entries") or []
        by_sku: dict[str, dict[str, Any]] = {}
        for item in movements:
            sku = str(item.get("sku") or item.get("item_name") or "unknown")
            qty = _float(item.get("quantity"))
            total = _float(item.get("total_cost") or (_float(item.get("unit_cost")) * qty))
            movement_type = str(item.get("movement_type") or "in").lower()
            bucket = by_sku.setdefault(sku, {"sku": sku, "in_qty": 0.0, "out_qty": 0.0, "cost": 0.0})
            if movement_type in {"out", "sale", "cogs"}:
                bucket["out_qty"] += abs(qty)
            else:
                bucket["in_qty"] += abs(qty)
                bucket["cost"] += abs(total)
        line_cogs = sum(_float(item.get("amount")) for item in line_items if _float(item.get("amount")) > 0)
        material_expense = sum(_float(item.get("amount")) for item in expenses if str(item.get("category") or "").lower() in {"materials", "inventory", "goods"})
        revenue = self.build_snapshot(dataset)["revenue"]["total"]
        cogs_estimate = max(line_cogs, material_expense, sum(item["cost"] for item in by_sku.values()))
        gross_margin = (revenue - cogs_estimate) / max(revenue, 1.0)
        alerts = []
        if revenue > 0 and gross_margin < 0.08:
            alerts.append({"severity": "high", "type": "low_margin", "message": "Bien loi nhuan gop thap bat thuong, can doi chieu gia von/chung tu dau vao."})
        if material_expense > line_cogs * 1.35 and material_expense > 0:
            alerts.append({"severity": "medium", "type": "missing_input_invoice", "message": "Chi phi hang hoa cao hon line item HĐĐT dau vao; co the thieu hoa don/bang ke."})
        for sku, item in by_sku.items():
            if item["out_qty"] > item["in_qty"] * 1.15 and item["out_qty"] > 0:
                alerts.append({"severity": "medium", "type": "negative_stock_risk", "sku": sku, "message": "Xuat ban vuot nhap kho trong du lieu hien co."})
        result = {
            "inventory_summary": list(by_sku.values())[:80],
            "cogs_estimate": _round(cogs_estimate),
            "gross_margin": _round(gross_margin, 4),
            "alerts": alerts,
            "method_stack": ["inventory_flow_reconciliation", "cogs_margin_anomaly", "peer_cohort_ready"],
            "explanation": {
                "reason_codes": ["line_item_cogs", "inventory_flow", "margin_anomaly"],
                "counterfactual": {"import_purchase_lines": "Nhap line item hoa don mua vao de tinh gia von/tong ton kho chinh xac hon."},
            },
        }
        return self._production_contract(result, {"payload": payload, "cogs": cogs_estimate}, "medium" if movements or line_items else "low", result["explanation"]["reason_codes"], bool(alerts))

    def evidence_bundle(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        purpose = str(payload.get("purpose") or "tax_audit_explanation")
        documents = dataset.get("documents") or []
        invoices = dataset.get("invoices") or []
        bank_transactions = dataset.get("bank_transactions") or []
        expenses = dataset.get("expense_entries") or []
        claims = dataset.get("claims") or []
        sections = [
            {"key": "identity", "title": "Thong tin nguoi nop thue", "items": [dataset.get("profile") or {}], "required": True},
            {"key": "filings", "title": "To khai/ho so da nop", "items": dataset.get("filings") or [], "required": purpose in {"appeal", "tax_audit_explanation"}},
            {"key": "einvoices", "title": "Hoa don dien tu lien quan", "items": invoices[:80], "required": True},
            {"key": "bank", "title": "Sao ke/chung tu thanh toan", "items": bank_transactions[:80], "required": True},
            {"key": "expenses", "title": "Chi phi va bang ke", "items": expenses[:80], "required": True},
            {"key": "documents", "title": "Hop dong/anh chung tu/bang ke", "items": documents[:80], "required": False},
            {"key": "claims", "title": "Quyet dinh/khieu nai lien quan", "items": claims[:20], "required": purpose == "appeal"},
        ]
        missing = [item["title"] for item in sections if item["required"] and not item["items"]]
        score = _round(_clamp(100 - len(missing) * 18 - sum(1 for expense in expenses if not expense.get("has_invoice")) * 4))
        result = {
            "purpose": purpose,
            "bundle_score": score,
            "readiness": "ready" if score >= 82 else "needs_review" if score >= 55 else "blocked",
            "sections": [{"key": item["key"], "title": item["title"], "item_count": len(item["items"]), "required": item["required"]} for item in sections],
            "missing_evidence": missing,
            "draft_outline": [
                "Tom tat van de va ky thue lien quan.",
                "Bang doi chieu doanh thu/hoa don/sao ke/so sach.",
                "Danh sach chi phi co chung tu va chi phi can bang ke thay the.",
                "Can cu phap ly/citation va kien nghi xu ly.",
            ],
            "export_plan": {"formats": ["pdf", "xlsx", "zip"], "status": "draft_ai_suggested"},
            "explanation": {
                "reason_codes": ["evidence_completeness", "document_grouping", "appeal_audit_bundle"],
                "counterfactual": {"upload_bank_invoice": "Bo sung sao ke va hoa don lien quan de tang diem bundle."},
            },
        }
        return self._production_contract(result, {"purpose": purpose, "score": score}, "medium", result["explanation"]["reason_codes"], score < 82)

    def legal_change_impact(self, payload: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        impact = self.policy_impact(payload, dataset)
        question = payload.get("question") or "Van ban moi anh huong den nganh/doanh thu/kenh ban nhu the nao?"
        rag = self.legal_graphrag({"question": question}, dataset)
        timeline = []
        for item in impact.get("citations") or []:
            timeline.append(
                {
                    "source_key": item.get("key"),
                    "title": item.get("title"),
                    "effective_from": item.get("effective_from"),
                    "article_ref": item.get("article_ref"),
                    "source_url": item.get("source_url"),
                }
            )
        result = {
            "context": impact.get("context"),
            "impacts": impact.get("impacts") or [],
            "citations": rag.get("citations") or impact.get("citations") or [],
            "effective_timeline": timeline,
            "needs_verification": rag.get("needs_verification", True),
            "change_alerts": [
                {
                    "severity": item.get("severity"),
                    "title": item.get("title"),
                    "message": item.get("message"),
                    "action": "Mo GraphRAG/citation va xac nhan voi van ban goc truoc khi nop ho so.",
                }
                for item in impact.get("impacts") or []
            ],
            "explanation": {
                "reason_codes": ["versioned_policy_rule", "profile_policy_matching", "citation_guard"],
                "counterfactual": {"add_industry_channel": "Nhap ro nganh/kenh ban/doanh thu de loc van ban lien quan hon."},
            },
        }
        return self._production_contract(result, {"payload": payload, "impact_count": len(result["impacts"])}, "medium", result["explanation"]["reason_codes"], True)

    def model_governance_production(self, dataset: dict[str, Any]) -> dict[str, Any]:
        base = self.model_governance(dataset)
        consents = {item.get("consent_key"): item.get("status") for item in dataset.get("privacy_consents") or []}
        connector_sources = {
            "bank_transactions": len(dataset.get("bank_transactions") or []),
            "einvoice_line_items": len(dataset.get("einvoice_line_items") or []),
            "platform_orders": len(dataset.get("platform_orders") or []),
            "inventory_movements": len(dataset.get("inventory_movements") or []),
            "reconciliation_cases": len(dataset.get("reconciliation_cases") or []),
        }
        gates = [
            {"gate": "database_source", "pass": any(connector_sources.values()), "required_for_promotion": True},
            {"gate": "bank_or_einvoice_consent", "pass": consents.get("bank_training") == "granted" or consents.get("ocr_training") == "granted", "required_for_promotion": True},
            {"gate": "drift_baseline", "pass": base["drift"]["score"] < 70, "required_for_promotion": True},
            {"gate": "feedback_loop", "pass": base["feedback_quality"]["feedback_count"] >= 20, "required_for_promotion": False},
        ]
        result = {
            **base,
            "production_gates": gates,
            "connector_readiness": connector_sources,
            "promotion_policy": {
                "synthetic_artifacts": "cannot_promote_by_default",
                "requires_model_card": True,
                "requires_quality_report": True,
                "requires_drift_baseline": True,
                "requires_privacy_consent": True,
            },
            "tracked_training_tracks": [
                "reconciliation_ranker",
                "channel_attribution",
                "tax_reserve_optimizer",
                "supplier_graph_risk",
                "inventory_anomaly",
                "evidence_classifier",
                "reminder_bandit",
            ],
            "explanation": {
                "reason_codes": ["production_gate", "privacy_consent", "connector_coverage", "drift_baseline"],
                "counterfactual": {"grant_consent_and_import": "Nhap Bank/HĐĐT va cap consent training de du dieu kien promote."},
            },
        }
        return self._production_contract(result, {"connector_sources": connector_sources, "gates": gates}, "medium", result["explanation"]["reason_codes"], True)

    def recommendations(self, dataset: dict[str, Any], snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        snapshot = snapshot or self.build_snapshot(dataset)
        items: list[dict[str, Any]] = []
        threshold_alert = snapshot["revenue"]["threshold"]["alert"]
        if threshold_alert in {"near_500m", "taxable", "einvoice_mandatory", "group3"}:
            items.append(
                self._rec(
                    "revenue_threshold_watch",
                    "Theo doi nguong doanh thu",
                    "high",
                    f"Doanh thu luy ke {snapshot['revenue']['total']:,.0f} VND dang o trang thai {threshold_alert}.",
                    "Mo lich va chuan bi ke khai",
                    "business_calendar.html",
                )
            )
        if snapshot["expenses"]["cash_payment_violations"] > 0:
            items.append(
                self._rec(
                    "cash_payment_fix",
                    "Xu ly giao dich tien mat tu 5 trieu",
                    "high",
                    "Co chi phi thanh toan tien mat vuot nguong, nguy co bi loai khi tinh chi phi duoc tru.",
                    "Kiem tra chi phi",
                    "business_expenses.html",
                )
            )
        if snapshot["expenses"]["evidence_gap_count"] > 0:
            items.append(
                self._rec(
                    "evidence_gap",
                    "Bo sung chung tu chi phi",
                    "medium",
                    f"{snapshot['expenses']['evidence_gap_count']} khoan chi can hoa don/bang ke/chung tu thanh toan.",
                    "Tai chung tu",
                    "business_accounting.html",
                )
            )
        if snapshot["compliance"]["deadline_overdue"] > 0 or snapshot["compliance"]["deadline_soon"] > 0:
            items.append(
                self._rec(
                    "deadline_attention",
                    "Uu tien han thue sap den",
                    "high",
                    f"{snapshot['compliance']['deadline_soon']} han sap den va {snapshot['compliance']['deadline_overdue']} han qua han.",
                    "Xem deadline",
                    "business_calendar.html",
                )
            )
        if snapshot["compliance"]["passport_ban"]["level"] in {"warning", "critical"}:
            items.append(
                self._rec(
                    "debt_repayment_plan",
                    "Lap ke hoach xu ly no thue",
                    "high",
                    snapshot["compliance"]["passport_ban"]["message"],
                    "Mo phan ky no",
                    "business_debts.html",
                )
            )
        if snapshot["invoices"]["risky_count"] > 0 or snapshot["invoices"]["duplicate_count"] > 0:
            items.append(
                self._rec(
                    "invoice_risk_review",
                    "Ra soat hoa don dau vao",
                    "high",
                    "Nhat ky co hoa don rui ro/trung lap can doi chieu truoc khi ke khai.",
                    "Kiem tra hoa don",
                    "business_invoices.html",
                )
            )
        if snapshot["data_quality_score"] < 85:
            items.append(
                self._rec(
                    "profile_data_quality",
                    "Hoan thien ho so taxpayer",
                    "medium",
                    f"Diem chat luong du lieu hien tai {snapshot['data_quality_score']:.0f}/100.",
                    "Cap nhat ho so",
                    "business_profile.html",
                )
            )
        if not items:
            items.append(
                self._rec(
                    "monthly_review",
                    "Duy tri so lieu hang thang",
                    "low",
                    "Chua co canh bao lon. Nen ghi doanh thu, chi phi va luu chung tu theo thang.",
                    "Mo so ke toan",
                    "business_accounting.html",
                )
            )
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        items.sort(key=lambda item: priority_rank.get(item["priority"], 9))
        return {"recommendations": items, "model": self.model_meta(snapshot, confidence="medium")}

    def _production_contract(
        self,
        result: dict[str, Any],
        payload: dict[str, Any],
        confidence: str,
        reason_codes: list[str],
        needs_human_confirmation: bool,
    ) -> dict[str, Any]:
        meta = self.model_meta(payload, confidence=confidence)
        explanation = result.get("explanation") or {}
        explanation.setdefault("reason_codes", reason_codes)
        return {
            **result,
            "model_name": meta["model_name"],
            "model_version": meta["model_version"],
            "confidence": meta["confidence"],
            "confidence_score": meta["confidence_score"],
            "input_hash": meta["input_hash"],
            "reason_codes": reason_codes,
            "needs_human_confirmation": bool(needs_human_confirmation),
            "explanation": explanation,
            "model": meta,
        }

    def _heat(self, key: str, label: str, raw_score: float) -> dict[str, Any]:
        score = _round(_clamp(raw_score))
        return {"key": key, "label": label, "score": score, "severity": self._risk_level(score)}

    def _risk_level(self, score: float) -> str:
        value = _float(score)
        if value >= 70:
            return "high"
        if value >= 40:
            return "medium"
        return "low"

    def _amount_from_fields(self, fields: dict[str, Any]) -> float:
        if "amount" in fields:
            return _float(fields.get("amount"))
        text = str(fields.get("amount_text") or "")
        if not text:
            return 0.0
        normalized = re.sub(r"[^\d,\.]", "", text)
        if "," in normalized and "." in normalized:
            normalized = normalized.replace(".", "").replace(",", ".")
        elif "," in normalized:
            normalized = normalized.replace(",", "")
        else:
            normalized = normalized.replace(",", "")
        return _float(normalized)

    def _bookkeeping_entry_from_fields(self, fields: dict[str, Any], description: str, direction: str) -> dict[str, Any]:
        amount = self._amount_from_fields(fields)
        if direction == "revenue":
            return {
                "book_code": "S1a-HKD",
                "entry_type": "revenue",
                "amount": _round(amount),
                "channel": "direct",
                "description": description or str(fields.get("invoice_number") or "Doanh thu tu chung tu"),
                "source": "ocr_reconcile",
            }
        category = self._category_from_text(description, "expense")
        evaluation = evaluate_expense({"amount": amount, "description": description, "category": category, "has_invoice": bool(fields.get("invoice_number"))})
        return {
            "book_code": "S2a-HKD",
            "entry_type": "expense",
            "amount": _round(amount),
            "category": category,
            "description": description or str(fields.get("invoice_number") or "Chi phi tu chung tu"),
            "deductible_status": evaluation["status"],
            "required_evidence": evaluation["required_evidence"],
            "source": "ocr_reconcile",
        }

    def _infer_direction(self, text: str, amount: float) -> str:
        lower = text.lower()
        if any(token in lower for token in ["ban hang", "doanh thu", "thu tien", "sales", "revenue"]):
            return "revenue"
        if amount < 0:
            return "expense"
        return "expense"

    def _category_from_text(self, text: str, direction: str) -> str:
        lower = text.lower()
        if direction == "revenue":
            return "sales"
        if any(token in lower for token in ["thue mat bang", "mat bang", "rent"]):
            return "rent"
        if any(token in lower for token in ["luong", "nhan vien", "salary", "wage"]):
            return "salary"
        if any(token in lower for token in ["quang cao", "ads", "marketing"]):
            return "advertising"
        if any(token in lower for token in ["dien", "nuoc", "internet", "utility"]):
            return "utilities"
        if any(token in lower for token in ["nguyen lieu", "hang hoa", "mua hang", "materials", "inventory"]):
            return "materials"
        return "other"

    def _precheck_fixes(self, issues: list[dict[str, Any]]) -> list[str]:
        fixes = []
        for item in issues:
            kind = item.get("type")
            if kind == "revenue_mismatch":
                fixes.append("Doi chieu doanh thu khai bao voi so S1a va doanh thu tung kenh.")
            elif kind == "invoice_revenue_gap":
                fixes.append("Kiem tra hoa don dau ra, doanh thu san TMDT va giao dich da bi tinh trung/bo sot.")
            elif kind == "cash_payment":
                fixes.append("Loai khoan tien mat tu 5 trieu khoi chi phi duoc tru hoac bo sung xu ly hop le.")
            elif kind == "evidence_gap":
                fixes.append("Tai hoa don, bang ke thay the va chung tu thanh toan cho cac khoan chi can bo sung.")
            elif kind == "group_change":
                fixes.append("Cap nhat nhom HKD, lich ke khai va dieu kien hoa don dien tu.")
        return fixes or ["Ho so khong co loi lon; tiep tuc luu chung tu va nop dung han."]

    def _impact(self, source_key: str, title: str, severity: str, message: str) -> dict[str, Any]:
        source = next((item for item in BASELINE_SOURCES if item["key"] == source_key), {"key": source_key, "title": source_key})
        return {
            "source_key": source_key,
            "title": title,
            "severity": severity,
            "message": message,
            "citation": {
                "key": source.get("key"),
                "title": source.get("title"),
                "source_url": source.get("source_url"),
                "article_ref": source.get("article_ref"),
                "effective_from": source.get("effective_from"),
            },
        }

    def _alerts(self, snapshot: dict[str, Any], scores: dict[str, float]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        threshold = snapshot["revenue"]["threshold"]
        if threshold["alert"] != "normal":
            alerts.append(
                {
                    "type": threshold["alert"],
                    "severity": "high" if threshold["alert"] in {"einvoice_mandatory", "group3"} else "medium",
                    "title": "Canh bao nguong doanh thu",
                    "message": f"Con {threshold['distance_to_next_threshold']:,.0f} VND den nguong ke tiep.",
                    "source": "revenue_threshold_model",
                }
            )
        if snapshot["compliance"]["passport_ban"]["level"] != "normal":
            alerts.append(
                {
                    "type": "debt_enforcement",
                    "severity": snapshot["compliance"]["passport_ban"]["level"],
                    "title": "Rui ro cuong che/tam hoan xuat canh",
                    "message": snapshot["compliance"]["passport_ban"]["message"],
                    "source": "debt_risk_model",
                }
            )
        if snapshot["expenses"]["cash_payment_violations"]:
            alerts.append(
                {
                    "type": "cash_payment",
                    "severity": "high",
                    "title": "Chi phi tien mat vuot nguong",
                    "message": "Giao dich tien mat tu 5 trieu co nguy co khong duoc tru.",
                    "source": "expense_rule_model",
                }
            )
        if scores["cashflow"] < 45:
            alerts.append(
                {
                    "type": "cashflow",
                    "severity": "medium",
                    "title": "Dong tien can theo doi",
                    "message": "Loi nhuan/du no hien tai co the gay ap luc thanh toan ky toi.",
                    "source": "cashflow_baseline",
                }
            )
        return alerts

    def _rec(self, key: str, title: str, priority: str, reason: str, action: str, page: str) -> dict[str, Any]:
        return {
            "key": key,
            "title": title,
            "priority": priority,
            "reason": reason,
            "action_label": action,
            "target_page": page,
            "source_model": self.model_name,
            "confidence": "medium",
        }

    def _reason_codes(self, snapshot: dict[str, Any]) -> list[str]:
        reasons = []
        if snapshot["revenue"]["threshold"]["alert"] != "normal":
            reasons.append(f"revenue_{snapshot['revenue']['threshold']['alert']}")
        if snapshot["expenses"]["cash_payment_violations"]:
            reasons.append("cash_payment_ge_5m")
        if snapshot["expenses"]["evidence_gap_count"]:
            reasons.append("expense_evidence_gap")
        if snapshot["invoices"]["risky_count"] or snapshot["invoices"]["duplicate_count"]:
            reasons.append("invoice_risk_or_duplicate")
        if snapshot["compliance"]["debt_total"]:
            reasons.append("tax_debt_pressure")
        if snapshot["data_quality_score"] < 85:
            reasons.append("profile_data_quality")
        return reasons or ["normal_monitoring"]

    def _nba_impact(self, key: str | None) -> str:
        mapping = {
            "revenue_threshold_watch": "Giam rui ro vuot nguong ma chua chuan bi ke khai/HDDT.",
            "cash_payment_fix": "Giam nguy co bi loai chi phi do thanh toan tien mat tu 5 trieu.",
            "evidence_gap": "Tang kha nang chi phi duoc chap nhan khi precheck to khai.",
            "deadline_attention": "Giam phat cham nop/cham ke khai va ap luc dong tien.",
            "debt_repayment_plan": "Giam hazard no thue, cuong che hoac tam hoan xuat canh.",
            "invoice_risk_review": "Giam rui ro hoa don dau vao bi tu choi/bi nghi van.",
            "profile_data_quality": "Tang do tin cay cua forecast, RAG va auto-fill ho so.",
            "monthly_review": "Duy tri du lieu tot cho sổ sach va canh bao som.",
        }
        return mapping.get(key or "", "Cai thien chat luong tuan thu va du lieu cho lan du bao tiep theo.")

    def _expense_action(self, evaluation: dict[str, Any]) -> str:
        status = evaluation.get("status")
        if status == "non_deductible":
            return "Khong dua vao chi phi duoc tru; can dieu chinh phuong thuc/chung tu neu co the."
        if status == "needs_invoice":
            return "Yeu cau hoa don dien tu hop le hoac chuyen sang bang ke neu thuoc truong hop duoc phep."
        if status == "needs_evidence":
            return "Lap bang ke va dinh kem chung tu thanh toan truoc khi ke khai."
        return "Luu hoa don, hop dong va chung tu thanh toan vao kho chung tu."

    def _suggest_document_category(self, filename: str, text: str, doc_type: str) -> str:
        lower = f"{filename} {text[:500]} {doc_type}".lower()
        if "hoa don" in lower or "invoice" in lower or "hddt" in lower:
            return "invoice"
        if "hop dong" in lower or "contract" in lower:
            return "contract"
        if "bang ke" in lower:
            return "no_invoice_statement"
        if "thanh toan" in lower or "payment" in lower:
            return "payment_evidence"
        return doc_type or "evidence"

    def _series_trend(self, values: list[float]) -> str:
        non_zero = [float(value or 0) for value in values if float(value or 0) != 0]
        if len(non_zero) < 2:
            return "insufficient_data"
        first = sum(non_zero[: max(1, len(non_zero) // 2)]) / max(1, len(non_zero[: max(1, len(non_zero) // 2)]))
        second = sum(non_zero[max(1, len(non_zero) // 2) :]) / max(1, len(non_zero[max(1, len(non_zero) // 2) :]))
        if second > first * 1.12:
            return "up"
        if second < first * 0.88:
            return "down"
        return "stable"

    def _spike_anomalies(self, metric: str, values: list[float], year: int) -> list[dict[str, Any]]:
        active = [float(value or 0) for value in values if float(value or 0) > 0]
        if len(active) < 3:
            return []
        avg = sum(active) / len(active)
        variance = sum((value - avg) ** 2 for value in active) / len(active)
        std = math.sqrt(variance)
        if std <= 0:
            return []
        anomalies = []
        for idx, value in enumerate(values, start=1):
            if value > avg + 1.8 * std:
                anomalies.append(
                    {
                        "type": f"{metric}_spike",
                        "severity": "medium",
                        "period": f"{year}-{idx:02d}",
                        "title": f"{metric.title()} tang bat thuong",
                        "description": f"{metric} thang {idx:02d} cao hon dang ke so voi mat bang cac thang co du lieu.",
                        "recommended_action": "Doi chieu hoa don/chung tu va nguyen nhan mua ban theo mua vu.",
                    }
                )
        return anomalies

    def _simple_progressive_profit_tax(self, profit: float) -> dict[str, Any]:
        brackets = [
            (60_000_000, 0.05),
            (120_000_000, 0.10),
            (216_000_000, 0.15),
            (384_000_000, 0.20),
            (float("inf"), 0.25),
        ]
        remaining = max(0.0, profit)
        previous = 0.0
        tax = 0.0
        parts = []
        for cap, rate in brackets:
            taxable = max(0.0, min(remaining, cap - previous))
            if taxable:
                amount = taxable * rate
                parts.append({"from": previous, "to": cap if math.isfinite(cap) else None, "rate": rate, "taxable": _round(taxable), "tax": _round(amount)})
                tax += amount
            remaining -= taxable
            previous = cap
            if remaining <= 0:
                break
        return {
            "profit": _round(profit),
            "tax": _round(tax),
            "effective_rate": _round(tax / profit if profit else 0.0, 4),
            "brackets": parts,
            "note": "Sandbox estimate de so sanh phuong phap; can gan voi quy dinh TNCN chinh thuc khi trien khai production.",
        }
