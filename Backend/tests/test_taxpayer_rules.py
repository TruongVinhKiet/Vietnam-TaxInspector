import sys
from datetime import date
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.taxpayer_rules import (
    build_deadlines,
    build_ics,
    e_invoice_requirement,
    evaluate_expense,
    legal_answer,
    passport_ban_risk,
    revenue_threshold_summary,
)


def test_deadlines_follow_household_groups() -> None:
    group1 = build_deadlines(2026, 1, today=date(2026, 5, 29))
    assert {item["form_code"] for item in group1} >= {"01/TKN-CNKD", "LPMB"}
    assert any(item["due_date"] == "2026-07-31" for item in group1)
    assert any(item["due_date"] == "2027-01-31" for item in group1)

    group2 = build_deadlines(2026, 2, today=date(2026, 5, 29))
    assert any(item["code"] == "2026-q1-01cnkd" and item["due_date"] == "2026-04-30" for item in group2)
    assert any(item["code"] == "2026-q4-01cnkd" and item["due_date"] == "2027-01-31" for item in group2)

    group3 = build_deadlines(2026, 3, today=date(2026, 5, 29))
    assert len([item for item in group3 if item["form_code"] == "01/CNKD"]) == 12
    assert any(item["due_date"] == "2026-02-20" for item in group3)


def test_einvoice_thresholds_and_revenue_alerts() -> None:
    assert e_invoice_requirement(499_000_000)["status"] == "single_issue_on_request"
    assert e_invoice_requirement(700_000_000)["status"] == "voluntary_recommended"
    assert e_invoice_requirement(1_000_000_000)["status"] == "mandatory"

    near = revenue_threshold_summary(460_000_000, 480_000_000)
    assert near["alert"] == "near_500m"
    assert near["group"]["group"] == 1

    large = revenue_threshold_summary(3_100_000_000, 3_100_000_000)
    assert large["alert"] == "group3"
    assert large["group"]["group"] == 3


def test_debt_exit_ban_and_expense_rule_engine() -> None:
    risk = passport_ban_risk(55_000_000, 121)
    assert risk["triggered"] is True
    assert risk["level"] == "critical"

    cash_expense = evaluate_expense(
        {
            "description": "Mua hang dau vao",
            "amount": 5_000_000,
            "payment_method": "cash",
            "has_invoice": True,
        }
    )
    assert cash_expense["deductible"] is False
    assert cash_expense["status"] == "non_deductible"

    no_invoice = evaluate_expense(
        {
            "description": "Mua nong san",
            "amount": 3_000_000,
            "payment_method": "bank_transfer",
            "has_invoice": False,
            "no_invoice_case": "farm_producer",
        }
    )
    assert no_invoice["deductible"] is True
    assert no_invoice["status"] == "needs_evidence"


def test_legal_answer_requires_citations_and_ics_export_shape() -> None:
    answer = legal_answer("Chi phi tien mat tu 5 trieu co duoc tru khong?")
    assert answer["status"] == "grounded"
    assert answer["citations"]
    assert any(item["key"] == "nd68_2026" for item in answer["citations"])

    ics = build_ics(build_deadlines(2026, 2))
    assert "BEGIN:VCALENDAR" in ics
    assert "20260430" in ics
