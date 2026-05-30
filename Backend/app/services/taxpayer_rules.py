# -*- coding: utf-8 -*-
"""
Pure taxpayer rule helpers for household-business workflows.

These functions intentionally avoid database and network dependencies so policy
thresholds can be tested independently from the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import ceil
from typing import Any


VND_500M = 500_000_000.0
VND_1B = 1_000_000_000.0
VND_3B = 3_000_000_000.0
PASSPORT_BAN_DEBT_THRESHOLD = 50_000_000.0
PASSPORT_BAN_DAY_THRESHOLD = 120
CASH_PAYMENT_LIMIT = 5_000_000.0
LATE_PAYMENT_RATE_PER_DAY = 0.0003


BASELINE_SOURCES: list[dict[str, Any]] = [
    {
        "key": "nd68_2026",
        "title": "Nghi dinh 68/2026/ND-CP ve chinh sach thue, quan ly thue doi voi HKD",
        "source_url": "https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-68-2026-nd-cp-quy-dinh-ve-chinh-sach-thue-quan-ly-thue-voi-ho-kinh-doanh-119260306102906789.htm",
        "effective_from": "2026-03-05",
        "category": "household_tax",
        "article_ref": "Chuong II-III",
        "confidence": 0.92,
    },
    {
        "key": "tt18_2026",
        "title": "Thong tu 18/2026/TT-BTC ve ho so, thu tuc quan ly thue HKD",
        "source_url": "https://vanban.chinhphu.vn/?docid=217174&pageid=27160",
        "effective_from": "2026-03-05",
        "category": "filing_procedure",
        "article_ref": "Ho so, thu tuc",
        "confidence": 0.9,
    },
    {
        "key": "tt152_2025",
        "title": "Thong tu 152/2025/TT-BTC huong dan ke toan cho HKD",
        "source_url": "https://vanban.chinhphu.vn/?docid=216533&pageid=27160",
        "effective_from": "2026-01-01",
        "category": "accounting",
        "article_ref": "So ke toan HKD",
        "confidence": 0.9,
    },
    {
        "key": "pit_109_2025",
        "title": "Luat Thue thu nhap ca nhan 109/2025/QH15",
        "source_url": "https://vanban.chinhphu.vn/?docid=216495&pageid=27160",
        "effective_from": "2026-07-01",
        "category": "personal_income_tax",
        "article_ref": "Bieu thue TNCN",
        "confidence": 0.9,
    },
    {
        "key": "nq110_2025",
        "title": "Nghi quyet 110/2025/UBTVQH15 ve giam tru gia canh TNCN",
        "source_url": "https://vanban.chinhphu.vn/?docid=215927&pageid=27160",
        "effective_from": "2026-01-01",
        "category": "personal_income_tax",
        "article_ref": "Giam tru gia canh",
        "confidence": 0.9,
    },
    {
        "key": "nd70_2025",
        "title": "Nghi dinh 70/2025/ND-CP sua doi quy dinh hoa don, chung tu",
        "source_url": "https://vanban.chinhphu.vn/?classid=1&docid=213179&pageid=27160&typegroupid=4",
        "effective_from": "2025-06-01",
        "category": "einvoice",
        "article_ref": "Hoa don dien tu",
        "confidence": 0.88,
    },
    {
        "key": "nd49_2025",
        "title": "Nghi dinh 49/2025/ND-CP ve nguong tam hoan xuat canh do no thue",
        "source_url": "https://xaydungchinhsach.chinhphu.vn/chinh-phu-quy-dinh-nguong-tien-thue-no-bi-tam-hoan-xuat-canh-119250301065602164.htm",
        "effective_from": "2025-02-28",
        "category": "debt_enforcement",
        "article_ref": "Nguong no thue va thoi gian no",
        "confidence": 0.92,
    },
]


INDUSTRY_TAX_RATES: dict[str, dict[str, Any]] = {
    "commerce": {
        "name": "Phan phoi, ban buon, ban le hang hoa",
        "isic_hint": "G",
        "gtgt_rate_pct": 1.0,
        "tncn_rate_pct": 0.5,
        "source": "TT40/2021/TT-BTC - phu luc ty le tren doanh thu",
    },
    "service": {
        "name": "Dich vu, moi gioi, dai ly",
        "isic_hint": "I, J, M, N, S",
        "gtgt_rate_pct": 5.0,
        "tncn_rate_pct": 2.0,
        "source": "TT40/2021/TT-BTC - phu luc ty le tren doanh thu",
    },
    "manufacture": {
        "name": "San xuat, van tai, dich vu co gan voi hang hoa, xay dung co bao thau",
        "isic_hint": "C, H, F",
        "gtgt_rate_pct": 3.0,
        "tncn_rate_pct": 1.5,
        "source": "TT40/2021/TT-BTC - phu luc ty le tren doanh thu",
    },
    "other": {
        "name": "Hoat dong kinh doanh khac",
        "isic_hint": "Khac",
        "gtgt_rate_pct": 2.0,
        "tncn_rate_pct": 1.0,
        "source": "TT40/2021/TT-BTC - phu luc ty le tren doanh thu",
    },
    "rental": {
        "name": "Cho thue tai san",
        "isic_hint": "L, N77",
        "gtgt_rate_pct": 5.0,
        "tncn_rate_pct": 5.0,
        "source": "TT40/2021/TT-BTC - dieu 9",
    },
}


NO_INVOICE_ALLOWED_CASES: list[dict[str, Any]] = [
    {
        "key": "farm_producer",
        "label": "Mua nong san, thuy san cua nguoi truc tiep san xuat",
        "required_evidence": ["Bang ke mua hang", "CCCD/ten nguoi ban", "Chung tu thanh toan"],
    },
    {
        "key": "non_business_individual",
        "label": "Mua hang hoa/dich vu cua ca nhan khong kinh doanh",
        "required_evidence": ["Hop dong hoac bien nhan", "Bang ke thay the hoa don", "Chung tu thanh toan"],
    },
    {
        "key": "street_vendor",
        "label": "Mua hang rong, ve so, qua vat hoac dich vu nho le",
        "required_evidence": ["Bang ke chi tiet", "Xac nhan nguoi mua", "Quy che chi tieu noi bo neu co"],
    },
    {
        "key": "seasonal_worker",
        "label": "Thue lao dong thoi vu duoi 01 thang",
        "required_evidence": ["Hop dong/khoan viec", "Bang cham cong", "Chung tu chi tra va khau tru neu phat sinh"],
    },
    {
        "key": "small_asset_rental",
        "label": "Thue tai san cua ca nhan theo giao dich nho le",
        "required_evidence": ["Hop dong thue", "Bang ke thanh toan", "Chung tu chuyen khoan neu tu 5 trieu dong"],
    },
]


@dataclass(frozen=True)
class Deadline:
    code: str
    title: str
    due_date: date
    group: int
    form_code: str
    priority: str
    description: str


def _today(value: date | None = None) -> date:
    return value or date.today()


def classify_household_group(annual_revenue: float) -> dict[str, Any]:
    revenue = max(0.0, float(annual_revenue or 0))
    if revenue < VND_500M:
        return {
            "group": 1,
            "label": "Nhom 1",
            "threshold_label": "duoi 500 trieu dong/nam",
            "filing_method": "Thong bao doanh thu, mien GTGT va TNCN neu duoi nguong",
            "accounting_books": ["S1a-HKD"],
        }
    if revenue <= VND_3B:
        return {
            "group": 2,
            "label": "Nhom 2",
            "threshold_label": "tu 500 trieu den 3 ty dong/nam",
            "filing_method": "Ke khai theo quy/quy dinh ho so day du",
            "accounting_books": ["S1a-HKD", "S2a-HKD", "S2b-HKD", "S2c-HKD"],
        }
    return {
        "group": 3,
        "label": "Nhom 3",
        "threshold_label": "tren 3 ty dong/nam",
        "filing_method": "Ke khai theo thang, theo doi doanh thu - chi phi va tai san",
        "accounting_books": ["S1a-HKD", "S2a-HKD", "S2b-HKD", "S2c-HKD", "So hang ton kho", "So TSCĐ"],
    }


def e_invoice_requirement(annual_revenue: float) -> dict[str, Any]:
    revenue = max(0.0, float(annual_revenue or 0))
    if revenue >= VND_1B:
        status = "mandatory"
        label = "Bat buoc dang ky hoa don dien tu"
        action = "Dang ky nha cung cap HDDT va phat hanh hoa don co ma CQT."
    elif revenue >= VND_500M:
        status = "voluntary_recommended"
        label = "Khuyen khich/nen dang ky hoa don dien tu"
        action = "Co the dang ky tu nguyen de chuan bi khi tang doanh thu."
    else:
        status = "single_issue_on_request"
        label = "Chua bat buoc dung thuong xuyen"
        action = "Co the xin cap hoa don dien tu tung lan khi co giao dich lon."
    return {
        "annual_revenue": revenue,
        "status": status,
        "label": label,
        "action": action,
        "thresholds": {"voluntary_from": VND_500M, "mandatory_from": VND_1B},
        "source_key": "nd68_2026",
    }


def deadline_status(due_date: date, today: date | None = None) -> dict[str, Any]:
    current = _today(today)
    days_left = (due_date - current).days
    if days_left < 0:
        return {"status": "overdue", "days_left": days_left, "badge": f"Qua han {abs(days_left)} ngay"}
    if days_left == 0:
        return {"status": "due_today", "days_left": 0, "badge": "Den han hom nay"}
    if days_left <= 7:
        return {"status": "soon", "days_left": days_left, "badge": f"Con {days_left} ngay"}
    return {"status": "upcoming", "days_left": days_left, "badge": "Chua den han"}


def build_deadlines(year: int, household_group: int, today: date | None = None) -> list[dict[str, Any]]:
    year = int(year or date.today().year)
    group = int(household_group or 2)
    deadlines: list[Deadline] = [
        Deadline(
            code=f"{year}-license-fee",
            title=f"Le phi mon bai nam {year}",
            due_date=date(year, 1, 30),
            group=group,
            form_code="LPMB",
            priority="low",
            description="Theo doi mien/giam le phi mon bai neu nam dau hoac nam duoc mien.",
        )
    ]

    if group == 1:
        deadlines.extend(
            [
                Deadline(
                    code=f"{year}-revenue-notice-h1",
                    title=f"Thong bao doanh thu 6 thang dau nam {year}",
                    due_date=date(year, 7, 31),
                    group=1,
                    form_code="01/TKN-CNKD",
                    priority="high",
                    description="Nhom 1 chi thong bao doanh thu, khong nop to khai chinh thuc neu doanh thu duoi nguong.",
                ),
                Deadline(
                    code=f"{year}-revenue-notice-final",
                    title=f"Thong bao doanh thu cuoi nam {year}",
                    due_date=date(year + 1, 1, 31),
                    group=1,
                    form_code="01/TKN-CNKD",
                    priority="high",
                    description="Tong hop doanh thu thuc te phat sinh trong nam duong lich.",
                ),
            ]
        )
    elif group == 2:
        for quarter, due_month, due_day in ((1, 4, 30), (2, 7, 31), (3, 10, 31), (4, 1, 31)):
            due_year = year if quarter < 4 else year + 1
            deadlines.append(
                Deadline(
                    code=f"{year}-q{quarter}-01cnkd",
                    title=f"To khai thue quy {quarter}/{year} (Mau 01/CNKD)",
                    due_date=date(due_year, due_month, due_day),
                    group=2,
                    form_code="01/CNKD",
                    priority="high",
                    description="Ke khai GTGT va TNCN tam tinh theo doanh thu quy.",
                )
            )
    else:
        for month in range(1, 13):
            due_year = year if month < 12 else year + 1
            due_month = month + 1 if month < 12 else 1
            deadlines.append(
                Deadline(
                    code=f"{year}-m{month:02d}-01cnkd",
                    title=f"To khai thue thang {month:02d}/{year}",
                    due_date=date(due_year, due_month, 20),
                    group=3,
                    form_code="01/CNKD",
                    priority="high",
                    description="Nhom 3 ke khai theo thang va theo doi day du doanh thu - chi phi.",
                )
            )

    return [
        {
            **deadline.__dict__,
            "due_date": deadline.due_date.isoformat(),
            **deadline_status(deadline.due_date, today),
        }
        for deadline in sorted(deadlines, key=lambda item: item.due_date)
    ]


def revenue_threshold_summary(
    cumulative_revenue: float,
    annual_revenue_plan: float | None = None,
) -> dict[str, Any]:
    revenue = max(0.0, float(cumulative_revenue or 0))
    planned = max(revenue, float(annual_revenue_plan or 0))
    next_threshold = VND_500M
    if revenue >= VND_500M:
        next_threshold = VND_1B
    if revenue >= VND_1B:
        next_threshold = VND_3B
    if revenue >= VND_3B:
        next_threshold = revenue
    distance = max(0.0, next_threshold - revenue)
    ratio = 1.0 if next_threshold == 0 else min(1.0, revenue / next_threshold)
    alert = "normal"
    if revenue >= VND_500M:
        alert = "taxable"
    elif ratio >= 0.9:
        alert = "near_500m"
    if revenue >= VND_1B:
        alert = "einvoice_mandatory"
    if revenue >= VND_3B:
        alert = "group3"
    return {
        "cumulative_revenue": revenue,
        "annual_revenue_plan": planned,
        "next_threshold": next_threshold,
        "distance_to_next_threshold": distance,
        "progress_ratio": round(ratio, 4),
        "alert": alert,
        "group": classify_household_group(max(revenue, planned)),
        "einvoice": e_invoice_requirement(max(revenue, planned)),
    }


def calculate_tax_by_industry(revenue: float, industry: str = "commerce") -> dict[str, Any]:
    revenue = max(0.0, float(revenue or 0))
    rate = INDUSTRY_TAX_RATES.get(industry, INDUSTRY_TAX_RATES["commerce"])
    gtgt = revenue * float(rate["gtgt_rate_pct"]) / 100.0
    tncn = revenue * float(rate["tncn_rate_pct"]) / 100.0
    return {
        "industry": industry,
        "industry_name": rate["name"],
        "gtgt_rate_pct": rate["gtgt_rate_pct"],
        "tncn_rate_pct": rate["tncn_rate_pct"],
        "gtgt_tax": round(gtgt, 2),
        "tncn_tax": round(tncn, 2),
        "total_tax": round(gtgt + tncn, 2),
        "source": rate["source"],
    }


def late_payment_penalty(amount: float, days: int) -> dict[str, Any]:
    principal = max(0.0, float(amount or 0))
    overdue_days = max(0, int(days or 0))
    penalty = principal * overdue_days * LATE_PAYMENT_RATE_PER_DAY
    return {
        "principal": principal,
        "days": overdue_days,
        "rate_per_day": LATE_PAYMENT_RATE_PER_DAY,
        "penalty": round(penalty, 2),
        "total": round(principal + penalty, 2),
    }


def passport_ban_risk(total_debt: float, max_days_overdue: int, is_forced_collection: bool = True) -> dict[str, Any]:
    debt = max(0.0, float(total_debt or 0))
    days = max(0, int(max_days_overdue or 0))
    meets_amount = debt >= PASSPORT_BAN_DEBT_THRESHOLD
    meets_days = days > PASSPORT_BAN_DAY_THRESHOLD
    triggered = bool(is_forced_collection and meets_amount and meets_days)
    if triggered:
        level = "critical"
        message = "Da cham nguong tam hoan xuat canh theo NĐ49/2025 neu dang bi cuong che."
    elif debt >= PASSPORT_BAN_DEBT_THRESHOLD * 0.8 or days >= 90:
        level = "warning"
        message = "Sap cham nguong can xu ly truoc khi bi ap dung bien phap cuong che."
    else:
        level = "normal"
        message = "Chua cham nguong tam hoan xuat canh."
    return {
        "total_debt": debt,
        "max_days_overdue": days,
        "is_forced_collection": is_forced_collection,
        "meets_amount_threshold": meets_amount,
        "meets_day_threshold": meets_days,
        "triggered": triggered,
        "level": level,
        "message": message,
        "source_key": "nd49_2025",
    }


def evaluate_expense(payload: dict[str, Any]) -> dict[str, Any]:
    amount = max(0.0, float(payload.get("amount") or 0))
    description = str(payload.get("description") or "").strip()
    category = str(payload.get("category") or "other").lower()
    payment_method = str(payload.get("payment_method") or "bank_transfer").lower()
    has_invoice = bool(payload.get("has_invoice", False))
    supplier_type = str(payload.get("supplier_type") or "").lower()
    no_invoice_case = str(payload.get("no_invoice_case") or "").lower()

    reasons: list[str] = []
    required_evidence: list[str] = []
    deductible = True
    status = "deductible"

    if "luong chu" in description.lower() or category in {"owner_salary", "personal"}:
        deductible = False
        status = "non_deductible"
        reasons.append("Luong/thu lao chu ho khong duoc tinh la chi phi duoc tru.")

    if amount >= CASH_PAYMENT_LIMIT and payment_method in {"cash", "tien_mat"}:
        deductible = False
        status = "non_deductible"
        reasons.append("Giao dich tu 5 trieu dong thanh toan tien mat khong du dieu kien chi phi duoc tru.")

    if not has_invoice and deductible:
        allowed = next((item for item in NO_INVOICE_ALLOWED_CASES if item["key"] == no_invoice_case), None)
        if allowed:
            status = "needs_evidence"
            reasons.append("Khoan chi co the duoc tru neu lap bang ke/chung tu thay the day du.")
            required_evidence.extend(allowed["required_evidence"])
        elif supplier_type in {"farmer", "non_business_individual", "street_vendor"}:
            status = "needs_evidence"
            reasons.append("Nha cung cap thuoc nhom co the lap bang ke thay hoa don.")
            required_evidence.extend(["Bang ke thay the hoa don", "Thong tin nguoi ban", "Chung tu thanh toan"])
        else:
            status = "needs_invoice"
            reasons.append("Can hoa don hop le hoac bang ke thay the theo truong hop duoc chap nhan.")
            required_evidence.extend(["Hoa don dien tu", "Hop dong/bien nhan", "Chung tu thanh toan"])

    if deductible and not reasons:
        reasons.append("Khoan chi co mo ta kinh doanh, chung tu va phuong thuc thanh toan phu hop.")
        required_evidence.extend(["Hoa don/chung tu", "Chung tu thanh toan", "Hop dong neu co"])

    return {
        "amount": amount,
        "category": category,
        "deductible": deductible,
        "status": status,
        "reasons": reasons,
        "required_evidence": list(dict.fromkeys(required_evidence)),
        "cash_payment_limit": CASH_PAYMENT_LIMIT,
        "source_key": "nd68_2026",
    }


def depreciation_schedule(cost: float, purchase_date: str | date, useful_life_months: int) -> dict[str, Any]:
    value = max(0.0, float(cost or 0))
    months = max(1, int(useful_life_months or 1))
    monthly = round(value / months, 2)
    if isinstance(purchase_date, str):
        start = datetime.fromisoformat(purchase_date[:10]).date()
    else:
        start = purchase_date
    entries = []
    for idx in range(months):
        month_number = start.month + idx
        year = start.year + (month_number - 1) // 12
        month = ((month_number - 1) % 12) + 1
        entries.append({"period": f"{year}-{month:02d}", "amount": monthly})
    return {
        "cost": value,
        "useful_life_months": months,
        "monthly_depreciation": monthly,
        "schedule": entries,
    }


def seasonal_tax_allocation(annual_tax: float, active_days: int, year: int | None = None) -> dict[str, Any]:
    yr = year or date.today().year
    days_in_year = 366 if (yr % 400 == 0 or (yr % 4 == 0 and yr % 100 != 0)) else 365
    days = min(days_in_year, max(0, int(active_days or 0)))
    tax = max(0.0, float(annual_tax or 0))
    allocated = tax * days / days_in_year
    return {
        "year": yr,
        "active_days": days,
        "days_in_year": days_in_year,
        "annual_tax": tax,
        "allocated_tax": round(allocated, 2),
    }


def build_ics(deadlines: list[dict[str, Any]], calendar_name: str = "TaxInspector Taxpayer Deadlines") -> str:
    def fmt(d: str) -> str:
        return d.replace("-", "")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//TaxInspector//Taxpayer Calendar//VI",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{calendar_name}",
    ]
    for item in deadlines:
        uid = f"{item.get('code')}@taxinspector.local"
        due = fmt(str(item.get("due_date")))
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
                f"DTSTART;VALUE=DATE:{due}",
                f"SUMMARY:{item.get('title', 'Deadline thue')}",
                f"DESCRIPTION:{item.get('description', '')}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def search_industry_rates(query: str | None = None) -> list[dict[str, Any]]:
    needle = (query or "").strip().lower()
    results = []
    for key, info in INDUSTRY_TAX_RATES.items():
        haystack = " ".join([key, str(info["name"]), str(info["isic_hint"])]).lower()
        if not needle or needle in haystack:
            results.append({"industry": key, **info})
    return results


def legal_answer(query: str, rules: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    text = (query or "").lower()
    citations = []
    answer_parts = []

    def add(source_key: str, sentence: str) -> None:
        source = next((item for item in BASELINE_SOURCES if item["key"] == source_key), None)
        if source:
            citations.append(source)
        answer_parts.append(sentence)

    if any(token in text for token in ["hoa don", "hddt", "invoice"]):
        add(
            "nd70_2025",
            "Hoa don dien tu can duoc kiem tra tinh hop le, trang thai huy/thay the va thong tin nguoi ban truoc khi dua vao chi phi.",
        )
        add(
            "nd68_2026",
            "Neu doanh thu dat nguong cao, he thong can chuan bi dang ky va su dung hoa don dien tu theo lo trinh HKD.",
        )
    if any(token in text for token in ["chi phi", "duoc tru", "tien mat", "5 trieu"]):
        add(
            "nd68_2026",
            "Chi phi duoc tru can lien quan hoat dong kinh doanh, co chung tu hop le va giao dich tu 5 trieu dong nen thanh toan khong dung tien mat.",
        )
    if any(token in text for token in ["no thue", "xuat canh", "cuong che"]):
        add(
            "nd49_2025",
            "Ca nhan kinh doanh/chu ho co the bi tam hoan xuat canh khi no tu 50 trieu dong va qua han tren 120 ngay trong truong hop bi cuong che.",
        )
    if any(token in text for token in ["so sach", "ke toan", "s1a", "s2a"]):
        add(
            "tt152_2025",
            "So sach ke toan cua HKD nen lap theo nhom doanh thu, toi thieu theo doi doanh thu va tang them chi phi, hang ton kho, TSCĐ khi quy mo lon.",
        )
    if any(token in text for token in ["to khai", "01/cnkd", "tam ngung", "thay doi dia diem"]):
        add(
            "tt18_2026",
            "Ho so, thu tuc quan ly thue voi HKD can theo dung mau bieu va thoi han nop ho so tuong ung tung nghiep vu.",
        )
    if any(token in text for token in ["giam tru", "nguoi phu thuoc", "tncn"]):
        add(
            "nq110_2025",
            "Tu nam 2026, muc giam tru gia canh moi can duoc ap dung khi tinh TNCN theo phuong phap tren thu nhap tinh thue.",
        )

    if not answer_parts:
        add(
            "nd68_2026",
            "Chua co du tin hieu de ket luan chac chan. Nen doi chieu quy dinh HKD hien hanh va bo sung chi tiet ve doanh thu, nganh nghe, chung tu.",
        )
        confidence = "needs_verification"
    else:
        confidence = "grounded"

    deduped_citations = []
    seen = set()
    for item in citations:
        if item["key"] not in seen:
            deduped_citations.append(item)
            seen.add(item["key"])
    return {
        "answer": " ".join(answer_parts),
        "citations": deduped_citations,
        "status": confidence,
    }


def hkd_vs_llc_comparison(revenue: float, expenses: float) -> dict[str, Any]:
    revenue = max(0.0, float(revenue or 0))
    expenses = max(0.0, float(expenses or 0))
    profit = max(0.0, revenue - expenses)
    hkd_tax = calculate_tax_by_industry(revenue, "commerce")["total_tax"]
    llc_cit = profit * 0.20
    dividend_tax = max(0.0, profit - llc_cit) * 0.05
    return {
        "revenue": revenue,
        "expenses": expenses,
        "profit": profit,
        "hkd_estimated_tax": round(hkd_tax, 2),
        "llc_cit_20": round(llc_cit, 2),
        "llc_dividend_tax": round(dividend_tax, 2),
        "llc_total_tax": round(llc_cit + dividend_tax, 2),
        "non_tax_factors": [
            "TNHH tach bach tai san va trach nhiem phap ly tot hon.",
            "HKD don gian hon ve quan tri nhung han che khi mo rong/ky hop dong lon.",
            "Khi doanh thu vuot 3 ty, nen chuan bi so sach va quy trinh gan voi doanh nghiep.",
        ],
    }


def debt_days_overdue(due_date: date | str, today: date | None = None) -> int:
    if isinstance(due_date, str):
        due = datetime.fromisoformat(due_date[:10]).date()
    else:
        due = due_date
    return max(0, (_today(today) - due).days)


def installment_plan(amount: float, months: int) -> dict[str, Any]:
    principal = max(0.0, float(amount or 0))
    count = max(1, int(months or 1))
    monthly = ceil(principal / count)
    return {
        "amount": principal,
        "months": count,
        "monthly_amount": monthly,
        "note": "Ke hoach sandbox; ho so that can duoc co quan thue chap thuan.",
    }
