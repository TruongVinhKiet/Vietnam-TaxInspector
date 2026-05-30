# -*- coding: utf-8 -*-
"""
calculator.py – Taxpayer Automatic Tax Calculator Services
===================================================================
Contains all calculators for Nhóm 2 - Tính thuế tự động:
1. GTGT & TNCN according to Circular 40/2021/TT-BTC.
2. Giảm trừ gia cảnh (Deductions 2026: Person 15.5M, Dependent 6.2M).
3. 7-step Progressive PIT (Article 22).
4. Comparison of Khoán vs Kê khai vs Transitioning to TNHH (TNDN 20%).
5. Late Payment Penalty (Law 38/2019/QH14 - 0.03% / day).
6. License Fee (NĐ 139/2016).
7. Property Rental Tax (5% GTGT + 5% TNCN).
8. Foreign Contractor Tax (FCT).
9. Transitioning LLC Corporate Income Tax (TNDN 20%).
10. Calculation History (JSONB database logger).
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict
import json
import datetime

from ..database import get_db

router = APIRouter(prefix="/api/calculator", tags=["Taxpayer Calculator"])


def ensure_calculator_schema(conn):
    """Ensure all required tables for logging calculations are present."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS tax_calculation_history (
            id SERIAL PRIMARY KEY,
            tax_code VARCHAR(20) NOT NULL,
            calc_type VARCHAR(50) NOT NULL,
            inputs JSONB NOT NULL,
            results JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))


# ---- ENDPOINTS ----

@router.post("/gtgt-tncn")
def calculate_gtgt_tncn(
    payload: Dict = Body(...)
):
    """
    Calculate GTGT & TNCN for household businesses based on revenue and sector (Circular 40/2021/TT-BTC).
    """
    revenue = float(payload.get("revenue", 0))
    industry = payload.get("industry", "commerce")

    # Define rates based on Circular 40
    rates = {
        "commerce": {"gtgt": 1.0, "tncn": 0.5, "name": "Phân phối, bán buôn, bán lẻ hàng hóa"},
        "service": {"gtgt": 5.0, "tncn": 2.0, "name": "Dịch vụ, môi giới, đại lý"},
        "manufacture": {"gtgt": 3.0, "tncn": 1.5, "name": "Sản xuất, gia công, xây dựng có bao thầu"},
        "other": {"gtgt": 2.0, "tncn": 1.0, "name": "Hoạt động kinh doanh khác"}
    }

    info = rates.get(industry, rates["commerce"])
    gtgt_rate = info["gtgt"] / 100.0
    tncn_rate = info["tncn"] / 100.0

    gtgt_tax = revenue * gtgt_rate
    tncn_tax = revenue * tncn_rate
    total_tax = gtgt_tax + tncn_tax

    return {
        "status": "success",
        "industry_name": info["name"],
        "gtgt_rate_pct": info["gtgt"],
        "tncn_rate_pct": info["tncn"],
        "gtgt_tax": round(gtgt_tax, 2),
        "tncn_tax": round(tncn_tax, 2),
        "total_tax": round(total_tax, 2)
    }


@router.post("/deductions")
def calculate_deductions(
    payload: Dict = Body(...)
):
    """
    Calculate PIT family deductions and taxable income (2026 Thresholds: Person 15.5M, Dependent 6.2M).
    """
    monthly_income = float(payload.get("monthly_income", 0))
    dependents = int(payload.get("dependents", 0))

    # 2026 thresholds
    person_deduction = 15500000.0
    dependent_deduction = 6200000.0

    total_dependent_deduction = dependents * dependent_deduction
    total_deduction = person_deduction + total_dependent_deduction
    
    taxable_income = max(0.0, monthly_income - total_deduction)

    return {
        "status": "success",
        "monthly_income": monthly_income,
        "dependents": dependents,
        "person_deduction": person_deduction,
        "dependent_deduction_rate": dependent_deduction,
        "total_dependent_deduction": total_dependent_deduction,
        "total_deduction": total_deduction,
        "taxable_income": round(taxable_income, 2)
    }


@router.post("/progressive-pit")
def calculate_progressive_pit(
    payload: Dict = Body(...)
):
    """
    Calculate progressive Personal Income Tax based on Article 22 PIT Law (7 tax brackets).
    Input: taxable_income (yearly or monthly)
    """
    taxable_income = float(payload.get("taxable_income", 0))
    is_yearly = payload.get("is_yearly", False)

    # Monthly tax brackets
    brackets = [
        {"limit": 5000000.0, "rate": 0.05, "sub": 0.0},
        {"limit": 10000000.0, "rate": 0.10, "sub": 250000.0},
        {"limit": 18000000.0, "rate": 0.15, "sub": 750000.0},
        {"limit": 32000000.0, "rate": 0.20, "sub": 1650000.0},
        {"limit": 52000000.0, "rate": 0.25, "sub": 3250000.0},
        {"limit": 80000000.0, "rate": 0.30, "sub": 5850000.0},
        {"limit": float('inf'), "rate": 0.35, "sub": 9850000.0}
    ]

    income_for_calc = taxable_income
    if is_yearly:
        income_for_calc = taxable_income / 12.0

    # Calculate PIT using quick subtraction formula
    pit_monthly = 0.0
    bracket_applied = 0
    
    if income_for_calc > 0:
        if income_for_calc <= 5000000.0:
            pit_monthly = income_for_calc * 0.05
            bracket_applied = 1
        elif income_for_calc <= 10000000.0:
            pit_monthly = income_for_calc * 0.10 - 250000.0
            bracket_applied = 2
        elif income_for_calc <= 18000000.0:
            pit_monthly = income_for_calc * 0.15 - 750000.0
            bracket_applied = 3
        elif income_for_calc <= 32000000.0:
            pit_monthly = income_for_calc * 0.20 - 1650000.0
            bracket_applied = 4
        elif income_for_calc <= 52000000.0:
            pit_monthly = income_for_calc * 0.25 - 3250000.0
            bracket_applied = 5
        elif income_for_calc <= 80000000.0:
            pit_monthly = income_for_calc * 0.30 - 5850000.0
            bracket_applied = 6
        else:
            pit_monthly = income_for_calc * 0.35 - 9850000.0
            bracket_applied = 7

    pit_total = pit_monthly * 12.0 if is_yearly else pit_monthly

    return {
        "status": "success",
        "taxable_income": taxable_income,
        "is_yearly": is_yearly,
        "bracket_applied": bracket_applied,
        "pit_monthly": round(pit_monthly, 2),
        "pit_total": round(pit_total, 2)
    }


@router.post("/compare-methods")
def compare_tax_methods(
    payload: Dict = Body(...)
):
    """
    Compare tax liabilities:
    1. Revenue Tax (Khoán): e.g. 1.5% - 7% on total revenue
    2. Declaration Accounting (Kê khai): tax based on progressive PIT on profit
    3. Transition to LLC: Corporate Income Tax (TNDN 20%) + Div / TNCN
    """
    revenue = float(payload.get("revenue", 0))  # annual
    expenses = float(payload.get("expenses", 0))  # annual
    industry = payload.get("industry", "commerce")

    # 1. Revenue Tax calculation
    rates = {
        "commerce": 1.5,
        "service": 7.0,
        "manufacture": 4.5,
        "other": 3.0
    }
    pct_rate = rates.get(industry, 1.5)
    tax_khoan = revenue * (pct_rate / 100.0)

    # 2. Declaration Accounting (PIT progressive on annual net income)
    profit = max(0.0, revenue - expenses)
    # Estimate PIT progressive by simulating it
    # We deduct standard family deductions: Person 15.5M/mo * 12 = 186M/year
    taxable_pit = max(0.0, profit - 186000000.0)
    
    # Calculate progressive pit on annual taxable income
    pit_calc = calculate_progressive_pit({"taxable_income": taxable_pit, "is_yearly": True})
    tax_kekhai = pit_calc["pit_total"]

    # 3. Transition to TNHH (TNDN 20%)
    # Profit * 20%
    tax_tndn = profit * 0.20
    # Additional tax for dividend payout (usually 5% TNCN)
    dividend_tax = (profit - tax_tndn) * 0.05
    tax_llc = tax_tndn + dividend_tax

    # Determine advice
    lowest = min(tax_khoan, tax_kekhai, tax_llc)
    advice = ""
    if lowest == tax_khoan:
        advice = f"Phương pháp Nộp khoán (%) là tối ưu nhất. Bạn nên duy trì mô hình Hộ kinh doanh nộp khoán."
    elif lowest == tax_kekhai:
        advice = f"Kê khai kế toán (Doanh thu - Chi phí) tối ưu nhất vì mức chi phí thực tế của bạn khá lớn. Bạn nên chuyển từ Khoán sang Kê khai."
    else:
        advice = f"Quyết định chuyển đổi sang doanh nghiệp TNHH (TNDN 20%) là tối ưu nhất cho hoạt động lâu dài và hưởng ưu đãi thuế TNDN."

    return {
        "status": "success",
        "tax_khoan": round(tax_khoan, 2),
        "tax_kekhai": round(tax_kekhai, 2),
        "tax_tndn_20": round(tax_tndn, 2),
        "tax_llc_total": round(tax_llc, 2),
        "profit": round(profit, 2),
        "advice": advice
    }


@router.post("/late-penalty")
def calculate_late_penalty(
    payload: Dict = Body(...)
):
    """
    Calculate late payment interest (phạt chậm nộp) under Law 38/2019/QH14.
    Supports historical rate parameter (e.g. 0.03 or 0.05 % per day).
    """
    amount = float(payload.get("amount", 0))
    days = int(payload.get("days", 0))
    rate_val = float(payload.get("rate", 0.03))  # in percent, default 0.03%

    rate = rate_val / 100.0
    penalty = amount * days * rate
    total = amount + penalty

    return {
        "status": "success",
        "origin_amount": round(amount, 2),
        "days": days,
        "penalty_rate": rate_val,
        "penalty_amount": round(penalty, 2),
        "total_amount": round(total, 2)
    }


@router.post("/license-fee")
def calculate_license_fee(
    payload: Dict = Body(...)
):
    """
    Calculate License Fee (Lệ phí môn bài) based on capital or revenue (Decree 139/2016/NĐ-CP).
    Types: 'business' (company), 'household' (HKD)
    """
    capital = float(payload.get("capital", 0))  # For companies
    revenue = float(payload.get("revenue", 0))  # For households
    biz_type = payload.get("type", "household")

    fee = 0.0
    level = "Miễn lệ phí môn bài"

    if biz_type == "business":
        # Companies: capital > 10 Billion -> Level 1: 3M, capital <= 10B -> Level 2: 2M
        if capital > 10000000000.0:
            fee = 3000000.0
            level = "Bậc 1 (Vốn điều lệ trên 10 tỷ VNĐ)"
        else:
            fee = 2000000.0
            level = "Bậc 2 (Vốn điều lệ từ 10 tỷ VNĐ trở xuống)"
    else:
        # Households: revenue > 500M -> 1M, revenue 300M - 500M -> 500k, revenue 100M - 300M -> 300k, revenue <= 100M -> Miễn
        if revenue > 500000000.0:
            fee = 1000000.0
            level = "Bậc 1 (Doanh thu trên 500 triệu VNĐ/năm)"
        elif revenue > 300000000.0:
            fee = 500000.0
            level = "Bậc 2 (Doanh thu từ 300 đến 500 triệu VNĐ/năm)"
        elif revenue > 100000000.0:
            fee = 300000.0
            level = "Bậc 3 (Doanh thu từ 100 đến 300 triệu VNĐ/năm)"
        else:
            fee = 0.0
            level = "Miễn lệ phí môn bài (Doanh thu <= 100 triệu VNĐ/năm)"

    return {
        "status": "success",
        "type": biz_type,
        "fee": fee,
        "level": level
    }


@router.post("/rental-tax")
def calculate_rental_tax(
    payload: Dict = Body(...)
):
    """
    Calculate Property Rental Tax (5% GTGT + 5% TNCN).
    Threshold: Exempt if annual rent <= 100M VND.
    """
    monthly_rent = float(payload.get("monthly_rent", 0))
    months = int(payload.get("months", 12))

    total_rent = monthly_rent * months
    is_taxable = total_rent > 100000000.0

    gtgt_tax = 0.0
    tncn_tax = 0.0
    
    if is_taxable:
        gtgt_tax = total_rent * 0.05
        tncn_tax = total_rent * 0.05

    total_tax = gtgt_tax + tncn_tax

    return {
        "status": "success",
        "total_rent": round(total_rent, 2),
        "is_taxable": is_taxable,
        "gtgt_rate_pct": 5.0,
        "tncn_rate_pct": 5.0,
        "gtgt_tax": round(gtgt_tax, 2),
        "tncn_tax": round(tncn_tax, 2),
        "total_tax": round(total_tax, 2)
    }


@router.post("/contractor-tax")
def calculate_contractor_tax(
    payload: Dict = Body(...)
):
    """
    Calculate Foreign Contractor Tax (FCT).
    Sectors:
    - 'services': GTGT 5% + TNDN 5% = 10%
    - 'goods': GTGT miễn/exempt + TNDN 1%
    - 'construction_goods': GTGT 3% + TNDN 2%
    """
    contract_value = float(payload.get("contract_value", 0))
    service_type = payload.get("service_type", "services")

    # Define rates (Circular 103/2014/TT-BTC)
    rates = {
        "services": {"gtgt": 5.0, "tndn": 5.0, "name": "Cung cấp dịch vụ"},
        "goods": {"gtgt": 0.0, "tndn": 1.0, "name": "Cung cấp hàng hóa"},
        "construction_goods": {"gtgt": 3.0, "tndn": 2.0, "name": "Xây dựng có bao thầu vật tư"}
    }

    info = rates.get(service_type, rates["services"])
    gtgt_tax = contract_value * (info["gtgt"] / 100.0)
    tndn_tax = contract_value * (info["tndn"] / 100.0)
    total_tax = gtgt_tax + tndn_tax

    return {
        "status": "success",
        "service_name": info["name"],
        "gtgt_rate_pct": info["gtgt"],
        "tndn_rate_pct": info["tndn"],
        "gtgt_tax": round(gtgt_tax, 2),
        "tndn_tax": round(tndn_tax, 2),
        "total_tax": round(total_tax, 2)
    }


@router.post("/tndn")
def calculate_tndn_tax(
    payload: Dict = Body(...)
):
    """
    Corporate Income Tax (TNDN) 20% calculation for transitioning to TNHH.
    Inputs:
        revenue: Annual revenue in VND
        expenses: Annual allowable expenses in VND
    """
    revenue = float(payload.get("revenue", 0))
    expenses = float(payload.get("expenses", 0))

    profit = max(0.0, revenue - expenses)
    tax_tndn = profit * 0.20

    return {
        "status": "success",
        "revenue": round(revenue, 2),
        "expenses": round(expenses, 2),
        "profit": round(profit, 2),
        "tax_rate_pct": 20.0,
        "tax_tndn": round(tax_tndn, 2)
    }


@router.post("/history")
def save_calculation_history(
    payload: Dict = Body(...),
    db: Session = Depends(get_db)
):
    """Log a completed tax calculation in database for user history."""
    ensure_calculator_schema(db.connection())
    
    tax_code = payload.get("tax_code", "").strip()
    calc_type = payload.get("calc_type", "gtgt_tncn")
    inputs = payload.get("inputs", {})
    results = payload.get("results", {})

    if not tax_code:
        raise HTTPException(status_code=400, detail="MST bắt buộc phải có để lưu lịch sử.")

    db.execute(text("""
        INSERT INTO tax_calculation_history (tax_code, calc_type, inputs, results)
        VALUES (:tax, :ctype, :inputs, :results)
    """), {
        "tax": tax_code,
        "ctype": calc_type,
        "inputs": json.dumps(inputs),
        "results": json.dumps(results)
    })
    
    db.commit()
    
    return {"status": "success", "message": "Lưu lịch sử tính toán thuế thành công."}


@router.get("/history/{tax_code}")
def get_calculation_history(
    tax_code: str,
    db: Session = Depends(get_db)
):
    """Retrieve the last 20 calculations recorded for a taxpayer."""
    ensure_calculator_schema(db.connection())
    
    rows = db.execute(text("""
        SELECT id, calc_type, inputs, results, created_at 
        FROM tax_calculation_history 
        WHERE tax_code = :tax
        ORDER BY created_at DESC
        LIMIT 20
    """), {"tax": tax_code}).all()

    calc_names = {
        "gtgt_tncn": "Thuế GTGT & TNCN Hộ kinh doanh",
        "deductions": "Giảm trừ gia cảnh & TN chịu thuế",
        "compare": "So sánh khoán vs kê khai vs TNHH",
        "penalty": "Tính phạt chậm nộp thuế",
        "progressive": "Thuế TNCN lũy tiến 7 bậc",
        "license_fee": "Lệ phí môn bài năm",
        "rental": "Thuế cho thuê tài sản",
        "contractor": "Thuế nhà thầu nước ngoài",
        "tndn": "Thuế TNDN 20% Doanh nghiệp"
    }

    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "calc_type": r[1],
            "calc_name": calc_names.get(r[1], r[1]),
            "inputs": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
            "results": r[3] if isinstance(r[3], dict) else json.loads(r[3]),
            "created_at": r[4].strftime("%d/%m/%Y %H:%M:%S") if r[4] else "N/A"
        })

    return {"history": history}
