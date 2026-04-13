"""
generate_mock_data.py – Script sinh dữ liệu giả lập 5,000 doanh nghiệp
=========================================================================
- ~95% doanh nghiệp bình thường
- ~5% (250 DN) có dấu hiệu bất thường (fraud_label = 1)
  với các pattern trốn thuế thật:
    * Đội chi phí đầu vào
    * Lệch pha doanh thu-chi phí
    * VAT vòng lặp
    * Biên lợi nhuận quá thấp so với ngành

Sinh dữ liệu 3 năm liên tiếp (2022, 2023, 2024) cho MỖI doanh nghiệp
=> Tổng cộng: 5000 * 3 = 15,000 rows

Columns:
    tax_code, company_name, industry, province, year,
    revenue, cost_of_goods, operating_expenses, total_expenses,
    gross_profit, net_profit, tax_paid,
    vat_input, vat_output,
    num_employees, registered_capital,
    industry_avg_profit_margin,
    fraud_label   (0 | 1)
"""

import numpy as np
import pandas as pd
import os
import sys

np.random.seed(42)

# =====================================================================
# 1. CONSTANTS & LOOKUP TABLES
# =====================================================================

NUM_COMPANIES = 5000
YEARS = [2022, 2023, 2024]
FRAUD_RATIO = 0.05  # 5% anomalous

INDUSTRIES = [
    "Xây dựng", "Bất động sản", "Thương mại XNK", "Sản xuất công nghiệp",
    "Nông nghiệp", "Vận tải & Logistics", "Công nghệ thông tin",
    "Dịch vụ tài chính", "Y tế & Dược phẩm", "Giáo dục & Đào tạo",
    "Thực phẩm & Đồ uống", "May mặc & Giầy da", "Khoáng sản & Năng lượng",
    "Du lịch & Khách sạn", "Viễn thông",
]

PROVINCES = [
    "Hà Nội", "TP.HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Bình Dương", "Đồng Nai", "Bắc Ninh", "Quảng Ninh", "Nghệ An",
    "Thanh Hóa", "Khánh Hòa", "Lâm Đồng", "Bà Rịa-VT", "Long An",
]

# Biên lợi nhuận trung bình ngành (%)
INDUSTRY_AVG_MARGINS = {
    "Xây dựng": 0.06,
    "Bất động sản": 0.12,
    "Thương mại XNK": 0.04,
    "Sản xuất công nghiệp": 0.08,
    "Nông nghiệp": 0.05,
    "Vận tải & Logistics": 0.07,
    "Công nghệ thông tin": 0.15,
    "Dịch vụ tài chính": 0.18,
    "Y tế & Dược phẩm": 0.14,
    "Giáo dục & Đào tạo": 0.10,
    "Thực phẩm & Đồ uống": 0.09,
    "May mặc & Giầy da": 0.06,
    "Khoáng sản & Năng lượng": 0.11,
    "Du lịch & Khách sạn": 0.08,
    "Viễn thông": 0.13,
}

COMPANY_PREFIXES = [
    "Công ty TNHH", "Công ty CP", "Công ty TNHH MTV", "DNTN",
    "Tập đoàn", "Xí nghiệp", "HTX",
]

COMPANY_NAMES = [
    "Hoàng Gia", "Phú Thịnh", "Minh Đức", "Thành Đạt", "An Phát",
    "Việt Hùng", "Đại Phong", "Thái Bình", "Trường Phát", "Hòa Bình",
    "Tiến Đạt", "Tân Phong", "Bảo An", "Kim Long", "Phúc Lợi",
    "Quang Minh", "Đông Á", "Nam Việt", "Bắc Sơn", "Tây Nguyên",
    "Sao Mai", "Hải Đăng", "Sơn Hà", "Liên Minh", "Thăng Long",
    "Đức Tín", "Tín Nghĩa", "Thuận Phát", "Duy Tân", "Văn Minh",
]


def gen_tax_code(idx: int) -> str:
    """Generate a 10-digit tax code."""
    province_prefix = np.random.choice(["01", "02", "03", "04", "05",
                                         "06", "07", "08", "09", "10"])
    return f"{province_prefix}{idx:08d}"


def gen_company_name() -> str:
    prefix = np.random.choice(COMPANY_PREFIXES)
    name = np.random.choice(COMPANY_NAMES)
    suffix = np.random.choice(["", " Việt Nam", " Sài Gòn", " Hà Nội",
                                " Miền Nam", " Quốc tế"])
    return f"{prefix} {name}{suffix}"


# =====================================================================
# 2. GENERATE NORMAL COMPANIES (95%)
# =====================================================================

def generate_normal_company(tax_code: str, name: str, industry: str,
                            province: str) -> list[dict]:
    """Generate 3 years of financial data for a normal company."""
    rows = []
    avg_margin = INDUSTRY_AVG_MARGINS[industry]
    registered_capital = np.random.lognormal(mean=22, sigma=1.5)
    num_employees = int(np.random.lognormal(mean=3.5, sigma=1.0))
    num_employees = max(5, min(num_employees, 5000))

    # Base revenue (in VND)
    base_revenue = np.random.lognormal(mean=23, sigma=1.2)

    for i, year in enumerate(YEARS):
        # Revenue grows moderately 5-15%/y
        growth = np.random.uniform(0.95, 1.18)
        revenue = base_revenue * (growth ** i)

        # Normal cost structure: COGS = 55-80% of revenue
        cogs_ratio = np.random.uniform(0.55, 0.80)
        cost_of_goods = revenue * cogs_ratio

        # Operating expenses 5-15% of revenue
        opex_ratio = np.random.uniform(0.05, 0.15)
        operating_expenses = revenue * opex_ratio

        total_expenses = cost_of_goods + operating_expenses
        gross_profit = revenue - cost_of_goods
        net_profit = revenue - total_expenses

        # CIT (Corporate Income Tax) ~20% of net_profit (if positive)
        tax_paid = max(0, net_profit * 0.20 * np.random.uniform(0.9, 1.1))

        # VAT Output ~ 10% of revenue, VAT Input ~ 10% of COGS
        vat_output = revenue * 0.10 * np.random.uniform(0.95, 1.05)
        vat_input = cost_of_goods * 0.10 * np.random.uniform(0.90, 1.05)

        rows.append({
            "tax_code": tax_code,
            "company_name": name,
            "industry": industry,
            "province": province,
            "year": year,
            "revenue": round(revenue, 2),
            "cost_of_goods": round(cost_of_goods, 2),
            "operating_expenses": round(operating_expenses, 2),
            "total_expenses": round(total_expenses, 2),
            "gross_profit": round(gross_profit, 2),
            "net_profit": round(net_profit, 2),
            "tax_paid": round(tax_paid, 2),
            "vat_input": round(vat_input, 2),
            "vat_output": round(vat_output, 2),
            "num_employees": num_employees,
            "registered_capital": round(registered_capital, 2),
            "industry_avg_profit_margin": round(avg_margin, 4),
            "fraud_label": 0,
        })

    return rows


# =====================================================================
# 3. GENERATE FRAUDULENT COMPANIES (5%)
# =====================================================================

def generate_fraud_company(tax_code: str, name: str, industry: str,
                           province: str) -> list[dict]:
    """
    Generate 3 years of data with embedded fraud patterns:
      - Pattern A: Cost inflation (F1 divergence) – chi phí tăng tốc hơn doanh thu
      - Pattern B: Ratio limit abuse (F2 > 0.98) – triệt tiêu lợi nhuận
      - Pattern C: VAT carousel (F3 extreme) – VAT đầu vào ≥ đầu ra
      - Pattern D: Revenue spike with no substance
    """
    rows = []
    avg_margin = INDUSTRY_AVG_MARGINS[industry]
    registered_capital = np.random.lognormal(mean=21, sigma=1.0)
    num_employees = int(np.random.lognormal(mean=2.8, sigma=0.8))
    num_employees = max(3, min(num_employees, 500))

    base_revenue = np.random.lognormal(mean=22.5, sigma=1.0)

    # Pick 1-2 fraud patterns
    fraud_patterns = np.random.choice(["A", "B", "C", "D"],
                                       size=np.random.randint(1, 3),
                                       replace=False)

    for i, year in enumerate(YEARS):
        # Revenue growth – may spike artificially
        if "D" in fraud_patterns and i >= 1:
            growth = np.random.uniform(2.5, 5.0)  # 250-500% spike
        else:
            growth = np.random.uniform(1.0, 1.25)

        revenue = base_revenue * (growth ** i)

        # --- Pattern A: Cost inflation divergence ---
        if "A" in fraud_patterns:
            # Costs grow MUCH faster than revenue
            cogs_ratio = np.random.uniform(0.75, 0.90) + (i * 0.05)
            cogs_ratio = min(cogs_ratio, 0.97)
        else:
            cogs_ratio = np.random.uniform(0.55, 0.78)

        cost_of_goods = revenue * cogs_ratio

        # --- Pattern B: Expense/Revenue ratio ~0.98-0.995 ---
        if "B" in fraud_patterns:
            target_total_ratio = np.random.uniform(0.975, 0.998)
            total_expenses = revenue * target_total_ratio
            operating_expenses = total_expenses - cost_of_goods
            if operating_expenses < 0:
                operating_expenses = revenue * 0.02
                total_expenses = cost_of_goods + operating_expenses
        else:
            opex_ratio = np.random.uniform(0.05, 0.15)
            operating_expenses = revenue * opex_ratio
            total_expenses = cost_of_goods + operating_expenses

        gross_profit = revenue - cost_of_goods
        net_profit = revenue - total_expenses

        # Tax paid: fraudulent companies pay almost no tax
        if net_profit > 0:
            tax_paid = net_profit * 0.20 * np.random.uniform(0.1, 0.5)
        else:
            tax_paid = 0

        # --- Pattern C: VAT carousel ---
        vat_output = revenue * 0.10 * np.random.uniform(0.95, 1.05)
        if "C" in fraud_patterns:
            # VAT input ≈ or > VAT output → near-zero VAT payable
            vat_input = vat_output * np.random.uniform(0.95, 1.15)
        else:
            vat_input = cost_of_goods * 0.10 * np.random.uniform(0.90, 1.05)

        rows.append({
            "tax_code": tax_code,
            "company_name": name,
            "industry": industry,
            "province": province,
            "year": year,
            "revenue": round(revenue, 2),
            "cost_of_goods": round(cost_of_goods, 2),
            "operating_expenses": round(operating_expenses, 2),
            "total_expenses": round(total_expenses, 2),
            "gross_profit": round(gross_profit, 2),
            "net_profit": round(net_profit, 2),
            "tax_paid": round(tax_paid, 2),
            "vat_input": round(vat_input, 2),
            "vat_output": round(vat_output, 2),
            "num_employees": num_employees,
            "registered_capital": round(registered_capital, 2),
            "industry_avg_profit_margin": round(avg_margin, 4),
            "fraud_label": 1,
        })

    return rows


# =====================================================================
# 4. MAIN EXECUTION
# =====================================================================

def main():
    print("=" * 60)
    print("  TaxInspector – Mock Data Generator")
    print("  5,000 doanh nghiệp × 3 năm = 15,000 bản ghi")
    print("=" * 60)

    num_fraud = int(NUM_COMPANIES * FRAUD_RATIO)
    num_normal = NUM_COMPANIES - num_fraud

    all_rows = []

    # ---- Generate Normal Companies ----
    print(f"\n[1/3] Sinh {num_normal} doanh nghiệp bình thường...")
    for idx in range(num_normal):
        tc = gen_tax_code(idx)
        name = gen_company_name()
        ind = np.random.choice(INDUSTRIES)
        prov = np.random.choice(PROVINCES)
        all_rows.extend(generate_normal_company(tc, name, ind, prov))

    # ---- Generate Fraud Companies ----
    print(f"[2/3] Sinh {num_fraud} doanh nghiệp có dấu hiệu bất thường (fraud)...")
    for idx in range(num_normal, NUM_COMPANIES):
        tc = gen_tax_code(idx)
        name = gen_company_name()
        ind = np.random.choice(INDUSTRIES)
        prov = np.random.choice(PROVINCES)
        all_rows.extend(generate_fraud_company(tc, name, ind, prov))

    # ---- Build DataFrame ----
    print("[3/3] Xây dựng DataFrame và lưu CSV...")
    df = pd.DataFrame(all_rows)

    # Shuffle rows so fraud is not all at bottom
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Output path
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "tax_data_mock.csv")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Đã lưu: {output_path}")
    print(f"  📊 Tổng số bản ghi: {len(df):,}")
    print(f"  🏢 Số doanh nghiệp: {df['tax_code'].nunique():,}")
    print(f"  🔴 Số DN gian lận:   {df[df['fraud_label']==1]['tax_code'].nunique():,}")
    print(f"  🟢 Số DN bình thường: {df[df['fraud_label']==0]['tax_code'].nunique():,}")
    print(f"  📉 Doanh thu trung vị: {df['revenue'].median():,.0f} VND")
    print(f"  📁 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    main()
