"""Create tax-rate policy what-if scenarios for macro retraining.

These scenarios are not crawled news. They are controlled policy stress tests
that make tax-rate elasticity observable to the local macro-event model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"
OUTPUT = DATA_DIR / "tax_policy_scenarios.json"


SCENARIOS = [
    ("vat_cut_10_to_8_stimulus", "Giảm VAT từ 10% xuống 8% để kích cầu", -1.5, 1.0, -0.020, 0.018, -0.20, 1.2),
    ("vat_cut_10_to_5_recession", "Giảm VAT mạnh còn 5% trong suy thoái", -5.0, 1.8, -0.050, 0.025, -0.55, 2.0),
    ("vat_raise_10_to_12_budget", "Tăng VAT từ 10% lên 12% để bù hụt thu", 2.0, -0.8, 0.020, -0.012, 0.15, -1.5),
    ("vat_raise_10_to_15_shock", "Tăng VAT mạnh lên 15% gây áp lực cầu tiêu dùng", 5.0, -2.2, 0.050, -0.030, 0.65, -3.8),
    ("cit_cut_20_to_17_fdi", "Giảm thuế TNDN từ 20% xuống 17% để hút FDI", -3.0, 1.4, -0.030, 0.012, -0.35, 4.8),
    ("cit_cut_20_to_15_competition", "Giảm thuế TNDN xuống 15% trong cạnh tranh đầu tư khu vực", -5.0, 2.0, -0.050, 0.018, -0.55, 7.0),
    ("cit_raise_20_to_23_budget", "Tăng thuế TNDN lên 23% để tăng thu ngân sách", 3.0, -0.9, 0.030, -0.014, 0.22, -2.3),
    ("cit_raise_20_to_25_pressure", "Tăng thuế TNDN lên 25% làm giảm lợi nhuận và đầu tư", 5.0, -1.7, 0.050, -0.025, 0.46, -4.5),
    ("dual_vat_cit_cut_growth", "Giảm đồng thời VAT và TNDN để kích hoạt phục hồi", -4.0, 2.5, -0.040, 0.020, -0.70, 6.0),
    ("dual_vat_cit_raise_austerity", "Tăng đồng thời VAT và TNDN trong kịch bản thắt chặt tài khóa", 4.0, -2.8, 0.040, -0.035, 0.90, -6.0),
    ("targeted_vat_relief_services", "Giảm VAT có mục tiêu cho dịch vụ và du lịch", -2.0, 1.2, -0.020, 0.015, -0.25, 2.0),
    ("exporter_cit_incentive", "Ưu đãi TNDN cho doanh nghiệp xuất khẩu chịu thuế quan ngoài nước", -2.5, 0.8, -0.025, 0.012, -0.15, 3.5),
    ("green_tax_incentive", "Ưu đãi thuế cho đầu tư xanh và năng lượng tái tạo", -1.0, 0.9, -0.010, 0.010, -0.12, 2.8),
    ("real_estate_tax_tightening", "Siết ưu đãi thuế bất động sản để hạ rủi ro bong bóng", 1.5, -0.6, 0.015, 0.018, 0.25, -1.0),
    ("small_business_vat_relief", "Giảm VAT cho hộ kinh doanh và doanh nghiệp nhỏ", -1.0, 0.7, -0.010, 0.030, -0.18, 0.8),
    ("luxury_vat_surcharge", "Phụ thu VAT nhóm hàng xa xỉ", 1.0, -0.2, 0.010, 0.006, 0.05, 0.2),
    ("regional_cit_incentive_highlands", "Ưu đãi TNDN vùng Tây Nguyên và miền núi", -2.0, 1.1, -0.020, 0.014, -0.20, 2.2),
    ("tax_compliance_digital_invoice", "Mở rộng hóa đơn điện tử làm tăng tuân thủ dù không đổi thuế suất", 0.0, 0.6, 0.000, 0.040, -0.08, 0.8),
    ("vat_refund_tightening", "Siết hoàn thuế VAT để giảm gian lận nhưng tăng chi phí vốn", 0.8, -0.4, 0.008, 0.025, 0.12, -0.7),
    ("tax_holiday_phased_exit", "Rút dần ưu đãi thuế sau giai đoạn miễn giảm", 2.0, -0.7, 0.020, -0.010, 0.18, -1.8),
]


def build_scenarios() -> list[dict]:
    rows = []
    for key, name, tax_delta_pp, gdp, tax_rate_delta, compliance, unemployment, fdi in SCENARIOS:
        tax_revenue = gdp + tax_delta_pp * 1.15 + compliance * 100.0 * 0.55 - unemployment * 0.4
        rows.append({
            "event_key": f"policy_{key}",
            "event_name": name,
            "event_name_vi": name,
            "event_type": "policy",
            "severity": "medium" if abs(tax_delta_pp) <= 2.5 else "high",
            "start_date": "2026-01-01",
            "impact_gdp_pct": round(gdp, 3),
            "impact_tax_revenue_pct": round(tax_revenue, 3),
            "impact_unemployment_pct": round(unemployment, 3),
            "impact_fdi_pct": round(fdi, 3),
            "tax_rate_delta": round(tax_rate_delta, 4),
            "compliance_delta": round(compliance, 4),
            "affected_provinces": [],
            "affected_sectors": ["Chính sách thuế", "Doanh nghiệp", "Hộ kinh doanh"],
            "scope": "national",
            "description_vi": f"Kịch bản policy lab: {name}. Dùng để hiệu chỉnh độ nhạy mô hình với thay đổi thuế suất.",
            "source": "TaxInspector policy scenario lab",
            "review_status": "approved_controlled_scenario",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Write controlled tax-rate policy scenarios.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = build_scenarios()
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "path": str(args.output), "count": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

