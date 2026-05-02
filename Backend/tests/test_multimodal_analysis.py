import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.multimodal_analysis import detect_csv_schema


def test_detect_risk_scoring_csv_contract():
    content = (
        "tax_code,company_name,industry,year,revenue,total_expenses,net_profit,vat_input,vat_output\n"
        "0300000001,A,Trade,2025,1000,700,300,70,100\n"
    ).encode("utf-8")

    result = detect_csv_schema(content)

    assert result["detected_schema"] == "risk_scoring_csv"
    assert result["risk_missing"] == []


def test_detect_vat_graph_csv_contract():
    content = (
        "invoice_number,seller_tax_code,buyer_tax_code,amount,vat_rate,date\n"
        "INV1,0300000001,0300000002,1000,10,2025-01-01\n"
    ).encode("utf-8")

    result = detect_csv_schema(content)

    assert result["detected_schema"] == "vat_graph_csv"
    assert result["vat_missing"] == []


def test_detect_unknown_csv_contract():
    content = "foo,bar\n1,2\n".encode("utf-8")

    result = detect_csv_schema(content)

    assert result["detected_schema"] == "unknown_csv"
    assert "tax_code" in result["risk_missing"]
    assert "seller_tax_code" in result["vat_missing"]
