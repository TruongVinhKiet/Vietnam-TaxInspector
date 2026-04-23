import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.graph_feature_pipeline import GraphFeaturePipeline


def test_integrity_signal_computation():
    pipeline = GraphFeaturePipeline()
    invoices = [{"invoice_number": "INV-1"}]
    events = [
        {"invoice_id": 1, "event_type": "issued", "event_time": __import__("datetime").datetime(2024, 3, 28, 10, 0, 0)},
        {"invoice_id": 1, "event_type": "canceled", "event_time": __import__("datetime").datetime(2024, 4, 2, 12, 0, 0)},
    ]
    payments = [{"invoice_id": 1, "paid_amount": 1000.0}]
    result = pipeline._compute_integrity(invoices, events, payments, {1: "INV-1"})
    assert result["payload"]["available"] is True
    assert result["payload"]["cancel_rate"] > 0
    assert result["payload"]["cross_party_mismatch_count"] >= 1
    assert result["invoice_state"]["INV-1"] == "canceled"


def test_payment_consistency_signal():
    pipeline = GraphFeaturePipeline()
    invoices = [
        {
            "invoice_number": "INV-1",
            "seller_tax_code": "0100000001",
            "buyer_tax_code": "0100000002",
            "amount": 1000.0,
        }
    ]
    payments = [
        {
            "invoice_id": 10,
            "payer_tax_code": "0100000003",
            "payee_tax_code": "0100000001",
            "paid_amount": 200.0,
        }
    ]
    result = pipeline._compute_payment_consistency(invoices, payments, {"INV-1": 10})
    assert result["payload"]["available"] is True
    assert result["payload"]["invoice_payment_match_rate"] <= 1.0
    assert result["mismatch_count"] >= 1
