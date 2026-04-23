import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.graph_intelligence import RingScorer


def test_ring_scorer_uses_global_ring_window_for_span():
    scorer = RingScorer()
    cycles = [["A", "B", "C"]]
    invoices = [
        {"seller_tax_code": "A", "buyer_tax_code": "B", "amount": 1_000_000, "date": "2026-01-01"},
        {"seller_tax_code": "A", "buyer_tax_code": "B", "amount": 1_500_000, "date": "2026-01-10"},
        {"seller_tax_code": "B", "buyer_tax_code": "C", "amount": 2_000_000, "date": "2026-01-10"},
        {"seller_tax_code": "C", "buyer_tax_code": "A", "amount": 2_000_000, "date": "2026-01-11"},
    ]

    rings = scorer.score_rings(cycles, invoices)
    assert len(rings) == 1
    ring = rings[0]
    assert ring["start_date"] == "2026-01-01"
    assert ring["end_date"] == "2026-01-11"
    assert ring["time_span_days"] == 10
    assert ring["span_method"] == "global_ring_window"
