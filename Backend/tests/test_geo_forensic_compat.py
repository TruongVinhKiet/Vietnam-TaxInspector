import pytest
from app.routers.graph import _normalize_country, _is_high_risk_country, _classify_zero_semantics, _build_cross_border_signals

def test_normalize_country():
    assert _normalize_country("vn") == "Vietnam"
    assert _normalize_country("việt nam") == "Vietnam"
    assert _normalize_country("  SINGAPORE  ") == "SINGAPORE"
    assert _normalize_country(None) == ""
    assert _normalize_country("") == ""

def test_is_high_risk_country():
    assert _is_high_risk_country("Cayman Islands") is True
    assert _is_high_risk_country("British Virgin Islands") is True
    assert _is_high_risk_country("Vietnam") is False
    assert _is_high_risk_country("Unknown") is False

def test_classify_zero_semantics():
    res1 = _classify_zero_semantics("ok", "ring_count", 5)
    assert res1["semantic_meaning"] == "observed_non_zero"

    res2 = _classify_zero_semantics("no_invoice_context", "ring_count", 0)
    assert res2["semantic_meaning"] == "zero_due_to_missing_coverage"

    res3 = _classify_zero_semantics("ok", "ring_count", 0, coverage_value=0.0)
    assert res3["semantic_meaning"] == "zero_with_low_coverage"
    assert res3["ambiguous"] is True

    res4 = _classify_zero_semantics("ok", "ring_count", 0, coverage_value=1.0)
    assert res4["semantic_meaning"] == "true_zero_observed"

def test_build_cross_border_signals():
    companies = [
        {"tax_code": "A", "country_inferred": "Vietnam", "is_within_vietnam": True},
        {"tax_code": "B", "country_inferred": "Singapore", "is_within_vietnam": False},
        {"tax_code": "C", "country_inferred": "Cayman", "is_within_vietnam": False},
        {"tax_code": "D"}
    ]
    invoices = [
        {"seller_tax_code": "A", "buyer_tax_code": "B"}, # Cross-border
        {"seller_tax_code": "A", "buyer_tax_code": "C"}, # Cross-border, high risk
        {"seller_tax_code": "B", "buyer_tax_code": "D"}, # Cross-border (D is unknown)
        {"seller_tax_code": "A", "buyer_tax_code": "A"}  # Domestic
    ]
    
    signals = _build_cross_border_signals(companies, invoices)
    assert signals["available"] is True
    assert signals["scope_companies_total"] == 4
    assert signals["companies_with_country"] == 3
    assert signals["companies_unknown_country"] == 1
    assert signals["companies_within_vietnam"] == 1
    assert signals["companies_outside_vietnam"] == 2
    assert signals["cross_border_invoice_count"] == 2
    assert signals["high_risk_country_exposure"]["count"] == 1
    assert "Cayman Islands" in signals["high_risk_country_exposure"]["countries"]
    assert signals["risk_level"] in ["medium", "high", "critical"]
