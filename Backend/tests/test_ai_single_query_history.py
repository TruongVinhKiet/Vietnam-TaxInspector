from datetime import date
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.routers import ai_analysis


class _TaxReturnRow:
    def __init__(self, year, revenue, expenses):
        self.filing_date = date(year, 12, 31)
        self.revenue = revenue
        self.expenses = expenses


class _AssessmentRow:
    def __init__(self, year=None, revenue=None, total_expenses=None, yearly_history=None):
        self.year = year
        self.revenue = revenue
        self.total_expenses = total_expenses
        self.yearly_history = yearly_history


def test_normalize_yearly_history_filters_and_sorts():
    raw = [
        {"year": 2024, "revenue": "1500", "total_expenses": "900"},
        {"year": "2022", "revenue": 1000, "total_expenses": 600},
        {"year": None, "revenue": 2000, "total_expenses": 1200},
        {"year": 2024, "revenue": 1600, "total_expenses": 950},
    ]

    normalized = ai_analysis._normalize_yearly_history(raw)

    assert [row["year"] for row in normalized] == [2022, 2024]
    assert normalized[1]["revenue"] == 1600.0
    assert normalized[1]["total_expenses"] == 950.0


def test_resolve_yearly_history_prefers_cache():
    cache_history = [{"year": 2023, "revenue": 1100, "total_expenses": 700}]
    tax_returns = [_TaxReturnRow(2022, 900, 500)]
    assessments = [_AssessmentRow(year=2021, revenue=800, total_expenses=400)]

    history, source = ai_analysis._resolve_yearly_history(cache_history, tax_returns, assessments)

    assert source == "cache"
    assert len(history) == 1
    assert history[0]["year"] == 2023


def test_resolve_yearly_history_falls_back_to_tax_returns():
    tax_returns = [
        _TaxReturnRow(2022, 1000, 650),
        _TaxReturnRow(2024, 1500, 920),
        _TaxReturnRow(2023, 1200, 710),
    ]

    history, source = ai_analysis._resolve_yearly_history([], tax_returns, [])

    assert source == "tax_returns"
    assert [row["year"] for row in history] == [2022, 2023, 2024]
    assert history[2]["revenue"] == 1500.0


def test_resolve_yearly_history_falls_back_to_assessment_rows():
    assessments = [
        _AssessmentRow(
            yearly_history=[
                {"year": 2024, "revenue": 1700, "total_expenses": 1010},
                {"year": 2023, "revenue": 1300, "total_expenses": 780},
            ]
        ),
        _AssessmentRow(year=2022, revenue=980, total_expenses=590),
    ]

    history, source = ai_analysis._resolve_yearly_history([], [], assessments)

    assert source == "assessment_history"
    assert [row["year"] for row in history] == [2022, 2023, 2024]
    assert history[0]["revenue"] == 980.0


def test_resolve_serving_model_version_prefers_explicit_value():
    resolved = ai_analysis._resolve_serving_model_version("fraud-hybrid-test")
    assert resolved == "fraud-hybrid-test"


def test_resolve_serving_model_version_reads_pipeline_metadata(monkeypatch):
    class _FakePipeline:
        def get_serving_metadata(self):
            return {"model_version": "fraud-hybrid-manifest-v9"}

    monkeypatch.setattr(ai_analysis, "get_pipeline", lambda: _FakePipeline())

    resolved = ai_analysis._resolve_serving_model_version()
    assert resolved == "fraud-hybrid-manifest-v9"


def test_resolve_serving_model_version_falls_back_to_legacy(monkeypatch):
    def _raise_pipeline_error():
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(ai_analysis, "get_pipeline", _raise_pipeline_error)

    resolved = ai_analysis._resolve_serving_model_version()
    assert resolved == "fraud-hybrid-legacy"


def test_build_company_rows_from_yearly_history_creates_inference_contract_rows():
    yearly_history = [
        {"year": 2022, "revenue": 1200, "total_expenses": 800},
        {"year": 2023, "revenue": 1500, "total_expenses": 900},
    ]

    rows = ai_analysis._build_company_rows_from_yearly_history(
        yearly_history=yearly_history,
        tax_code="01010001",
        company_name="Cong ty A",
        industry="Xay dung",
    )

    assert len(rows) == 2
    assert rows[0]["tax_code"] == "01010001"
    assert rows[1]["year"] == 2023
    assert rows[1]["net_profit"] == 600.0
    assert rows[1]["vat_output"] == 150.0


def test_extract_feature_analytics_from_rows_uses_pipeline_payload(monkeypatch):
    class _FakePipeline:
        def predict_single(self, _rows):
            return {
                "yearly_feature_scores": [{"year": 2024, "f1_divergence": 0.2}],
                "previous_year_features": {"year": 2023, "f1_divergence": 0.1},
                "feature_deltas": {"f1_divergence": 0.1},
            }

    monkeypatch.setattr(ai_analysis, "get_pipeline", lambda: _FakePipeline())

    payload = ai_analysis._extract_feature_analytics_from_rows(
        [{"tax_code": "0101", "year": 2024, "revenue": 100, "total_expenses": 50, "net_profit": 50, "vat_input": 5, "vat_output": 10}]
    )

    assert payload["yearly_feature_scores"][0]["year"] == 2024
    assert payload["previous_year_features"]["year"] == 2023
    assert payload["feature_deltas"]["f1_divergence"] == 0.1


def test_build_single_risk_tier_sankey_creates_nodes_and_links():
    yearly_scores = [
        {"year": 2022, "risk_score": 32.5},
        {"year": 2023, "risk_score": 65.0},
        {"year": 2024, "risk_score": 84.2},
    ]

    payload = ai_analysis._build_single_risk_tier_sankey(yearly_scores)

    assert len(payload["nodes"]) == 3
    assert len(payload["links"]) == 2
    assert payload["nodes"][0]["tier"] == "low"
    assert payload["nodes"][1]["tier"] == "high"
    assert payload["nodes"][2]["tier"] == "critical"


def test_build_single_cumulative_risk_curve_calculates_concentration():
    yearly_scores = [
        {"year": 2022, "risk_score": 20.0},
        {"year": 2023, "risk_score": 80.0},
    ]

    payload = ai_analysis._build_single_cumulative_risk_curve(yearly_scores)

    assert payload["total_periods"] == 2
    assert payload["total_risk"] == 100.0
    assert payload["points"][0]["percent_risk"] == 80.0
    assert payload["top_10pct_risk_share"] == 80.0


def test_build_single_red_flags_timeline_detects_feature_flags():
    yearly_scores = [
        {
            "year": 2022,
            "risk_score": 45.0,
            "f1_divergence": 0.10,
            "f2_ratio_limit": 0.80,
            "f3_vat_structure": 0.20,
            "f4_peer_comparison": 0.10,
        },
        {
            "year": 2023,
            "risk_score": 72.0,
            "f1_divergence": 0.42,
            "f2_ratio_limit": 1.35,
            "f3_vat_structure": 0.67,
            "f4_peer_comparison": 0.31,
        },
    ]

    payload = ai_analysis._build_single_red_flags_timeline(yearly_scores)

    assert len(payload["year_points"]) == 2
    assert payload["year_points"][1]["flag_count"] >= 4

    detected_flags = {item["flag_id"] for item in payload["flags"]}
    assert "risk_high" in detected_flags
    assert "f1_divergence" in detected_flags
    assert "f2_ratio_limit" in detected_flags
    assert "f3_vat_structure" in detected_flags
    assert "f4_peer_comparison" in detected_flags
