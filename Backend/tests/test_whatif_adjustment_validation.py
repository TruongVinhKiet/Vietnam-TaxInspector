import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.routers import ai_analysis


def test_validate_whatif_adjustments_normalizes_payload():
    payload = {
        "revenue": "-20",
        "total_expenses": 35.678,
    }

    normalized = ai_analysis._validate_whatif_adjustments(payload)

    assert normalized["revenue"] == -20.0
    assert normalized["total_expenses"] == 35.68


def test_validate_whatif_adjustments_rejects_unsupported_fields():
    payload = {
        "revenue": -10,
        "vat_output": 20,
    }

    with pytest.raises(ValueError, match="Chỉ hỗ trợ điều chỉnh"):
        ai_analysis._validate_whatif_adjustments(payload)


def test_validate_whatif_adjustments_rejects_out_of_range_values():
    payload = {"revenue": -99}

    with pytest.raises(ValueError, match="phải nằm trong khoảng"):
        ai_analysis._validate_whatif_adjustments(payload)


def test_validate_whatif_adjustments_rejects_non_object_payload():
    with pytest.raises(ValueError, match="object JSON"):
        ai_analysis._validate_whatif_adjustments([1, 2, 3])


def test_normalize_whatif_step_list_uses_default_when_missing():
    normalized = ai_analysis._normalize_whatif_step_list(
        None,
        field_name="revenue",
        default_steps=[-30, -20, -10, 0, 10],
    )

    assert normalized == [-30.0, -20.0, -10.0, 0.0, 10.0]


def test_normalize_whatif_step_list_rejects_invalid_value_type():
    with pytest.raises(ValueError, match="phai la so phan tram hop le"):
        ai_analysis._normalize_whatif_step_list(
            ["abc", 10],
            field_name="expense",
            default_steps=[0],
        )


def test_build_whatif_heatmap_values_generates_matrix_ordered_by_expense_then_revenue():
    class _FakePipeline:
        def predict_whatif(self, _base_data, adjustments):
            rev = float(adjustments.get("revenue", 0.0))
            exp = float(adjustments.get("total_expenses", 0.0))
            # deterministic pseudo-model for testing matrix structure
            return {"simulated_risk_score": 50.0 + (0.2 * exp) - (0.1 * rev)}

    values = ai_analysis._build_whatif_heatmap_values(
        pipeline=_FakePipeline(),
        base_data={"tax_code": "0101"},
        original_risk_score=50.0,
        revenue_steps=[-10.0, 0.0, 10.0],
        expense_steps=[10.0, 0.0],
    )

    assert len(values) == 6
    # First cell: expense=10, revenue=-10 => 50 + 2 + 1 = 53
    assert values[0] == [0, 0, 53.0, 3.0]
    # Last cell: expense=0, revenue=10 => 50 - 1 = 49
    assert values[-1] == [2, 1, 49.0, -1.0]


def test_what_if_grid_returns_backend_matrix(monkeypatch):
    class _CachedAssessment:
        tax_code = "01010001"
        company_name = "Cong ty A"
        industry = "Thuong mai"
        year = 2024
        revenue = 1_000_000
        total_expenses = 700_000
        risk_score = 60.0
        risk_level = "high"

    class _FakePipeline:
        def predict_whatif(self, _base_data, adjustments):
            rev = float(adjustments.get("revenue", 0.0))
            exp = float(adjustments.get("total_expenses", 0.0))
            return {"simulated_risk_score": 60.0 + (0.1 * exp) - (0.05 * rev)}

    monkeypatch.setattr(
        ai_analysis,
        "_get_latest_cached_assessment",
        lambda _tax_code, _db: _CachedAssessment(),
    )
    monkeypatch.setattr(ai_analysis, "get_pipeline", lambda: _FakePipeline())

    payload = {
        "revenue_steps": [-10, 0, 10],
        "expense_steps": [10, 0, -10],
    }

    result = ai_analysis.what_if_grid("01010001", payload=payload, db=object())

    assert result["source"] == "what_if_backend"
    assert result["grid_size"]["rows"] == 3
    assert result["grid_size"]["cols"] == 3
    assert result["grid_size"]["points"] == 9
    assert len(result["values"]) == 9
    assert result["values"][0][:2] == [0, 0]
