import sys
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_research_lab import (  # noqa: E402
    build_data_quality_report,
    build_macro_research_evaluation,
    build_model_card,
    build_research_state,
    run_causal_merger_effect,
    run_forecast_research,
    run_shock_propagation,
)


def test_macro_research_state_exposes_model_cards_and_quality():
    payload = build_research_state(boundary_version="vn_34_2025")

    assert payload["boundary_version"] == "vn_34_2025"
    assert payload["data_quality"]["province_count"] == 34
    assert len(payload["model_cards"]) >= 3
    assert "fan_chart" in payload["visualization_spec"]


def test_macro_research_data_quality_supports_legacy_63():
    payload = build_data_quality_report(boundary_version="vn_63_legacy")

    assert payload["province_count"] == 63
    assert payload["expected_provinces"] == 63
    assert payload["review_policy"] == "approved_sources_only"


def test_macro_research_forecast_has_intervals_and_fingerprint():
    payload = run_forecast_research({
        "boundary_version": "vn_34_2025",
        "province_code": "VN34-HCM",
        "horizon_quarters": 8,
        "scenario_params": {"gdp_delta_pct": -2.0, "compliance_delta": -0.02},
    })

    assert payload["run_id"].startswith("forecast-")
    assert payload["province_code"] == "VN34-HCM"
    assert len(payload["forecast_points"]) == 8
    assert payload["forecast_points"][0]["lower_tax_revenue"] < payload["forecast_points"][0]["upper_tax_revenue"]
    assert payload["data_fingerprint"]


def test_macro_shock_propagation_returns_timeline_and_edges():
    payload = run_shock_propagation({
        "boundary_version": "vn_34_2025",
        "source_province_code": "VN34-HCM",
        "shock_strength_pct": -3.0,
        "horizon_quarters": 6,
    })

    assert payload["run_id"].startswith("shock-")
    assert payload["source_province_code"] == "VN34-HCM"
    assert len(payload["timeline"]) == 6
    assert payload["edge_paths"]


def test_macro_causal_merger_effect_returns_counterfactual():
    payload = run_causal_merger_effect({
        "province_code": "VN34-CM",
        "boundary_version": "vn_34_2025",
        "outcome": "grdp_billion_vnd_est",
    })

    assert payload["run_id"].startswith("causal-")
    assert payload["province_code"] == "VN34-CM"
    assert payload["actual_series"]
    assert payload["counterfactual_series"]
    assert "difference_in_differences_pct" in payload["metrics"]


def test_macro_model_card_and_evaluation_contract():
    card = build_model_card("macro-ensemble-v2")
    report = build_macro_research_evaluation(boundary_version="vn_34_2025")

    assert card["model_key"] == "macro-ensemble-v2"
    assert report["model_version"] == "macro-research-lab-v1"
    assert report["forecast_backtest"]["sample_provinces"] > 0
    assert report["ablation_plan"]


def test_macro_research_router_contracts_without_db_session():
    from app.routers.simulation import (
        MacroCausalMergerInput,
        MacroResearchForecastInput,
        MacroShockPropagationInput,
        get_macro_data_quality,
        get_macro_model_card,
        get_macro_research_state,
        run_macro_causal_merger_effect,
        run_macro_forecast,
        run_macro_shock_propagation,
    )

    state = get_macro_research_state(boundary_version="vn_34_2025")
    quality = get_macro_data_quality(boundary_version="vn_34_2025")
    card = get_macro_model_card("macro-ensemble-v2")
    forecast = run_macro_forecast(MacroResearchForecastInput(
        province_code="VN34-HCM",
        horizon_quarters=4,
        scenario_params={"gdp_delta_pct": -1.0},
    ))
    shock = run_macro_shock_propagation(MacroShockPropagationInput(
        source_province_code="VN34-HCM",
        shock_strength_pct=-2.5,
        horizon_quarters=4,
    ))
    causal = run_macro_causal_merger_effect(MacroCausalMergerInput(province_code="VN34-CM"))

    assert state["research_title"]
    assert quality["expected_provinces"] == 34
    assert card["model_key"] == "macro-ensemble-v2"
    assert len(forecast["forecast_points"]) == 4
    assert len(shock["timeline"]) == 4
    assert causal["treatment_key"] == "admin_merger_2025"
    assert "granger_causality" in forecast
    assert "structural_break" in forecast
    assert "fevd" in forecast
    assert len(forecast["granger_causality"]["matrix"]) == 5
    assert len(forecast["structural_break"]["cusum"]) > 0
    assert len(forecast["fevd"]) == 20
