import sys
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_scenario_engine import (  # noqa: E402
    ScenarioParams,
    build_vietnam_geojson,
    compute_scenario,
    get_events_for_province,
    load_events,
    load_provinces,
)
from ml_engine.admin_boundary_manager import audit_boundary_readiness, load_boundary_geojson  # noqa: E402
from ml_engine.macro_event_ingest import MacroEventCandidate, ingest_macro_event_candidates  # noqa: E402
import ml_engine.macro_scenario_llm as macro_scenario_llm  # noqa: E402
from ml_engine.macro_retrain_pipeline import (  # noqa: E402
    build_event_training_matrix,
    collect_training_sources,
    predict_scenario_from_trained_model,
    run_retrain,
)


def test_macro_digital_twin_assets_cover_vn34_provinces_and_events():
    provinces = load_provinces("vn_34_2025")
    events = load_events()
    geojson = build_vietnam_geojson("vn_34_2025")

    assert len(provinces) == 34
    assert len(events) >= 100
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == len(provinces)
    assert geojson["metadata"]["boundary_status"] == "official_or_reviewed_geojson"


def test_no_change_scenario_is_anchored_to_observed_revenue():
    for province_code in ["01", "79", "48", "92"]:
        result = compute_scenario(province_code, ScenarioParams())
        assert abs(result.delta_revenue_pct) <= 0.01
        assert result.uncertainty_band_revenue["low"] <= result.projected_revenue
        assert result.uncertainty_band_revenue["high"] >= result.projected_revenue


def test_province_event_relevance_returns_ranked_context():
    events = get_events_for_province("79", limit=20)

    assert len(events) == 20
    assert all("relevance_score" in event for event in events)
    assert events == sorted(
        events,
        key=lambda event: event.get("relevance_score", 0.0),
        reverse=True,
    )


def test_boundary_versions_expose_2025_production_target():
    readiness = audit_boundary_readiness()
    assert readiness["active_version"] == "vn_34_2025"
    assert readiness["production_target_version"] == "vn_34_2025"
    assert "vn_34_2025" in readiness["versions"]

    target = load_boundary_geojson("vn_34_2025")
    assert target["metadata"]["expected_unit_count"] == 34
    assert target["metadata"]["boundary_status"] == "official_or_reviewed_geojson"
    assert len(target["features"]) == 34


def test_vn34_aggregate_profile_can_run_scenario():
    result = compute_scenario("VN34-HCM", ScenarioParams(gdp_delta_pct=-2.0, tax_rate_delta=0.01))
    assert result.province_name == "Hồ Chí Minh"
    assert result.delta_gdp_pct == -2.0


def test_simulation_map_state_contract_uses_vn34():
    from app.routers.simulation import get_map_state

    payload = get_map_state(boundary_version="vn_34_2025", include_geojson=True, event_limit=2)
    assert payload["boundary_version"] == "vn_34_2025"
    assert payload["total"] == 34
    assert payload["data_quality"]["profile_polygon_coverage_ok"] is True
    assert len(payload["geojson"]["features"]) == 34
    assert all("risk_score" in province for province in payload["provinces"])


def test_macro_event_ingest_deduplicates_review_queue(tmp_path):
    candidate = MacroEventCandidate(
        title="Xuất khẩu Việt Nam tăng mạnh nhờ đơn hàng điện tử",
        description="Nguồn tin mô tả tác động tích cực tới sản xuất và thu ngân sách.",
        source_name="unit-test",
        source_url="https://example.test/macro/1",
        published_at="2026-05-18T00:00:00Z",
        event_type="growth",
    )
    queue_path = tmp_path / "queue.jsonl"

    first = ingest_macro_event_candidates([candidate], queue_path=queue_path)
    second = ingest_macro_event_candidates([candidate], queue_path=queue_path)

    assert first["queued"] == 1
    assert second["duplicates"] == 1
    assert queue_path.exists()
    assert len(queue_path.read_text(encoding="utf-8").splitlines()) == 1


def test_text_scenario_interpreter_uses_hitl_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(macro_scenario_llm, "MEMORY_PATH", tmp_path / "memory.jsonl")
    for key in ["OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "COHERE_API_KEY", "GITHUB_TOKEN", "GITHUB_MODELS_TOKEN"]:
        monkeypatch.delenv(key, raising=False)

    first = macro_scenario_llm.interpret_text_scenario(
        "Mỹ đánh thuế hàng hóa Việt Nam 30%",
        force_llm=True,
    )
    assert first["source"] == "rule_fallback"
    assert first["macro_parameters"]["gdp_delta_pct"] < 0

    macro_scenario_llm.remember_scenario_feedback(
        text="Mỹ đánh thuế hàng hóa Việt Nam 30%",
        payload=first,
        rating=5,
        approved=True,
    )
    second = macro_scenario_llm.interpret_text_scenario(
        "My danh thue hang hoa Viet Nam 40%",
        force_llm=False,
    )
    assert second["source"] == "memory"
    assert second["memory_similarity"] >= 0.86


def test_macro_retrain_pipeline_builds_reviewed_training_dataset():
    source_rows, counts = collect_training_sources()
    assert counts["canonical_events"] >= 100
    assert source_rows

    import numpy as np

    X, y, meta = build_event_training_matrix(source_rows[:12], min_samples=80, rng=np.random.default_rng(42))
    assert X.shape[0] == 80
    assert y.shape[0] == 80
    assert len(meta) == 80
    assert y.shape[1] >= 5


def test_macro_retrain_pipeline_writes_local_model_artifacts(tmp_path, monkeypatch):
    import ml_engine.macro_retrain_pipeline as retrain

    monkeypatch.setattr(retrain, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(retrain, "EVENT_MODEL_PATH", tmp_path / "macro_event_impact_model.joblib")
    monkeypatch.setattr(retrain, "PROVINCE_MODEL_PATH", tmp_path / "macro_province_response_model.joblib")
    monkeypatch.setattr(retrain, "REPORT_PATH", tmp_path / "macro_retrain_report.json")
    monkeypatch.setattr(retrain, "DATASET_PREVIEW_PATH", tmp_path / "macro_retrain_dataset_preview.jsonl")

    report = run_retrain(min_samples=220, seed=7)
    assert report["status"] == "trained"
    assert (tmp_path / "macro_event_impact_model.joblib").exists()
    assert (tmp_path / "macro_province_response_model.joblib").exists()

    predicted = predict_scenario_from_trained_model("Mỹ áp thuế 35% lên hàng xuất khẩu Việt Nam")
    assert predicted is not None
    assert predicted["macro_parameters"]["gdp_delta_pct"] < 5
    assert predicted["llm_provider"] == "local_reviewed_model"
