from datetime import date, timedelta
import sys
from pathlib import Path
import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.routers import delinquency
from ml_engine.delinquency_model import DelinquencyPipeline


class _EmptyQuery:
    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def all(self):
        return []


class _EmptyDB:
    def query(self, *_args, **_kwargs):
        return _EmptyQuery()


class _StaticRowsQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def distinct(self):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows


class _BatchCandidateDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_args, **_kwargs):
        return _StaticRowsQuery(self._rows)


class _EarlyWarningQuery:
    def __init__(self, rows=None, scalar_value=0):
        self._rows = rows or []
        self._scalar_value = scalar_value

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar_value


class _EarlyWarningDB:
    def __init__(self, due_rows=None, filed_returns=0):
        self._due_rows = due_rows or []
        self._filed_returns = filed_returns

    def query(self, *args, **_kwargs):
        if len(args) >= 2:
            return _EarlyWarningQuery(rows=self._due_rows)
        return _EarlyWarningQuery(scalar_value=self._filed_returns)


def test_router_normalize_horizon_probs_monotonic():
    p30, p60, p90, adjusted = delinquency._normalize_horizon_probs(0.82, 0.33, 0.25)

    assert adjusted is True
    assert p30 == 0.82
    assert p60 == 0.82
    assert p90 == 0.82


def test_router_prediction_age_and_freshness():
    assert delinquency._prediction_age_days(date.today()) == 0
    assert delinquency._freshness_from_age(0) == "fresh"
    assert delinquency._freshness_from_age(10) == "aging"
    assert delinquency._freshness_from_age(90) == "stale"
    assert delinquency._freshness_from_age(None) == "unknown"


def test_baseline_prediction_no_data_contract_fields():
    result = delinquency._build_baseline_prediction(_EmptyDB(), "0101000001", "Cong ty A")

    assert result["score_source"] == "no_data"
    assert result["freshness"] == "unknown"
    assert result["prediction_age_days"] is None
    assert result["monotonic_adjusted"] is False
    assert "early_warning" in result
    assert "intervention_uplift" in result
    assert result["intervention_uplift"]["recommended_action"] == "monitor"


def test_early_warning_non_filer_priority_review():
    today = date.today()
    due_rows = [
        (today - timedelta(days=20), None),
        (today - timedelta(days=45), None),
        (today - timedelta(days=80), None),
    ]
    db = _EarlyWarningDB(due_rows=due_rows, filed_returns=0)
    summary = {
        "late_count": 3,
        "avg_days_late": 18,
    }

    warning = delinquency._build_early_warning_payload(db, "010123", summary)

    assert warning["has_warning"] is True
    assert warning["queue"] == "priority_review"
    assert warning["level"] in {"high", "critical"}
    assert "non_filer" in warning["tags"]


def test_early_warning_monitor_when_behavior_is_stable():
    today = date.today()
    due_rows = [
        (today - timedelta(days=25), today - timedelta(days=22)),
        (today - timedelta(days=70), today - timedelta(days=68)),
        (today - timedelta(days=120), today - timedelta(days=118)),
    ]
    db = _EarlyWarningDB(due_rows=due_rows, filed_returns=3)
    summary = {
        "late_count": 0,
        "avg_days_late": 0,
    }

    warning = delinquency._build_early_warning_payload(db, "010124", summary)

    assert warning["has_warning"] is False
    assert warning["queue"] == "monitor"
    assert warning["level"] == "low"


def test_intervention_uplift_escalated_for_priority_review_case():
    payload = delinquency._build_intervention_uplift_payload(
        probability=0.87,
        prob_30d=0.72,
        prob_60d=0.82,
        prob_90d=0.87,
        payment_history_summary={
            "total_periods": 8,
            "late_count": 5,
            "unpaid_count": 3,
            "avg_days_late": 22,
            "total_penalties": 24000000,
        },
        early_warning={
            "queue": "priority_review",
            "level": "critical",
            "reason": "Co nhieu ky qua han chua nop.",
        },
    )

    assert payload["recommended_action"] == "escalated_enforcement"
    assert payload["priority_score"] >= 80
    assert payload["expected_risk_reduction_pp"] > 0
    assert payload["expected_penalty_saving"] > 0
    assert payload["expected_collection_uplift"] > 0
    assert len(payload["next_steps"]) >= 2


def test_intervention_uplift_monitor_for_low_risk_case():
    payload = delinquency._build_intervention_uplift_payload(
        probability=0.08,
        prob_30d=0.05,
        prob_60d=0.06,
        prob_90d=0.08,
        payment_history_summary={
            "total_periods": 4,
            "late_count": 0,
            "unpaid_count": 0,
            "avg_days_late": 0,
            "total_penalties": 0,
        },
        early_warning={
            "queue": "monitor",
            "level": "low",
            "reason": "Hanh vi on dinh.",
        },
    )

    assert payload["recommended_action"] == "monitor"
    assert payload["priority_score"] <= 20
    assert payload["expected_risk_reduction_pp"] >= 0


def test_model_normalize_horizon_probs_monotonic():
    pipeline = DelinquencyPipeline()
    p30, p60, p90, adjusted = pipeline._normalize_horizon_probs(0.7, 0.4, 0.2)

    assert adjusted is True
    assert p30 == 0.7
    assert p60 == 0.7
    assert p90 == 0.7


def test_normalize_freshness_filter_accepts_known_values():
    assert delinquency._normalize_freshness_filter("fresh") == "fresh"
    assert delinquency._normalize_freshness_filter("AGING") == "aging"
    assert delinquency._normalize_freshness_filter(" stale ") == "stale"
    assert delinquency._normalize_freshness_filter(None) is None


def test_normalize_freshness_filter_rejects_invalid_values():
    with pytest.raises(HTTPException):
        delinquency._normalize_freshness_filter("very_old")


def test_normalize_tax_code_for_contract_accepts_10_and_legacy_9_digits():
    assert delinquency._normalize_tax_code_for_contract("0101000001") == "0101000001"
    assert delinquency._normalize_tax_code_for_contract("101000001") == "0101000001"


def test_normalize_tax_code_for_response_rejects_invalid_values():
    assert delinquency._normalize_tax_code_for_response("abc") is None
    assert delinquency._normalize_tax_code_for_response("0101") is None


def test_build_tax_code_lookup_candidates_handles_legacy_aliases():
    assert delinquency._build_tax_code_lookup_candidates("0101000001") == ["0101000001", "101000001"]
    assert delinquency._build_tax_code_lookup_candidates("101000001") == ["101000001", "0101000001"]


def test_build_batch_candidate_tax_codes_with_requested_codes():
    db = _BatchCandidateDB(rows=[("0101000001",), ("101000002",), ("INVALID",)])
    candidates = delinquency._build_batch_candidate_tax_codes(
        db,
        requested_tax_codes=["0101000001", "", "0101000002", "0101000001"],
        limit=10,
        refresh_existing=True,
    )

    assert candidates == ["0101000001", "101000002"]


def test_infer_score_source_from_model_version():
    assert delinquency._infer_score_source_from_model_version("baseline-statistical-v1") == "statistical_baseline"
    assert delinquency._infer_score_source_from_model_version("delinquency-temporal-v1") == "ml_model"
    assert delinquency._infer_score_source_from_model_version(None) == "unknown"


def test_cache_health_summary_flags_critical_coverage_and_stale_ratio():
    latest_predictions = [
        {
            "tax_code": "0101",
            "prediction_ref": date.today(),
            "model_version": "delinquency-temporal-v1",
            "prob_90d": 0.8,
        },
        {
            "tax_code": "0102",
            "prediction_ref": date.today() - timedelta(days=12),
            "model_version": "baseline-statistical-v1",
            "prob_90d": 0.4,
        },
        {
            "tax_code": "0103",
            "prediction_ref": date.today() - timedelta(days=45),
            "model_version": None,
            "prob_90d": 0.2,
        },
    ]

    summary = delinquency._build_cache_health_summary(
        latest_predictions=latest_predictions,
        total_companies_with_payments=5,
        fresh_days=7,
        stale_days=30,
        sample_stale=10,
    )

    assert summary["status"] == "critical"
    assert summary["coverage"]["coverage_ratio"] == 0.6
    assert summary["freshness"]["counts"]["stale"] == 1
    assert summary["sources"]["ml_model"] == 1
    assert summary["sources"]["statistical_baseline"] == 1
    assert summary["sources"]["unknown"] == 1
    assert summary["stale_samples"][0]["tax_code"] == "0103"


def test_cache_health_summary_no_data_status_when_no_payments_and_no_predictions():
    summary = delinquency._build_cache_health_summary(
        latest_predictions=[],
        total_companies_with_payments=0,
        fresh_days=7,
        stale_days=30,
        sample_stale=5,
    )

    assert summary["status"] == "no_data"
    assert summary["coverage"]["companies_with_payments"] == 0
    assert summary["coverage"]["companies_with_latest_prediction"] == 0
    assert any(alert["code"] == "no_tax_payment_data" for alert in summary["alerts"])


def test_cache_health_summary_rejects_invalid_threshold_order():
    with pytest.raises(ValueError):
        delinquency._build_cache_health_summary(
            latest_predictions=[],
            total_companies_with_payments=1,
            fresh_days=30,
            stale_days=7,
            sample_stale=5,
        )


def test_extract_health_threshold_alerts_filters_only_coverage_and_stale_codes():
    summary = {
        "alerts": [
            {"code": "low_coverage", "severity": "critical"},
            {"code": "stale_ratio_warning", "severity": "warning"},
            {"code": "unknown_freshness", "severity": "warning"},
            {"code": "partial_coverage", "severity": "warning"},
            {"code": "high_stale_ratio", "severity": "critical"},
        ]
    }

    result = delinquency._extract_health_threshold_alerts(summary)
    codes = {item["code"] for item in result}

    assert codes == {
        "low_coverage",
        "partial_coverage",
        "high_stale_ratio",
        "stale_ratio_warning",
    }
    assert "unknown_freshness" not in codes


def test_extract_health_threshold_alerts_handles_missing_alerts_field():
    assert delinquency._extract_health_threshold_alerts({}) == []
    assert delinquency._extract_health_threshold_alerts({"alerts": None}) == []
